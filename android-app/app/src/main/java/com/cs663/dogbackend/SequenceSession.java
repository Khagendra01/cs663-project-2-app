package com.cs663.dogbackend;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;

// Java port of the Python temporal state machine that is the source of
// truth for Project 2's video layer:
//   files/video-sequence-layer/state_machine.py
// (Python version passes 17/17 unit tests in test_state_machine.py.)
//
// Same logic, same names, same verdicts, same reason strings (adapted
// only where Android has no confidence score - see below):
//
//   A command interval starts explicitly (Start Sequence). A frame-level
//   classifier emits one posture label per sampled frame, and this
//   machine turns the SEQUENCE into one verdict:
//     SUCCESS      target posture reached and held for the target duration
//     EARLY_BREAK  target held, then left (sustained, after smoothing)
//                  before the target duration was reached
//     UNCERTAIN    target never established, too many unclear frames,
//                  the dog was lost for too long, or the interval ended
//                  before the target duration
//
// Smoothing is a majority vote over a sliding window of raw labels, so
// one wrong frame never flips a verdict, and a break must be sustained.
//
// Updated proposal (Oct 26, 2026) rules implemented here:
//   - Two command shapes. HOLD commands (Stay / Stand / Walk): reach
//     the target (settle grace 2 frames) and hold it for the full
//     target duration. REACH-THEN-HOLD commands (Sit / Down): the dog
//     must REACH the target within a 3.0s response window (RESPONSE
//     phase), then HOLD it for 0.6s (HOLD phase). If the window
//     expires unreached, the verdict is EARLY_BREAK with held 0s -
//     the dog failed the command, it did not merely run out of time.
//   - Break timestamp = the FIRST raw observation of the wrong
//     posture, not the confirmation frame (proposal 5.3). The smoother
//     lags the raw stream, so the break is timestamped when the dog
//     actually moved, and held time is measured to that frame. The
//     Python reference implements the same rule; a single wrong raw
//     frame the smoother votes down is forgotten as soon as a raw
//     frame shows the dog back in the target posture.
//   - Lost / unclear: 2 consecutive no-label frames end the attempt
//     as UNCERTAIN (proposal 5.3: lost/occluded for >1.0s).
//
// 1 FPS MAPPING (SequenceActivity samples one frame per second):
// the proposal assumes ~5 fps analysis; the live loop cannot do that
// because each frame is a cloud call (~1s+ round trip), so the app
// samples at 1 fps and every proposal time is mapped to frames:
//   smooth window 3 frames   (sequence_evaluate.py runs the Python with
//                             smooth_window=3; the Python dataclass
//                             default of 5, and the proposal's 5, would
//                             lag half of a 5s stay at 1 fps -
//                             deliberately NOT changed to 5)
//   settle grace 2           target must appear within the first 2 frames
//   response window 3        Sit/Down: 3 frames to reach the target
//   hold after reach 1       Sit/Down: 1 more frame (~0.6s at 5 fps)
//   break confirm 1          1 further smoothed non-target frame confirms
//                            a break that started 1 frame ago (same 1s
//                            rule: break time is when it started)
//   lost frames 2            2 no-label frames in a row -> UNCERTAIN
//   max uncertain ratio 0.4  checked in finish(), same as Python
//
// One real difference from Python: the Gemini endpoints return only a
// text label, no confidence, so min_confidence never triggers here
// and the proposal's "<0.65 confidence -> Undefined" rule cannot be
// applied - there is no score to threshold (see parseLabel() in
// SequenceActivity). A frame whose reply cannot be parsed, or whose
// network call failed, is fed as a no-label frame - exactly how Python
// treats a frame with no dog / an unknown label (UNCERTAIN_FRAME).
public class SequenceSession {

    public static final String UNCERTAIN_FRAME = "uncertain_frame";

    // canonical_label() from the Python file: aliases -> canonical label,
    // anything unknown -> UNCERTAIN_FRAME.
    public static String canonicalLabel(String raw) {
        if (raw == null) return UNCERTAIN_FRAME;
        switch (raw.trim().toLowerCase(Locale.US)) {
            case "sit":
            case "sitting":
                return "sitting";
            case "stand":
            case "standing":
                return "standing";
            case "down":
            case "lie":
            case "lying":
            case "lying down":
            case "lying_down":
                return "lying_down";
            case "walk":
            case "walking":
                return "walking";
            default: // "transition", "undefined", "none", "", anything else
                return UNCERTAIN_FRAME;
        }
    }

    // SessionConfig from the Python file, with the values above, plus
    // the updated proposal's response-window shape (Sit/Down only).
    public static class Config {
        public double targetDurationS;
        public double minConfidence = 0.6;   // unused here (no scores)
        public int smoothWindow = 3;         // frames (see header)
        public double settleGraceS = 2.0;    // frames at 1 fps
        public double breakConfirmS = 1.0;   // frames at 1 fps
        public int lostFrames = 2;           // consecutive no-label frames
                                             // -> UNCERTAIN (see header)
        public double maxUncertainRatio = 0.4;
        // Sit/Down shape (proposal 5.2): reach the target within
        // responseWindowS, then hold it for holdAfterReachS. At 1 fps
        // the 0.6s hold is 1 frame.
        public boolean reachThenHold = false;
        public double responseWindowS = 3.0; // frames at 1 fps
        public double holdAfterReachS = 1.0; // frames at 1 fps

        public Config(double targetDurationS) {
            this.targetDurationS = targetDurationS;
        }
    }

    // Verdict dataclass from the Python file.
    public static class Verdict {
        public String verdict;            // SUCCESS | EARLY_BREAK | UNCERTAIN
        public String reason;
        public String targetLabel;
        public double targetDurationS;
        public Double targetReachedAtS;   // null if never reached
        public double heldS;
        public Double breakAtS;           // null if no break
        public int nFrames;
        public double uncertainRatio;
        public List<String> smoothedSequence = new ArrayList<>(); // "t label"
    }

    // TemporalSmoother from the Python file: majority vote over the last
    // `window` usable labels. Uncertain frames return UNCERTAIN_FRAME and
    // do not enter the window.
    static class TemporalSmoother {
        private final int window;
        private final double minConfidence;
        private final Deque<String> buf = new ArrayDeque<>();

        TemporalSmoother(int window, double minConfidence) {
            this.window = Math.max(1, window);
            this.minConfidence = minConfidence;
        }

        String push(String rawLabel, Double confidence, boolean dogPresent) {
            String label = canonicalLabel(rawLabel);
            if (!dogPresent || label.equals(UNCERTAIN_FRAME)
                    || (confidence != null && confidence < minConfidence)) {
                return UNCERTAIN_FRAME;
            }
            buf.addLast(label);
            while (buf.size() > window) buf.removeFirst();

            Map<String, Integer> counts = new HashMap<>();
            for (String x : buf) {
                Integer c = counts.get(x);
                counts.put(x, c == null ? 1 : c + 1);
            }
            // Most common label; ties go to the most recent (the Python
            // max() walks the buffer newest-first and keeps the first
            // maximum it sees, so we do the same).
            String best = null;
            int bestCount = -1;
            Iterator<String> it = ((ArrayDeque<String>) buf).descendingIterator();
            while (it.hasNext()) {
                String x = it.next();
                int c = counts.get(x);
                if (c > bestCount) {
                    bestCount = c;
                    best = x;
                }
            }
            return best;
        }
    }

    // CommandSession from the Python file.
    private final String target;
    private final Config cfg;
    private final String command;
    private final TemporalSmoother smoother;

    private Verdict verdict = null;
    private Double reachedAt = null;
    private Double breakSince = null;
    // t of the first RAW (unsmoothed) wrong-posture observation while
    // holding; becomes breakAtS if the break confirms (proposal 5.3).
    // Reset only when a raw frame shows the dog back in the target -
    // never on a smoothed target alone, because the smoother lags the
    // raw stream and would erase the break's true start time.
    private Double pendingBreakStart = null;
    private int consecutiveUncertain = 0;
    private Double lastT = null;
    private int n = 0;
    private int nUncertain = 0;
    private final List<String> seq = new ArrayList<>();
    private String lastSmoothed = null;

    public SequenceSession(String targetLabel, Config config, String command) {
        this.target = canonicalLabel(targetLabel);
        if (this.target.equals(UNCERTAIN_FRAME)) {
            throw new IllegalArgumentException("unknown target label: " + targetLabel);
        }
        this.cfg = config;
        this.command = command;
        this.smoother = new TemporalSmoother(cfg.smoothWindow, cfg.minConfidence);
    }

    // Feed one sampled frame. t is the frame's time in the sequence
    // (at 1 fps: t = frame index). label == null means no usable label
    // (unparseable reply or network error). Returns a Verdict as soon as
    // one is decided, otherwise null - same contract as Python feed().
    public Verdict feed(double t, String label) {
        return feed(t, label, null, label != null);
    }

    // Full Python feed() signature: (t, label, confidence, dogPresent).
    public Verdict feed(double t, String label, Double confidence, boolean dogPresent) {
        if (verdict != null) return verdict;
        n++;
        lastT = t;
        // The raw label, if it is a usable observation at all - this is
        // what the break timestamp is taken from (proposal 5.3), before
        // smoothing delays it.
        String raw = canonicalLabel(label);
        if (!dogPresent || raw.equals(UNCERTAIN_FRAME)
                || (confidence != null && confidence < cfg.minConfidence)) {
            raw = UNCERTAIN_FRAME;
        }
        String smoothed = smoother.push(label, confidence, dogPresent);
        lastSmoothed = smoothed;
        if (smoothed.equals(UNCERTAIN_FRAME)) nUncertain++;
        seq.add(String.format(Locale.US, "%.0f %s", t, smoothed));

        // Lost / unclear for too long -> UNCERTAIN. Proposal 5.3:
        // lost/occluded for >1.0s; at 1 fps that is 2 consecutive
        // no-label frames. Applies in any phase - an attempt that
        // cannot be seen cannot be judged.
        if (smoothed.equals(UNCERTAIN_FRAME)) {
            consecutiveUncertain++;
            if (consecutiveUncertain >= cfg.lostFrames) {
                return make("UNCERTAIN",
                        "dog lost / unclear for " + cfg.lostFrames
                                + " frames in a row - cannot judge the attempt",
                        null, null);
            }
        } else {
            consecutiveUncertain = 0;
        }

        // First raw sighting of a wrong posture during the hold.
        // (Before the target is reached the dog is simply not there
        // yet, so nothing is recorded.)
        if (reachedAt != null && !raw.equals(UNCERTAIN_FRAME)
                && !raw.equals(target) && pendingBreakStart == null) {
            pendingBreakStart = t;
        }

        if (reachedAt == null) {
            if (smoothed.equals(target)) {
                reachedAt = t;
            } else if (cfg.reachThenHold) {
                // RESPONSE phase (Sit/Down, proposal 5.2): the dog must
                // reach the target within the response window. t is the
                // frame index at 1 fps, so t >= 3 means the 3 frames
                // t=0,1,2 passed without it. Missing the window is a
                // failed command, hence EARLY_BREAK with held 0s -
                // not UNCERTAIN.
                if (t >= cfg.responseWindowS) {
                    return make("EARLY_BREAK",
                            "target posture '" + target + "' not reached within the "
                                    + fmt0(cfg.responseWindowS) + "s response window",
                            null, 0.0);
                }
            } else if (t >= cfg.settleGraceS) {
                return make("UNCERTAIN",
                        "target posture '" + target + "' never established within the "
                                + fmt0(cfg.settleGraceS) + "s settle grace", null, null);
            }
            return null;
        }

        // Tracking the hold.
        if (smoothed.equals(target)) {
            breakSince = null;
            // A wrong raw frame the smoother voted down is a glitch:
            // forget it only once a RAW frame shows the dog back in the
            // target posture (see the pendingBreakStart comment above).
            if (raw.equals(target)) {
                pendingBreakStart = null;
            }
            double holdNeeded = holdRequiredS();
            if (t - reachedAt >= holdNeeded) {
                if (cfg.reachThenHold) {
                    return make("SUCCESS",
                            "reached '" + target + "' within the "
                                    + fmt0(cfg.responseWindowS) + "s response window and held it "
                                    + "for the 0.6s confirmation hold (1 frame at 1 fps)",
                            null, null);
                }
                return make("SUCCESS",
                        "held '" + target + "' for the full " + fmt0(cfg.targetDurationS)
                                + "s target duration", null, null);
            }
            return null;
        }

        if (!smoothed.equals(UNCERTAIN_FRAME)) {
            if (breakSince == null) breakSince = t;
            if (t - breakSince >= cfg.breakConfirmS) {
                // Break timestamp = first raw sighting of the wrong
                // posture (proposal 5.3), not the confirmation frame.
                double breakAt = pendingBreakStart != null
                        ? pendingBreakStart : breakSince;
                if (cfg.reachThenHold) {
                    return make("EARLY_BREAK",
                            "reached '" + target + "' but left it after holding "
                                    + fmt1(breakAt - reachedAt) + "s of the 0.6s "
                                    + "confirmation hold",
                            breakAt, breakAt - reachedAt);
                }
                return make("EARLY_BREAK",
                        "left '" + target + "' after holding "
                                + fmt1(breakAt - reachedAt) + "s of the "
                                + fmt0(cfg.targetDurationS) + "s target",
                        breakAt, breakAt - reachedAt);
            }
        }
        return null;
    }

    // How long the target must be held once reached: the full target
    // duration for HOLD commands, the 0.6s confirmation hold (1 frame
    // at 1 fps) for REACH-THEN-HOLD commands.
    public double holdRequiredS() {
        return cfg.reachThenHold ? cfg.holdAfterReachS : cfg.targetDurationS;
    }

    // Current phase, for the live UI: RESPONSE (Sit/Down, target not
    // yet reached) or HOLD (target reached, confirming the hold).
    public String getPhase() {
        if (verdict != null) return "DONE";
        if (!cfg.reachThenHold) return "HOLD";
        return reachedAt == null ? "RESPONSE" : "HOLD";
    }

    public boolean isReachThenHold() {
        return cfg.reachThenHold;
    }

    // finish() from the Python file: verdict for an interval that ended
    // (STOP tapped / stream over) with no decision yet.
    public Verdict finish() {
        if (verdict != null) return verdict;
        if (n > 0 && (double) nUncertain / n > cfg.maxUncertainRatio) {
            return make("UNCERTAIN",
                    nUncertain + "/" + n + " frames had no usable label (network "
                            + "error or unclear reply) - too unclear to judge",
                    null, null);
        }
        if (reachedAt == null) {
            return make("UNCERTAIN",
                    "target posture '" + target + "' was never established",
                    null, null);
        }
        if (cfg.reachThenHold) {
            return make("UNCERTAIN",
                    "interval ended after holding " + fmt1(lastT - reachedAt)
                            + "s of the 0.6s confirmation hold - "
                            + "incomplete, not a success and not a break", null, null);
        }
        return make("UNCERTAIN",
                "interval ended after holding " + fmt1(lastT - reachedAt) + "s of the "
                        + fmt0(cfg.targetDurationS) + "s target duration - "
                        + "incomplete, not a success and not a break", null, null);
    }

    private Verdict make(String verdictText, String reason, Double breakAt, Double held) {
        Verdict v = new Verdict();
        v.verdict = verdictText;
        v.reason = reason;
        v.targetLabel = target;
        v.targetDurationS = cfg.targetDurationS;
        v.targetReachedAtS = reachedAt;
        // Python _make(): held defaults to lastT - reachedAt when a hold
        // was in progress, else the explicit value, else 0.
        if (held == null && reachedAt != null && lastT != null) {
            v.heldS = lastT - reachedAt;
        } else {
            v.heldS = held == null ? 0.0 : held;
        }
        v.breakAtS = breakAt;
        v.nFrames = n;
        v.uncertainRatio = n == 0 ? 0.0 : (double) nUncertain / n;
        v.smoothedSequence = new ArrayList<>(seq);
        this.verdict = v;
        return v;
    }

    // Python formats durations with %.0f and held times with %.1f.
    private static String fmt0(double x) {
        return String.format(Locale.US, "%.0f", x);
    }

    private static String fmt1(double x) {
        return String.format(Locale.US, "%.1f", x);
    }

    // Small getters so the live UI can show progress without reaching
    // into the machine's state.
    public String getTargetLabel() {
        return target;
    }

    public String getCommand() {
        return command;
    }

    public String getLastSmoothed() {
        return lastSmoothed;
    }

    public Double getReachedAt() {
        return reachedAt;
    }

    public Double getLastT() {
        return lastT;
    }

    public int getFrameCount() {
        return n;
    }

    public boolean isDecided() {
        return verdict != null;
    }
}
