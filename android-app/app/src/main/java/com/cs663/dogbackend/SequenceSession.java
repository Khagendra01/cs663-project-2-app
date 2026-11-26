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
//   cs663-project-2-vision/video-sequence-layer/state_machine.py
// (Python version passes 17/17 unit tests in test_state_machine.py;
// the Java port has its own JUnit tests in app/src/test.)
//
// Same logic, same names, same verdicts:
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
//     the target (settle grace 2 s) and hold it for the full
//     target duration. REACH-THEN-HOLD commands (Sit / Down): the dog
//     must REACH the target within a 3.0s response window (RESPONSE
//     phase), then HOLD it for 0.6s (HOLD phase). If the window
//     expires unreached, the verdict is EARLY_BREAK with held 0s -
//     the dog failed the command, it did not merely run out of time -
//     UNLESS the dog was unclear / not visible at the deadline (or within
//     breakConfirmS of it): then nobody can say whether it responded, so
//     the verdict is UNCERTAIN ("dog not clearly visible").
//   - Both ends of the hold are on the RAW time base. The smoother lags
//     the raw stream by ~3 frames, so:
//       reach time = first raw target frame of the run the smoother then
//                    confirmed (not the lagged smoothed flip). A run that
//                    started inside the window but is confirmed up to
//                    confirmGraceS after it still counts as in time.
//       break time = first frame after the LAST raw target frame, i.e.
//                    the first wrong raw frame, or the start of the
//                    unclear gap before it (proposal 5.3: timestamped
//                    when the dog actually moved, not when the smoother
//                    confirmed). The hold is only proven up to the last
//                    frame that actually showed the target posture.
//     A single wrong raw frame the smoother votes down is forgotten as
//     soon as a raw frame shows the dog back in the target posture.
//   - SUCCESS needs the smoothed AND the raw label to be the target on
//     the deciding frame, so a stale smoother buffer (e.g. right after an
//     unclear gap) cannot declare a hold the dog has already broken.
//   - Lost / unclear: no dog or no confident label for 1.5 s ends the
//     attempt as UNCERTAIN (proposal 5.3: lost/occluded for >1.0s) -
//     unless the wrong posture was already seen (smoothed, or on >= 2
//     CONSECUTIVE raw frames that are the last usable frames before the
//     unclear run) after the target was reached: then the dog left the
//     posture and then the view, which is EARLY_BREAK at that first
//     wrong frame.
//
// TIME BASE (v2, Nov 2026): the app now classifies frames ON THE PHONE
// (Model C dog detector -> Model A posture classifier, ~5 fps), so t is
// real elapsed wall-clock time in seconds since START, not a frame index.
// Every proposal time is used as-is:
//   smooth window 5 frames   (~1 s at 5 fps; proposal 5.3)
//   min confidence 0.65      Model A softmax below this -> unclear frame
//                            (proposal 5.3 "Undefined" rule, now possible
//                            because Model A returns a probability)
//   settle grace 2.0 s       HOLD commands: target must appear quickly
//   response window 3.0 s    Sit/Down: reach the target within 3 s
//   confirm grace 1.0 s      smoother may confirm an in-window reach
//                            this long after the deadline
//   hold after reach 0.6 s   Sit/Down: then hold it 0.6 s
//   break confirm 0.6 s      a smoothed non-target run this long confirms
//                            a break (timestamped at its first raw frame)
//   lost 1.5 s               unclear / no dog for this long -> UNCERTAIN
//                            (proposal says >1.0 s; 1.5 s leaves room for
//                            a few low-confidence frames at 5 fps)
//   max uncertain ratio 0.4  checked in finish(), same as Python
//
// Python reference (state_machine.py) uses the same time-based rules
// with uncertain_timeout_s for the lost rule.
public class SequenceSession {

    public Config getConfig() {
        return cfg;
    }

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
        public double minConfidence = 0.65;  // Model A softmax threshold
        public int smoothWindow = 5;         // frames (~1 s at 5 fps)
        public double settleGraceS = 2.0;    // seconds
        public double breakConfirmS = 0.6;   // seconds
        public double lostS = 1.5;           // seconds unclear -> UNCERTAIN
        public double maxUncertainRatio = 0.4;
        // Sit/Down shape (proposal 5.2): reach the target within
        // responseWindowS, then hold it for holdAfterReachS.
        public boolean reachThenHold = false;
        public double responseWindowS = 3.0; // seconds
        public double holdAfterReachS = 0.6; // seconds
        // A raw target run that began inside the response window (or the
        // settle grace) may be confirmed by the smoother up to this long
        // after the deadline. The 5-frame smoother needs 3 target frames
        // to flip: ~0.4 s at 5 fps but ~1 s on a 3 fps phone, so without
        // this grace a dog that responded at 2.6 s could fail a 3 s window.
        public double confirmGraceS = 1.0;   // seconds

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
    // Length of the current run of CONSECUTIVE usable wrong raw frames
    // while holding. A raw target frame resets it to 0, and an unclear
    // frame ends the run: the next wrong frame starts a new run at 1.
    // Two or more in a row as the last usable frames before the dog
    // vanishes count as a break (one is just as likely a glitch on the way
    // out of view). It must be consecutive: sporadic misreads
    // with unclear frames between them (Model A flickering on a dog that
    // never moved), followed by an occlusion, are not evidence of a break.
    private int wrongRawStreak = 0;
    // Whether the previous frame had no usable label (ends a wrong run).
    private boolean prevRawUnclear = false;
    // t of the first RAW target frame of the current run while the
    // target is not yet reached; becomes reachedAt when the smoother
    // confirms. Unclear frames leave it alone, a usable non-target frame
    // resets it.
    private Double pendingReachStart = null;
    // Raw target frames in that pending run (unclear frames between them
    // do not break the run). One frame can be a glitch; two or more is the
    // same evidence the break rules accept as a real posture.
    private int pendingReachFrames = 0;
    // t of the latest raw target frame, and t of the first frame (of any
    // kind) after it. The latter is where a break starts: the hold is
    // only backed by frames that actually showed the target posture.
    private Double lastRawTargetAt = null;
    private Double firstAfterRawTarget = null;
    // t of the latest frame with no usable label (RESPONSE-phase verdict).
    private Double lastUnclearT = null;
    private Double uncertainSince = null; // t of the first frame of the
                                          // current unclear run
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

    // Feed one sampled frame. t is seconds since START (wall clock).
    // label == null means no usable label. Returns a Verdict as soon as
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
        seq.add(String.format(Locale.US, "%.1f %s", t, smoothed));

        // Raw-time bookkeeping (see the header: both ends of the hold are
        // measured on raw frames, not on the lagged smoothed stream).
        boolean rawUnclear = raw.equals(UNCERTAIN_FRAME);
        if (rawUnclear) lastUnclearT = t;
        if (raw.equals(target)) {
            lastRawTargetAt = t;
            firstAfterRawTarget = null;
        } else if (lastRawTargetAt != null && firstAfterRawTarget == null) {
            firstAfterRawTarget = t;
        }
        if (reachedAt == null) {
            if (raw.equals(target)) {
                if (pendingReachStart == null) {
                    pendingReachStart = t;
                    pendingReachFrames = 0;
                }
                pendingReachFrames++;
            } else if (!rawUnclear) {
                pendingReachStart = null;
                pendingReachFrames = 0;
            }
        }
        // Consecutive wrong-raw run (see wrongRawStreak). Wrong frames are
        // counted below, once the target is reached; a target frame ends
        // the run here.
        boolean afterUnclear = prevRawUnclear;
        prevRawUnclear = rawUnclear;
        if (raw.equals(target)) wrongRawStreak = 0;

        // Lost / unclear for too long -> UNCERTAIN (proposal 5.3:
        // lost/occluded for >1.0s; see the header for why 1.5 s).
        // Applies in any phase - an attempt that cannot be seen cannot
        // be judged.
        if (smoothed.equals(UNCERTAIN_FRAME)) {
            if (uncertainSince == null) uncertainSince = t;
            if (t - uncertainSince >= cfg.lostS) {
                // The dog left the target posture and THEN left the view
                // (e.g. stood up and trotted off faster than the ~1 s it
                // takes to confirm a break). That is a broken hold, not an
                // unjudgeable attempt - but only with real evidence: a
                // smoothed wrong posture, or >= 2 wrong raw frames.
                if (reachedAt != null && (breakSince != null
                        || (pendingBreakStart != null && wrongRawStreak >= 2))) {
                    double breakAt = pendingBreakStart != null ? pendingBreakStart : breakSince;
                    return make("EARLY_BREAK",
                            "left '" + target + "' after holding " + fmt1(breakAt - reachedAt)
                                    + "s, then left the view", breakAt, breakAt - reachedAt);
                }
                return make("UNCERTAIN",
                        "dog lost / unclear for " + fmt1(t - uncertainSince)
                                + "s in a row - cannot judge the attempt",
                        null, null);
            }
        } else {
            uncertainSince = null;
        }

        // A pending reach must not survive an unclear run of
        // breakConfirmS or longer. Without this, one glitch target frame
        // (e.g. 'sitting' at 1.6 s), then ~1 s of no dog, then the dog
        // really sitting at 3.2 s would be backdated to 1.6 s and judged
        // "reached within the window". After a gap that long the old raw
        // frame says nothing about when the dog actually got there, so the
        // reach restarts at the first target frame after the gap. At the
        // deadline this leaves the unclear-at-deadline rule in charge: dog unclear at
        // (or just before) the deadline -> UNCERTAIN, since it may have
        // responded out of sight. (smoothed is UNCERTAIN_FRAME exactly
        // when raw is, so uncertainSince is the start of the raw run.)
        // Only a LONE target frame is discarded: two or more confident
        // target frames before a short occlusion (owner, leash) are a real
        // sit, and the reach keeps its in-window start time. The 1e-6 lets
        // an exactly-breakConfirmS run count despite float rounding.
        if (reachedAt == null && rawUnclear && pendingReachStart != null
                && pendingReachFrames < 2
                && uncertainSince != null && t - uncertainSince >= cfg.breakConfirmS - 1e-6) {
            pendingReachStart = null;
            pendingReachFrames = 0;
        }

        // First raw sighting of a wrong posture during the hold.
        // (Before the target is reached the dog is simply not there
        // yet, so nothing is recorded.)
        // Timestamped at the first frame after the last raw target frame:
        // with no gap that is this frame; after an unclear gap it is the
        // gap's start, because the dog was not seen holding during it.
        if (reachedAt != null && !rawUnclear && !raw.equals(target)) {
            if (pendingBreakStart == null) {
                pendingBreakStart = (firstAfterRawTarget != null && firstAfterRawTarget >= reachedAt)
                        ? firstAfterRawTarget : t;
            }
            // Consecutive only: an unclear frame since the
            // previous wrong frame starts a new run.
            wrongRawStreak = afterUnclear ? 1 : wrongRawStreak + 1;
        }

        if (reachedAt == null) {
            double deadline = cfg.reachThenHold ? cfg.responseWindowS : cfg.settleGraceS;
            if (smoothed.equals(target)) {
                // Reach time on the raw time base: when the dog actually
                // got there, not when the smoother caught up.
                reachedAt = pendingReachStart != null ? pendingReachStart : t;
                pendingReachStart = null;
                pendingReachFrames = 0;
                if (cfg.reachThenHold && reachedAt > cfg.responseWindowS) {
                    return make("EARLY_BREAK",
                            "target posture '" + target + "' reached at " + fmt1(reachedAt)
                                    + "s, after the " + fmt0(cfg.responseWindowS)
                                    + "s response window", null, 0.0);
                }
                // Fall through: with a slow smoother the hold may already
                // be complete on this very frame.
            } else {
                if (t < deadline) return null;
                // A raw target run that began inside the window is still
                // being confirmed by the smoother: wait, but only up to
                // confirmGraceS past the deadline so this cannot defer
                // the decision forever.
                if (pendingReachStart != null && pendingReachStart <= deadline
                        && t < deadline + cfg.confirmGraceS) {
                    return null;
                }
                if (!cfg.reachThenHold) {
                    return make("UNCERTAIN",
                            "target posture '" + target + "' never established within the "
                                    + fmt0(cfg.settleGraceS) + "s settle grace", null, null);
                }
                // RESPONSE phase (Sit/Down, proposal 5.2). If the dog is
                // unclear now, or was within breakConfirmS, it may well be
                // mid-transition into the target out of sight (Model A's
                // confidence drops exactly then), so the command cannot be
                // judged: UNCERTAIN.
                boolean recentlyUnclear = smoothed.equals(UNCERTAIN_FRAME)
                        || (lastUnclearT != null && t - lastUnclearT < cfg.breakConfirmS);
                if (recentlyUnclear) {
                    return make("UNCERTAIN",
                            "dog not clearly visible at the end of the "
                                    + fmt0(cfg.responseWindowS)
                                    + "s response window - cannot judge the response",
                            null, null);
                }
                // Clearly seen in another posture to the end: a failed
                // command, hence EARLY_BREAK with held 0s - not UNCERTAIN.
                return make("EARLY_BREAK",
                        "target posture '" + target + "' not reached within the "
                                + fmt0(cfg.responseWindowS) + "s response window",
                        null, 0.0);
            }
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
            // The deciding frame must itself show the target (raw), so the
            // hold is backed by a real sighting at the full duration. If
            // the smoother says "target" only because its buffer is stale
            // (lag, or an unclear gap that never flushed it), wait: either
            // the break confirms (timestamped above) or a fresh raw target
            // frame arrives.
            if (raw.equals(target) && t - reachedAt >= holdNeeded) {
                if (cfg.reachThenHold) {
                    return make("SUCCESS",
                            "reached '" + target + "' within the "
                                    + fmt0(cfg.responseWindowS) + "s response window and held it "
                                    + "for the " + fmt1(cfg.holdAfterReachS) + "s confirmation hold",
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
                                    + fmt1(breakAt - reachedAt) + "s of the "
                                    + fmt1(cfg.holdAfterReachS) + "s confirmation hold",
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
    // duration for HOLD commands, the 0.6s confirmation hold for
    // REACH-THEN-HOLD commands.
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
                    nUncertain + "/" + n + " frames had no usable label (no dog "
                            + "detected or low model confidence) - too unclear to judge",
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
                            + "s of the " + fmt1(cfg.holdAfterReachS) + "s confirmation hold - "
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
