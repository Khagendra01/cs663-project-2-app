package com.cs663.dogbackend;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

// Unit tests for the on-phone temporal state machine at the v2 time base
// (real seconds, ~5 fps, Model A confidences). Run: ./gradlew test
public class SequenceSessionTest {

    private static final double DT = 0.2; // 5 fps

    private static SequenceSession stay(double seconds) {
        SequenceSession.Config c = new SequenceSession.Config(seconds);
        return new SequenceSession("sitting", c, "Stay");
    }

    private static SequenceSession sit() {
        SequenceSession.Config c = new SequenceSession.Config(5);
        c.reachThenHold = true;
        return new SequenceSession("sitting", c, "Sit");
    }

    // Feed `label` at 5 fps from t0 for `seconds`; returns the verdict if
    // one was decided, and leaves the next free t in tOut[0].
    private static SequenceSession.Verdict run(SequenceSession s, double[] t, String label,
                                               double conf, boolean dog, double seconds) {
        return run(s, t, label, conf, dog, seconds, DT);
    }

    // Same, at a custom frame interval (slower phones run below 5 fps).
    private static SequenceSession.Verdict run(SequenceSession s, double[] t, String label,
                                               double conf, boolean dog, double seconds,
                                               double dt) {
        SequenceSession.Verdict v = null;
        int n = (int) Math.round(seconds / dt);
        for (int i = 0; i < n && v == null; i++) {
            v = s.feed(t[0], label, conf, dog);
            t[0] = Math.round((t[0] + dt) * 1000) / 1000.0;
        }
        return v;
    }

    private static SequenceSession down() {
        SequenceSession.Config c = new SequenceSession.Config(5);
        c.reachThenHold = true;
        return new SequenceSession("lying_down", c, "Down");
    }

    @Test
    public void staySuccessAfterFullHold() {
        SequenceSession s = stay(5);
        double[] t = {0};
        SequenceSession.Verdict v = run(s, t, "sitting", 0.9, true, 6);
        assertNotNull(v);
        assertEquals("SUCCESS", v.verdict);
        assertEquals(0.0, v.targetReachedAtS, 1e-9);
        assertTrue(v.heldS >= 5.0);
    }

    @Test
    public void stayBreakIsTimestampedAtFirstWrongFrame() {
        SequenceSession s = stay(10);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 3.0));   // t = 0.0 .. 2.8
        double breakStart = t[0];                            // 3.0
        SequenceSession.Verdict v = run(s, t, "standing", 0.9, true, 3.0);
        assertNotNull(v);
        assertEquals("EARLY_BREAK", v.verdict);
        assertEquals(breakStart, v.breakAtS, 1e-9);
        assertEquals(3.0, v.heldS, 1e-9);
    }

    @Test
    public void oneGlitchFrameDoesNotBreakAStay() {
        SequenceSession s = stay(4);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 2.0));
        assertNull(run(s, t, "standing", 0.9, true, 0.2)); // single wrong frame
        SequenceSession.Verdict v = run(s, t, "sitting", 0.9, true, 3.0);
        assertNotNull(v);
        assertEquals("SUCCESS", v.verdict);
    }

    @Test
    public void sitNotReachedInThreeSecondsIsEarlyBreak() {
        SequenceSession s = sit();
        double[] t = {0};
        SequenceSession.Verdict v = run(s, t, "standing", 0.9, true, 4.0);
        assertNotNull(v);
        assertEquals("EARLY_BREAK", v.verdict);
        assertNull(v.targetReachedAtS);
        assertEquals(0.0, v.heldS, 1e-9);
    }

    @Test
    public void sitReachedThenHeldIsSuccess() {
        SequenceSession s = sit();
        double[] t = {0};
        assertNull(run(s, t, "standing", 0.9, true, 1.0));
        SequenceSession.Verdict v = run(s, t, "sitting", 0.9, true, 3.0);
        assertNotNull(v);
        assertEquals("SUCCESS", v.verdict);
        assertTrue(v.targetReachedAtS < 3.0);
    }

    @Test
    public void lowConfidenceFramesCountAsUnclear() {
        SequenceSession s = stay(5);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 1.0));
        // Model A below the 0.65 threshold for 2 s -> lost -> UNCERTAIN
        SequenceSession.Verdict v = run(s, t, "sitting", 0.4, true, 2.0);
        assertNotNull(v);
        assertEquals("UNCERTAIN", v.verdict);
    }

    @Test
    public void noDogForTooLongIsUncertain() {
        SequenceSession s = stay(5);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 1.0));
        SequenceSession.Verdict v = run(s, t, null, 0, false, 2.0);
        assertNotNull(v);
        assertEquals("UNCERTAIN", v.verdict);
        assertTrue(v.reason.contains("lost"));
    }

    @Test
    public void shortOcclusionIsTolerated() {
        SequenceSession s = stay(4);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 1.0));
        assertNull(run(s, t, null, 0, false, 1.0)); // 1.0 s < 1.5 s lost rule
        SequenceSession.Verdict v = run(s, t, "sitting", 0.9, true, 4.0);
        assertNotNull(v);
        assertEquals("SUCCESS", v.verdict);
    }

    @Test
    public void stoppingEarlyIsUncertainNotSuccess() {
        SequenceSession s = stay(10);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 3.0));
        SequenceSession.Verdict v = s.finish();
        assertEquals("UNCERTAIN", v.verdict);
    }

    @Test
    public void readinessSmootherRespectsConfidence() {
        SequenceSession.TemporalSmoother sm = new SequenceSession.TemporalSmoother(5, 0.65);
        assertEquals(SequenceSession.UNCERTAIN_FRAME, sm.push("sitting", 0.5, true));
        assertEquals("sitting", sm.push("sitting", 0.8, true));
        assertEquals(SequenceSession.UNCERTAIN_FRAME, sm.push("sitting", 0.9, false));
    }

    // The dog is mid-transition (low confidence) when the
    // Sit response window ends. The machine cannot see whether it sat, so
    // the verdict must be UNCERTAIN, not "not reached" (EARLY_BREAK).
    @Test
    public void sitWindowEndingWhileUnclearIsUncertain() {
        SequenceSession s = sit();
        double[] t = {0};
        assertNull(run(s, t, "standing", 0.9, true, 2.0));  // t = 0.0 .. 1.8
        SequenceSession.Verdict v = run(s, t, "sitting", 0.5, true, 2.0); // 2.0 .. unclear
        assertNotNull(v);
        assertEquals("UNCERTAIN", v.verdict);
        assertEquals(3.0, s.getLastT(), 1e-9); // decided at the deadline
        assertTrue(v.reason.contains("not clearly visible"));
    }

    // Counterpart: one unclear frame well before the deadline, then
    // clearly standing to the end -> still a failed Sit (EARLY_BREAK).
    @Test
    public void sitClearlyStandingAtDeadlineIsStillEarlyBreak() {
        SequenceSession s = sit();
        double[] t = {0};
        assertNull(run(s, t, "standing", 0.9, true, 2.0));
        assertNull(run(s, t, null, 0, false, 0.2));         // t = 2.0 unclear
        SequenceSession.Verdict v = run(s, t, "standing", 0.9, true, 2.0);
        assertNotNull(v);
        assertEquals("EARLY_BREAK", v.verdict);
        assertNull(v.targetReachedAtS);
    }

    // After an unclear gap the smoother buffer is all
    // stale "sitting", so the first standing frames are outvoted. The Stay
    // must not be called SUCCESS on a frame that shows the dog standing; the
    // hold is only proven up to the last sitting frame (3.4 s).
    @Test
    public void staleSmootherAfterGapDoesNotGiveSuccess() {
        SequenceSession s = stay(5);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 3.6));   // t = 0.0 .. 3.4
        assertNull(run(s, t, "sitting", 0.5, true, 1.4));   // t = 3.6 .. 4.8 unclear
        SequenceSession.Verdict v = run(s, t, "standing", 0.9, true, 3.0);
        assertNotNull(v);
        assertEquals("EARLY_BREAK", v.verdict);
        assertEquals(3.6, v.breakAtS, 1e-9);
        assertTrue(v.heldS < 5.0);
    }

    // Same without occlusion: the dog stands at 4.8 s of a 5 s Stay.
    // The smoother still says "sitting" at 5.0 s, but that is lag, not a hold.
    @Test
    public void standingJustBeforeTargetIsEarlyBreakNotSuccess() {
        SequenceSession s = stay(5);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 4.8));   // t = 0.0 .. 4.6
        SequenceSession.Verdict v = run(s, t, "standing", 0.9, true, 3.0);
        assertNotNull(v);
        assertEquals("EARLY_BREAK", v.verdict);
        assertEquals(4.8, v.breakAtS, 1e-9);
    }

    // The dog stands (two raw frames) and walks out of
    // view before the smoother confirms the break. That is a broken Stay at
    // 6.6 s, not "could not see the dog".
    @Test
    public void standingThenLeavingViewIsEarlyBreak() {
        SequenceSession s = stay(10);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 6.6));   // t = 0.0 .. 6.4
        assertNull(run(s, t, "standing", 0.9, true, 0.4));  // t = 6.6, 6.8
        SequenceSession.Verdict v = run(s, t, null, 0, false, 3.0);
        assertNotNull(v);
        assertEquals("EARLY_BREAK", v.verdict);
        assertEquals(6.6, v.breakAtS, 1e-9);
        assertEquals(6.6, v.heldS, 1e-9);
    }

    // Counterpart: a single wrong frame followed by an occlusion is
    // too little evidence of a break, so the lost rule still says UNCERTAIN.
    @Test
    public void singleWrongFrameThenLostIsUncertain() {
        SequenceSession s = stay(10);
        double[] t = {0};
        assertNull(run(s, t, "sitting", 0.9, true, 6.6));
        assertNull(run(s, t, "standing", 0.9, true, 0.2));  // t = 6.6 only
        SequenceSession.Verdict v = run(s, t, null, 0, false, 3.0);
        assertNotNull(v);
        assertEquals("UNCERTAIN", v.verdict);
    }

    // At 4 fps the dog lies down at raw t = 2.75 s, but
    // the 5-frame smoother only confirms it at 3.25 s. The reach happened
    // inside the 3 s window, so the Down is judged on raw time: reached at
    // 2.75 s, then the 0.6 s hold -> SUCCESS.
    @Test
    public void reachConfirmedJustAfterWindowCountsOnRawTime() {
        SequenceSession s = down();
        double[] t = {0};
        double dt = 0.25;
        assertNull(run(s, t, "sitting", 0.9, true, 2.75, dt)); // t = 0.0 .. 2.5
        SequenceSession.Verdict v = run(s, t, "lying_down", 0.9, true, 3.0, dt);
        assertNotNull(v);
        assertEquals("SUCCESS", v.verdict);
        assertEquals(2.75, v.targetReachedAtS, 1e-9);
    }

    // Bounded wait: a raw target frame just inside the window that is
    // never confirmed (the dog goes unclear) cannot defer the decision past
    // responseWindowS + confirmGraceS.
    @Test
    public void pendingReachDoesNotDeferForever() {
        SequenceSession s = sit();
        double[] t = {0};
        assertNull(run(s, t, "standing", 0.9, true, 2.8));  // t = 0.0 .. 2.6
        assertNull(run(s, t, "sitting", 0.9, true, 0.2));   // t = 2.8 raw sit
        SequenceSession.Verdict v = run(s, t, null, 0, false, 1.4); // 3.0 .. 4.2
        assertNotNull(v);
        assertNull(v.targetReachedAtS);
        assertTrue(s.getLastT() <= 3.0 + s.getConfig().confirmGraceS + 1e-9);
    }

    // One glitch 'sitting' frame at 1.6 s, then no dog 1.8-3.0 s,
    // then the dog really sits at 3.2 s. The stale 1.6 s reach must not be
    // carried across the gap and backdated into the window (that gave
    // SUCCESS). The dog is unclear at the 3 s deadline, so by the
    // unclear-at-deadline rule the response cannot be judged: UNCERTAIN, decided at 3.0 s.
    @Test
    public void staleReachAcrossUnclearGapIsNotBackdated() {
        SequenceSession s = sit();
        double[] t = {0};
        assertNull(run(s, t, "standing", 0.9, true, 1.6));  // t = 0.0 .. 1.4
        assertNull(run(s, t, "sitting", 0.9, true, 0.2));   // t = 1.6 glitch
        SequenceSession.Verdict v = run(s, t, null, 0, false, 1.4); // 1.8 .. 3.0
        assertNotNull(v);
        assertEquals("UNCERTAIN", v.verdict);
        assertEquals(3.0, s.getLastT(), 1e-9);
        assertNull(v.targetReachedAtS);
        assertTrue(v.reason.contains("not clearly visible"));
        // The late sit at 3.2 / 3.4 does not change the decided verdict.
        SequenceSession.Verdict after = run(s, t, "sitting", 0.9, true, 0.4);
        assertEquals("UNCERTAIN", after.verdict);
        assertNull(after.targetReachedAtS);
    }

    // Two confident sitting frames inside the window, then a short
    // occlusion (owner / leash) past the deadline, then the dog clearly
    // sitting: a real in-time sit. Only a LONE target frame is discarded
    // across a gap, so this is SUCCESS reached at 2.4 s.
    @Test
    public void realSitBeforeShortOcclusionKeepsItsReachTime() {
        SequenceSession s = sit();
        double[] t = {0};
        assertNull(run(s, t, "standing", 0.9, true, 2.4));  // 0.0 .. 2.2
        assertNull(run(s, t, "sitting", 0.9, true, 0.4));   // 2.4, 2.6
        assertNull(run(s, t, null, 0, false, 0.8));         // 2.8 .. 3.4
        SequenceSession.Verdict v = run(s, t, "sitting", 0.9, true, 2.0);
        assertNotNull(v);
        assertEquals("SUCCESS", v.verdict);
        assertEquals(2.4, v.targetReachedAtS, 1e-9);
    }

    // Model A misreads a lying dog as 'sitting' at 5.2 and 5.6 s
    // with an unclear frame between (smoother still lying_down), then the
    // owner steps in front from 5.8 s. The two misreads are not consecutive,
    // so they are no evidence that the dog moved: UNCERTAIN, not EARLY_BREAK.
    @Test
    public void sporadicMisreadsThenOcclusionIsUncertain() {
        SequenceSession s = new SequenceSession("lying_down",
                new SequenceSession.Config(10), "Stay");
        double[] t = {0};
        assertNull(run(s, t, "lying_down", 0.9, true, 5.2)); // t = 0.0 .. 5.0
        assertNull(run(s, t, "sitting", 0.9, true, 0.2));     // t = 5.2 misread
        assertNull(run(s, t, null, 0, false, 0.2));           // t = 5.4 unclear
        assertNull(run(s, t, "sitting", 0.9, true, 0.2));     // t = 5.6 misread
        assertEquals("lying_down", s.getLastSmoothed());
        SequenceSession.Verdict v = run(s, t, null, 0, false, 3.0); // 5.8 ..
        assertNotNull(v);
        assertEquals("UNCERTAIN", v.verdict);
        assertNull(v.breakAtS);
        assertTrue(v.reason.contains("lost"));
    }
}
