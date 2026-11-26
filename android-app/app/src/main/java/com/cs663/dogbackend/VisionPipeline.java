package com.cs663.dogbackend;

import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;

import java.util.List;

// The on-device half of the system, run once per analysed camera frame
// (~5 fps) on the analysis thread:
//
//   upright frame
//     -> Model C (DogDetector): dog boxes
//     -> tracking: while a track is live, only a box that overlaps the
//        previous one (or, failing that, whose centre is close to the
//        previous centre - fast dogs) continues it; if the detector misses
//        (or only sees some other animal), hold the last box for up to HOLD_BOX_MS
//        (detectors flicker on lying / curled dogs)
//     -> ROI: square-ish crop around the box, padded 15% so the legs
//        and tail stay in it (the crop Model B checks; Model A's crop
//        reading is logged for comparison)
//     -> Model A (PostureClassifier) on the WHOLE frame: label +
//        confidence, used for judging only when Model C found a dog
//        this frame (FrameResult.judged says why not the crop)
//
// Model A is only trusted on a box the detector found THIS frame. A held box
// is kept for the overlay (yellow "tracking..." box) but its crop and
// cropBitmap are null: the dog may have walked out of that region, and
// Model A has no "no dog" class, so it would happily call empty floor
// "lying_down" and keep a Stay counting or hand Model B an empty crop.
// The caller treats held frames as unclear and lets the session's
// "lost" rule absorb short detector gaps.
//
// If there is no dog box (none detected and none held), the frame is
// marked dogPresent=false and is NOT judged - the state machine sees an
// unclear frame. Model A is still run on the whole frame for the log,
// so the evaluation can measure what the crop step adds (crop vs full
// frame on the same frames).
public class VisionPipeline {

    static final long HOLD_BOX_MS = 1000;
    static final float PAD = 0.15f;
    static final float MIN_IOU_TRACK = 0.2f;
    // Fallback gate when no box passes MIN_IOU_TRACK: a box whose centre is
    // within this many last-box diagonals of the last centre continues the
    // track (see process()).
    static final float CENTER_TRACK_DIAGS = 1.5f;

    public static class FrameResult {
        public long elapsedMs;           // capture time: wall clock live, clip time in replay
        public int frameW, frameH;
        public boolean dogPresent;
        public String boxSource;         // detected | held | none
        public RectF box;                // frame pixels, null if none
        public float detScore;
        public int nDogs;
        public PostureClassifier.Result crop;   // Model A on the ROI (null unless boxSource is "detected"); logged only
        public PostureClassifier.Result full;   // Model A on the whole frame
        // The Model A reading the app judges with: the WHOLE frame, but only
        // when Model C found a dog this frame (null otherwise). Model A was
        // trained on whole photos, and on its held-out test set (235 photos)
        // it scores 75.3% on the whole image vs 68.5% on our padded crop
        // (walking: 43/62 vs 17/62 - a tight box loses the stride and the
        // path). See eval/model_a_crop_vs_full.md. Model C still decides
        // whether there is a dog at all, tracks it, draws the box, and
        // supplies the crop that Model B checks.
        public PostureClassifier.Result judged;
        public Bitmap cropBitmap;        // for Model B keyframe checks (null unless "detected")
        public long detMs, clsMs, totalMs;
    }

    private final DogDetector detector;
    private final PostureClassifier classifier;
    private RectF lastBox;
    private float lastScore;
    private long lastBoxMs;

    public VisionPipeline(DogDetector detector, PostureClassifier classifier) {
        this.detector = detector;
        this.classifier = classifier;
    }

    public void resetTracking() {
        lastBox = null;
    }

    public FrameResult process(Bitmap frame) {
        return process(frame, SystemClock.elapsedRealtime());
    }

    // clockMs is the time the frame was taken on the caller's clock. Live
    // that is the wall clock; replay passes CLIP time, so the box hold
    // (HOLD_BOX_MS) expires after the same stretch of video however fast
    // or slow the frames are played back. The stage timings (detMs etc.)
    // stay on the wall clock: they measure the phone, not the video.
    public FrameResult process(Bitmap frame, long clockMs) {
        long t0 = SystemClock.elapsedRealtime();
        FrameResult r = new FrameResult();
        r.elapsedMs = clockMs;
        r.frameW = frame.getWidth();
        r.frameH = frame.getHeight();

        List<DogDetector.Box> boxes = detector.detect(frame);
        long t1 = SystemClock.elapsedRealtime();
        r.detMs = t1 - t0;
        r.nDogs = boxes.size();

        // A track is live while the last accepted box is recent. Only then
        // does the previous box get a say in which detection we follow.
        boolean haveTrack = lastBox != null && clockMs - lastBoxMs <= HOLD_BOX_MS;
        DogDetector.Box pick = null;
        if (haveTrack) {
            // Only a box that overlaps the dog we were already following
            // may continue the track. Falling back to the top-scoring box
            // here would let a cat on the sofa (or a second dog) take over
            // the attempt the moment Model C misses our dog for one frame,
            // and IoU tracking would then follow the wrong animal for good.
            float bestIou = MIN_IOU_TRACK;
            for (DogDetector.Box b : boxes) {
                float iou = iou(b.rect, lastBox);
                if (iou > bestIou) {
                    bestIou = iou;
                    pick = b;
                }
            }
            if (pick == null && boxes.size() == 1) {
                // No box overlaps enough, but a fast dog can move more than
                // ~0.65 box widths between frames at ~3-5 fps, which drops
                // IoU below MIN_IOU_TRACK even though it is the same dog.
                // Requiring overlap alone turned every fast run into held,
                // unclear frames (and the stale box never caught up). So a
                // box may also continue the track by position: its centre
                // within CENTER_TRACK_DIAGS x the last box's diagonal, and a
                // similar size (area ratio 0.5-2). Only when it is the ONLY
                // detection: with two animals in view, a one-frame miss of
                // ours must not hand the track to the one lying beside it,
                // so an ambiguous frame stays "held".
                DogDetector.Box b = boxes.get(0);
                float diag = (float) Math.hypot(lastBox.width(), lastBox.height());
                float d = (float) Math.hypot(b.rect.centerX() - lastBox.centerX(),
                        b.rect.centerY() - lastBox.centerY());
                float lastArea = lastBox.width() * lastBox.height();
                float ratio = lastArea > 0 ? b.rect.width() * b.rect.height() / lastArea : 0f;
                if (d <= CENTER_TRACK_DIAGS * diag && ratio >= 0.5f && ratio <= 2f) pick = b;
            }
        } else if (!boxes.isEmpty()) {
            // No live track (attempt start, or the hold expired): take the
            // best-scoring box. Boxes are sorted by score, highest first.
            pick = boxes.get(0);
        }
        if (pick != null) {
            r.box = pick.rect;
            r.detScore = pick.score;
            r.boxSource = "detected";
            lastBox = new RectF(pick.rect);
            lastScore = pick.score;
            lastBoxMs = clockMs;
        } else if (haveTrack) {
            // lastBoxMs is deliberately NOT refreshed: the hold must expire
            // HOLD_BOX_MS after the last real detection, otherwise a dog that
            // left the view would be "held" forever.
            r.box = new RectF(lastBox);
            r.detScore = lastScore;
            r.boxSource = "held";
        } else {
            r.boxSource = "none";
        }
        // Held frames still count as "dog present" for the overlay and the
        // UI chip; the caller decides what the state machine sees.
        r.dogPresent = r.box != null;

        if ("detected".equals(r.boxSource)) {
            r.cropBitmap = cropPadded(frame, r.box);
            r.crop = classifier.classify(r.cropBitmap);
        }
        r.full = classifier.classify(frame);
        if ("detected".equals(r.boxSource)) r.judged = r.full;
        long t2 = SystemClock.elapsedRealtime();
        r.clsMs = t2 - t1;
        r.totalMs = t2 - t0;
        return r;
    }

    static Bitmap cropPadded(Bitmap frame, RectF box) {
        float w = box.width(), h = box.height();
        // Pad, then grow the short side toward a square so a long, low
        // lying dog is not squashed flat when resized to 224x224.
        float side = Math.max(w, h) * (1 + 2 * PAD);
        float cw = Math.max(w * (1 + 2 * PAD), side * 0.75f);
        float ch = Math.max(h * (1 + 2 * PAD), side * 0.75f);
        int l = clamp(Math.round(box.centerX() - cw / 2), 0, frame.getWidth() - 1);
        int t = clamp(Math.round(box.centerY() - ch / 2), 0, frame.getHeight() - 1);
        int r = clamp(Math.round(box.centerX() + cw / 2), l + 1, frame.getWidth());
        int b = clamp(Math.round(box.centerY() + ch / 2), t + 1, frame.getHeight());
        return Bitmap.createBitmap(frame, l, t, r - l, b - t);
    }

    static float iou(RectF a, RectF b) {
        float ix = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        float iy = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        float inter = ix * iy;
        float union = a.width() * a.height() + b.width() * b.height() - inter;
        return union <= 0 ? 0 : inter / union;
    }

    private static int clamp(int v, int lo, int hi) {
        return Math.max(lo, Math.min(hi, v));
    }
}
