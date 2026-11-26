package com.cs663.dogbackend;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.Detection;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector;
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

// MODEL C - pretrained foundation detector, NOT fine-tuned.
//
//   Model:     EfficientDet-Lite0 (COCO, 80 classes), int8, from the
//              MediaPipe model zoo (assets/efficientdet_lite0.tflite)
//   Runs:      on the phone, MediaPipe Tasks ObjectDetector, CPU
//   Input:     one upright camera frame (ARGB_8888 Bitmap)
//   Output:    dog boxes in frame pixels + score (+ the COCO label)
//   Purpose:   (1) is there a dog in view at all? (no dog -> the frame
//              is not judged, so an empty room can never open the READY
//              gate or "hold" a Stay); (2) WHERE the dog is: the user
//              sees a tracked box on the live preview, and the box crop
//              is what Model B (tuned Gemini) checks after the verdict.
//   Next step: dog found -> Model A judges the frame (VisionPipeline).
//
// Why not Model A or Gemini for this: Model A is a 4-way posture
// classifier with no "no dog" class - it always answers one of the four
// postures, even for an empty room. Gemini is a ~5 s cloud call. A
// detector answers "dog or not, and where" in tens of milliseconds.
public class DogDetector implements AutoCloseable {

    public static final String MODEL_ASSET = "efficientdet_lite0.tflite";

    // COCO has one "dog" class, and EfficientDet-Lite0 often gives a real
    // dog a different four-legged label instead: lying or curled-up dogs
    // come out as "cat", and big, dark or long-legged dogs as "horse",
    // "cow" or "sheep". The app only ever looks at one animal (the dog
    // being trained), so any four-legged animal box is treated as the dog.
    // "dog" is accepted at a lower score than the stand-in classes.
    // Offline check on the six project clips (111 frames, all with a dog):
    // "dog" or "cat" only found the dog in 61% of frames; this rule finds
    // it in 96% (see README, Model C).
    private static final float DOG_MIN_SCORE = 0.25f;
    private static final float OTHER_MIN_SCORE = 0.30f;
    static final List<String> ACCEPTED = java.util.Arrays.asList(
            "dog", "cat", "horse", "sheep", "cow", "bear");

    public static class Box {
        public final RectF rect;   // frame pixels
        public final float score;
        public final String category;

        Box(RectF rect, float score, String category) {
            this.rect = rect;
            this.score = score;
            this.category = category;
        }
    }

    private final ObjectDetector detector;

    public DogDetector(Context ctx) {
        ObjectDetector.ObjectDetectorOptions options =
                ObjectDetector.ObjectDetectorOptions.builder()
                        .setBaseOptions(BaseOptions.builder()
                                .setModelAssetPath(MODEL_ASSET).build())
                        .setRunningMode(RunningMode.IMAGE)
                        .setMaxResults(5)
                        .setScoreThreshold(DOG_MIN_SCORE)
                        .setCategoryAllowlist(ACCEPTED)
                        .build();
        detector = ObjectDetector.createFromOptions(ctx, options);
    }

    // All accepted dog boxes, best score first.
    public List<Box> detect(Bitmap frame) {
        MPImage img = new BitmapImageBuilder(frame).build();
        ObjectDetectorResult r = detector.detect(img);
        List<Box> out = new ArrayList<>();
        for (Detection d : r.detections()) {
            if (d.categories().isEmpty()) continue;
            Category c = d.categories().get(0);
            String name = c.categoryName();
            float s = c.score();
            if (s >= ("dog".equals(name) ? DOG_MIN_SCORE : OTHER_MIN_SCORE)) {
                out.add(new Box(new RectF(d.boundingBox()), s, name));
            }
        }
        Collections.sort(out, (a, b) -> Float.compare(b.score, a.score));
        return out;
    }

    @Override
    public void close() {
        detector.close();
    }
}
