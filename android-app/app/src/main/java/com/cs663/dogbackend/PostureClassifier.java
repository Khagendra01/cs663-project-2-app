package com.cs663.dogbackend;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

// MODEL A - the group's own fine-tuned vision model.
//
//   Model:     MobileNetV3Small (ImageNet weights) fine-tuned on dataset
//              v4 (1,000 images, 4 classes) - cs663-project-2-vision/
//              model-a/train_model_a.py; exported as float16 TFLite
//              (assets/model_a_posture.tflite, same file as
//              model-a/model_a.tflite)
//   Runs:      on the phone, TensorFlow Lite interpreter, CPU, 2 threads
//   Input:     the dog crop from Model C, resized to 224x224 RGB,
//              float32 0..255 (rescaling is built into the model)
//   Output:    softmax over [sitting, standing, lying_down, walking]
//   Purpose:   fast per-frame posture (~5 fps) with a real confidence
//              score - what the temporal state machine needs to time a
//              Stay or catch a break. Gemini (~5 s per call) cannot do
//              this in real time.
//   Next step: label + confidence -> SequenceSession (state machine);
//              frames below 0.65 confidence are treated as unclear.
public class PostureClassifier implements AutoCloseable {

    public static final String MODEL_ASSET = "model_a_posture.tflite";
    // Class order from train_model_a.py CLASS_NAMES.
    public static final String[] LABELS = {"sitting", "standing", "lying_down", "walking"};
    public static final int SIZE = 224;

    public static class Result {
        public final String label;
        public final float confidence;
        public final float[] probs;

        Result(String label, float confidence, float[] probs) {
            this.label = label;
            this.confidence = confidence;
            this.probs = probs;
        }
    }

    private final Interpreter interpreter;
    private final ByteBuffer input;
    private final float[][] output = new float[1][LABELS.length];
    private final int[] pixels = new int[SIZE * SIZE];

    public PostureClassifier(Context ctx) throws IOException {
        Interpreter.Options opts = new Interpreter.Options();
        opts.setNumThreads(2);
        interpreter = new Interpreter(loadAsset(ctx, MODEL_ASSET), opts);
        input = ByteBuffer.allocateDirect(4 * SIZE * SIZE * 3).order(ByteOrder.nativeOrder());
    }

    // Classify a crop of any size. Training used tf.image.resize to
    // 224x224 without keeping the aspect ratio, so the crop is squashed
    // the same way here.
    public synchronized Result classify(Bitmap crop) {
        Bitmap in = crop.getWidth() == SIZE && crop.getHeight() == SIZE
                ? crop : Bitmap.createScaledBitmap(crop, SIZE, SIZE, true);
        in.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE);
        if (in != crop) in.recycle();
        input.rewind();
        for (int p : pixels) {
            input.putFloat((p >> 16) & 0xFF);
            input.putFloat((p >> 8) & 0xFF);
            input.putFloat(p & 0xFF);
        }
        input.rewind();
        interpreter.run(input, output);
        float[] probs = output[0].clone();
        int best = 0;
        for (int i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
        return new Result(LABELS[best], probs[best], probs);
    }

    private static MappedByteBuffer loadAsset(Context ctx, String name) throws IOException {
        try (AssetFileDescriptor fd = ctx.getAssets().openFd(name);
             FileInputStream fis = new FileInputStream(fd.getFileDescriptor())) {
            return fis.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    fd.getStartOffset(), fd.getDeclaredLength());
        }
    }

    @Override
    public synchronized void close() {
        interpreter.close();
    }
}
