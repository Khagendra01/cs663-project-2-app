package com.cs663.dogbackend;

import android.Manifest;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicInteger;

// Model lab: one photo through every model, side by side.
//   On the phone: Model C (EfficientDet-Lite0) finds the dog, Model A
//   (our fine-tuned MobileNetV3Small) classifies the crop.
//   In the cloud: base Gemini vs fine-tuned Gemini (Model B), same photo,
//   same prompt - the base-vs-tuned comparison the spec asks for, live.
public class MainActivity extends Activity {

    // Backend URL. The text field on screen starts with this value and
    // can be edited, so a new deploy does not need a rebuild.
    public static final String BACKEND_URL = "https://cs663-gemini-service-917815040923.us-central1.run.app";

    private static final int REQ_CAMERA = 1;
    private static final int REQ_PICK = 2;
    private static final int REQ_CAMERA_PERM = 3;
    private static final int MAX_SIDE = 1600;

    // All model work (load, runLocal, close) runs on ONE thread, in order.
    // That way close() always waits for an in-flight load or inference
    // (closing MediaPipe / LiteRT mid-call crashes the app, and closing
    // before a load finishes leaked the native models), and the model
    // fields are only ever touched from that one thread.
    private final ExecutorService modelExec = Executors.newSingleThreadExecutor();
    // Picture decoding and the two Gemini calls. Kept off modelExec so a
    // slow network call never delays the on-device models.
    private final ExecutorService pool = Executors.newFixedThreadPool(3);

    private EditText etUrl;
    private EditText etPrompt;
    private ImageView imgPreview;
    private TextView tvStatus;
    private TextView tvLocal;
    private TextView tvBase;
    private TextView tvTuned;
    private Button btnAsk;
    private Button btnCamera;
    private Button btnPick;

    private Bitmap photo;
    // Model fields: read and written only on modelExec.
    private DogDetector detector;
    private PostureClassifier classifier;
    private String modelLoadError;

    // Set in onDestroy so background tasks stop posting to dead views.
    private volatile boolean destroyed;
    // UI-thread state. loadRequest makes sure only the latest pick is shown
    // if the user picks again before the previous decode finishes.
    private int loadRequest;
    private boolean loading;
    // Decodes still reading capture.jpg (superseded ones included, since
    // they keep reading until they finish). Only these block the camera.
    private int captureDecodes;
    private boolean asking;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        etUrl = findViewById(R.id.etUrl);
        etPrompt = findViewById(R.id.etPrompt);
        imgPreview = findViewById(R.id.imgPreview);
        tvStatus = findViewById(R.id.tvStatus);
        tvLocal = findViewById(R.id.tvLocal);
        tvBase = findViewById(R.id.tvBase);
        tvTuned = findViewById(R.id.tvTuned);
        btnAsk = findViewById(R.id.btnAsk);
        btnCamera = findViewById(R.id.btnCamera);
        btnPick = findViewById(R.id.btnPick);

        String url = getIntent().getStringExtra(SequenceActivity.EXTRA_BACKEND_URL);
        etUrl.setText(url == null || url.isEmpty() ? BACKEND_URL : url);
        etPrompt.setText(GeminiClient.POSTURE_PROMPT);

        btnCamera.setOnClickListener(v -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                launchCamera();
            } else {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA_PERM);
            }
        });

        btnPick.setOnClickListener(v -> {
            Intent i = new Intent(Intent.ACTION_GET_CONTENT);
            i.setType("image/*");
            startActivityForResult(Intent.createChooser(i, "Pick a picture"), REQ_PICK);
        });

        btnAsk.setOnClickListener(v -> askBoth());
        findViewById(R.id.btnSequence).setOnClickListener(v -> finish());

        modelExec.execute(() -> {
            if (destroyed) return;
            try {
                detector = new DogDetector(this);
                classifier = new PostureClassifier(this);
            } catch (Throwable e) {
                modelLoadError = String.valueOf(e.getMessage());
                ui(() -> tvLocal.setText("On-device models failed to load: " + modelLoadError));
            }
        });
    }

    // Post to the UI thread, but drop the update once the activity is gone
    // (Back / rotation while a model or network call is still running).
    private void ui(Runnable r) {
        runOnUiThread(() -> {
            if (destroyed || isDestroyed()) return;
            r.run();
        });
    }

    private void refreshAskButton() {
        btnAsk.setEnabled(!loading && !asking);
        // Camera is off while a decode may still be reading capture.jpg:
        // launchCamera deletes that file and lets the camera app overwrite
        // it. Gallery stays on - a picked picture is copied to its own temp
        // file, and a stuck cloud pick must be replaceable by picking again
        // (loadRequest keeps only the latest). Always called on the UI
        // thread (directly, or via ui(), which drops it once destroyed).
        btnCamera.setEnabled(captureDecodes == 0);
        btnPick.setEnabled(true);
    }

    private File photosDir() {
        File dir = new File(getCacheDir(), "photos");
        if (!dir.exists()) dir.mkdirs();
        return dir;
    }

    // Fixed path, rebuilt on demand instead of kept in a field: the camera
    // app is memory-hungry, so this activity is often recreated (rotation or
    // process death) before the result comes back, and a field would be
    // null by then. Must stay inside the FileProvider's cache-path
    // "photos/" (res/xml/file_paths.xml).
    private File captureFile() {
        return new File(photosDir(), "capture.jpg");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_CAMERA_PERM) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                launchCamera();
            } else {
                tvStatus.setText("Camera permission denied - pick a picture instead.");
            }
        }
    }

    // Full-resolution capture into a cache file (the 'data' extra is only
    // a ~200 px thumbnail, too small for a fair model comparison).
    private void launchCamera() {
        // Second guard for the permission-dialog path (button pressed, then
        // a load started before the grant came back): never delete
        // capture.jpg while a decode may be reading it.
        if (captureDecodes > 0) return;
        try {
            File f = captureFile();
            // Remove the previous capture so a camera app that says OK
            // without writing anything cannot make us reload an old photo.
            if (f.exists()) f.delete();
            Uri out = FileProvider.getUriForFile(this, getPackageName() + ".fileprovider", f);
            Intent i = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            i.putExtra(MediaStore.EXTRA_OUTPUT, out);
            i.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
            startActivityForResult(i, REQ_CAMERA);
        } catch (ActivityNotFoundException e) {
            tvStatus.setText("No camera app on this device - pick a picture instead.");
        } catch (Exception e) {
            tvStatus.setText("Could not open the camera: " + e.getMessage());
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK) return;   // user cancelled
        File local = null;
        Uri uri = null;
        if (requestCode == REQ_CAMERA) {
            File f = captureFile();
            if (f.length() > 0) local = f;
        } else if (requestCode == REQ_PICK) {
            if (data != null) uri = data.getData();
        } else {
            return;
        }
        if (local == null && uri == null) {
            tvStatus.setText(requestCode == REQ_CAMERA
                    ? "No photo came back from the camera - try again."
                    : "No picture came back - try again.");
            return;
        }
        loadPhoto(local, uri);
    }

    // Decoding runs on the pool: a gallery pick can be a cloud item that
    // downloads inside openInputStream, and a big photo takes a while to
    // decode, both long enough to freeze the UI (ANR).
    private void loadPhoto(File local, Uri uri) {
        final int req = ++loadRequest;
        final boolean readsCapture = local != null;
        if (readsCapture) captureDecodes++;
        loading = true;
        refreshAskButton();
        tvStatus.setText("Loading picture...");
        try {
            pool.execute(() -> {
                Bitmap bmp = null;
                String err = null;
                File tmp = null;
                try {
                    File src = local;
                    if (src == null) {
                        // Read the content stream once into a local file, then
                        // decode bounds, pixels and EXIF from that file, so a
                        // cloud picture is downloaded only once.
                        tmp = File.createTempFile("picked", ".jpg", photosDir());
                        copyTo(uri, tmp);
                        src = tmp;
                    }
                    bmp = loadUpright(src);
                } catch (Throwable e) {   // includes OutOfMemoryError
                    err = e.getMessage() != null ? e.getMessage() : e.toString();
                } finally {
                    if (tmp != null) tmp.delete();
                }
                final Bitmap result = bmp;
                final String error = err;
                ui(() -> {
                    if (readsCapture) captureDecodes--;
                    if (req != loadRequest) {          // a newer pick replaced this one
                        refreshAskButton();
                        return;
                    }
                    loading = false;
                    refreshAskButton();
                    if (result != null) {
                        photo = result;
                        imgPreview.setImageBitmap(result);
                        tvStatus.setText("Photo ready. Running the on-device models...");
                        runLocal(result);
                    } else if (error != null) {
                        tvStatus.setText("Could not load that picture: " + error);
                    } else {
                        tvStatus.setText("Could not read that picture.");
                    }
                });
            });
        } catch (RejectedExecutionException e) {
            // onDestroy already shut the pool down; nothing left to show.
        }
    }

    private void copyTo(Uri uri, File dst) throws Exception {
        try (InputStream in = getContentResolver().openInputStream(uri);
             OutputStream out = new FileOutputStream(dst)) {
            if (in == null) throw new java.io.IOException("could not open the picture");
            byte[] buf = new byte[64 * 1024];
            int n;
            while ((n = in.read(buf)) > 0) out.write(buf, 0, n);
        }
    }

    // Decode at most MAX_SIDE px (no OOM on 48 MP photos) and apply the
    // EXIF orientation, so every model sees the photo upright.
    private Bitmap loadUpright(File f) throws Exception {
        String path = f.getAbsolutePath();
        BitmapFactory.Options b = new BitmapFactory.Options();
        b.inJustDecodeBounds = true;
        BitmapFactory.decodeFile(path, b);
        if (b.outWidth <= 0 || b.outHeight <= 0) return null;
        int sample = 1;
        while (Math.max(b.outWidth, b.outHeight) / (sample * 2) >= MAX_SIDE) sample *= 2;
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inSampleSize = sample;
        o.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bmp = BitmapFactory.decodeFile(path, o);
        if (bmp == null) return null;
        // getRotationDegrees/isFlipped cover all 8 EXIF orientations,
        // including TRANSPOSE/TRANSVERSE (flip + 90) and the plain flips,
        // which a switch on ROTATE_90/180/270 alone left sideways or mirrored.
        int deg = 0;
        boolean flip = false;
        try {
            ExifInterface exif = new ExifInterface(path);
            deg = exif.getRotationDegrees();
            flip = exif.isFlipped();
        } catch (Exception ignored) {
        }
        if (deg != 0 || flip) {
            Matrix m = new Matrix();
            // Flip first, then rotate: AndroidX defines getRotationDegrees()
            // as the rotation applied after the horizontal flip.
            if (flip) m.postScale(-1, 1);
            if (deg != 0) m.postRotate(deg);
            Bitmap upright = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), m, true);
            if (upright != bmp) bmp.recycle();
            bmp = upright;
        }
        return bmp;
    }

    // Runs on modelExec, so it always runs after the model load queued in
    // onCreate (no "still loading" dead end) and before any close().
    private void runLocal(Bitmap bmp) {
        try {
            modelExec.execute(() -> {
                try {
                    runLocalOnModelThread(bmp);
                } catch (Throwable e) {
                    // An uncaught error on an executor thread kills the app.
                    String msg = e.getMessage() != null ? e.getMessage() : e.toString();
                    ui(() -> tvLocal.setText("On-device models failed on this photo: " + msg));
                }
            });
        } catch (RejectedExecutionException e) {
            // Activity is being destroyed; the models are closing.
        }
    }

    private void runLocalOnModelThread(Bitmap bmp) {
        if (destroyed) return;
        if (detector == null || classifier == null) {
            String why = modelLoadError != null ? modelLoadError : "not loaded";
            ui(() -> tvLocal.setText("On-device models failed to load: " + why));
            return;
        }
        long t0 = System.currentTimeMillis();
        List<DogDetector.Box> boxes = detector.detect(bmp);
        long t1 = System.currentTimeMillis();
        StringBuilder sb = new StringBuilder();
        Bitmap shown = bmp.copy(Bitmap.Config.ARGB_8888, true);
        if (boxes.isEmpty()) {
            PostureClassifier.Result full = classifier.classify(bmp);
            sb.append(String.format(Locale.US, "Model C: no dog found (%d ms).\n", t1 - t0));
            sb.append(String.format(Locale.US,
                    "Model A on the whole photo (not trusted without a dog box): %s %.0f%%",
                    SequenceActivity.pretty(full.label), full.confidence * 100));
        } else {
            DogDetector.Box best = boxes.get(0);
            Bitmap crop = VisionPipeline.cropPadded(bmp, best.rect);
            PostureClassifier.Result a = classifier.classify(crop);
            PostureClassifier.Result full = classifier.classify(bmp);
            long t2 = System.currentTimeMillis();
            sb.append(String.format(Locale.US, "Model C: %d dog box(es), best %s %.0f%% (%d ms)\n",
                    boxes.size(), best.category, best.score * 100, t1 - t0));
            sb.append(String.format(Locale.US, "Model A on the dog crop: %s %.0f%% (%d ms)%s\n",
                    SequenceActivity.pretty(a.label), a.confidence * 100, t2 - t1,
                    a.confidence < 0.65 ? " - below 0.65, the app would call this unclear" : ""));
            sb.append("   all classes: ");
            for (int i = 0; i < a.probs.length; i++) {
                sb.append(String.format(Locale.US, "%s %.0f%%%s", SequenceActivity.pretty(PostureClassifier.LABELS[i]),
                        a.probs[i] * 100, i < a.probs.length - 1 ? ", " : ""));
            }
            sb.append(String.format(Locale.US, "\nModel A on the whole photo: %s %.0f%%",
                    SequenceActivity.pretty(full.label), full.confidence * 100));
            Canvas c = new Canvas(shown);
            Paint p = new Paint();
            p.setStyle(Paint.Style.STROKE);
            p.setColor(Color.GREEN);
            p.setStrokeWidth(Math.max(4, shown.getWidth() / 150f));
            c.drawRect(best.rect, p);
        }
        ui(() -> {
            // A newer photo was loaded while this one ran; do not
            // overwrite its preview with this older result.
            if (bmp != photo) return;
            imgPreview.setImageBitmap(shown);
            tvLocal.setText(sb.toString());
            tvStatus.setText("On-device result below. Tap Ask both models for the Gemini comparison.");
        });
    }

    private void askBoth() {
        if (photo == null) {
            tvStatus.setText("Take or pick a picture first.");
            return;
        }
        GeminiClient gc = new GeminiClient(etUrl.getText().toString());
        if (!gc.configured()) {
            tvStatus.setText("Set the real backend URL first.");
            return;
        }
        String prompt = etPrompt.getText().toString().trim();
        if (prompt.isEmpty()) {
            tvStatus.setText("Type a prompt first.");
            return;
        }
        asking = true;
        refreshAskButton();
        tvStatus.setText("Calling both models...");
        tvBase.setText("Calling...");
        tvTuned.setText("Calling...");
        String b64 = GeminiClient.toJpegB64(photo, 1024);
        AtomicInteger pending = new AtomicInteger(2);
        Runnable done = () -> {
            if (pending.decrementAndGet() == 0) {
                ui(() -> {
                    asking = false;
                    refreshAskButton();
                    tvStatus.setText("Done. Compare the results.");
                });
            }
        };
        pool.execute(() -> {
            GeminiClient.Reply r = gc.post("/api/gemini", prompt, b64);
            ui(() -> tvBase.setText(show(r)));
            done.run();
        });
        pool.execute(() -> {
            GeminiClient.Reply r = gc.post("/api/fine-tuned-gemini", prompt, b64);
            ui(() -> tvTuned.setText(show(r)));
            done.run();
        });
    }

    private static String show(GeminiClient.Reply r) {
        return r.ok() ? r.text + String.format(Locale.US, "\n(%.1f s)", r.latencyMs / 1000.0)
                : "Failed: " + r.error;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        destroyed = true;
        // Queued behind any load / runLocal already on modelExec, so the
        // models are never closed while they are in use.
        modelExec.execute(() -> {
            if (detector != null) detector.close();
            if (classifier != null) classifier.close();
            detector = null;
            classifier = null;
        });
        modelExec.shutdown();
        pool.shutdown();
    }
}
