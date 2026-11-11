package com.cs663.dogbackend;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Base64;
import android.util.Size;
import android.view.Surface;
import android.view.TextureView;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

// Sequence mode: the Project 2 core. Instead of judging one photo, the
// app judges a command over TIME.
//
// How it works:
//   1. The trainer picks a command (Sit / Down / Stand / Stay / Walk) and
//      a target duration, then taps START SEQUENCE.
//   2. The camera (Camera2, live preview on a TextureView) also streams
//      JPEG frames into an ImageReader. Once per second the sampler
//      takes the newest frame, shrinks it to ~640px wide, and POSTs it
//      to /api/fine-tuned-gemini with the fixed posture prompt.
//   3. Each reply is parsed to a canonical label and fed into
//      SequenceSession - the Java port of the Python temporal state
//      machine in files/video-sequence-layer/state_machine.py - which
//      decides SUCCESS / EARLY_BREAK / UNCERTAIN with smoothing, a
//      settle grace, and a sustained-break rule.
//   4. When a verdict is decided, the app asks the base model
//      (/api/gemini) for short trainer feedback about the attempt.
public class SequenceActivity extends Activity {

    public static final String EXTRA_BACKEND_URL = "backendUrl";

    // Fixed prompt for every frame, identical to the dataset prompt.
    private static final String FRAME_PROMPT =
            "Look at the image of the dog. What is the dog doing? "
                    + "Answer with exactly one of: sitting, standing, lying down, walking.";

    private static final int REQ_CAMERA_PERM = 77;

    private static final String[] COMMANDS = {"Sit", "Down", "Stand", "Stay", "Walk"};
    private static final String[] STAY_POSTURES = {"Sitting", "Lying down", "Standing"};
    private static final String[] DURATIONS = {"5 seconds", "10 seconds", "20 seconds"};
    private static final int[] DURATION_SECONDS = {5, 10, 20};

    // UI
    private EditText etSeqUrl;
    private Spinner spCommand;
    private Spinner spStayPosture;
    private Spinner spDuration;
    private TextureView texturePreview;
    private TextView tvSeqCurrent;
    private TextView tvSeqProgress;
    private ProgressBar progressSeq;
    private Button btnStartSeq;
    private Button btnStopSeq;
    private TextView tvSeqStatus;
    private LinearLayout layoutVerdict;
    private TextView tvSeqVerdict;
    private TextView tvSeqDetails;
    private Button btnFeedback;
    private TextView tvSeqFeedback;

    // Camera2 state. The preview is live as soon as this screen opens
    // (and permission is granted); START only starts the judging.
    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private ImageReader imageReader;
    private Surface previewSurface;
    private HandlerThread cameraThread;
    private Handler cameraHandler;
    // Newest JPEG frame from the ImageReader, replaced as frames arrive.
    private volatile byte[] latestJpeg;

    // Sequence state.
    private final ScheduledExecutorService sampler =
            Executors.newSingleThreadScheduledExecutor();
    private ScheduledFuture<?> sampleFuture;
    private volatile boolean running = false;
    private SequenceSession session;
    private int sampleIndex;              // next frame's t (1 fps: t == index)
    private String runCommand;            // e.g. "Stay"
    private String runTargetLabel;        // e.g. "sitting"
    private int runDurationSec;
    private String backendBase;           // URL captured at START
    private volatile String lastFrameB64; // last captured frame, for feedback
    private SequenceSession.Verdict lastVerdict;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sequence);

        etSeqUrl = findViewById(R.id.etSeqUrl);
        spCommand = findViewById(R.id.spCommand);
        spStayPosture = findViewById(R.id.spStayPosture);
        spDuration = findViewById(R.id.spDuration);
        texturePreview = findViewById(R.id.texturePreview);
        tvSeqCurrent = findViewById(R.id.tvSeqCurrent);
        tvSeqProgress = findViewById(R.id.tvSeqProgress);
        progressSeq = findViewById(R.id.progressSeq);
        btnStartSeq = findViewById(R.id.btnStartSeq);
        btnStopSeq = findViewById(R.id.btnStopSeq);
        tvSeqStatus = findViewById(R.id.tvSeqStatus);
        layoutVerdict = findViewById(R.id.layoutVerdict);
        tvSeqVerdict = findViewById(R.id.tvSeqVerdict);
        tvSeqDetails = findViewById(R.id.tvSeqDetails);
        btnFeedback = findViewById(R.id.btnFeedback);
        tvSeqFeedback = findViewById(R.id.tvSeqFeedback);

        String url = getIntent().getStringExtra(EXTRA_BACKEND_URL);
        etSeqUrl.setText(url == null || url.isEmpty() ? MainActivity.BACKEND_URL : url);

        spCommand.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, COMMANDS));
        spStayPosture.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, STAY_POSTURES));
        spDuration.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, DURATIONS));
        spDuration.setSelection(1); // default 10 seconds

        btnStartSeq.setOnClickListener(v -> startSequence());
        btnStopSeq.setOnClickListener(v -> stopSequence(true));
        btnFeedback.setOnClickListener(v -> {
            if (lastVerdict != null) fetchFeedback(lastVerdict);
        });

        cameraThread = new HandlerThread("CameraThread");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());

        texturePreview.setSurfaceTextureListener(new TextureView.SurfaceTextureListener() {
            @Override
            public void onSurfaceTextureAvailable(SurfaceTexture st, int w, int h) {
                ensureCameraPermissionThenOpen();
            }

            @Override
            public void onSurfaceTextureSizeChanged(SurfaceTexture st, int w, int h) {
            }

            @Override
            public boolean onSurfaceTextureDestroyed(SurfaceTexture st) {
                closeCamera();
                return true;
            }

            @Override
            public void onSurfaceTextureUpdated(SurfaceTexture st) {
            }
        });
    }

    // ---- camera permission + open ------------------------------------

    private void ensureCameraPermissionThenOpen() {
        if (checkSelfPermission(Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA_PERM);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions,
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_CAMERA_PERM) {
            if (grantResults.length > 0
                    && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                if (texturePreview.isAvailable()) openCamera();
            } else {
                tvSeqStatus.setText("Camera permission denied - sequence mode "
                        + "needs the camera to watch the dog.");
            }
        }
    }

    @SuppressLint("MissingPermission") // checked in ensureCameraPermissionThenOpen
    private void openCamera() {
        if (cameraDevice != null || !texturePreview.isAvailable()) return;
        try {
            CameraManager manager = (CameraManager) getSystemService(CAMERA_SERVICE);
            if (manager == null) return;

            // Prefer the back camera; fall back to any camera.
            String cameraId = null;
            CameraCharacteristics chosen = null;
            for (String id : manager.getCameraIdList()) {
                CameraCharacteristics cc = manager.getCameraCharacteristics(id);
                Integer facing = cc.get(CameraCharacteristics.LENS_FACING);
                if (facing != null && facing == CameraCharacteristics.LENS_FACING_BACK) {
                    cameraId = id;
                    chosen = cc;
                    break;
                }
                if (cameraId == null) {
                    cameraId = id;
                    chosen = cc;
                }
            }
            if (cameraId == null || chosen == null) {
                tvSeqStatus.setText("No camera found on this device.");
                return;
            }

            StreamConfigurationMap map = chosen.get(
                    CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            if (map == null) return;
            Size previewSize = chooseSize(map.getOutputSizes(SurfaceTexture.class), 1280, 720);
            Size jpegSize = chooseSize(map.getOutputSizes(ImageFormat.JPEG), 1280, 1280);

            SurfaceTexture st = texturePreview.getSurfaceTexture();
            st.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());
            previewSurface = new Surface(st);

            // The ImageReader gives us the actual frame bytes (JPEG) that
            // the sampler sends to the backend. We only keep the newest.
            imageReader = ImageReader.newInstance(
                    jpegSize.getWidth(), jpegSize.getHeight(), ImageFormat.JPEG, 2);
            imageReader.setOnImageAvailableListener(reader -> {
                Image img = reader.acquireLatestImage();
                if (img == null) return;
                try {
                    ByteBuffer buf = img.getPlanes()[0].getBuffer();
                    byte[] bytes = new byte[buf.remaining()];
                    buf.get(bytes);
                    latestJpeg = bytes;
                } finally {
                    img.close();
                }
            }, cameraHandler);

            manager.openCamera(cameraId, new CameraDevice.StateCallback() {
                @Override
                public void onOpened(CameraDevice device) {
                    cameraDevice = device;
                    startPreviewSession();
                }

                @Override
                public void onDisconnected(CameraDevice device) {
                    device.close();
                    cameraDevice = null;
                }

                @Override
                public void onError(CameraDevice device, int error) {
                    device.close();
                    cameraDevice = null;
                    runOnUiThread(() -> tvSeqStatus.setText(
                            "Camera error " + error + " - preview unavailable."));
                }
            }, cameraHandler);
        } catch (Exception e) {
            tvSeqStatus.setText("Could not open camera: " + e.getMessage());
        }
    }

    private void startPreviewSession() {
        if (cameraDevice == null) return;
        try {
            List<Surface> surfaces = Arrays.asList(previewSurface, imageReader.getSurface());
            cameraDevice.createCaptureSession(surfaces,
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(CameraCaptureSession session) {
                            captureSession = session;
                            try {
                                CaptureRequest.Builder b = cameraDevice.createCaptureRequest(
                                        CameraDevice.TEMPLATE_PREVIEW);
                                b.addTarget(previewSurface);
                                b.addTarget(imageReader.getSurface());
                                session.setRepeatingRequest(b.build(), null, cameraHandler);
                                runOnUiThread(() -> tvSeqStatus.setText(
                                        "Camera live. Pick a command and tap START SEQUENCE."));
                            } catch (CameraAccessException e) {
                                runOnUiThread(() -> tvSeqStatus.setText(
                                        "Preview failed: " + e.getMessage()));
                            }
                        }

                        @Override
                        public void onConfigureFailed(CameraCaptureSession session) {
                            runOnUiThread(() -> tvSeqStatus.setText(
                                    "Camera session failed to configure."));
                        }
                    }, cameraHandler);
        } catch (CameraAccessException e) {
            tvSeqStatus.setText("Preview failed: " + e.getMessage());
        }
    }

    // Largest size that fits the caps, or the smallest available.
    private Size chooseSize(Size[] choices, int maxW, int maxH) {
        Size best = null;
        for (Size s : choices) {
            if (s.getWidth() <= maxW && s.getHeight() <= maxH
                    && (best == null || s.getWidth() * s.getHeight()
                    > best.getWidth() * best.getHeight())) {
                best = s;
            }
        }
        if (best != null) return best;
        Size smallest = choices[0];
        for (Size s : choices) {
            if (s.getWidth() * s.getHeight() < smallest.getWidth() * smallest.getHeight()) {
                smallest = s;
            }
        }
        return smallest;
    }

    private void closeCamera() {
        if (captureSession != null) {
            captureSession.close();
            captureSession = null;
        }
        if (cameraDevice != null) {
            cameraDevice.close();
            cameraDevice = null;
        }
        if (imageReader != null) {
            imageReader.close();
            imageReader = null;
        }
        if (previewSurface != null) {
            previewSurface.release();
            previewSurface = null;
        }
    }

    // ---- the sequence run ---------------------------------------------

    // Command -> target posture. Stay holds whatever posture the trainer
    // picked in the Stay posture spinner (a "stay" is a hold of a
    // posture, not a posture of its own).
    private String targetFor(String command, String stayPosture) {
        switch (command) {
            case "Sit":
                return "sitting";
            case "Down":
                return "lying_down";
            case "Stand":
                return "standing";
            case "Walk":
                return "walking";
            case "Stay":
                return SequenceSession.canonicalLabel(stayPosture);
            default:
                return "sitting";
        }
    }

    private void startSequence() {
        String base = etSeqUrl.getText().toString().trim();
        if (base.isEmpty() || base.contains("YOUR-CLOUD-RUN-URL")) {
            tvSeqStatus.setText("Set the real backend URL first.");
            return;
        }
        if (base.endsWith("/")) base = base.substring(0, base.length() - 1);
        backendBase = base;

        runCommand = COMMANDS[spCommand.getSelectedItemPosition()];
        runTargetLabel = targetFor(runCommand,
                STAY_POSTURES[spStayPosture.getSelectedItemPosition()]);
        runDurationSec = DURATION_SECONDS[spDuration.getSelectedItemPosition()];

        // State machine config: window 3, grace 2, break confirm 1,
        // lost timeout 3, uncertain ratio 0.4 - at 1 fps these numbers
        // are frames = seconds (see SequenceSession's header comment).
        session = new SequenceSession(runTargetLabel,
                new SequenceSession.Config(runDurationSec), runCommand);
        sampleIndex = 0;
        lastFrameB64 = null;
        lastVerdict = null;
        running = true;

        layoutVerdict.setVisibility(View.GONE);
        tvSeqCurrent.setText("Current posture: -");
        progressSeq.setMax(runDurationSec);
        progressSeq.setProgress(0);
        tvSeqProgress.setText("Running... waiting for frames.");
        tvSeqStatus.setText("Sequence running: command " + runCommand
                + ", target '" + pretty(runTargetLabel) + "' for "
                + runDurationSec + "s. One frame per second is judged.");
        setControlsRunning(true);

        // Fixed-delay so a slow network call never overlaps the next
        // sample; a timed-out call just makes that frame a no-label one.
        sampleFuture = sampler.scheduleWithFixedDelay(
                this::sampleOnce, 0, 1, TimeUnit.SECONDS);
    }

    // One sampler tick: classify the newest frame and feed the machine.
    // Runs on the sampler thread; each fed frame advances the sequence
    // clock by 1 (t = sample index), so the state machine's seconds are
    // frames - the timeline in SequenceSession's comments.
    private void sampleOnce() {
        if (!running || session == null) return;
        byte[] jpeg = latestJpeg;
        if (jpeg == null) {
            runOnUiThread(() -> tvSeqStatus.setText(
                    "Sequence running... waiting for the first camera frame."));
            return;
        }

        String label = null;
        Bitmap bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        if (bmp != null) {
            String b64 = Base64.encodeToString(toJpeg640(bmp), Base64.NO_WRAP);
            lastFrameB64 = b64;
            String text = postModel(backendBase + "/api/fine-tuned-gemini",
                    FRAME_PROMPT, b64);
            if (text != null) label = parseLabel(text);
        }

        double t = sampleIndex++;
        SequenceSession.Verdict v = session.feed(t, label);
        runOnUiThread(() -> updateLiveUi());
        if (v != null) onVerdict(v);
    }

    private void updateLiveUi() {
        if (session == null) return;
        String cur = session.getLastSmoothed();
        tvSeqCurrent.setText("Current posture: "
                + (cur == null ? "-" : pretty(cur)));
        double held = 0;
        if (session.getReachedAt() != null && session.getLastT() != null) {
            held = Math.max(0, session.getLastT() - session.getReachedAt());
        }
        progressSeq.setProgress((int) Math.min(runDurationSec, held));
        tvSeqProgress.setText(String.format(Locale.US,
                "Frames sampled: %d (t = %.0fs) - held %.0fs / %ds",
                session.getFrameCount(),
                session.getLastT() == null ? 0 : session.getLastT(),
                held, runDurationSec));
    }

    // STOP tapped (fromUser) or the screen is going away: finish() gives
    // the verdict for an interval that ended undecided.
    private void stopSequence(boolean fromUser) {
        if (!running || session == null) return;
        running = false;
        if (sampleFuture != null) sampleFuture.cancel(false);
        onVerdict(session.finish());
    }

    private void onVerdict(SequenceSession.Verdict v) {
        running = false;
        if (sampleFuture != null) sampleFuture.cancel(false);
        lastVerdict = v;
        runOnUiThread(() -> {
            setControlsRunning(false);
            updateLiveUi();
            layoutVerdict.setVisibility(View.VISIBLE);
            tvSeqVerdict.setText("Verdict: " + v.verdict);
            tvSeqStatus.setText("Sequence finished: " + v.verdict);

            StringBuilder sb = new StringBuilder();
            sb.append("Command: ").append(runCommand)
                    .append(" - target '").append(pretty(v.targetLabel))
                    .append("' for ").append(String.format(Locale.US, "%.0f", v.targetDurationS))
                    .append("s\n");
            sb.append("Held: ").append(String.format(Locale.US, "%.1f", v.heldS)).append("s");
            if (v.breakAtS != null) {
                sb.append(" - broke at ").append(String.format(Locale.US, "%.0f", v.breakAtS))
                        .append("s");
            }
            sb.append("\nFrames judged: ").append(v.nFrames)
                    .append(" - unclear frames: ")
                    .append(String.format(Locale.US, "%.0f", v.uncertainRatio * 100))
                    .append("%\n");
            sb.append("Reason: ").append(v.reason).append("\n");
            sb.append("Smoothed label sequence (1 per second):\n");
            List<String> prettySeq = new ArrayList<>();
            for (String step : v.smoothedSequence) {
                // stored as "t label"
                int sp = step.indexOf(' ');
                prettySeq.add(pretty(step.substring(sp + 1)));
            }
            sb.append(String.join(", ", prettySeq));
            tvSeqDetails.setText(sb.toString());
            tvSeqFeedback.setText("(getting trainer feedback...)");
        });
        // Model B at runtime: auto-run the feedback request; the button
        // stays available to re-run it.
        fetchFeedback(v);
    }

    private void setControlsRunning(boolean isRunning) {
        btnStartSeq.setEnabled(!isRunning);
        btnStopSeq.setEnabled(isRunning);
        spCommand.setEnabled(!isRunning);
        spStayPosture.setEnabled(!isRunning);
        spDuration.setEnabled(!isRunning);
        etSeqUrl.setEnabled(!isRunning);
    }

    // ---- Model B feedback ----------------------------------------------
    // The deployed backend requires imageBase64 on EVERY call (there is
    // no text-only endpoint), so the last captured frame rides along with
    // the feedback prompt. The prompt itself carries the attempt's facts:
    // command, target duration, verdict, held time, and the full label
    // sequence, followed by the expert-trainer instruction.
    private void fetchFeedback(SequenceSession.Verdict v) {
        String b64 = lastFrameB64;
        if (b64 == null) {
            runOnUiThread(() -> tvSeqFeedback.setText(
                    "No frame was captured, so no feedback call was made."));
            return;
        }
        List<String> labels = new ArrayList<>();
        for (String step : v.smoothedSequence) {
            labels.add(pretty(step.substring(step.indexOf(' ') + 1)));
        }
        String prompt = "Command: " + runCommand
                + ". Target duration: " + runDurationSec + " seconds. "
                + "Verdict: " + v.verdict + ". "
                + "The dog held the target posture for "
                + String.format(Locale.US, "%.1f", v.heldS) + " seconds. "
                + "Posture sequence (one label per second): "
                + String.join(", ", labels) + ". "
                + "You are an expert dog training assistant. "
                + "In 2-3 short sentences, evaluate this attempt and give "
                + "the trainer one concrete next step.";
        String url = backendBase;
        new Thread(() -> {
            String text = postModel(url + "/api/gemini", prompt, b64);
            runOnUiThread(() -> tvSeqFeedback.setText(text == null
                    ? "Feedback request failed (network or backend error)."
                    : text));
        }).start();
    }

    // ---- backend calls ---------------------------------------------------
    // POST {prompt, imageBase64, mimeType}; returns the "text" field, or
    // null on any failure. (MainActivity's callModel returns error
    // strings for display; here null is meaningful - it marks the frame
    // as a no-label frame for the state machine.)
    private String postModel(String endpoint, String prompt, String imageBase64) {
        HttpURLConnection conn = null;
        try {
            JSONObject payload = new JSONObject();
            payload.put("prompt", prompt);
            payload.put("mimeType", "image/jpeg");
            payload.put("imageBase64", imageBase64);

            conn = (HttpURLConnection) new URL(endpoint).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            // Shorter than the photo flow's 60s: a hung call must not
            // stall the 1 fps sampler for a whole minute. A timeout just
            // makes this frame a no-label frame.
            conn.setConnectTimeout(20000);
            conn.setReadTimeout(20000);
            conn.setDoOutput(true);
            OutputStream out = conn.getOutputStream();
            out.write(payload.toString().getBytes(StandardCharsets.UTF_8));
            out.close();

            int code = conn.getResponseCode();
            InputStream in = code >= 200 && code < 300
                    ? conn.getInputStream() : conn.getErrorStream();
            String raw = readAll(in);
            if (code < 200 || code >= 300) return null;
            return new JSONObject(raw).optString("text", null);
        } catch (Exception e) {
            return null;
        } finally {
            if (conn != null) conn.disconnect();
        }
    }

    private String readAll(InputStream in) throws Exception {
        if (in == null) return "{}";
        BufferedReader r = new BufferedReader(
                new InputStreamReader(in, StandardCharsets.UTF_8));
        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = r.readLine()) != null) sb.append(line);
        r.close();
        return sb.toString();
    }

    // Parse a model reply to a canonical label. The tuned model was asked
    // for exactly one word, but the v3 evaluation showed it sometimes
    // answers in a sentence ("The dog is standing."), so this is a simple
    // keyword scan; anything unrecognisable is a no-label frame (null).
    private String parseLabel(String text) {
        String s = text.trim().toLowerCase(Locale.US);
        if (s.contains("lying") || s.equals("down")) return "lying_down";
        if (s.contains("sitting")) return "sitting";
        if (s.contains("standing")) return "standing";
        if (s.contains("walking")) return "walking";
        return null;
    }

    // Downscale to ~640px wide (the dataset/backend frame size) and
    // re-encode as JPEG so the JSON body stays small.
    private byte[] toJpeg640(Bitmap bmp) {
        Bitmap use = bmp;
        if (bmp.getWidth() > 640) {
            float scale = 640f / bmp.getWidth();
            use = Bitmap.createScaledBitmap(bmp, 640,
                    Math.round(bmp.getHeight() * scale), true);
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        use.compress(Bitmap.CompressFormat.JPEG, 80, baos);
        return baos.toByteArray();
    }

    private String pretty(String label) {
        if (label == null) return "-";
        if (label.equals("lying_down")) return "lying down";
        if (label.equals(SequenceSession.UNCERTAIN_FRAME)) return "no label";
        return label;
    }

    // ---- lifecycle ------------------------------------------------------

    @Override
    protected void onPause() {
        super.onPause();
        // Judging stops with the screen; finish() still leaves a verdict
        // on screen for when the user comes back.
        if (running) stopSequence(false);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        running = false;
        if (sampleFuture != null) sampleFuture.cancel(false);
        sampler.shutdownNow();
        closeCamera();
        if (cameraThread != null) cameraThread.quitSafely();
    }
}
