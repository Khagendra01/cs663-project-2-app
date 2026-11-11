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
// app judges a command over TIME, following the updated proposal
// (Oct 26, 2026) sections 3, 5 and 7:
//
// How it works:
//   0. READY gate (proposal 5.2): START SEQUENCE starts disabled. While
//      this screen is open, a readiness sampler classifies one frame
//      per second and shows the live posture. Start enables only once
//      the required starting posture is observed stable - Sit starts
//      from Stand, Down starts from Sit, Stay/Stand start from the
//      selected posture (Walk has no starting-posture requirement; it
//      is outside the proposal's MVP table). At 1 fps, "stable >=0.6s"
//      maps to 1 confirmed (smoothed) frame of the required posture.
//      The moment the trainer taps Start is t=0.
//   1. The trainer picks a command (Sit / Down / Stand / Stay / Walk)
//      and a target duration (Stay default 5s), then taps START SEQUENCE.
//   2. The camera (Camera2, live preview on a TextureView) also streams
//      JPEG frames into an ImageReader. Once per second the sampler
//      takes the newest frame, shrinks it to ~640px wide, and POSTs it
//      to /api/fine-tuned-gemini with the fixed posture prompt.
//   3. Each reply is parsed to a canonical label and fed into
//      SequenceSession - the Java port of the Python temporal state
//      machine in files/video-sequence-layer/state_machine.py - which
//      decides SUCCESS / EARLY_BREAK / UNCERTAIN with smoothing, a
//      settle grace / response window, and a sustained-break rule.
//      Sit/Down run as RESPONSE then HOLD: reach the target within
//      3.0s, then hold it 0.6s (1 frame at 1 fps).
//   4. When a verdict is decided, the app asks the base model
//      (/api/gemini) for feedback as a JSON object (proposal 7):
//      the prompt carries the command, target duration and the
//      timestamped posture timeline - but NOT the engine's verdict
//      and NOT the break time, so Model B judges the evidence
//      independently. If the model's status disagrees with the local
//      verdict, or the reply cannot be parsed, the LOCAL verdict
//      stands with a deterministic template tip; the model's verdict
//      is never displayed over the local one.
//   The app stays native Android (owner decision, Nov 11): no Flutter
//   rebuild - the outcome is the same working app, and the proposal's
//   ~5 fps analysis rate is mapped to 1 fps here because each frame
//   is a cloud call (~1s+ round trip). The mapping is documented in
//   SequenceSession's header comment, not hidden.
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

    // READY-gate state (proposal 5.2). The readiness sampler runs while
    // this screen is open and no sequence is running: it classifies one
    // frame per second through the same smoother the session uses, and
    // START enables only while the required starting posture is the
    // current smoothed label. Changing the command or Stay posture
    // resets the check.
    private ScheduledFuture<?> readyFuture;
    private volatile boolean readyRunning = false;
    private volatile boolean readyGateOpen = false;
    private volatile boolean paused = false;
    private SequenceSession.TemporalSmoother readinessSmoother;
    private String readyRequiredLabel;    // null = no requirement (Walk)
    private volatile String readyLastSmoothed;
    private volatile boolean readyFrameSeen = false;

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
        spDuration.setSelection(0); // default 5 seconds (proposal 5.2)

        // The READY gate owns the Start button: it starts disabled and
        // is enabled only by the readiness sampler (see startReadiness).
        btnStartSeq.setEnabled(false);

        // Changing the command or the Stay posture changes the required
        // starting posture, so the readiness check restarts from scratch.
        // Sit/Down use the fixed 3s response window + 0.6s hold, so the
        // duration spinner does not apply to them and is disabled.
        android.widget.AdapterView.OnItemSelectedListener readyReset =
                new android.widget.AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(
                            android.widget.AdapterView<?> parent, View view,
                            int position, long id) {
                        updateDurationEnabled();
                        resetReadiness();
                    }

                    @Override
                    public void onNothingSelected(
                            android.widget.AdapterView<?> parent) {
                    }
                };
        spCommand.setOnItemSelectedListener(readyReset);
        spStayPosture.setOnItemSelectedListener(readyReset);
        updateDurationEnabled();

        btnStartSeq.setOnClickListener(v -> startSequence());
        btnStopSeq.setOnClickListener(v -> stopSequence(true));
        btnFeedback.setOnClickListener(v -> {
            if (lastVerdict != null) fetchFeedback(lastVerdict);
        });

        cameraThread = new HandlerThread("CameraThread");
        cameraThread.start();
        cameraHandler = new Handler(cameraThread.getLooper());

        startReadiness();

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

    // Command -> required STARTING posture for the READY gate
    // (proposal 5.2: Sit starts from Stand, Down starts from Sit, Stay
    // starts from the selected posture). Stand is not in the proposal's
    // MVP table; it is a hold of the standing posture, so its starting
    // posture is standing, like Stay (Standing). Walk has no starting
    // posture requirement (null) - it is outside the proposal's MVP
    // table and is kept as an extra command.
    private String requiredStartingLabel(String command, String stayPosture) {
        switch (command) {
            case "Sit":
                return "standing";
            case "Down":
                return "sitting";
            case "Stand":
                return "standing";
            case "Stay":
                return SequenceSession.canonicalLabel(stayPosture);
            case "Walk":
                return null;
            default:
                return "standing";
        }
    }

    private boolean isReachThenHoldCommand(String command) {
        return "Sit".equals(command) || "Down".equals(command);
    }

    private void updateDurationEnabled() {
        String cmd = COMMANDS[spCommand.getSelectedItemPosition()];
        // Sit/Down use the fixed 3s response window + 0.6s hold; the
        // duration spinner does not apply to them.
        spDuration.setEnabled(!isReachThenHoldCommand(cmd));
    }

    // ---- READY gate ----------------------------------------------------
    // The readiness sampler runs while the screen is open and no
    // sequence is running. Same 1 fps classification as a run, through
    // the same majority-vote smoother - but nothing is judged; the only
    // output is whether START may be tapped. At 1 fps, the proposal's
    // "starting posture stable >=0.6s" maps to 1 confirmed (smoothed)
    // frame of the required posture. The gate is a UI enablement, not
    // a verdict: the session still judges the attempt from t=0, so a
    // single noisy frame opening the gate cannot fake a result.
    private void startReadiness() {
        if (readyRunning || running || paused) return;
        readyRequiredLabel = requiredStartingLabel(
                COMMANDS[spCommand.getSelectedItemPosition()],
                STAY_POSTURES[spStayPosture.getSelectedItemPosition()]);
        readinessSmoother = new SequenceSession.TemporalSmoother(3, 0.6);
        readyGateOpen = false;
        readyFrameSeen = false;
        readyLastSmoothed = null;
        readyRunning = true;
        updateStartButton();
        readyFuture = sampler.scheduleWithFixedDelay(
                this::readySampleOnce, 0, 1, TimeUnit.SECONDS);
    }

    private void stopReadiness() {
        readyRunning = false;
        if (readyFuture != null) readyFuture.cancel(false);
        readyFuture = null;
    }

    private void resetReadiness() {
        stopReadiness();
        updateStartButton();
        startReadiness();
    }

    private void updateStartButton() {
        btnStartSeq.setEnabled(readyGateOpen && !running);
    }

    private void readySampleOnce() {
        if (!readyRunning || running || paused) return;
        byte[] jpeg = latestJpeg;
        if (jpeg == null) {
            runOnUiThread(() -> {
                tvSeqCurrent.setText("Current posture: -");
                tvSeqStatus.setText("Waiting for the first camera frame...");
            });
            return;
        }
        String base = etSeqUrl.getText().toString().trim();
        if (base.isEmpty() || base.contains("YOUR-CLOUD-RUN-URL")) {
            runOnUiThread(() -> {
                readyGateOpen = false;
                updateStartButton();
                tvSeqStatus.setText("Set the real backend URL first.");
            });
            return;
        }
        if (base.endsWith("/")) base = base.substring(0, base.length() - 1);

        String label = null;
        Bitmap bmp = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.length);
        if (bmp != null) {
            String b64 = Base64.encodeToString(toJpeg640(bmp), Base64.NO_WRAP);
            String text = postModel(base + "/api/fine-tuned-gemini",
                    FRAME_PROMPT, b64);
            if (text != null) label = parseLabel(text);
        }

        String smoothed = readinessSmoother.push(label, null, label != null);
        readyLastSmoothed = smoothed;
        readyFrameSeen = true;
        // Gate: the required starting posture is the current smoothed
        // label (Walk: no requirement, any classified frame will do).
        readyGateOpen = readyRequiredLabel == null
                || smoothed.equals(readyRequiredLabel);
        final String showSmoothed = smoothed;
        runOnUiThread(() -> {
            tvSeqCurrent.setText("Current posture: " + pretty(showSmoothed));
            if (readyGateOpen) {
                tvSeqStatus.setText(readyRequiredLabel == null
                        ? "Ready - Walk has no starting posture. Tap START SEQUENCE."
                        : "Ready: the dog is " + pretty(readyRequiredLabel)
                                + ". Tap START SEQUENCE.");
            } else {
                tvSeqStatus.setText("Waiting for starting posture: "
                        + pretty(readyRequiredLabel) + "...");
            }
            updateStartButton();
        });
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

        // The READY gate has done its job; t=0 is now.
        stopReadiness();
        readyGateOpen = false;

        // State machine config: window 3, grace 2, break confirm 1,
        // lost after 2 unclear frames, uncertain ratio 0.4 - at 1 fps
        // these numbers are frames = seconds (see SequenceSession's
        // header comment). Sit/Down run as RESPONSE then HOLD
        // (proposal 5.2): reach the target within 3s, then hold 0.6s.
        SequenceSession.Config cfg =
                new SequenceSession.Config(runDurationSec);
        cfg.reachThenHold = isReachThenHoldCommand(runCommand);
        session = new SequenceSession(runTargetLabel, cfg, runCommand);
        sampleIndex = 0;
        lastFrameB64 = null;
        lastVerdict = null;
        running = true;

        layoutVerdict.setVisibility(View.GONE);
        tvSeqCurrent.setText("Current posture: -");
        progressSeq.setMax(runDurationSec);
        progressSeq.setProgress(0);
        tvSeqProgress.setText("Running... waiting for frames.");
        if (cfg.reachThenHold) {
            tvSeqStatus.setText("Sequence running: command " + runCommand
                    + " - reach '" + pretty(runTargetLabel) + "' within 3s, "
                    + "then hold it for 0.6s. One frame per second is judged.");
        } else {
            tvSeqStatus.setText("Sequence running: command " + runCommand
                    + ", target '" + pretty(runTargetLabel) + "' for "
                    + runDurationSec + "s. One frame per second is judged.");
        }
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
        if (session.isReachThenHold()) {
            // Sit/Down: the progress line is the response window /
            // confirmation hold, not the duration spinner.
            progressSeq.setMax(3);
            progressSeq.setProgress((int) Math.min(3,
                    session.getLastT() == null ? 0 : session.getLastT()));
            if ("RESPONSE".equals(session.getPhase())) {
                tvSeqProgress.setText(String.format(Locale.US,
                        "Frames sampled: %d (t = %.0fs) - waiting for the dog "
                                + "to reach '%s' (3s window)",
                        session.getFrameCount(),
                        session.getLastT() == null ? 0 : session.getLastT(),
                        pretty(session.getTargetLabel())));
            } else {
                tvSeqProgress.setText(String.format(Locale.US,
                        "Frames sampled: %d (t = %.0fs) - target reached, "
                                + "confirming the 0.6s hold",
                        session.getFrameCount(),
                        session.getLastT() == null ? 0 : session.getLastT()));
            }
            return;
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
            sb.append("Command: ").append(runCommand);
            if (isReachThenHoldCommand(runCommand)) {
                // Sit/Down rule (proposal 5.2): reach within 3s, hold
                // 0.6s - the duration spinner does not apply here.
                sb.append(" - rule: reach '").append(pretty(v.targetLabel))
                        .append("' within 3s, then hold 0.6s\n");
            } else {
                sb.append(" - target '").append(pretty(v.targetLabel))
                        .append("' for ")
                        .append(String.format(Locale.US, "%.0f", v.targetDurationS))
                        .append("s\n");
            }
            sb.append("Held: ").append(String.format(Locale.US, "%.1f", v.heldS)).append("s");
            if (v.breakAtS != null) {
                // Break time is when the dog FIRST left the posture
                // (proposal 5.3), not when the break was confirmed.
                sb.append(" - broke at ").append(String.format(Locale.US, "%.1f", v.breakAtS))
                        .append("s (first wrong posture)");
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
        // Back to the READY gate so the trainer can run again once the
        // dog is back in the required starting posture.
        startReadiness();
    }

    private void setControlsRunning(boolean isRunning) {
        // The Start button is owned by the READY gate, never enabled
        // directly here - after a run it stays disabled until the gate
        // re-opens (startReadiness, above).
        if (isRunning) {
            btnStartSeq.setEnabled(false);
        } else {
            updateStartButton();
        }
        btnStopSeq.setEnabled(isRunning);
        spCommand.setEnabled(!isRunning);
        spStayPosture.setEnabled(!isRunning);
        // Sit/Down never use the duration spinner (fixed 3s + 0.6s rule).
        spDuration.setEnabled(!isRunning
                && !(runCommand != null && isReachThenHoldCommand(runCommand)));
        etSeqUrl.setEnabled(!isRunning);
    }

    // ---- Model B feedback (proposal 7) --------------------------------
    // The deployed backend requires imageBase64 on EVERY call (there is
    // no text-only endpoint), so the last captured frame rides along
    // with the feedback prompt. The prompt itself carries ONLY the
    // evidence: command, target duration, and the timestamped smoothed
    // posture timeline. It deliberately does NOT carry the engine's
    // verdict and does NOT carry the break time, so Model B judges the
    // evidence independently instead of copying the local decision.
    // The reply contract is a JSON object with exactly: status
    // (success / early_break / uncertain), break_time_s (number or
    // null), evidence_summary, coaching_tip, and optional caution.
    //
    // Authority rule (proposal 7): the on-device temporal engine is
    // the authority for status and timing. resolveFeedback() compares
    // the model's status with the LOCAL verdict: if they agree, the
    // model's summary and tip are shown; if they disagree, or the
    // reply cannot be parsed, or the call fails, the LOCAL verdict is
    // shown with a deterministic template tip. The model's verdict is
    // NEVER displayed when it disagrees with the local one.
    private void fetchFeedback(SequenceSession.Verdict v) {
        String b64 = lastFrameB64;
        if (b64 == null) {
            runOnUiThread(() -> tvSeqFeedback.setText(
                    "No frame was captured, so no feedback call was made."));
            return;
        }
        List<String> timeline = new ArrayList<>();
        for (String step : v.smoothedSequence) {
            int sp = step.indexOf(' ');
            timeline.add(step.substring(0, sp) + "s: "
                    + pretty(step.substring(sp + 1)));
        }
        String prompt = "You are an expert dog training assistant. "
                + "Review this training attempt and reply with a JSON "
                + "object only, no other text.\n"
                + "Command: " + runCommand
                + ". Target duration: " + runDurationSec + " seconds.\n"
                + "Timestamped smoothed posture timeline "
                + "(one label per second): "
                + String.join(", ", timeline) + ".\n"
                + "Reply with a JSON object with exactly these fields: "
                + "\"status\" (one of: success, early_break, uncertain), "
                + "\"break_time_s\" (a number, or null), "
                + "\"evidence_summary\" (one sentence), "
                + "\"coaching_tip\" (one short sentence), "
                + "\"caution\" (optional, one short sentence).";
        String url = backendBase;
        new Thread(() -> {
            String text = postModel(url + "/api/gemini", prompt, b64);
            runOnUiThread(() -> tvSeqFeedback.setText(
                    resolveFeedback(v, text)));
        }).start();
    }

    // Parse the model's JSON reply (tolerating a ```json code fence),
    // compare its status with the local verdict, and decide what to
    // show. See the authority rule in the comment above fetchFeedback.
    private String resolveFeedback(SequenceSession.Verdict v, String text) {
        String modelStatus = null;
        String summary = null;
        String tip = null;
        String caution = null;
        if (text != null) {
            try {
                String s = text.trim();
                if (s.startsWith("```")) {
                    int nl = s.indexOf('\n');
                    if (nl >= 0) s = s.substring(nl + 1);
                    int end = s.lastIndexOf("```");
                    if (end > 0) s = s.substring(0, end);
                    s = s.trim();
                }
                JSONObject j = new JSONObject(s);
                String rawStatus = j.optString("status", "")
                        .trim().toLowerCase(Locale.US);
                if (rawStatus.equals("success")) modelStatus = "SUCCESS";
                else if (rawStatus.equals("early_break")) modelStatus = "EARLY_BREAK";
                else if (rawStatus.equals("uncertain")) modelStatus = "UNCERTAIN";
                summary = j.optString("evidence_summary", null);
                tip = j.optString("coaching_tip", null);
                caution = j.optString("caution", null);
                if (summary != null && summary.isEmpty()) summary = null;
                if (tip != null && tip.isEmpty()) tip = null;
                if (caution != null && caution.isEmpty()) caution = null;
            } catch (Exception e) {
                // Unparseable reply: fall through to the local template.
                modelStatus = null;
            }
        }
        if (modelStatus != null && modelStatus.equals(v.verdict)
                && summary != null && tip != null) {
            StringBuilder sb = new StringBuilder();
            sb.append("Model agrees with the local verdict (")
                    .append(v.verdict).append(").\n");
            sb.append(summary).append("\nTip: ").append(tip);
            if (caution != null) sb.append("\nNote: ").append(caution);
            return sb.toString();
        }
        return templateFeedback(v);
    }

    // Deterministic fallback: the local verdict plus a fixed tip per
    // verdict. Used when the model disagrees with the local verdict,
    // when its reply cannot be parsed, or when the call fails.
    private String templateFeedback(SequenceSession.Verdict v) {
        switch (v.verdict) {
            case "SUCCESS":
                return "Local verdict: SUCCESS. The dog held '"
                        + pretty(v.targetLabel)
                        + "' for the full target. Praise and reward right "
                        + "away. Next time, try a slightly longer target.";
            case "EARLY_BREAK":
                if (isReachThenHoldCommand(runCommand)) {
                    if (v.targetReachedAtS == null) {
                        return "Local verdict: EARLY_BREAK. The dog did "
                                + "not reach '" + pretty(v.targetLabel)
                                + "' within the 3s response window. Do not "
                                + "reward. Next time, lure the dog into "
                                + "position first and try again.";
                    }
                    return "Local verdict: EARLY_BREAK. The dog reached '"
                            + pretty(v.targetLabel) + "' but left it after "
                            + String.format(Locale.US, "%.1f", v.heldS)
                            + "s, before the 0.6s hold was complete. Do not "
                            + "reward the break. Next time, reward the "
                            + "moment the dog reaches the posture.";
                }
                return "Local verdict: EARLY_BREAK. The dog held '"
                        + pretty(v.targetLabel) + "' for "
                        + String.format(Locale.US, "%.1f", v.heldS)
                        + "s of the "
                        + String.format(Locale.US, "%.0f", v.targetDurationS)
                        + "s target. Do not reward the break. Next time "
                        + "set a shorter target, below the held time, and "
                        + "reduce distractions.";
            default: // UNCERTAIN
                return "Local verdict: UNCERTAIN. The attempt could not "
                        + "be judged. Reposition the camera so the whole "
                        + "dog is in frame and try again. Do not treat "
                        + "this as a failure.";
        }
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
    //
    // Confidence thresholding (proposal 5.3: ignore classifications
    // under 0.65 confidence as Undefined) is NOT implementable here:
    // the tuned endpoint returns text only, with no confidence score
    // to threshold. Nothing is faked - an unparseable reply simply
    // becomes a no-label frame for the state machine.
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
        paused = true;
        stopReadiness();
        // Judging stops with the screen; finish() still leaves a verdict
        // on screen for when the user comes back.
        if (running) stopSequence(false);
    }

    @Override
    protected void onResume() {
        super.onResume();
        paused = false;
        // If no sequence is running, the READY gate resumes watching
        // for the starting posture - even with a verdict still showing,
        // so START can be tapped again once the dog is back in position.
        // (startReadiness is a no-op if it is already running.)
        if (!running) {
            startReadiness();
        }
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
