package com.cs663.dogbackend;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Size;
import android.view.OrientationEventListener;
import android.view.Surface;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.ComponentActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.core.resolutionselector.AspectRatioStrategy;
import androidx.camera.core.resolutionselector.ResolutionSelector;
import androidx.camera.core.resolutionselector.ResolutionStrategy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.FileProvider;

import com.google.common.util.concurrent.ListenableFuture;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

// Sequence mode: the Project 2 Dog Training Assistant.
//
// The trainer puts the phone on a tripod, picks a command (Sit / Down /
// Stand / Stay / Walk) and taps START when the app says Ready. The app
// watches the dog and judges the whole attempt: SUCCESS, EARLY_BREAK or
// UNCERTAIN, with the time the dog broke, spoken out loud.
//
// THE INTEGRATED SYSTEM (one pipeline, three models):
//
//   CameraX ImageAnalysis, upright 640x480 frames, ~5 fps
//     |
//     v  [phone]  Model C  EfficientDet-Lite0 (pretrained, COCO)
//     |           -> is there a dog? where? (box, tracked between frames)
//     v  [phone]  ROI crop around the box (padded)
//     v  [phone]  Model A  our fine-tuned MobileNetV3Small (TFLite)
//     |           -> posture + confidence for the crop
//     v  [phone]  temporal state machine (SequenceSession)
//     |           smoothing, 0.65 confidence rule, response window,
//     |           hold timer, break detection, real wall-clock seconds
//     v           VERDICT (spoken + vibration + on screen)
//     |
//     v  [cloud]  Model B  our fine-tuned Gemini (Vertex endpoint via
//     |           Cloud Run): re-checks the decisive frames (target
//     |           reached / first wrong posture / last frame) against
//     |           what the VERDICT claims about each of them ->
//     |           VERIFIED / DISPUTED / PARTLY CHECKED / NOT CHECKED.
//     |           Also asked during the READY gate when Model A stays
//     |           unsure.
//     v  [cloud]  base Gemini: coaching tip from verdict + timeline +
//                 the decisive frame.
//
// The on-device verdict never waits on the network: if the backend is
// down, the verdict still shows, marked "not checked by Model B".
public class SequenceActivity extends ComponentActivity {

    public static final String EXTRA_BACKEND_URL = "backendUrl";

    private static final int REQ_CAMERA_PERM = 77;
    private static final long MIN_FRAME_INTERVAL_MS = 180;   // ~5 fps cap
    private static final double MIN_CONF = 0.65;             // proposal 5.3
    private static final long READY_STABLE_MS = 600;         // proposal 5.2
    private static final long ESCALATE_AFTER_MS = 2000;      // A unsure this long -> ask B
    private static final long ESCALATE_EVERY_MS = 6000;
    private static final long B_READY_VALID_MS = 5000;       // counted from the reply's arrival
    private static final int ROTATION_HYSTERESIS_DEG = 15;   // see quantizeRotation()
    private static final int KEYFRAME_MAX_SIDE = 320;
    private static final int MAX_KEYFRAMES = 600;

    private static final String[] COMMANDS = {"Sit", "Down", "Stand", "Stay", "Walk"};
    private static final String[] STAY_POSTURES = {"Stay: sitting", "Stay: lying down", "Stay: standing"};
    private static final String[] STAY_LABELS = {"sitting", "lying_down", "standing"};
    private static final String[] DURATIONS = {"5 s", "10 s", "20 s"};
    private static final int[] DURATION_SECONDS = {5, 10, 20};

    // UI
    private PreviewView previewView;
    private OverlayView overlay;
    private TextView tvSeqCurrent, tvTimer, tvFps, tvBanner;
    private Spinner spCommand, spStayPosture, spDuration;
    private Button btnStartSeq, btnStopSeq, btnFeedback, btnShareLog;
    private TextView tvSeqStatus, tvSeqProgress, tvSeqVerdict, tvSeqDetails, tvModelB, tvSeqFeedback;
    private ProgressBar progressSeq;
    private LinearLayout layoutVerdict;
    private EditText etSeqUrl;
    private CheckBox cbVoice;

    // Threads. Everything below marked [A] is owned by the analysis
    // thread and only touched there.
    private final ExecutorService analysisExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService netPool = Executors.newCachedThreadPool();
    private Cues cues;

    // [A] models
    private DogDetector detector;
    private PostureClassifier classifier;
    private VisionPipeline pipeline;
    private long lastAnalysedMs = 0;
    private long fpsWindowStart = 0;
    private int fpsFrames = 0;

    // [A] run state
    private SequenceSession session;
    private long runT0Ms;
    private String runCommand;
    private int runDurationSec;
    private final List<KeyFrame> keyframes = new ArrayList<>();
    private FrameLog log;
    private boolean reachedCued;
    private final List<Long> frameLatencies = new ArrayList<>();

    // [A] READY gate
    private SequenceSession.TemporalSmoother readinessSmoother;
    private String readyLastSmoothed = SequenceSession.UNCERTAIN_FRAME; // [A] last detected-frame vote
    private String readyRequired;          // null = any posture (Walk)
    private Long readySinceMs;
    // Detected (not held) frames since readySinceMs, and whether a detected
    // frame has completed the stable span. See readinessStep.
    private int readyDetectedFrames;
    private boolean readyStable;
    private Long unsureSinceMs;
    private long lastEscalationMs;
    private boolean gateOpen;
    private String gateSource = "";
    private volatile String bReadyLabel;   // Model B's answer for the READY gate
    private volatile long bReadyAtMs;      // when that answer ARRIVED (nowMs clock)
    private final AtomicBoolean bReadyInFlight = new AtomicBoolean(false);
    // Bumped by every readiness reset (command change, attempt end,
    // rotation change), so a Model B reply for a crop taken before the
    // reset cannot open the gate for the new situation.
    private final AtomicInteger readyGen = new AtomicInteger();

    // Bumped by every cloud check and every START. Each cloud-check task
    // only writes the result card while its generation is still current,
    // so a slow reply for an earlier attempt cannot overwrite a newer
    // attempt's card.
    private final AtomicInteger cloudGen = new AtomicInteger();

    // Orientation. The activity is locked to portrait, so CameraX's
    // getRotationDegrees() only makes a frame upright relative to the
    // SCREEN. On a phone clamped sideways on a tripod the dog would then
    // be lying on its side in every frame - but Models C, A and B were
    // all trained on upright photos. So the physical rotation is tracked
    // here and the frame is turned the rest of the way to world-upright
    // in analyze().
    private OrientationEventListener orientationListener;
    private volatile int deviceRotDeg = 0;   // [UI writes] 0/90/180/270, OrientationEventListener convention
    private volatile int displayRotDeg = 0;  // [UI writes] display rotation of the portrait screen
    private volatile boolean frontLens = false;
    private int appliedExtraRotDeg = 0;      // [A] extra clockwise rotation used on the last frame
    private int runExtraRotDeg = 0;          // [A] latched at START for the whole attempt

    // Camera permission (UI thread only).
    private boolean cameraStarted = false;
    private boolean cameraDenied = false;
    private boolean replayRequested = false;

    // Replay clock: in replay mode every time-based rule runs on clip
    // time (frame index / clipFps), not on how fast the phone replays.
    private volatile long replayClockMs = 0;

    // UI-thread mirrors
    private volatile boolean running = false;
    private volatile boolean modelsReady = false;
    private volatile boolean uiGateOpen = false;
    private volatile int selCommand = 0;   // spinner positions, set on the UI thread
    private volatile int selStay = 0;

    // Last finished attempt (for re-check / share)
    private volatile SequenceSession.Verdict lastVerdict;
    private volatile List<KeyFrame> lastKeyframes = new ArrayList<>();
    private volatile String lastCommand;
    private volatile int lastDuration;
    private volatile FrameLog lastLog;

    static class KeyFrame {
        double t;
        String label;      // Model A label fed to the machine (null = unclear)
        float conf;
        byte[] jpeg;       // small crop for Model B
        String role;       // set when chosen as a decisive frame
        // What the verdict claims about this frame: TRUE = the dog was in
        // the target posture, FALSE = it was not, null = no claim (an
        // UNCERTAIN verdict claims nothing). Model B is checked against
        // this, not against Model A's label.
        Boolean expectTarget;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sequence);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        previewView = findViewById(R.id.previewView);
        overlay = findViewById(R.id.overlay);
        tvSeqCurrent = findViewById(R.id.tvSeqCurrent);
        tvTimer = findViewById(R.id.tvTimer);
        tvFps = findViewById(R.id.tvFps);
        tvBanner = findViewById(R.id.tvBanner);
        spCommand = findViewById(R.id.spCommand);
        spStayPosture = findViewById(R.id.spStayPosture);
        spDuration = findViewById(R.id.spDuration);
        btnStartSeq = findViewById(R.id.btnStartSeq);
        btnStopSeq = findViewById(R.id.btnStopSeq);
        btnFeedback = findViewById(R.id.btnFeedback);
        btnShareLog = findViewById(R.id.btnShareLog);
        tvSeqStatus = findViewById(R.id.tvSeqStatus);
        tvSeqProgress = findViewById(R.id.tvSeqProgress);
        tvSeqVerdict = findViewById(R.id.tvSeqVerdict);
        tvSeqDetails = findViewById(R.id.tvSeqDetails);
        tvModelB = findViewById(R.id.tvModelB);
        tvSeqFeedback = findViewById(R.id.tvSeqFeedback);
        progressSeq = findViewById(R.id.progressSeq);
        layoutVerdict = findViewById(R.id.layoutVerdict);
        etSeqUrl = findViewById(R.id.etSeqUrl);
        cbVoice = findViewById(R.id.cbVoice);

        previewView.setScaleType(PreviewView.ScaleType.FIT_CENTER);
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        progressSeq.setMax(100);

        String url = getIntent().getStringExtra(EXTRA_BACKEND_URL);
        etSeqUrl.setText(url == null || url.isEmpty() ? MainActivity.BACKEND_URL : url);
        etSeqUrlText = etSeqUrl.getText().toString();
        etSeqUrl.addTextChangedListener(new android.text.TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence c, int a, int b, int d) {
            }

            @Override
            public void onTextChanged(CharSequence c, int a, int b, int d) {
            }

            @Override
            public void afterTextChanged(android.text.Editable e) {
                etSeqUrlText = e.toString();
            }
        });

        spCommand.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, COMMANDS));
        spStayPosture.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, STAY_POSTURES));
        spDuration.setAdapter(new ArrayAdapter<>(this,
                android.R.layout.simple_spinner_dropdown_item, DURATIONS));

        android.widget.AdapterView.OnItemSelectedListener reset =
                new android.widget.AdapterView.OnItemSelectedListener() {
                    @Override
                    public void onItemSelected(android.widget.AdapterView<?> p, View v, int pos, long id) {
                        selCommand = spCommand.getSelectedItemPosition();
                        selStay = spStayPosture.getSelectedItemPosition();
                        updateSpinnersEnabled();
                        requestReadinessReset();
                    }

                    @Override
                    public void onNothingSelected(android.widget.AdapterView<?> p) {
                    }
                };
        spCommand.setOnItemSelectedListener(reset);
        spStayPosture.setOnItemSelectedListener(reset);
        updateSpinnersEnabled();

        cues = new Cues(this);
        cbVoice.setOnCheckedChangeListener((b, on) -> cues.setVoiceOn(on));

        // START stays ENABLED while the gate is closed (updateStartButton)
        // and the gate is checked here instead: a disabled View gets no
        // long-clicks, so disabling it would make the override below
        // unreachable in exactly the case it exists for.
        btnStartSeq.setOnClickListener(v -> {
            if (!uiGateOpen) {
                Toast.makeText(this, "Waiting for READY - long-press START to override",
                        Toast.LENGTH_SHORT).show();
                return;
            }
            startSequence(false);
        });
        // Long-press = start without the READY gate (e.g. the dog is in
        // position but Model A keeps misreading it). Logged as such.
        btnStartSeq.setOnLongClickListener(v -> {
            if (!modelsReady || running) return false;
            Toast.makeText(this, "Started without the READY check", Toast.LENGTH_SHORT).show();
            startSequence(true);
            return true;
        });
        btnStopSeq.setOnClickListener(v -> stopSequence());
        btnFeedback.setOnClickListener(v -> {
            if (lastVerdict != null) runCloudChecks(lastVerdict, lastKeyframes, lastCommand, lastDuration, lastLog);
        });
        btnShareLog.setOnClickListener(v -> shareLog());
        findViewById(R.id.btnModelLab).setOnClickListener(v -> {
            Intent i = new Intent(this, MainActivity.class);
            i.putExtra(SequenceActivity.EXTRA_BACKEND_URL, etSeqUrl.getText().toString().trim());
            startActivity(i);
        });

        analysisExecutor.execute(this::loadModels);

        orientationListener = new OrientationEventListener(this) {
            @Override
            public void onOrientationChanged(int deg) {
                // UNKNOWN = phone lying flat: keep the last rotation.
                if (deg == ORIENTATION_UNKNOWN) return;
                // deg is the device's clockwise rotation from its NATURAL
                // orientation. The screen-upright frame is upright for the
                // current display rotation (0 on a phone; 90/270 on a
                // landscape-natural tablet locked to portrait), so take the
                // rotation relative to the screen.
                int rel = (deg + displayRotDeg) % 360;
                deviceRotDeg = quantizeRotation(rel, deviceRotDeg);
            }
        };

        // Debug builds only: replay a folder of frames through the exact
        // same pipeline instead of the camera (used to test the app on the
        // emulator and to evaluate recorded clips on a phone). See
        // README "Replay mode".
        //   am start -n com.cs663.dogbackend/.SequenceActivity --es replay <folder>
        //       --ei cmd <0..4: Sit, Down, Stand, Stay, Walk>
        //       --ei stay <0..2: sitting, lying down, standing>
        //       --ef fps <playback speed, frames per wall-clock second>
        //       --ef clipFps <rate the frames were extracted at; default = fps>
        //       --ez autostart true
        boolean debuggable = (getApplicationInfo().flags
                & android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0;
        String replay = getIntent().getStringExtra("replay");
        if (debuggable && replay != null) {
            replayRequested = true;
            int cmd = getIntent().getIntExtra("cmd", 0);
            int stay = getIntent().getIntExtra("stay", 0);
            spCommand.setSelection(cmd);
            spStayPosture.setSelection(stay);
            selCommand = cmd;
            selStay = stay;
            // clipFps defaults to the playback fps, not to 2: before clipFps
            // existed, "--ef fps 5" meant a 5 fps clip played in real time,
            // and a fixed default of 2 would now stretch that clip's time
            // 2.5x (breaking the 3 s window / hold rules) for old commands.
            float fpsExtra = getIntent().getFloatExtra("fps", 2f);
            float clipFpsExtra = getIntent().hasExtra("clipFps")
                    ? getIntent().getFloatExtra("clipFps", fpsExtra) : fpsExtra;
            android.util.Log.i("SequenceActivity", String.format(Locale.US,
                    "Replay %s: playback fps=%.2f, clipFps=%.2f%s", replay, fpsExtra, clipFpsExtra,
                    getIntent().hasExtra("clipFps") ? "" : " (defaulted to fps)"));
            startReplay(replay, fpsExtra, clipFpsExtra,
                    getIntent().getBooleanExtra("autostart", false));
            return;
        }

        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, REQ_CAMERA_PERM);
        }
    }

    // ---- models + camera ----------------------------------------------

    // [A]
    private void loadModels() {
        try {
            long t0 = SystemClock.elapsedRealtime();
            detector = new DogDetector(this);
            classifier = new PostureClassifier(this);
            pipeline = new VisionPipeline(detector, classifier);
            resetReadinessOnAnalysis();
            modelsReady = true;
            long ms = SystemClock.elapsedRealtime() - t0;
            runOnUiThread(() -> {
                // A permanently denied permission is answered almost at
                // once, often before the models finish loading: keep the
                // denial on screen, or the user is told to point a camera
                // that will never start.
                if (cameraDenied && !replayMode) {
                    tvSeqStatus.setText(String.format(Locale.US, "%s (On-device models loaded in %d ms.)",
                            CAMERA_DENIED_MSG, ms));
                } else {
                    tvSeqStatus.setText(String.format(Locale.US,
                            "On-device models loaded in %d ms. %s", ms,
                            replayMode ? "Replaying frames." : "Point the camera at the dog."));
                }
                updateStartButton();
            });
        } catch (Throwable e) {
            runOnUiThread(() -> tvSeqStatus.setText(
                    "Could not load the on-device models: " + e.getMessage()));
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_CAMERA_PERM) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                cameraDenied = false;
                if (!cameraStarted) startCamera();
            } else {
                cameraDenied = true;
                tvSeqStatus.setText(CAMERA_DENIED_MSG);
            }
        }
    }

    private static final String CAMERA_DENIED_MSG = "Camera permission denied. The assistant needs the "
            + "camera to watch the dog - allow it in Settings > Apps.";

    private void startCamera() {
        cameraStarted = true;
        ListenableFuture<ProcessCameraProvider> f = ProcessCameraProvider.getInstance(this);
        f.addListener(() -> {
            try {
                ProcessCameraProvider provider = f.get();
                ResolutionSelector rs = new ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(new ResolutionStrategy(new Size(640, 480),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                        .build();
                Preview preview = new Preview.Builder()
                        .setResolutionSelector(new ResolutionSelector.Builder()
                                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                                .build())
                        .build();
                preview.setSurfaceProvider(previewView.getSurfaceProvider());
                ImageAnalysis analysis = new ImageAnalysis.Builder()
                        .setResolutionSelector(rs)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build();
                analysis.setAnalyzer(analysisExecutor, this::analyze);
                CameraSelector selector = provider.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)
                        ? CameraSelector.DEFAULT_BACK_CAMERA : CameraSelector.DEFAULT_FRONT_CAMERA;
                // PreviewView mirrors a front camera's preview but the
                // analysis frames are not mirrored; the overlay flips the
                // box to match (setDisplayTransform in handleFrame). The
                // world rotation also runs the other way for a front lens.
                frontLens = selector == CameraSelector.DEFAULT_FRONT_CAMERA;
                provider.unbindAll();
                provider.bindToLifecycle(this, selector, preview, analysis);
            } catch (Exception e) {
                cameraStarted = false; // let the next onResume try again
                tvSeqStatus.setText("Could not start the camera: " + e.getMessage());
            }
        }, androidx.core.content.ContextCompat.getMainExecutor(this));
    }

    // ---- replay (debug) ----------------------------------------------------

    private final java.util.concurrent.ScheduledExecutorService replayTimer =
            Executors.newSingleThreadScheduledExecutor();

    // fps = playback speed (frames per wall-clock second); clipFps = the
    // rate the frames were extracted from the video at. They are kept
    // apart because the state machine's rules (3 s response window, 0.6 s
    // hold, 1.5 s lost rule, READY timers) must run on CLIP time: replaying
    // a 5 fps clip at 2 fps would otherwise stretch every second of the
    // video to 2.5 s and turn a 2 s sit into "not reached within 3 s".
    private void startReplay(String name, float fps, float clipFps, boolean autostart) {
        // Internal storage first (adb: push to /data/local/tmp, then
        // `run-as com.cs663.dogbackend cp -r ... files/replay/`), then the
        // external files dir.
        java.io.File dir = new java.io.File(new java.io.File(getFilesDir(), "replay"), name);
        if (!dir.isDirectory()) {
            dir = new java.io.File(new java.io.File(getExternalFilesDir(null), "replay"), name);
        }
        java.io.File[] files = dir.listFiles((d, n) -> n.toLowerCase(Locale.US).endsWith(".jpg")
                || n.toLowerCase(Locale.US).endsWith(".png"));
        if (files == null || files.length == 0) {
            tvSeqStatus.setText("Replay: no frames in " + dir);
            return;
        }
        Arrays.sort(files);
        replayMode = true;
        final double clipRate = clipFps > 0 ? clipFps : 2.0;
        tvSeqStatus.setText(String.format(Locale.US, "Replay mode: %d frames from %s (clip %.1f fps)",
                files.length, name, clipRate));
        final int[] idx = {0};
        final boolean[] started = {false};
        // Any large base works (only differences are used); starting at
        // "now" keeps the escalation timer (lastEscalationMs = 0) from
        // treating the first seconds of the clip specially.
        final long replayBaseMs = SystemClock.elapsedRealtime();
        replayClockMs = replayBaseMs;
        long period = Math.max(50, Math.round(1000 / (fps > 0 ? fps : 2f)));
        // Fixed DELAY, and the timer waits for each frame to finish on the
        // analysis thread, so a slow device never builds a backlog (a
        // backlog would push START behind the whole clip).
        replayTimer.scheduleWithFixedDelay(() -> {
            try {
                analysisExecutor.submit(() -> {
                    if (!modelsReady) return;
                    if (idx[0] >= files.length) {
                        if (!replayDone) {
                            replayDone = true;
                            if (session != null && !session.isDecided()) onVerdict(session.finish());
                            runOnUiThread(() -> tvFps.setText("replay finished"));
                        }
                        return;
                    }
                    final int frameIdx = idx[0]++;
                    Bitmap b = android.graphics.BitmapFactory.decodeFile(files[frameIdx].getPath());
                    if (b == null) return;
                    // Clip time for this frame. Every time-based rule reads
                    // r.elapsedMs or nowMs(), so all of them follow it -
                    // including VisionPipeline's box hold, which is why the
                    // clock is passed in rather than overwritten afterwards.
                    replayClockMs = replayBaseMs + Math.round(frameIdx * 1000.0 / clipRate);
                    VisionPipeline.FrameResult r = pipeline.process(b, replayClockMs);
                    final Bitmap shownFrame = b;
                    runOnUiThread(() -> overlay.setFrame(shownFrame));
                    handleFrame(r);
                    if (autostart && !started[0] && gateOpen) {
                        started[0] = true;
                        runOnUiThread(() -> startSequence(false));
                    }
                }).get();
            } catch (Exception ignored) {
            }
            if (replayDone) replayTimer.shutdown();
        }, 500, period, java.util.concurrent.TimeUnit.MILLISECONDS);
    }

    // [A] One camera frame. CameraX hands frames in sensor orientation;
    // they are rotated upright here, because Models C, A and B were all
    // trained on upright photos.
    //
    // Two rotations, combined into one bitmap copy:
    //  1. getRotationDegrees(): sensor -> SCREEN-upright. For this
    //     portrait-locked activity that is only right while the phone is
    //     physically upright.
    //  2. extra (worldRotationDeg): SCREEN-upright -> WORLD-upright, from
    //     the OrientationEventListener. Its convention: 90 means the
    //     phone is turned 90 deg CLOCKWISE (its left edge is now on top).
    //     The back camera turns with the phone, so world-up is now the
    //     frame's LEFT edge - the scene is turned 90 deg counter-clockwise
    //     in the frame - and we rotate the frame back 90 deg CLOCKWISE
    //     (Matrix.postRotate(+90) with y down is clockwise) to bring the
    //     left edge to the top. So extra = device rotation for the back
    //     lens. A front lens looks the other way, so left and right swap
    //     in its (unmirrored) frame and the same phone turn needs the
    //     opposite correction: extra = 360 - device rotation. (This is the
    //     same as CameraX's rule: back = sensor - target rotation, front =
    //     sensor + target rotation.)
    // Rotations compose additively, so one postRotate(rot + extra) does
    // both. The overlay is told `extra` (OverlayView.setDisplayTransform) to
    // map the box back onto the portrait preview.
    private void analyze(ImageProxy img) {
        try {
            if (!modelsReady) return;
            long now = SystemClock.elapsedRealtime();
            if (now - lastAnalysedMs < MIN_FRAME_INTERVAL_MS) return;
            lastAnalysedMs = now;

            // Latched for a running attempt: changing it mid-attempt
            // would flip the frame's aspect ratio and box coordinates and
            // mix two different views in the temporal state machine.
            boolean attemptRunning = session != null && !session.isDecided();
            int extra = attemptRunning ? runExtraRotDeg : worldRotationDeg();
            if (extra != appliedExtraRotDeg) {
                appliedExtraRotDeg = extra;
                // The tracked box is in the old frame's coordinates, and
                // the READY decision was made on the old view.
                pipeline.resetTracking();
                resetReadinessOnAnalysis();
            }

            Bitmap frame = img.toBitmap();
            int rot = (img.getImageInfo().getRotationDegrees() + extra) % 360;
            if (rot != 0) {
                Matrix m = new Matrix();
                m.postRotate(rot);
                Bitmap r = Bitmap.createBitmap(frame, 0, 0, frame.getWidth(), frame.getHeight(), m, true);
                frame.recycle();
                frame = r;
            }
            VisionPipeline.FrameResult r = pipeline.process(frame);
            handleFrame(r);
        } catch (Throwable e) {
            runOnUiThread(() -> tvSeqStatus.setText("Frame error: " + e.getMessage()));
        } finally {
            img.close();
        }
    }

    // [A]
    private void handleFrame(VisionPipeline.FrameResult r) {
        // Only a box Model C actually found this frame is judged. A "held"
        // box (detector missed, last box reused) has no crop/Model A
        // result (see VisionPipeline), and is fed to the state machine as an
        // unclear, dog-not-seen frame: the session's lost rule absorbs
        // short detector gaps, and a stale box never decides a verdict.
        final boolean detected = "detected".equals(r.boxSource);
        String fed = null;
        Double conf = null;
        if (detected && r.judged != null) {   // whole-frame Model A, see FrameResult.judged
            fed = r.judged.label;
            conf = (double) r.judged.confidence;
        }
        boolean confident = conf != null && conf >= MIN_CONF;

        // fps over a 2 s window
        if (fpsWindowStart == 0) fpsWindowStart = r.elapsedMs;
        fpsFrames++;
        double fps = -1;
        if (r.elapsedMs - fpsWindowStart >= 2000) {
            fps = fpsFrames * 1000.0 / (r.elapsedMs - fpsWindowStart);
            fpsFrames = 0;
            fpsWindowStart = r.elapsedMs;
        }

        if (session != null && !session.isDecided()) {
            double t = Math.max(0, (r.elapsedMs - runT0Ms) / 1000.0);
            SequenceSession.Verdict v = session.feed(t, fed, conf, detected);
            frameLatencies.add(r.totalMs);
            // Key frames for Model B only from detected boxes, so Model B
            // never re-checks a crop of a stale box.
            if (detected && r.cropBitmap != null) {
                if (keyframes.size() >= MAX_KEYFRAMES) keyframes.remove(0);
                KeyFrame kf = new KeyFrame();
                kf.t = t;
                kf.label = confident ? fed : null;
                kf.conf = conf == null ? 0f : conf.floatValue();
                kf.jpeg = smallJpeg(r.cropBitmap);
                keyframes.add(kf);
            }
            if (log != null) log.frame(t, session.getPhase(), r, confident ? fed : null, session.getLastSmoothed());
            if (!reachedCued && session.getReachedAt() != null && v == null) {
                reachedCued = true;
                cues.reached();
            }
            final SequenceSession s = session;
            runOnUiThread(() -> updateLiveUi(s));
            if (v != null) onVerdict(v);
        } else {
            readinessStep(r, fed, conf, detected);
        }

        final double fpsShow = fps;
        final String chip = !r.dogPresent ? "No dog"
                : r.judged == null ? "Dog not detected this frame (holding last box)"
                : (confident ? pretty(fed) : "Unsure (" + pretty(r.judged.label) + "?)");
        // Replay frames are never rotated and never mirrored, and the
        // overlay draws them itself.
        final int dispRot = replayMode ? 0 : appliedExtraRotDeg;
        final boolean mirror = !replayMode && frontLens;
        runOnUiThread(() -> {
            overlay.setDisplayTransform(dispRot, mirror);
            overlay.show(r, confident);
            tvSeqCurrent.setText("Posture: " + chip);
            if (fpsShow > 0) {
                tvFps.setText(String.format(Locale.US, "%.1f fps · C %d ms · A %d ms",
                        fpsShow, r.detMs, r.clsMs));
            }
        });
    }

    // ---- READY gate ----------------------------------------------------

    private void requestReadinessReset() {
        analysisExecutor.execute(this::resetReadinessOnAnalysis);
    }

    // [A]
    private void resetReadinessOnAnalysis() {
        readyRequired = requiredStartingLabel(COMMANDS[selCommand], STAY_LABELS[selStay]);
        readinessSmoother = new SequenceSession.TemporalSmoother(5, MIN_CONF);
        readyLastSmoothed = SequenceSession.UNCERTAIN_FRAME;
        readySinceMs = null;
        readyDetectedFrames = 0;
        readyStable = false;
        unsureSinceMs = null;
        gateOpen = false;
        readyGen.incrementAndGet();
        bReadyLabel = null;
        uiGateOpen = false;
        runOnUiThread(this::updateStartButton);
    }

    // [A] Gate opens when the required starting posture has been the
    // smoothed Model A label for 0.6 s - or, when Model A stays unsure,
    // when Model B (fine-tuned Gemini) confirms it on a crop.
    private void readinessStep(VisionPipeline.FrameResult r, String fed, Double conf, boolean detected) {
        if (readinessSmoother == null || session != null && !session.isDecided()) return;
        long now = r.elapsedMs;
        // A held frame (detector missed, last box reused for <= 1 s) has no
        // Model A label (see VisionPipeline). It is not pushed into the smoother:
        // the smoother returns "unclear" for a not-detected frame, which
        // would reset readySinceMs and drop the gate on every single
        // detector miss - and re-fire the READY cue each time it reopens.
        // Instead a held frame carries the last detected frame's vote
        // forward, so it neither adds evidence nor wipes it. Once the hold
        // expires the frame is "none" (dogPresent=false) and resets as usual.
        boolean held = r.dogPresent && !detected;
        String smoothed = held ? readyLastSmoothed : readinessSmoother.push(fed, conf, detected);
        if (!held) readyLastSmoothed = smoothed;
        boolean aMatch = r.dogPresent && (readyRequired == null
                ? !smoothed.equals(SequenceSession.UNCERTAIN_FRAME)
                : smoothed.equals(readyRequired));
        // A held frame carries no new evidence, so it may keep a running
        // stable span alive but must never START it or COMPLETE it:
        // otherwise one detected frame followed by ~0.6 s of held frames
        // would open READY. Start/complete are judged on detected frames
        // only, and the span needs >= 3 detected frames. Once a detected
        // frame has completed it, held frames keep it (no gate flicker).
        if (aMatch) {
            if (!held) {
                if (readySinceMs == null) {
                    readySinceMs = now;
                    readyDetectedFrames = 0;
                }
                readyDetectedFrames++;
                if (now - readySinceMs >= READY_STABLE_MS && readyDetectedFrames >= 3) {
                    readyStable = true;
                }
            }
        } else {
            readySinceMs = null;
            readyDetectedFrames = 0;
            readyStable = false;
        }
        boolean aStable = readyStable;

        boolean aUnsure = r.dogPresent && smoothed.equals(SequenceSession.UNCERTAIN_FRAME);
        if (aUnsure) {
            if (unsureSinceMs == null) unsureSinceMs = now;
        } else {
            unsureSinceMs = null;
        }
        GeminiClient gc = gemini();
        if (aUnsure && readyRequired != null && unsureSinceMs != null
                && now - unsureSinceMs >= ESCALATE_AFTER_MS
                && now - lastEscalationMs >= ESCALATE_EVERY_MS
                && r.cropBitmap != null && gc.configured()
                && bReadyInFlight.compareAndSet(false, true)) {
            lastEscalationMs = now;
            final Bitmap crop = r.cropBitmap;
            final int gen = readyGen.get();
            runOnUiThread(() -> tvSeqStatus.setText("Model A is unsure - asking Model B (fine-tuned Gemini)..."));
            boolean queued = netExecute(() -> {
                GeminiClient.Reply rep = null;
                try {
                    rep = gc.tunedPosture(crop);
                    // A reset (command change, attempt end, rotation)
                    // since the crop was taken makes the answer stale.
                    if (rep.ok() && gen == readyGen.get()) {
                        // Validity is counted from the reply's ARRIVAL (a
                        // Gemini call takes ~5 s, which would otherwise use
                        // up the whole window). Time first, then label:
                        // readers check the label first, so they never pair
                        // a new label with an old time.
                        bReadyAtMs = nowMs();
                        bReadyLabel = rep.text;
                    }
                } catch (RuntimeException e) {
                    rep = null;
                } finally {
                    bReadyInFlight.set(false);
                }
                final GeminiClient.Reply fr = rep;
                runOnUiThread(() -> {
                    if (running || gen != readyGen.get()) return;
                    tvSeqStatus.setText(fr != null && fr.ok()
                            ? "Model B says the dog is " + pretty(fr.text)
                            + String.format(Locale.US, " (%.1f s).", fr.latencyMs / 1000.0)
                            : "Model B unavailable: " + (fr == null ? "request failed" : fr.error));
                });
            });
            if (!queued) bReadyInFlight.set(false);
        }
        String bl = bReadyLabel;
        // Model B's confirmation holds for the whole window unless Model A
        // CONFIDENTLY sees a different posture (not only while A is unsure:
        // one confident frame would otherwise flicker the gate shut).
        boolean aSeesOther = !smoothed.equals(SequenceSession.UNCERTAIN_FRAME)
                && readyRequired != null && !smoothed.equals(readyRequired);
        boolean bConfirms = readyRequired != null && bl != null && bl.equals(readyRequired)
                && r.dogPresent && now - bReadyAtMs <= B_READY_VALID_MS && !aSeesOther;

        boolean open = aStable || bConfirms;
        if (open && !gateOpen) cues.ready();
        gateOpen = open;
        gateSource = aStable ? "Model A" : bConfirms ? "Model B" : "";
        uiGateOpen = open;

        final String msg;
        if (!r.dogPresent) {
            msg = "No dog in view. Frame the whole dog, side-on if possible.";
        } else if (open) {
            msg = readyRequired == null ? "Ready. Tap START, then give the command."
                    : "Ready: the dog is " + pretty(readyRequired) + " (" + gateSource
                    + "). Tap START, then give the command.";
        } else if (aUnsure) {
            msg = null; // keep any Model B status message on screen
        } else {
            msg = "Waiting for the starting posture: " + pretty(readyRequired)
                    + " (now: " + pretty(smoothed) + ").";
        }
        runOnUiThread(() -> {
            if (!running) {
                if (msg != null) tvSeqStatus.setText(msg);
                tvTimer.setText(uiGateOpen ? "READY" : "");
                updateStartButton();
            }
        });
    }

    // START is enabled whenever an attempt could start, even with the gate
    // closed: a disabled button gets no long-press, and long-press is the
    // override for a gate that will not open. The click handler refuses a
    // plain tap while the gate is closed; the dimmed look says "waiting".
    private void updateStartButton() {
        btnStartSeq.setEnabled(modelsReady && !running);
        boolean waiting = !uiGateOpen && !running;
        btnStartSeq.setText(waiting ? "START (waiting)" : "START");
        btnStartSeq.setAlpha(waiting ? 0.6f : 1f);
    }

    // ---- the run --------------------------------------------------------

    private String targetFor(String command, String stayLabel) {
        switch (command) {
            case "Sit": return "sitting";
            case "Down": return "lying_down";
            case "Stand": return "standing";
            case "Walk": return "walking";
            default: return stayLabel; // Stay holds the chosen posture
        }
    }

    // Proposal 5.2: Sit starts from Stand, Down starts from Sit, Stay
    // starts from the posture it holds. Walk has no requirement.
    private String requiredStartingLabel(String command, String stayLabel) {
        switch (command) {
            case "Sit": return "standing";
            case "Down": return "sitting";
            case "Stand": return "standing";
            case "Walk": return null;
            default: return stayLabel;
        }
    }

    private static boolean isReachThenHold(String command) {
        return "Sit".equals(command) || "Down".equals(command);
    }

    private void updateSpinnersEnabled() {
        String cmd = COMMANDS[spCommand.getSelectedItemPosition()];
        spStayPosture.setEnabled(!running && "Stay".equals(cmd));
        spDuration.setEnabled(!running && !isReachThenHold(cmd));
    }

    private void startSequence(boolean override) {
        if (!modelsReady || running) return;
        final String cmd = COMMANDS[spCommand.getSelectedItemPosition()];
        final String target = targetFor(cmd, STAY_LABELS[spStayPosture.getSelectedItemPosition()]);
        final int dur = DURATION_SECONDS[spDuration.getSelectedItemPosition()];
        running = true;
        // Any cloud check still in flight belongs to the previous attempt:
        // it must not fill the card this attempt will show.
        cloudGen.incrementAndGet();
        setControlsRunning(true);
        layoutVerdict.setVisibility(View.GONE);
        tvBanner.setVisibility(View.GONE);
        tvSeqStatus.setText(isReachThenHold(cmd)
                ? "Running: say \"" + cmd + "\" now. The dog has 3 s to reach "
                + pretty(target) + ", then holds 0.6 s."
                : "Running: " + cmd + " - hold " + pretty(target) + " for " + dur + " s.");
        cues.go();
        analysisExecutor.execute(() -> {
            SequenceSession.Config cfg = new SequenceSession.Config(dur);
            cfg.reachThenHold = isReachThenHold(cmd);
            cfg.minConfidence = MIN_CONF;
            session = new SequenceSession(target, cfg, cmd);
            runCommand = cmd;
            runDurationSec = dur;
            runT0Ms = nowMs();
            // The view the READY gate was judged on is kept for the whole
            // attempt (see analyze()).
            runExtraRotDeg = appliedExtraRotDeg;
            keyframes.clear();
            frameLatencies.clear();
            reachedCued = false;
            pipeline.resetTracking();
            try {
                log = new FrameLog(this, cmd);
            } catch (Exception e) {
                log = null;
            }
            startedWithOverride = override;
            startGateSource = override ? "override" : gateSource;
        });
    }

    private volatile boolean replayMode = false;
    private volatile boolean replayDone = false;
    private boolean startedWithOverride;  // [A]
    private String startGateSource = "";  // [A]

    private void stopSequence() {
        try {
            analysisExecutor.execute(() -> {
                // A plain Runnable: an exception here would kill the
                // analysis thread and, on Android, the app.
                try {
                    if (session != null && !session.isDecided()) onVerdict(session.finish());
                } catch (Throwable e) {
                    runOnUiThread(() -> tvSeqStatus.setText("Could not finish the attempt: " + e.getMessage()));
                }
            });
        } catch (RejectedExecutionException ignored) {
            // Activity already destroyed.
        }
    }

    // [A]
    private void onVerdict(SequenceSession.Verdict v) {
        final String cmd = runCommand;
        final int dur = runDurationSec;
        final List<KeyFrame> decisive = pickDecisiveFrames(v, keyframes);
        final FrameLog lg = log;
        if (lg != null) lg.close();
        log = null;
        long sum = 0;
        List<Long> sorted = new ArrayList<>(frameLatencies);
        java.util.Collections.sort(sorted);
        for (long x : sorted) sum += x;
        final long medianMs = sorted.isEmpty() ? 0 : sorted.get(sorted.size() / 2);
        final long p90Ms = sorted.isEmpty() ? 0 : sorted.get((int) Math.min(sorted.size() - 1, sorted.size() * 0.9));
        final double runS = v.nFrames == 0 ? 0 : (nowMs() - runT0Ms) / 1000.0;
        final double fps = runS > 0 ? v.nFrames / runS : 0;
        final boolean override = startedWithOverride;
        final String gateBy = startGateSource;
        session = null;
        resetReadinessOnAnalysis();

        lastVerdict = v;
        lastKeyframes = decisive;
        lastCommand = cmd;
        lastDuration = dur;
        lastLog = lg;

        runOnUiThread(() -> {
            running = false;
            setControlsRunning(false);
            showVerdict(v, cmd, medianMs, p90Ms, fps, override, gateBy);
            cues.verdict(v.verdict, spokenVerdict(v, cmd));
        });
        if (lg != null) {
            try {
                JSONObject s = verdictJson(v, cmd, dur);
                s.put("on_device_median_frame_ms", medianMs);
                s.put("on_device_p90_frame_ms", p90Ms);
                s.put("analysed_fps", fps);
                s.put("ready_gate", gateBy);
                lg.summary(s);
            } catch (Exception ignored) {
            }
        }
        runCloudChecks(v, decisive, cmd, dur, lg);
    }

    // [A] The frames that decided the verdict, for Model B to re-check.
    // Each one is also given what the VERDICT claims about it
    // (expectTarget), because that claim - not Model A's label on the
    // frame - is what Model B has to confirm. A wrong verdict usually
    // comes from smoothing lag, and then Model A's raw label on the
    // decisive frame already contradicts the verdict: comparing Model B
    // with Model A would print VERIFIED for exactly those wrong verdicts.
    static List<KeyFrame> pickDecisiveFrames(SequenceSession.Verdict v, List<KeyFrame> all) {
        List<KeyFrame> out = new ArrayList<>();
        if (all.isEmpty()) return out;
        if (v.targetReachedAtS != null) addNear(out, all, v.targetReachedAtS, "target reached");
        if (v.breakAtS != null) addBreakFrame(out, all, v.breakAtS, v.targetLabel);
        KeyFrame last = all.get(all.size() - 1);
        boolean have = false;
        for (KeyFrame k : out) if (k.t == last.t) have = true;
        if (!have) out.add(copy(last, "last frame"));
        for (KeyFrame k : out) {
            k.expectTarget = expectationFor(v, k.role);
            // Fallback break frame (no confident non-target crop at/after
            // breakAt): Model A itself did not see a wrong posture on it, so
            // the verdict makes no claim about it - Model B's view is
            // reported but not counted as agree/dispute.
            if ("first wrong posture".equals(k.role) && !isConfidentNonTarget(k, v.targetLabel)) {
                k.expectTarget = null;
            }
        }
        return out;
    }

    // KeyFrame.label is only set when Model A was confident (>= MIN_CONF).
    private static boolean isConfidentNonTarget(KeyFrame k, String target) {
        return k.label != null && !k.label.equals(target);
    }

    // The "first wrong posture" frame. After an unclear gap breakAtS can
    // point at the START of the gap, so the first keyframe at/after it may
    // be a half-risen transition crop that Model A was unsure about; giving
    // it expectTarget=false let Model B call a correct EARLY_BREAK DISPUTED.
    // So pick the first keyframe at/after breakAt that Model A confidently
    // labelled as NOT the target; if there is none, fall back to addNear
    // (pickDecisiveFrames then gives that frame no expectation).
    private static void addBreakFrame(List<KeyFrame> out, List<KeyFrame> all, double t, String target) {
        for (KeyFrame k : all) {
            if (k.t >= t - 1e-6 && isConfidentNonTarget(k, target)) {
                for (KeyFrame o : out) if (o.t == k.t) return;
                out.add(copy(k, "first wrong posture"));
                return;
            }
        }
        addNear(out, all, t, "first wrong posture");
    }

    // What the verdict says the dog was doing on a decisive frame:
    //   target reached               -> in the target posture
    //   first wrong posture          -> NOT in the target posture
    //   last frame of a SUCCESS      -> in the target posture
    //   last frame of an EARLY_BREAK -> NOT in it (after a break, or the
    //                                   target was never reached)
    //   last frame of a LATE reach   -> no claim: the dog did get there,
    //     (EARLY_BREAK, reached, no     only after the 3 s window. Lateness
    //      break time)                  is timing Model B cannot check.
    //   any frame of an UNCERTAIN    -> no claim (null)
    static Boolean expectationFor(SequenceSession.Verdict v, String role) {
        if ("UNCERTAIN".equals(v.verdict)) return null;
        if ("target reached".equals(role)) return Boolean.TRUE;
        if ("first wrong posture".equals(role)) return Boolean.FALSE;
        if ("last frame".equals(role)) {
            if ("SUCCESS".equals(v.verdict)) return Boolean.TRUE;
            if ("EARLY_BREAK".equals(v.verdict)) {
                boolean lateReach = v.breakAtS == null && v.targetReachedAtS != null;
                return lateReach ? null : Boolean.FALSE;
            }
        }
        return null;
    }

    // Does Model B's label support the verdict's claim on this frame?
    // null when there is no claim or no answer.
    static Boolean supportsVerdict(Boolean expectTarget, String bLabel, String target) {
        if (expectTarget == null || bLabel == null) return null;
        return expectTarget.booleanValue() == bLabel.equals(target);
    }

    static String expectedText(Boolean expectTarget, String target) {
        if (expectTarget == null) return null;
        return expectTarget ? pretty(target) : "not " + pretty(target);
    }

    // The ONE status line for Model B's check of a verdict, used for the
    // card, the coaching prompt and the attempt JSON. bLabels.get(i) is
    // Model B's canonical label for frames.get(i), null = no answer.
    //   NOT CHECKED    no answer at all; or the verdict is UNCERTAIN (it
    //                  claims nothing to check); or a frame that decided
    //                  the verdict (target reached / first wrong posture)
    //                  got no answer
    //   DISPUTED       Model B contradicts the verdict on some key frame
    //   PARTLY CHECKED Model B supports every key frame it answered, but
    //                  some other key frame got no answer
    //   VERIFIED       Model B answered every key frame and supports the
    //                  verdict on all of them
    // An unanswered frame is never silently dropped: "1/1 agree" when the
    // break frame timed out would claim a check that did not happen.
    static String buildCheckStatus(SequenceSession.Verdict v, List<KeyFrame> frames, List<String> bLabels) {
        int answered = 0, withClaim = 0, support = 0;
        String dispute = null;
        List<String> missing = new ArrayList<>();
        List<String> missingDeciding = new ArrayList<>();
        for (int i = 0; i < frames.size(); i++) {
            KeyFrame k = frames.get(i);
            String b = bLabels.get(i);
            if (b == null) {
                missing.add(k.role);
                if ("target reached".equals(k.role) || "first wrong posture".equals(k.role)) {
                    missingDeciding.add(k.role);
                }
                continue;
            }
            answered++;
            Boolean sup = supportsVerdict(k.expectTarget, b, v.targetLabel);
            if (sup == null) continue;
            withClaim++;
            if (sup) {
                support++;
            } else if (dispute == null) {
                dispute = String.format(Locale.US,
                        "at %.1f s (%s) Model B sees %s, but the verdict says the dog was %s",
                        k.t, k.role, pretty(b), expectedText(k.expectTarget, v.targetLabel));
            }
        }
        if (answered == 0) {
            return "NOT CHECKED - Model B gave no answer. The on-device verdict stands unchecked.";
        }
        if ("UNCERTAIN".equals(v.verdict)) {
            return "NOT CHECKED - the verdict is UNCERTAIN, so there is nothing for Model B to confirm. "
                    + "What Model B sees on the key frames is listed for information only.";
        }
        if (dispute != null) {
            return "DISPUTED - " + dispute + ". Treat this verdict with caution."
                    + (missing.isEmpty() ? "" : " (" + missing.size() + " key frame(s) not answered.)");
        }
        String agrees = "Model B agrees with the verdict on " + support + "/" + withClaim
                + " key frame" + (withClaim == 1 ? "" : "s");
        if (!missingDeciding.isEmpty()) {
            return "NOT CHECKED - no Model B answer for the " + String.join(" and ", missingDeciding)
                    + " frame, which decided this verdict. " + agrees + " that it did answer.";
        }
        if (!missing.isEmpty()) {
            return "PARTLY CHECKED - " + agrees + "; no answer for the " + String.join(", ", missing)
                    + ". The verdict is not fully verified.";
        }
        if (withClaim < answered) {
            return "PARTLY CHECKED - " + agrees + ".";
        }
        return "VERIFIED - " + agrees + ".";
    }

    private static void addNear(List<KeyFrame> out, List<KeyFrame> all, double t, String role) {
        KeyFrame best = null;
        for (KeyFrame k : all) {
            if (k.t >= t - 1e-6) {
                best = k;
                break;
            }
        }
        if (best == null) best = all.get(all.size() - 1);
        for (KeyFrame k : out) if (k.t == best.t) return;
        out.add(copy(best, role));
    }

    private static KeyFrame copy(KeyFrame k, String role) {
        KeyFrame c = new KeyFrame();
        c.t = k.t;
        c.label = k.label;
        c.conf = k.conf;
        c.jpeg = k.jpeg;
        c.role = role;
        c.expectTarget = k.expectTarget;
        return c;
    }

    // ---- cloud checks: Model B verification + base Gemini coaching -------

    private void runCloudChecks(SequenceSession.Verdict v, List<KeyFrame> decisive,
                                String cmd, int dur, FrameLog lg) {
        final GeminiClient gc = gemini();
        // This check's generation. onVerdict, the re-check button and START
        // all bump it, so only the newest check may write the card. The log
        // has its own per-attempt generation (FrameLog.beginCheck), so a
        // check still running when the next attempt starts still saves its
        // results to its own attempt's file.
        final int gen = cloudGen.incrementAndGet();
        final int logGen = lg != null ? lg.beginCheck() : 0;
        runOnUiThread(() -> {
            if (gen != cloudGen.get()) return;
            tvModelB.setText(decisive.isEmpty() ? "No dog frames to check."
                    : "Checking " + decisive.size() + " decisive frame(s) with Model B...");
            tvSeqFeedback.setText("(waiting for Gemini...)");
        });
        if (!gc.configured()) {
            runOnUiThread(() -> {
                if (gen != cloudGen.get()) return;
                tvModelB.setText("Backend URL not set - verdict not checked by Model B.");
                tvSeqFeedback.setText(templateTip(v, cmd));
            });
            return;
        }
        boolean queued = netExecute(() -> {
            List<Future<GeminiClient.Reply>> futures = new ArrayList<>();
            for (KeyFrame k : decisive) {
                Future<GeminiClient.Reply> f;
                try {
                    f = netPool.submit(() -> gc.post("/api/fine-tuned-gemini",
                            GeminiClient.POSTURE_PROMPT, android.util.Base64.encodeToString(k.jpeg,
                                    android.util.Base64.NO_WRAP)));
                } catch (RejectedExecutionException e) {
                    // Pool shut down between submits (activity closing): a
                    // failed answer keeps futures index-aligned with decisive.
                    GeminiClient.Reply fail = new GeminiClient.Reply();
                    fail.error = "app closing";
                    f = CompletableFuture.completedFuture(fail);
                }
                futures.add(f);
            }
            StringBuilder sb = new StringBuilder();
            int failed = 0;
            List<String> bLabels = new ArrayList<>();
            JSONArray checks = new JSONArray();
            for (int i = 0; i < decisive.size(); i++) {
                KeyFrame k = decisive.get(i);
                GeminiClient.Reply rep;
                try {
                    rep = futures.get(i).get();
                } catch (Exception e) {
                    rep = new GeminiClient.Reply();
                    rep.error = e.getMessage();
                }
                String bLabel = rep.ok() ? GeminiClient.parseLabel(rep.text) : null;
                bLabels.add(bLabel);
                String expected = expectedText(k.expectTarget, v.targetLabel);
                Boolean supports = supportsVerdict(k.expectTarget, bLabel, v.targetLabel);
                JSONObject c = new JSONObject();
                try {
                    c.put("role", k.role);
                    c.put("t_s", k.t);
                    // What the verdict claims on this frame, and whether
                    // Model B's answer supports it (null = no claim / no answer).
                    c.put("verdict_expects", expected == null ? JSONObject.NULL : expected);
                    c.put("model_b_supports_verdict", supports == null ? JSONObject.NULL : supports);
                    c.put("model_a", k.label == null ? JSONObject.NULL : k.label);
                    c.put("model_a_conf", k.conf);
                    c.put("model_b", bLabel == null ? JSONObject.NULL : bLabel);
                    c.put("model_b_agrees_with_model_a", bLabel == null || k.label == null
                            ? JSONObject.NULL : bLabel.equals(k.label));
                    c.put("model_b_raw", rep.text == null ? JSONObject.NULL : rep.text);
                    c.put("latency_ms", rep.latencyMs);
                    c.put("error", rep.error == null ? JSONObject.NULL : rep.error);
                } catch (Exception ignored) {
                }
                checks.put(c);
                String aShow = k.label == null ? "unsure" : pretty(k.label);
                String claim = expected == null ? "" : "verdict says " + expected + "; ";
                if (bLabel == null) {
                    failed++;
                    sb.append(String.format(Locale.US, "\u2022 %.1f s, %s: %sModel B gave no answer (%s). Model A: %s\n",
                            k.t, k.role, claim, rep.error == null ? "unparseable reply" : rep.error, aShow));
                    continue;
                }
                String outcome = supports == null ? "" : supports ? " - agrees" : " - CONTRADICTS the verdict";
                sb.append(String.format(Locale.US, "\u2022 %.1f s, %s: %sModel B sees %s%s (%.1f s). Model A: %s\n",
                        k.t, k.role, claim, pretty(bLabel), outcome, rep.latencyMs / 1000.0, aShow));
            }
            final String status = buildCheckStatus(v, decisive, bLabels);
            final String bText = status + "\n" + sb.toString().trim();
            runOnUiThread(() -> {
                if (gen == cloudGen.get()) tvModelB.setText(bText);
            });

            // Coaching from base Gemini, given the verdict (explain, do
            // not re-judge) and Model B's check.
            KeyFrame img = null;
            for (KeyFrame k : decisive) if ("first wrong posture".equals(k.role)) img = k;
            if (img == null && !decisive.isEmpty()) img = decisive.get(decisive.size() - 1);
            String tipText;
            long coachMs = -1;
            boolean jsonOk = false;
            if (img == null) {
                tipText = "App tip (no frame to send): " + templateTip(v, cmd);
            } else {
                GeminiClient.Reply rep = gc.post("/api/gemini", coachingPrompt(v, cmd, dur, status),
                        android.util.Base64.encodeToString(img.jpeg, android.util.Base64.NO_WRAP));
                coachMs = rep.latencyMs;
                JSONObject j = rep.ok() ? parseJsonLoose(rep.text) : null;
                if (j != null && !j.optString("coaching_tip", "").isEmpty()) {
                    jsonOk = true;
                    StringBuilder t = new StringBuilder();
                    String sum = j.optString("evidence_summary", "");
                    if (!sum.isEmpty()) t.append(sum).append("\n");
                    t.append("Tip: ").append(j.optString("coaching_tip"));
                    String caution = j.optString("caution", "");
                    if (!caution.isEmpty() && !"null".equals(caution)) t.append("\nNote: ").append(caution);
                    t.append(String.format(Locale.US, "\n(base Gemini, %.1f s)", rep.latencyMs / 1000.0));
                    tipText = t.toString();
                } else {
                    tipText = "App tip (Gemini " + (rep.ok() ? "reply was not valid JSON" : "unavailable: " + rep.error)
                            + "): " + templateTip(v, cmd);
                }
            }
            final String show = tipText;
            runOnUiThread(() -> {
                if (gen == cloudGen.get()) tvSeqFeedback.setText(show);
            });
            // FrameLog.summary() merges last-writer-wins, so an older check
            // finishing after a newer re-check of the SAME attempt would
            // overwrite the newer Model B fields: only that log's newest
            // check writes it.
            if (lg != null && lg.isLatestCheck(logGen)) {
                try {
                    // FrameLog.summary() merges, so the on-device fields
                    // written at the verdict survive this write.
                    JSONObject s = verdictJson(v, cmd, dur);
                    s.put("model_b_status", status);
                    s.put("model_b_checks", checks);
                    s.put("model_b_unanswered", failed);
                    s.put("coaching_json_valid", jsonOk);
                    s.put("coaching_latency_ms", coachMs);
                    s.put("coaching_text", show);
                    lg.summary(s);
                } catch (Exception ignored) {
                }
            }
        });
        if (!queued) {
            runOnUiThread(() -> {
                if (gen == cloudGen.get()) tvModelB.setText("NOT CHECKED - the app is closing.");
            });
        }
    }

    // netPool.execute that never throws: after onDestroy has shut the
    // pool down, a rejected task would otherwise escape a plain Runnable
    // on the analysis thread and crash the app. Returns false if rejected.
    private boolean netExecute(Runnable task) {
        try {
            netPool.execute(task);
            return true;
        } catch (RejectedExecutionException e) {
            return false;
        }
    }

    private String coachingPrompt(SequenceSession.Verdict v, String cmd, int dur, String bStatus) {
        List<String> timeline = new ArrayList<>();
        // Thin the ~5 fps timeline to ~2 labels per second for the prompt.
        double lastT = -1;
        for (String step : v.smoothedSequence) {
            int sp = step.indexOf(' ');
            double t = Double.parseDouble(step.substring(0, sp));
            if (t - lastT >= 0.5) {
                timeline.add(step.substring(0, sp) + "s " + pretty(step.substring(sp + 1)));
                lastT = t;
            }
        }
        String rule = isReachThenHold(cmd)
                ? "reach the target posture within 3 seconds, then hold it 0.6 seconds"
                : "hold the target posture for " + dur + " seconds";
        return "You are a positive-reinforcement dog trainer. A phone app already judged this "
                + "training attempt; do NOT re-judge it, explain it and coach the owner.\n"
                + "Command: " + cmd + ". Rule: " + rule + ". Target posture: " + pretty(v.targetLabel) + ".\n"
                + "App verdict: " + v.verdict + " - " + v.reason + ".\n"
                + String.format(Locale.US, "Held: %.1f s.", v.heldS)
                + (v.breakAtS != null ? String.format(Locale.US, " First wrong posture at %.1f s.", v.breakAtS) : "")
                + "\nPosture timeline: " + String.join(", ", timeline) + ".\n"
                + "Independent check by a second model: " + bStatus + "\n"
                + "The image is the decisive frame. Reply with a JSON object only, no other text, "
                + "with exactly these fields: \"evidence_summary\" (one sentence on what happened), "
                + "\"coaching_tip\" (one short, concrete next step), \"caution\" (one short sentence "
                + "or null).";
    }

    static JSONObject parseJsonLoose(String text) {
        if (text == null) return null;
        String s = text.trim();
        int a = s.indexOf('{'), b = s.lastIndexOf('}');
        if (a < 0 || b <= a) return null;
        try {
            return new JSONObject(s.substring(a, b + 1));
        } catch (Exception e) {
            return null;
        }
    }

    private String templateTip(SequenceSession.Verdict v, String cmd) {
        switch (v.verdict) {
            case "SUCCESS":
                return "Praise and reward right away. Next time, try a slightly longer target.";
            case "EARLY_BREAK":
                if (isReachThenHold(cmd) && v.targetReachedAtS == null) {
                    return "No reward. Lure the dog into " + pretty(v.targetLabel)
                            + " first, then fade the lure.";
                }
                return String.format(Locale.US, "No reward for the break. Next time set a target "
                        + "shorter than %.1f s and reduce distractions.", v.heldS);
            default:
                return "Move the phone so the whole dog is in frame, side-on, in good light, and try again.";
        }
    }

    // ---- UI -------------------------------------------------------------

    private void updateLiveUi(SequenceSession s) {
        if (s == null || !running) return;
        double t = s.getLastT() == null ? 0 : s.getLastT();
        if (s.isReachThenHold() && "RESPONSE".equals(s.getPhase())) {
            double win = s.getConfig().responseWindowS;
            tvTimer.setText(String.format(Locale.US, "respond %.1f / %.0f s", Math.min(t, win), win));
            progressSeq.setProgress((int) Math.min(100, 100 * t / win));
            tvSeqProgress.setText("Waiting for the dog to reach " + pretty(s.getTargetLabel()) + "...");
        } else {
            double held = s.getReachedAt() == null ? 0 : Math.max(0, t - s.getReachedAt());
            double need = s.holdRequiredS();
            tvTimer.setText(String.format(Locale.US, "held %.1f / %.1f s", Math.min(held, need), need));
            progressSeq.setProgress((int) Math.min(100, 100 * held / need));
            tvSeqProgress.setText(s.getReachedAt() == null
                    ? "Waiting for " + pretty(s.getTargetLabel()) + "..."
                    : String.format(Locale.US, "Holding %s · t = %.1f s · %d frames judged on the phone",
                    pretty(s.getTargetLabel()), t, s.getFrameCount()));
        }
    }

    private void showVerdict(SequenceSession.Verdict v, String cmd, long medianMs, long p90Ms,
                             double fps, boolean override, String gateBy) {
        layoutVerdict.setVisibility(View.VISIBLE);
        int color = "SUCCESS".equals(v.verdict) ? Color.rgb(46, 125, 50)
                : "EARLY_BREAK".equals(v.verdict) ? Color.rgb(198, 40, 40) : Color.rgb(97, 97, 97);
        tvBanner.setText(v.verdict.replace('_', ' '));
        tvBanner.setBackgroundColor(color);
        tvBanner.setVisibility(View.VISIBLE);
        tvBanner.postDelayed(() -> tvBanner.setVisibility(View.GONE), 4000);
        tvSeqVerdict.setText(v.verdict.replace('_', ' '));
        tvSeqVerdict.setTextColor(color);
        tvSeqStatus.setText("Attempt finished: " + v.verdict + ". Get the dog back into the starting posture to go again.");
        tvTimer.setText("");

        StringBuilder sb = new StringBuilder();
        sb.append(cmd).append(" → ").append(pretty(v.targetLabel)).append('\n');
        sb.append(String.format(Locale.US, "Held %.1f s", v.heldS));
        if (v.breakAtS != null) sb.append(String.format(Locale.US, " · first wrong posture at %.1f s", v.breakAtS));
        if (v.targetReachedAtS != null) sb.append(String.format(Locale.US, " · reached at %.1f s", v.targetReachedAtS));
        sb.append('\n').append("Why: ").append(v.reason).append('\n');
        sb.append(String.format(Locale.US, "%d frames judged on the phone (%.1f fps), %.0f%% unclear; "
                        + "on-device latency median %d ms, p90 %d ms per frame.",
                v.nFrames, fps, v.uncertainRatio * 100, medianMs, p90Ms));
        sb.append("\nStarted by: ").append(override ? "long-press override (no READY check)"
                : "READY gate (" + gateBy + ")");
        tvSeqDetails.setText(sb.toString());
    }

    private String spokenVerdict(SequenceSession.Verdict v, String cmd) {
        switch (v.verdict) {
            case "SUCCESS":
                return isReachThenHold(cmd) ? "Success. Good " + cmd.toLowerCase(Locale.US) + "."
                        : String.format(Locale.US, "Success. Held %.0f seconds.", v.targetDurationS);
            case "EARLY_BREAK":
                if (v.breakAtS != null) return String.format(Locale.US, "Break at %.1f seconds.", v.breakAtS);
                return "No " + cmd.toLowerCase(Locale.US) + " within three seconds.";
            default:
                return "Uncertain. I could not see the dog clearly.";
        }
    }

    private void setControlsRunning(boolean isRunning) {
        btnStopSeq.setEnabled(isRunning);
        spCommand.setEnabled(!isRunning);
        updateSpinnersEnabled();
        etSeqUrl.setEnabled(!isRunning);
        updateStartButton();
    }

    private void shareLog() {
        FrameLog lg = lastLog;
        if (lg == null) {
            Toast.makeText(this, "No attempt log yet", Toast.LENGTH_SHORT).show();
            return;
        }
        ArrayList<Uri> uris = new ArrayList<>();
        String auth = getPackageName() + ".fileprovider";
        for (java.io.File f : Arrays.asList(lg.csvFile(), lg.jsonFile())) {
            if (f.exists()) uris.add(FileProvider.getUriForFile(this, auth, f));
        }
        Intent i = new Intent(Intent.ACTION_SEND_MULTIPLE);
        i.setType("text/plain");
        i.putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris);
        i.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        startActivity(Intent.createChooser(i, "Share attempt log"));
    }

    // ---- helpers ----------------------------------------------------------

    private GeminiClient gemini() {
        return new GeminiClient(etSeqUrlText);
    }

    // The one clock every time-based rule uses: wall clock live, clip
    // time (frame index / clipFps) in replay. r.elapsedMs is on the same
    // clock (startReplay overwrites it per frame).
    private long nowMs() {
        return replayMode ? replayClockMs : SystemClock.elapsedRealtime();
    }

    // The extra clockwise rotation that turns a screen-upright frame
    // world-upright (see analyze() for the direction).
    private int worldRotationDeg() {
        int dev = deviceRotDeg;
        return frontLens ? (360 - dev) % 360 : dev;
    }

    // Snap an OrientationEventListener angle (0-359, clockwise) to 0/90/
    // 180/270, with hysteresis: the current value is kept until the phone
    // is ROTATION_HYSTERESIS_DEG past the 45 deg midpoint, so a tripod
    // mounted near 45 deg does not flip the frame (and reset the READY
    // gate) on every bit of sensor noise.
    static int quantizeRotation(int deg, int current) {
        int diff = Math.abs(((deg - current) % 360 + 540) % 360 - 180);
        if (diff <= 45 + ROTATION_HYSTERESIS_DEG) return current;
        return ((deg + 45) / 90 % 4) * 90;
    }

    @SuppressWarnings("deprecation")
    private int currentDisplayRotationDeg() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_90: return 90;
            case Surface.ROTATION_180: return 180;
            case Surface.ROTATION_270: return 270;
            default: return 0;
        }
    }

    private volatile String etSeqUrlText = MainActivity.BACKEND_URL;

    private static JSONObject verdictJson(SequenceSession.Verdict v, String cmd, int dur) throws Exception {
        JSONObject s = new JSONObject();
        s.put("command", cmd);
        s.put("target", v.targetLabel);
        s.put("target_duration_s", dur);
        s.put("verdict", v.verdict);
        s.put("reason", v.reason);
        s.put("held_s", v.heldS);
        s.put("reached_at_s", v.targetReachedAtS == null ? JSONObject.NULL : v.targetReachedAtS);
        s.put("break_at_s", v.breakAtS == null ? JSONObject.NULL : v.breakAtS);
        s.put("n_frames", v.nFrames);
        s.put("uncertain_ratio", v.uncertainRatio);
        return s;
    }

    private static byte[] smallJpeg(Bitmap crop) {
        Bitmap use = crop;
        int m = Math.max(crop.getWidth(), crop.getHeight());
        if (m > KEYFRAME_MAX_SIDE) {
            float s = KEYFRAME_MAX_SIDE / (float) m;
            use = Bitmap.createScaledBitmap(crop, Math.max(1, Math.round(crop.getWidth() * s)),
                    Math.max(1, Math.round(crop.getHeight() * s)), true);
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        use.compress(Bitmap.CompressFormat.JPEG, 85, baos);
        if (use != crop) use.recycle();
        return baos.toByteArray();
    }

    static String pretty(String label) {
        if (label == null) return "-";
        if (label.equals("lying_down")) return "lying down";
        if (label.equals(SequenceSession.UNCERTAIN_FRAME)) return "unclear";
        return label;
    }

    // ---- lifecycle --------------------------------------------------------

    @Override
    protected void onResume() {
        super.onResume();
        displayRotDeg = currentDisplayRotationDeg();
        if (orientationListener != null && orientationListener.canDetectOrientation()) {
            orientationListener.enable();
        }
        // The denial message sends the user to Settings; when they come
        // back with the permission granted, start the camera here, since
        // onCreate (the only other place) does not run again.
        if (!replayRequested && !cameraStarted
                && checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            cameraDenied = false;
            startCamera();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (orientationListener != null) orientationListener.disable();
        // An attempt the trainer cannot see is ended, not left running.
        if (running) stopSequence();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        replayTimer.shutdownNow();
        // netPool is shut down from the analysis thread AFTER the close
        // task: the executor is single-threaded and FIFO, so a verdict
        // still queued (onPause's stop task, or the frame in progress)
        // hands its Model B work to netPool while netPool still accepts it.
        // (runCloudChecks also survives a rejected submit on its own.)
        analysisExecutor.execute(() -> {
            if (detector != null) detector.close();
            if (classifier != null) classifier.close();
            netPool.shutdown();
        });
        analysisExecutor.shutdown();
        if (cues != null) cues.shutdown();
    }
}
