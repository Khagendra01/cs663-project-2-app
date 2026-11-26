# Dog Training Assistant — Application (GitHub B)

CS663 Computer Vision, Project 2 · Group 5: Arnav Verma, Khagendra Khatri

This repository is the **application** half of the project: the Android app and
the Cloud Run backend it talks to. The models' training code, datasets and
offline testing live in GitHub A:
https://github.com/Khagendra01/cs663-project-2-vision

The app watches a dog through the phone's camera and judges one training
attempt: did the dog Sit / Down / Stand when told, or hold a Stay for the
target time? It answers **SUCCESS**, **EARLY_BREAK** (with the second it
broke) or **UNCERTAIN** (it could not see well enough to judge). It also has the
fine-tuned Gemini check the frames the verdict rests on, and gives the trainer
one coaching tip.

---

## The three models and what each one does

All three run in the live loop on every attempt. Each has a job the
others cannot do.

| | Model | Purpose in the app | Input | Output | Where it runs | Customization | Next step |
|---|---|---|---|---|---|---|---|
| **C** | EfficientDet-Lite0 (COCO), int8, MediaPipe Tasks | Is a dog in view, and **where**? Without a dog, no frame is judged: an empty room cannot open READY or "hold" a Stay | One upright camera frame | Dog box + score | **On the phone**, every frame (~5 fps cap) | Pretrained, **not** fine-tuned. Accept rule tuned on our clips: COCO "dog" ≥ 0.25, or cat/horse/sheep/cow/bear ≥ 0.30 (EfficientDet often labels real dogs as other four-legged animals) | Dog found → Model A's reading counts. Box → tracked overlay + the crop Model B checks |
| **A** | MobileNetV3-Small, fine-tuned by us (GitHub A `model-a/`), LiteRT | Posture of the dog **every frame**, fast enough for timing | The whole frame at 224×224, the way it was trained. Judged only on frames where Model C found a dog | sitting / standing / lying_down / walking + softmax confidence | **On the phone**, every frame | Student fine-tuned on dataset v4 (1,000 verified images) | Temporal state machine (smoothing, 0.65 confidence gate, timing rules) → verdict |
| **B** | Gemini, fine-tuned on Vertex AI (GitHub A `gemini-finetune/`) | (1) **Checks the verdict**: re-reads the 2–3 decisive frames and says VERIFIED / DISPUTED / PARTLY CHECKED / NOT CHECKED; (2) **READY tie-break**: when Model A is unsure for 2 s before a start, B is asked once | One JPEG crop + the dataset prompt | One posture word | **Cloud** (`/api/fine-tuned-gemini` on our Cloud Run service) | Supervised fine-tuning on our labelled posture images | Status line on the verdict card + attempt log |
| (base) | Gemini 3 Flash (base) | Coaching text only. It is told the verdict and must **not** re-judge it | Verdict facts + last frame | JSON: evidence summary, coaching tip, caution | Cloud (`/api/gemini`) | none (prompted) | Coaching card; falls back to a built-in tip on error |

Why this split: Model A has no "no dog" class, so it always names one of
the four postures, even for an empty room. That is why Model C gates it.
Gemini takes ~4–6 s per call, far too slow to judge timing frame by
frame, so it checks the result instead of producing it. When Model B
disagrees with the verdict on a decisive frame, the card says so instead
of hiding it.

```
camera ─► Model C (dog? where?) ─► Model A (posture of the frame, conf) ─► state machine ─► verdict
              on phone                       on phone                         on phone        │
                                                                                         ├─► Model B: VERIFIED / DISPUTED (cloud)
                                                                                         └─► base Gemini: coaching tip (cloud)
```

---

## Section 1 — App execution instructions

### What you need
- An Android phone with a 64-bit or 32-bit ARM CPU (nearly every phone), Android 7.0 (API 24)
  or newer, a camera, and internet for the Gemini parts. Models C and A run
  offline on the phone.
- The APK (build it as below; `DogBackendLab.apk` in the repo root is the
  **old** v1 build and does not contain this version).
- No accounts or keys: the backend is already deployed.

### Use it
1. Install the APK and open **Dog Training Assistant**. Allow the camera.
2. Pick a command (Sit / Down / Stand / Stay / Walk). For Stay, also pick the
   posture to hold and the duration (5 / 10 / 20 s).
3. Point the phone at the dog, side-on, whole dog in frame. Hold the phone
   upright (portrait); landscape frames are rotated back for the models.
   A box appears on the dog: green = Model A confident, orange = unsure,
   yellow = detector briefly lost it and the box is held.
4. **START** unlocks ("READY" cue + vibration) once the dog has been in the
   required starting posture for 0.6 s. Tap START, then give your command.
   A long press on START starts without waiting for READY.
5. The app speaks/vibrates when the target is reached and at the verdict. The
   card shows the verdict, timing, the Model B check, and a coaching tip.
   **Share log** sends the attempt's per-frame CSV + JSON summary.
6. **Model lab** (button at the bottom) runs one photo through Model C + A
   on the phone and compares base vs fine-tuned Gemini on it.

### Build from source
```bash
cd android-app
export JAVA_HOME="/Applications/Android Studio.app/Contents/jbr/Contents/Home"   # any JDK 17–21
./gradlew assembleDebug testDebugUnitTest
```
Output: `android-app/app/build/outputs/apk/debug/app-debug.apk`. Gradle 8.7 /
AGP 8.5.2, compileSdk 35, 16 KB page-size aligned (Android 15+ ready). Both
`.tflite` models ship in `app/src/main/assets/`. The unit tests cover the
state machine (`SequenceSessionTest`, 21 tests).

### Backend
Already deployed, nothing to run: Cloud Run service `cs663-gemini-service`,
`https://cs663-gemini-service-917815040923.us-central1.run.app`
(`GET /health` → `{"ok": true, "mockMode": false, ...}`). The URL is
prefilled and editable in the app. To redeploy, `cloud-run-backend/` is a plain
Node service (`npm start`, listens on `$PORT`); it needs Vertex AI access
for the GCP project (Application Default Credentials). If the backend is
started with `MOCK_MODE=true`, the app shows "backend is in MOCK_MODE" instead of
treating stub replies as Gemini answers.

---

## Section 2 — Code description

`android-app/app/src/main/java/com/cs663/dogbackend/`:

- `SequenceActivity.java` — the training screen. CameraX preview +
  ImageAnalysis (frames made upright, including a phone held in landscape),
  one analysis thread that owns all model state, the READY gate, the attempt
  life cycle, and the post-verdict cloud checks. Each check is tagged with a
  generation number, so a late reply can never land on a newer attempt's card.
  It also has a debug-only replay mode (below).
- `DogDetector.java` — Model C wrapper (MediaPipe ObjectDetector, accept rule above).
- `PostureClassifier.java` — Model A wrapper (LiteRT Interpreter, 224×224 RGB 0–255 input).
- `VisionPipeline.java` — per frame: Model C → track the same dog (IoU, or
  nearest centre when it is the only animal) → padded crop for Model B →
  Model A on the whole frame. Model A on the crop is also logged, for
  comparison. If the detector misses for up to 1 s, the box is held for
  display only, and those frames count as "unclear" instead of being judged.
- `SequenceSession.java` — the temporal state machine (time-based, any fps):
  majority-vote smoothing, 0.65 confidence gate, 2 s settle grace, a break
  must persist 0.6 s, 1.5 s unclear = UNCERTAIN, and a 3 s response window
  for Sit/Down/Stand. The reach time and break time are taken from raw
  frames, not from the lagged smoothed stream.
- `GeminiClient.java` — HTTP client for both Gemini endpoints (timeouts, error
  classes, mock detection, label parsing).
- `OverlayView.java` — box + posture label over the preview.
- `Cues.java` — text-to-speech + vibration cues (READY, GO, reached, verdict).
- `FrameLog.java` — per-attempt CSV (one row per frame: box, both Model A
  readings, what was fed to the state machine, per-stage latency) + merged
  JSON summary, under `Android/data/com.cs663.dogbackend/files/logs/`.
- `MainActivity.java` — Model lab (single photo: on-device C + A, base vs tuned Gemini).
- `cloud-run-backend/index.js` — `GET /health`, `POST /api/gemini`,
  `POST /api/fine-tuned-gemini`; body `{prompt, imageBase64, mimeType}` →
  `{text}`.
- `eval/model_c_recall.py` — offline check of Model C's accept rule on the
  six project clips (results in `eval/model_c_recall_results.md`).

### Replay mode (debug builds only; for repeatable testing without a dog)
Copy a folder of extracted frames to the app's private storage, then start an
attempt from adb:
```bash
adb push seq02_down_batumi_frames /data/local/tmp/dogreplay/seq02_down_batumi
```
```bash
adb shell run-as com.cs663.dogbackend sh -c "mkdir -p files/replay && cp -r /data/local/tmp/dogreplay/seq02_down_batumi files/replay/"
```
```bash
adb shell am start -S -n com.cs663.dogbackend/.SequenceActivity --es replay seq02_down_batumi --ei cmd 3 --ei stay 1 --ef fps 2 --ef clipFps 2 --ez autostart true
```
`cmd` is 0 Sit, 1 Down, 2 Stand, 3 Stay, 4 Walk; `stay` is 0 sitting, 1 lying
down, 2 standing. `clipFps` is the rate the frames were extracted at. All timing
rules run on clip time, so a slow device does not change the verdict.
`fps` is playback speed only.

---

## Section 3 — App start and results

Verified on Nov 25, 2026 on the Android emulator (Pixel 10 Pro image, arm64),
against the live Cloud Run backend, using replay mode on the project clips
(GitHub A `video-sequence-layer/video-sequences/`, 2 frames/s). **No physical
phone yet**: see Section 4.

**Full system, one attempt per row** (all three models, from camera frame to the
coaching tip):

| Clip | Command | Verdict | Timing | Model B check | On-device ms/frame (median / p90) |
|---|---|---|---|---|---|
| seq02 (dog lying, holds) | Stay: lying down, 5 s | **SUCCESS** | reached 0.5 s, held 5.0 s | VERIFIED 2/2 | 94 / 166 |
| seq05 (dog lying, panting) | Stay: lying down, 5 s | **SUCCESS** | reached 0.5 s, held 5.0 s | VERIFIED 2/2 | 114 / 128 |
| seq03 (dog walking) | Walk, 5 s | **SUCCESS** | held 6.5 s | VERIFIED 2/2 | 102 / 138 |
| splice: seq05 lying → seq01 standing | Stay: lying down, 5 s | **EARLY_BREAK** at 1.5 s | held 1.0 s | VERIFIED 3/3 | 149 / 335 |
| seq01 (dog standing) | Stand | never started | READY never opened | — | — |

- **What the rows show:**
  - Model B (tuned Gemini) answered in 2.7–4.8 s per frame.
  - Base Gemini coaching returned valid JSON every time, in 3.1–8.4 s.
  - Every attempt wrote its CSV + JSON log.
- **The failure row is real:**
  - Model A reads the standing africanis dog as *sitting* with 91–99% confidence. So a Stand attempt never gets READY, which needs the dog standing first.
  - In the spliced break, the verdict is still right, because the dog left *lying down* either way. Model B, asked about the same frames, says *standing*: correct where Model A is not. Its check agrees with the verdict, but it exposes the wrong label in the log.

**Per-model offline checks behind two design choices** (scripts and full tables in `eval/`):

| Model | Check | Result | What changed in the app |
|---|---|---|---|
| C | Dog found per frame, 6 clips / 111 frames (all contain a dog) | COCO "dog"/"cat" only: **61%**. Any four-legged class: **96%** | Accept rule widened (seq01: 15% → 95%) |
| A | Whole image vs Model C crop, Model A's held-out test split (235 photos) | Whole image **75.3%**, padded crop 68.5% (walking 43/62 vs 17/62) | Model A judges the whole frame; Model C gates it |
| A | Same, 111 clip frames | Whole image **73.9%**, crop 62.2% | (same) |

Earlier results kept from GitHub A's `TESTING.md` (held-out photos): fine-tuned
Gemini 88.4% vs base 86.0% (43 images, dataset v3). Model A 74.9% (235
images, dataset v4).

---

## Section 4 — Incomplete / known limits

Stated plainly:

1. **Not yet run on a physical phone.** Everything below Section 3 was run on
   the Android emulator (Apple-silicon Mac, Pixel 10 Pro image), which is far slower than a
   phone (≈1–1.5 analysed fps; Model C 100–270 ms/frame). The live camera
   path was checked only against the emulator's virtual scene (no dog → "No dog
   in view"). Real fps, real lighting and a real dog are untested until the demo.
2. **Model A misreads some standing dogs as sitting** with high confidence
   (91–99% on the africanis clip), so a Stand attempt with that dog cannot
   start. The state machine cannot fix a confident wrong label. Model B sees
   the same frames correctly, and when its answer contradicts what the verdict
   claims, the card says DISPUTED. But Model B never changes the verdict: the
   verdict is Model A's, and Model B's check is advisory.
3. **Model C is a pretrained COCO detector.** Accepting four-legged stand-in
   classes raised recall on our clips from 61% to 96% of frames, but a real
   cat or other pet in view could also be taken as "the dog". The tracker
   keeps following the first animal it locked onto.
4. **The deployed tuned Gemini is trained on dataset v3 (171 images)**. Model
   A is trained on v4 (1,000 images). Re-tuning Gemini on v4 is not done.
5. **Cloud parts need internet** and take ~3–6 s per call. Without a
   connection, the verdict still works (on-device), and the card says the
   check was NOT CHECKED.
6. **Debug build only**, not signed for release. `DogBackendLab.apk` in the
   repo root is the old v1 build.
7. **Dataset bias:** all imagery is licensed stock, not our own dogs.

---

## Section 5 — YouTube live demo

**Demo video:** _link to be added after the recording._

Plan: the app on a real phone with a real dog. It shows a Stay SUCCESS, a
broken Stay (EARLY_BREAK with its timestamp), the Model B check on the verdict
card, the coaching tip, and one failure case (dog half out of frame →
UNCERTAIN), plus Model lab comparing base vs tuned Gemini on one photo.

---

Maintained by Group 5. Models, data, and testing evidence:
https://github.com/Khagendra01/cs663-project-2-vision
