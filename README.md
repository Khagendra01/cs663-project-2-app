# Dog Training Assistant — Application (GitHub B)

CS663 Computer Vision, Project 2 · Group 5: Arnav Verma, Khagendra Khatri, Kokila Ganesan

This repository is the **application** half of the project: the Android app and
the Cloud Run backend it talks to. The models, datasets, and testing
documentation live in GitHub A:
https://github.com/Khagendra01/cs663-project-2-vision

The app helps a trainer judge whether a dog is holding a commanded posture
over time. The owner picks a command and a target duration, starts a sequence,
and the app samples the camera once per second, classifies each frame, and
judges the attempt with a temporal state machine — SUCCESS, EARLY_BREAK, or
UNCERTAIN — then offers feedback from a dog-training assistant model. A
single-photo mode (the Week 11 lab app) is included for comparing the base and
fine-tuned models on one image.

---

## Section 1 — App execution instructions

### What you need
- An Android phone, Android 7.0 (API 24) or newer, with a camera and internet.
- The APK: `DogBackendLab.apk` in this repository (debug build), or build it
  yourself (below).
- No accounts, keys, or setup: the backend is already deployed (see below).

### Install and run (prebuilt APK)
1. Copy `DogBackendLab.apk` to the phone and open it; allow install from your
   file manager/browser if Android asks.
2. Open **Dog Backend Lab**. The backend URL field is prefilled with the live
   service: `https://cs663-gemini-service-917815040923.us-central1.run.app`
3. **Single-photo mode:** tap *Take a photo* (grant the camera permission) or
   *Pick from gallery*, leave the prompt as is (or edit it), tap
   *Ask Both Models*. Two answers appear: base Gemini and the fine-tuned model.
4. **Sequence mode (the Project 2 flow):** tap *3. Open Sequence mode*.
   Choose a command (Sit / Down / Stand / Stay / Walk; for Stay also choose the
   posture to hold), choose a target duration (5s / 10s / 20s), and tap
   **Start Sequence**. Point the camera at the dog. The screen shows the live
   camera, the current (smoothed) posture, and held time vs the target.
   The sequence ends on its own with a verdict, or tap **Stop** to finish
   early. The verdict panel shows the result, held seconds, the reason, and
   the full label sequence, and trainer feedback is fetched automatically.

### Build from source
- Android Studio, or the command line with JDK 21 and the Android SDK
  (platform 34). This project builds fully offline with Gradle 8.7:
  `gradle assembleDebug` in `android-app/` → `app/build/outputs/apk/debug/app-debug.apk`.
- The backend URL is a constant in `MainActivity.java` and an editable field
  in the app; no rebuild is needed to point at another deployment.

### Backend
- Already deployed, nothing to run: Cloud Run service `cs663-gemini-service`
  (GCP project `cs663-dog-training`), base URL above. Health check:
  `GET /` or `GET /health` → `{"ok": true, "mockMode": false, ...}`.
- To redeploy: `cloud-run-backend/` is a plain Node service
  (`npm start`, listens on `$PORT`); it needs Vertex AI access for the GCP
  project (Application Default Credentials) and the tuned model endpoint set
  in `index.js`.

---

## Section 2 — Code description

### android-app/ (Java, no external libraries — platform APIs only)
- `MainActivity.java` — single-photo screen: camera/gallery capture, prompt,
  POSTs the image to both backend endpoints, shows both answers. Also carries
  the backend URL and the button into Sequence mode.
- `SequenceActivity.java` — the sequence screen: Camera2 live preview
  (TextureView) plus an ImageReader holding the newest JPEG frame; a 1-second
  sampler downscales that frame and POSTs it to the tuned endpoint with the
  fixed dataset prompt; parses the reply into one of
  sitting / standing / lying_down / walking; drives the live UI; on a verdict,
  requests trainer feedback from the base model (the last captured frame is
  attached because the backend requires an image on every call — commented
  in the code).
- `SequenceSession.java` — the temporal state machine, a Java port of the
  reference Python implementation in GitHub A (`video-sequence-layer/`):
  majority-vote smoothing (window 3), 2s settle grace, a break must persist
  1s to count, 3s of missing/unclear frames ends the attempt as UNCERTAIN,
  verdicts SUCCESS / EARLY_BREAK / UNCERTAIN with the same reason strings.
  At 1 fps, 1 frame = 1 second, so the timing constants carry over directly.
- Layouts: `activity_main.xml`, `activity_sequence.xml`. Permissions:
  INTERNET + CAMERA (runtime-requested).

### cloud-run-backend/ (Node, Express-style single file)
- `index.js` — three routes: `GET /` and `GET /health` (status, model names,
  mock flag), `POST /api/gemini` (also `/api/base-gemini`) for the base model,
  `POST /api/fine-tuned-gemini` for the tuned model. Request body is
  `{prompt, imageBase64, mimeType}`; response is `{text}` (or `{error}`).
  The base model runs on Vertex AI (`gemini-3-flash-preview`, global) with
  ADC — no API key in code. The tuned model is called at its Vertex tuning
  endpoint in `us`.

### How the pieces fit together
Camera → 1 fps frame sampling → tuned Gemini per-frame posture (Model B) →
temporal state machine verdict → base Gemini trainer feedback on the finished
sequence (Model B reasoning at runtime; its fine-tuning corpus is in GitHub A,
`gemini-finetune/model-b-scenarios/`). Model A (student-trained MobileNetV3,
GitHub A `model-a/`, with a TFLite export) and Model C+ (YOLOv8 dog detection,
GitHub A `model-c/`) are the proposal's other two models; their measured
results are in GitHub A's `TESTING.md`. The deployed per-frame classifier is
the tuned Gemini because it measured best (88.4%) and needs no on-device
model download.

---

## Section 3 — App start and results

Verified against the live deployment on Nov 10, 2026:

- `GET /health` → `{"ok": true, "mockMode": false, "baseModel":
  "gemini-3-flash-preview", "fineTunedConfigured": true}`
- `POST /api/gemini` with a held-out lying-down test photo →
  "lying down" ✓
- `POST /api/fine-tuned-gemini`, same photo → "lying down" ✓
- Live sequence proof (offline harness driving the same endpoints the app
  uses): 7 frames of a standing dog from a 3.6s clip, target 8s → all frames
  classified standing, state machine verdict **UNCERTAIN — interval incomplete** (held time is time-base dependent: 3.0s at the stored 2 fps, 6.0s at the app's 1 fps; see GitHub A video_evaluation.md) (GitHub A,
  `video-sequence-layer/results/live_seq06_tuned.json`).

Accuracy summary (full testing documentation in GitHub A, `TESTING.md`):
fine-tuned Gemini 88.4% vs base 86.0% (43 held-out images); Model A
MobileNetV3 74.9% (235 held-out images); YOLOv8 dog detection 81.7% on
held-out photos, 58.6% on video frames (weak on lying/top-down views).

Screenshots of the app itself are captured in the Section 5 demo recording;
the app runs only on a physical phone (no emulator was used in development).

---

## Section 4 — Incomplete / errors

Stated plainly, as required:

1. **Sequence mode has not yet run on a physical phone.** It compiles,
   packages, and its state machine is a line-checked port of the Python
   reference that passes 17/17 unit tests, but camera timing and per-frame
   network latency at 1 fps on real hardware are unverified until the demo
   recording (Section 5). If a frame call takes longer than a second the
   sampler simply skips that tick (fixed-delay), so a verdict can take a few
   seconds longer than the target duration on a slow connection.
2. **The deployed tuned Gemini is trained on dataset v3 (171 images).**
   Dataset v4 (1,000 images, verified, in GitHub A) is collected for the next
   tuning round but has not been trained on yet. Model A *is* trained on v4.
3. **YOLOv8 (Model C+) is not wired into the phone app.** It was evaluated
   offline (GitHub A, `model-c/`): 81.7% on photos but 58.6% on video frames,
   failing mainly on lying/curled/top-down dogs, so gating the pipeline on it
   would have made the live app worse. The app classifies full frames.
4. **EARLY_BREAK has no real-video demonstration.** None of the licensed
   clips we could use contain a genuine break, and we did not fake one; the
   break path is proven by synthetic scenarios (6/6) and unit tests, and by
   the app on a real dog in the demo if the dog cooperates — or doesn't.
5. **Posture ambiguity is real:** standing vs sitting vs walking confuse each
   other in single frames (see the confusion matrix in GitHub A). The
   sequence design (smoothing + held-time rules) exists because of this; a
   one-photo judgement would not be trustworthy.
6. **Known rough edges:** the backend occasionally drops a call under load
   (the app's sampler counts it as an unclear frame; the offline harness
   retries); the app is a debug build, unsigned for release; the backend URL
   is a text field, so a typo there fails calls until fixed.
7. **Dataset bias:** all imagery is licensed stock, not our own dogs; some
   breeds/viewpoints are over-represented (standing is show-photo heavy).

---

## Section 5 — YouTube live demo

**Demo video:** _link to be added here as soon as the recording is uploaded_
(recording is the final step of the project; this section is completed then).

What the video shows, in order:
1. The app starting on a real phone, backend URL prefilled.
2. Single-photo mode: a new photo of a dog taken in the video; base vs
   fine-tuned answers compared.
3. Sequence mode with a real dog: command + duration selected, Start
   Sequence, live posture and timer while the dog holds, and the verdict
   panel with trainer feedback.
4. Discussion of accuracy, using the measured numbers: fine-tuned Gemini
   88.4% vs base 86.0%; Model A 74.9% on the larger 235-image test; YOLOv8's
   81.7% / 58.6% split and what it taught us about lying and top-down views;
   and why the system judges sequences instead of single photos.

No code or notebooks appear in the video — it is the running app, filmed
full-screen.

---

Maintained by Group 5. Models, data, and testing evidence:
https://github.com/Khagendra01/cs663-project-2-vision
