package com.cs663.dogbackend;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Base64;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Locale;

// Calls to the Cloud Run backend (cloud-run-backend/index.js).
//
// MODEL B - fine-tuned Gemini (gemini-3.5-flash supervised tune, Vertex
// endpoint, served by POST /api/fine-tuned-gemini). In this app it is
// the slow, accurate second opinion, never the per-frame classifier:
//   - READY gate: when Model A stays unsure about the starting posture,
//     one crop goes to Model B, and its answer can open the gate.
//   - After every verdict: the key frames that decided the verdict
//     (target reached, first wrong posture, last frame) go to Model B in
//     parallel, and the app reports whether Model B's view of each frame
//     supports what the verdict claims about it (VERIFIED / DISPUTED).
// Base Gemini (POST /api/gemini) writes the coaching tip for the trainer
// from the verdict, the timeline and the decisive frame.
public class GeminiClient {

    public static final String POSTURE_PROMPT =
            "Look at the image of the dog. What is the dog doing? "
                    + "Answer with exactly one of: sitting, standing, lying down, walking.";

    public static class Reply {
        public String text;       // null on failure
        public String error;      // set on failure
        public long latencyMs;

        public boolean ok() {
            return text != null;
        }
    }

    private final String base;

    public GeminiClient(String baseUrl) {
        String b = baseUrl == null ? "" : baseUrl.trim();
        if (b.endsWith("/")) b = b.substring(0, b.length() - 1);
        base = b;
    }

    public boolean configured() {
        return !base.isEmpty() && !base.contains("YOUR-CLOUD-RUN-URL");
    }

    // Model B: one posture label for one crop. Returns the canonical
    // label in reply.text (or null if the reply could not be parsed).
    public Reply tunedPosture(Bitmap crop) {
        Reply r = post("/api/fine-tuned-gemini", POSTURE_PROMPT, toJpegB64(crop, 640));
        if (r.ok()) {
            String label = parseLabel(r.text);
            if (label == null) {
                r.error = "unparseable reply: " + r.text;
            }
            r.text = label;
        }
        return r;
    }

    public Reply baseGemini(String prompt, Bitmap image) {
        return post("/api/gemini", prompt, toJpegB64(image, 640));
    }

    Reply post(String path, String prompt, String imageBase64) {
        Reply reply = new Reply();
        long t0 = SystemClock.elapsedRealtime();
        HttpURLConnection conn = null;
        try {
            JSONObject payload = new JSONObject();
            payload.put("prompt", prompt);
            payload.put("mimeType", "image/jpeg");
            payload.put("imageBase64", imageBase64);
            conn = (HttpURLConnection) new URL(base + path).openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setConnectTimeout(10000);
            conn.setReadTimeout(25000);
            conn.setDoOutput(true);
            try (OutputStream out = conn.getOutputStream()) {
                out.write(payload.toString().getBytes(StandardCharsets.UTF_8));
            }
            int code = conn.getResponseCode();
            InputStream in = code >= 200 && code < 300 ? conn.getInputStream() : conn.getErrorStream();
            String raw = readAll(in);
            if (code < 200 || code >= 300) {
                String msg = raw;
                try {
                    msg = new JSONObject(raw).optString("error", raw);
                } catch (Exception ignored) {
                }
                reply.error = "HTTP " + code + ": " + msg;
            } else {
                JSONObject json = new JSONObject(raw);
                // With MOCK_MODE=true the backend answers 2xx with a stub
                // that echoes the prompt. The prompt lists every posture,
                // so parseLabel would read it as a real "lying down" and
                // fake a Model B answer (READY gate, VERIFIED/DISPUTED,
                // coaching). Treat it as an error so no stub reaches them.
                if (json.optBoolean("mock", false)) {
                    reply.error = "backend is in MOCK_MODE (stub reply, not Gemini)";
                } else {
                    reply.text = json.optString("text", "");
                }
            }
        } catch (java.net.UnknownHostException e) {
            reply.error = "no network (cannot resolve backend)";
        } catch (java.net.SocketTimeoutException e) {
            reply.error = "backend timed out";
        } catch (Exception e) {
            reply.error = e.getClass().getSimpleName() + ": " + e.getMessage();
        } finally {
            if (conn != null) conn.disconnect();
            reply.latencyMs = SystemClock.elapsedRealtime() - t0;
        }
        return reply;
    }

    // The tuned model was asked for one word but sometimes answers in a
    // sentence ("The dog is standing."), so this is a keyword scan.
    public static String parseLabel(String text) {
        if (text == null) return null;
        String s = text.trim().toLowerCase(Locale.US);
        // Belt and braces for a mock reply that slipped past post(): it
        // echoes the prompt, which names every label.
        if (s.startsWith("[mock")) return null;
        if (s.contains("lying") || s.equals("down") || s.contains("laying")) return "lying_down";
        if (s.contains("sitting") || s.equals("sit")) return "sitting";
        if (s.contains("standing") || s.equals("stand")) return "standing";
        if (s.contains("walking") || s.equals("walk")) return "walking";
        return null;
    }

    static String toJpegB64(Bitmap bmp, int maxSide) {
        Bitmap use = bmp;
        int m = Math.max(bmp.getWidth(), bmp.getHeight());
        if (m > maxSide) {
            float s = maxSide / (float) m;
            use = Bitmap.createScaledBitmap(bmp, Math.max(1, Math.round(bmp.getWidth() * s)),
                    Math.max(1, Math.round(bmp.getHeight() * s)), true);
        }
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        use.compress(Bitmap.CompressFormat.JPEG, 85, baos);
        if (use != bmp) use.recycle();
        return Base64.encodeToString(baos.toByteArray(), Base64.NO_WRAP);
    }

    private static String readAll(InputStream in) throws Exception {
        if (in == null) return "{}";
        StringBuilder sb = new StringBuilder();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            String line;
            while ((line = r.readLine()) != null) sb.append(line).append('\n');
        }
        return sb.toString().trim();
    }
}
