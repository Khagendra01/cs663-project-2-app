package com.cs663.dogbackend;

import android.content.Context;

import org.json.JSONObject;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Iterator;
import java.util.Locale;

// One CSV per attempt (every analysed frame) plus one JSON summary, in
// the app's external files dir:
//   Android/data/com.cs663.dogbackend/files/logs/attempt_<time>.csv
//   Android/data/com.cs663.dogbackend/files/logs/attempt_<time>.json
// Pull them with `adb pull /sdcard/Android/data/com.cs663.dogbackend/files/logs`
// or the Share log button. They are the raw data for the full-system
// evaluation: latency per stage, detector misses, Model A confidence,
// crop vs full-frame labels, and Model B agreement.
public class FrameLog {

    public static final String HEADER = "t_s,phase,dog_present,box_source,det_score,n_dogs,"
            + "box_l,box_t,box_r,box_b,a_crop_label,a_crop_conf,a_full_label,a_full_conf,"
            + "fed_label,smoothed,det_ms,cls_ms,total_ms";

    private final File csv;
    private final File json;
    private final JSONObject summary = new JSONObject(); // merged across summary() calls
    // Cloud checks on THIS attempt (the verdict's check, then any re-check).
    // summary() merges last-writer-wins, so only the newest check of this
    // log may write its Model B fields. Kept per log, not app-wide, so
    // starting the next attempt never suppresses this attempt's results.
    private int checkGen;

    public synchronized int beginCheck() { return ++checkGen; }

    public synchronized boolean isLatestCheck(int gen) { return gen == checkGen; }
    private FileWriter w;

    public FrameLog(Context ctx, String command) throws IOException {
        File dir = new File(ctx.getExternalFilesDir(null), "logs");
        if (!dir.exists() && !dir.mkdirs()) throw new IOException("cannot create " + dir);
        String stamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String name = "attempt_" + stamp + "_" + command.toLowerCase(Locale.US);
        csv = new File(dir, name + ".csv");
        json = new File(dir, name + ".json");
        w = new FileWriter(csv);
        w.write(HEADER + "\n");
    }

    public synchronized void frame(double t, String phase, VisionPipeline.FrameResult r,
                                   String fed, String smoothed) {
        if (w == null) return;
        try {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format(Locale.US, "%.3f,%s,%d,%s,%.3f,%d,", t, phase,
                    r.dogPresent ? 1 : 0, r.boxSource, r.detScore, r.nDogs));
            if (r.box != null) {
                sb.append(String.format(Locale.US, "%.4f,%.4f,%.4f,%.4f,",
                        r.box.left / r.frameW, r.box.top / r.frameH,
                        r.box.right / r.frameW, r.box.bottom / r.frameH));
            } else {
                sb.append(",,,,");
            }
            if (r.crop != null) {
                sb.append(r.crop.label).append(',')
                        .append(String.format(Locale.US, "%.4f", r.crop.confidence)).append(',');
            } else {
                sb.append(",,");
            }
            sb.append(r.full.label).append(',')
                    .append(String.format(Locale.US, "%.4f", r.full.confidence)).append(',');
            sb.append(fed == null ? "" : fed).append(',').append(smoothed == null ? "" : smoothed);
            sb.append(String.format(Locale.US, ",%d,%d,%d\n", r.detMs, r.clsMs, r.totalMs));
            w.write(sb.toString());
        } catch (IOException ignored) {
        }
    }

    public synchronized void close() {
        if (w == null) return;
        try {
            w.close();
        } catch (IOException ignored) {
        }
        w = null;
    }

    // The attempt summary is written in several steps (on-device stats at
    // the verdict, then Model B + coaching when the cloud replies, then
    // again on every re-check), so each call MERGES its keys into the
    // summary kept here instead of replacing the file - otherwise the
    // later write would drop the latency / fps / ready-gate fields.
    public synchronized void summary(JSONObject s) {
        try {
            Iterator<String> it = s.keys();
            while (it.hasNext()) {
                String k = it.next();
                summary.put(k, s.get(k));
            }
            // Write to a temp file and rename it over the real one, so
            // Share log (or adb pull) never sees a half-written JSON.
            File tmp = new File(json.getPath() + ".tmp");
            try (FileWriter jw = new FileWriter(tmp)) {
                jw.write(summary.toString(2));
            }
            if (!tmp.renameTo(json)) {
                // rename() over an existing file works on Android's
                // filesystems; this is only a fallback.
                if (json.delete() || !json.exists()) tmp.renameTo(json);
            }
        } catch (Exception ignored) {
        }
    }

    public File csvFile() {
        return csv;
    }

    public File jsonFile() {
        return json;
    }
}
