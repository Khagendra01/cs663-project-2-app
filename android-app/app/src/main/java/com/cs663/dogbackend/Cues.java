package com.cs663.dogbackend;

import android.content.Context;
import android.os.Build;
import android.os.VibrationEffect;
import android.os.Vibrator;
import android.speech.tts.TextToSpeech;

import java.util.Locale;

// Audio + haptic output. A trainer's eyes are on the dog, not on the
// phone on the tripod, so every state change the trainer must act on is
// spoken and has its own vibration pattern:
//   ready      one short buzz   + "Ready"
//   go         one buzz         + "Go"
//   reached    two quick ticks  (target posture reached)
//   success    long buzz        + "Success, ..."
//   break      three buzzes     + "Break at ... seconds"
//   uncertain  two slow buzzes  + "Uncertain, ..."
public class Cues implements TextToSpeech.OnInitListener {

    private final Vibrator vibrator;
    private volatile TextToSpeech tts;
    private volatile boolean ttsReady = false;
    private volatile boolean voiceOn = true;

    public Cues(Context ctx) {
        vibrator = (Vibrator) ctx.getSystemService(Context.VIBRATOR_SERVICE);
        tts = new TextToSpeech(ctx.getApplicationContext(), this);
    }

    @Override
    public void onInit(int status) {
        if (status == TextToSpeech.SUCCESS && tts != null) {
            tts.setLanguage(Locale.US);
            ttsReady = true;
        }
    }

    public void setVoiceOn(boolean on) {
        voiceOn = on;
    }

    public void ready() {
        buzz(new long[]{0, 60});
        say("Ready");
    }

    public void go() {
        buzz(new long[]{0, 150});
        say("Go");
    }

    public void reached() {
        buzz(new long[]{0, 40, 80, 40});
    }

    public void verdict(String verdict, String spoken) {
        switch (verdict) {
            case "SUCCESS":
                buzz(new long[]{0, 500});
                break;
            case "EARLY_BREAK":
                buzz(new long[]{0, 120, 100, 120, 100, 120});
                break;
            default:
                buzz(new long[]{0, 250, 250, 250});
        }
        say(spoken);
    }

    public void say(String text) {
        // say() is also called from the analysis thread while shutdown()
        // runs on the UI thread: read the field once so it cannot turn
        // null between the check and the call.
        TextToSpeech t = tts;
        if (voiceOn && ttsReady && t != null) {
            t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "cue");
        }
    }

    private void buzz(long[] pattern) {
        if (vibrator == null || !vibrator.hasVibrator()) return;
        if (Build.VERSION.SDK_INT >= 26) {
            vibrator.vibrate(VibrationEffect.createWaveform(pattern, -1));
        } else {
            vibrator.vibrate(pattern, -1);
        }
    }

    public void shutdown() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
            tts = null;
        }
    }
}
