package com.cs663.dogbackend;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import java.util.Locale;

// Draws Model C's dog box and Model A's posture label over the live
// preview. The PreviewView uses FIT_CENTER and the analysis frames have
// the same 4:3 aspect ratio as the preview, so frame pixels map to view
// pixels with one scale factor plus the letterbox offset.
//
// Two things can make the model frame differ from what the preview shows:
//  - rotation: the activity is portrait-locked, but when the phone sits
//    sideways on a tripod SequenceActivity turns the frame so the models
//    see the dog upright in world terms. The box is then in that rotated
//    frame and has to be turned back into the screen-upright (preview)
//    frame first, or it is drawn transposed over the preview.
//  - mirroring: on a front-camera-only device PreviewView mirrors the
//    preview but the analysis frame is not mirrored, so the box has to be
//    flipped horizontally or it sits over empty floor on the other side.
public class OverlayView extends View {

    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint textBg = new Paint();
    private VisionPipeline.FrameResult result;
    private boolean judged; // Model A confidence above the threshold
    // Clockwise rotation (0/90/180/270) SequenceActivity applied to the
    // screen-upright camera frame to get the model frame, and whether the
    // preview is mirrored. Replay mode always uses (0, false).
    private int rotationDeg = 0;
    private boolean mirror = false;

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(6f);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(44f);
        textPaint.setFakeBoldText(true);
        textBg.setColor(0xAA000000);
    }

    private android.graphics.Bitmap frame; // replay mode only: drawn under the box

    public void setFrame(android.graphics.Bitmap b) {
        frame = b;
        invalidate();
    }

    public void setDisplayTransform(int rotationDeg, boolean mirror) {
        int d = ((rotationDeg % 360) + 360) % 360;
        // Only quarter turns are meaningful for a camera frame; snap anything
        // else to the nearest one so a bad value cannot skew the box.
        this.rotationDeg = (Math.round(d / 90f) * 90) % 360;
        this.mirror = mirror;
        invalidate();
    }

    public void show(VisionPipeline.FrameResult r, boolean confident) {
        result = r;
        judged = confident;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        VisionPipeline.FrameResult r = result;
        if (r == null || r.frameW == 0) return;
        int rot = rotationDeg;
        // Screen-upright frame size: a quarter turn swaps width and height.
        boolean quarter = rot == 90 || rot == 270;
        int sw = quarter ? r.frameH : r.frameW;
        int sh = quarter ? r.frameW : r.frameH;
        float scale = Math.min(getWidth() / (float) sw, getHeight() / (float) sh);
        float ox = (getWidth() - sw * scale) / 2f;
        float oy = (getHeight() - sh * scale) / 2f;
        android.graphics.Bitmap f = frame;
        // The replay bitmap is the model frame itself and replay never
        // rotates, so only draw it when no rotation is in effect; otherwise
        // it would be stretched into the swapped screen frame.
        if (rot == 0 && f != null && !f.isRecycled() && f.getWidth() == r.frameW) {
            canvas.drawBitmap(f, null, new RectF(ox, oy, ox + sw * scale, oy + sh * scale), null);
        }
        if (r.box == null) {
            String msg = "No dog in view";
            float tw = textPaint.measureText(msg);
            float x = (getWidth() - tw) / 2f, y = getHeight() / 2f;
            canvas.drawRect(x - 16, y - 52, x + tw + 16, y + 16, textBg);
            canvas.drawText(msg, x, y, textPaint);
            return;
        }
        RectF s = toScreenFrame(r.box, r.frameW, r.frameH, rot);
        if (mirror) {
            // PreviewView mirrors the front camera; flip the box to match.
            float l = sw - s.right, rr = sw - s.left;
            s.left = l;
            s.right = rr;
        }
        RectF v = new RectF(ox + s.left * scale, oy + s.top * scale,
                ox + s.right * scale, oy + s.bottom * scale);
        boolean held = !"detected".equals(r.boxSource);
        int color = held ? Color.YELLOW
                : judged ? Color.rgb(0, 220, 90) : Color.rgb(255, 160, 0);
        boxPaint.setColor(color);
        canvas.drawRect(v, boxPaint);
        String label = null;
        if (held) {
            // A held box has no Model A result (the crop is stale), so say
            // what is happening instead of showing a posture.
            label = "tracking...";
        } else if (r.judged != null) {
            label = String.format(Locale.US, "%s %.0f%%",
                    r.judged.label.replace('_', ' '), r.judged.confidence * 100);
            if (!judged) label += " (unsure)";
        }
        if (label != null) {
            float tw = textPaint.measureText(label);
            float ty = Math.max(48, v.top - 12);
            // Keep the label on screen; a mirrored or rotated box can sit
            // right against the edge where the label would run off.
            float lx = Math.max(0, Math.min(v.left, getWidth() - tw - 20));
            canvas.drawRect(lx, ty - 46, lx + tw + 20, ty + 10, textBg);
            canvas.drawText(label, lx + 10, ty, textPaint);
        }
    }

    // Maps a box from the model frame (mw x mh, produced by rotating the
    // screen-upright frame clockwise by rot degrees) back into the
    // screen-upright frame by applying the inverse rotation.
    static RectF toScreenFrame(RectF b, int mw, int mh, int rot) {
        switch (rot) {
            case 90:
                // Forward: model (x', y') = (sh - y, x); inverse: (y', mw - x').
                return new RectF(b.top, mw - b.right, b.bottom, mw - b.left);
            case 180:
                return new RectF(mw - b.right, mh - b.bottom, mw - b.left, mh - b.top);
            case 270:
                // Forward: model (x', y') = (y, sw - x); inverse: (mh - y', x').
                return new RectF(mh - b.bottom, b.left, mh - b.top, b.right);
            default:
                return new RectF(b);
        }
    }
}
