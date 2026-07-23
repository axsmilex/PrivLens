// com/wt/ocr/ml/BlurUtils.java
package com.wt.ocr.ml;

import android.graphics.*;
import android.os.Build;

public class BlurUtils {

    public static Bitmap blurMasked(Bitmap src, float[][] mask, float thr, float radiusPx) {
        int w = src.getWidth(), h = src.getHeight();
        // upsample mask to w,h
        Bitmap maskBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ALPHA_8);
        Canvas mc = new Canvas(maskBmp);
        Paint mp = new Paint(Paint.ANTI_ALIAS_FLAG);
        // draw mask as scaled pixels
        int mh = mask.length, mw = mask[0].length;
        int[] line = new int[mw];
        for (int y=0; y<mh; y++) {
            for (int x=0; x<mw; x++) {
                int a = mask[y][x] >= thr ? 255 : 0;
                line[x] = Color.argb(a, 0, 0, 0);
            }
            Bitmap row = Bitmap.createBitmap(line, mw, 1, Bitmap.Config.ARGB_8888);
            Rect srcR = new Rect(0,0,mw,1);
            Rect dstR = new Rect(0, (int)((y*1.0f/mh)*h), w, (int)(((y+1)*1.0f/mh)*h));
            mc.drawBitmap(row, srcR, dstR, null);
        }

        // blurred copy
        Bitmap blurred = src.copy(Bitmap.Config.ARGB_8888, true);
        if (Build.VERSION.SDK_INT >= 31) {
            RenderEffect effect = RenderEffect.createBlurEffect(radiusPx, radiusPx, Shader.TileMode.CLAMP);
            Canvas c = new Canvas(blurred);
            Paint p = new Paint();
            p.setRenderEffect(effect);
            c.drawBitmap(blurred, 0, 0, p);
        } else {
            // simple stack blur fallback
            blurred = fastBoxBlur(blurred, 8);
        }

        // compose: draw blurred only where mask alpha=1
        Bitmap out = src.copy(Bitmap.Config.ARGB_8888, true);
        Canvas oc = new Canvas(out);
        Paint mpaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        mpaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.DST_IN));
        // layer: draw blurred
        oc.drawBitmap(blurred, 0, 0, null);
        // keep only masked area
        oc.drawBitmap(maskBmp, 0, 0, mpaint);
        mpaint.setXfermode(null);

        return out;
    }

    private static Bitmap fastBoxBlur(Bitmap src, int radius) {
        // very small & rough; good enough for redaction
        Bitmap bmp = src.copy(src.getConfig(), true);
        for (int i=0; i<radius; i++) bmp = boxBlurOnce(bmp);
        return bmp;
    }
    private static Bitmap boxBlurOnce(Bitmap src) {
        int w = src.getWidth(), h = src.getHeight();
        int[] p = new int[w*h];
        src.getPixels(p, 0, w, 0, 0, w, h);
        int[] q = new int[p.length];
        for (int y=1; y<h-1; y++) {
            for (int x=1; x<w-1; x++) {
                int i = y*w+x;
                int c00 = p[i]; int c01 = p[i-1]; int c02 = p[i+1];
                int c10 = p[i-w]; int c12 = p[i+w];
                int r = ((c00>>16)&255) + ((c01>>16)&255) + ((c02>>16)&255) + ((c10>>16)&255) + ((c12>>16)&255);
                int g = ((c00>>8)&255)  + ((c01>>8)&255)  + ((c02>>8)&255)  + ((c10>>8)&255)  + ((c12>>8)&255);
                int b = (c00&255)       + (c01&255)       + (c02&255)       + (c10&255)       + (c12&255);
                q[i] = Color.rgb(r/5, g/5, b/5);
            }
        }
        Bitmap out = src.copy(Bitmap.Config.ARGB_8888, true);
        out.setPixels(q, 0, w, 0, 0, w, h);
        return out;
    }
}
