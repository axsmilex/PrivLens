// com/wt/ocr/ml/ImageIO.java
// Preprocess helpers (Bitmap → float NHWC)
package com.wt.ocr.ml;

import android.graphics.*;
import java.nio.*;

public class ImageIO {
    // Resize + normalize to [0,1]; NHWC float32
    public static FloatBuffer bitmapToFloatNHWC(Bitmap src, int dstW, int dstH) {
        Bitmap bm = Bitmap.createScaledBitmap(src, dstW, dstH, true);
        FloatBuffer buf = ByteBuffer.allocateDirect(dstW * dstH * 3 * 4)
                .order(ByteOrder.nativeOrder()).asFloatBuffer();
        int[] px = new int[dstW * dstH];
        bm.getPixels(px, 0, dstW, 0, 0, dstW, dstH);
        int idx = 0;
        for (int i = 0; i < px.length; i++) {
            int c = px[i];
            float r = ((c >> 16) & 0xFF) / 255f;
            float g = ((c >> 8) & 0xFF) / 255f;
            float b = (c & 0xFF) / 255f;
            buf.put(idx++, r);
            buf.put(idx++, g);
            buf.put(idx++, b);
        }
        buf.rewind();
        return buf;
    }
}
