// com/wt/ocr/ml/OrtSegModel.java
package com.wt.ocr.ml;

import ai.onnxruntime.*;
import java.util.*;
import java.nio.*;

public class OrtSegModel {
    private final OrtSession session;
    private final int W, H;

    public OrtSegModel(OrtSession s, int inputSize) {
        this.session = s;
        this.W = inputSize; this.H = inputSize;
    }

    public float[][] run(FloatBuffer nhwc) throws Exception {
        long[] shape = new long[]{1, H, W, 3};
        OnnxTensor inp = OnnxTensor.createTensor(session.getEnvironment(), nhwc, shape);
        Map<String, OnnxTensor> feeds = new HashMap<>();
        feeds.put("input", inp);
        try (OrtSession.Result out = session.run(feeds)) {
            float[][][][] logits = (float[][][][]) out.get(0).getValue(); // [1,H,W,1]
            float[][] mask = new float[H][W];
            for (int y=0; y<H; y++) {
                for (int x=0; x<W; x++) {
                    float v = logits[0][y][x][0];
                    mask[y][x] = (float)(1.0 / (1.0 + Math.exp(-v)));
                }
            }
            return mask;
        }
    }

    // simple util: compute max prob
    public static float maxProb(float[][] m) {
        float mx = 0f;
        for (int y=0; y<m.length; y++)
            for (int x=0; x<m[0].length; x++)
                if (m[y][x] > mx) mx = m[y][x];
        return mx;
    }
}
