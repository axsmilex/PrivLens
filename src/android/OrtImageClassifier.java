// com/wt/ocr/ml/OrtImageClassifier.java
package com.wt.ocr.ml;

import ai.onnxruntime.*;
import java.util.*;
import java.nio.*;

public class OrtImageClassifier {
    private final OrtSession session;
    private final int W, H;

    public OrtImageClassifier(OrtSession s, int inputSize) {
        this.session = s;
        this.W = inputSize; this.H = inputSize;
    }

    public float predict(FloatBuffer nhwc) throws Exception {
        long[] shape = new long[]{1, H, W, 3}; // NHWC
        OnnxTensor inp = OnnxTensor.createTensor(session.getEnvironment(), nhwc, shape);
        Map<String, OnnxTensor> feeds = new HashMap<>();
        feeds.put("input", inp);
        try (OrtSession.Result out = session.run(feeds)) {
            float[][] logits = (float[][]) out.get(0).getValue(); // [1,1]
            float logit = logits[0][0];
            float prob = (float)(1.0 / (1.0 + Math.exp(-logit)));
            return prob;
        }
    }
}
