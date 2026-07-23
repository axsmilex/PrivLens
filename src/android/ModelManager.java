// com/wt/ocr/ml/ModelManager.java
// (lazy singleton) within Android
package com.wt.ocr.ml;

import android.content.Context;
import java.io.*;
import ai.onnxruntime.*;

public class ModelManager {
    private static OrtEnvironment env;
    private static OrtSession clsSession;
    private static OrtSession segSession;

    public static synchronized void init(Context ctx) throws Exception {
        if (env != null) return;
        env = OrtEnvironment.getEnvironment();

        clsSession = env.createSession(loadFromAssets(ctx, "models/privlens_cls.onnx"),
                new OrtSession.SessionOptions());
        segSession = env.createSession(loadFromAssets(ctx, "models/privlens_seg.onnx"),
                new OrtSession.SessionOptions());
    }

    public static OrtSession cls() { return clsSession; }
    public static OrtSession seg() { return segSession; }

    private static byte[] loadFromAssets(Context ctx, String assetPath) throws IOException {
        InputStream in = ctx.getAssets().open(assetPath);
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) != -1) bos.write(buf, 0, n);
        in.close();
        return bos.toByteArray();
    }
}
