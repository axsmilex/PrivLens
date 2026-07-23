// In PhotoObserverService.java (inside checkForNewPhoto, when you got imagePath)
ExecutorService exec = Executors.newSingleThreadExecutor();

exec.submit(() -> {
    try {
        // lazy init models once (e.g., in onCreate too)
        ModelManager.init(getApplicationContext());
        final int INPUT = 320;

        Bitmap original = BitmapFactory.decodeFile(imagePath);
        if (original == null) return;

        FloatBuffer fb = com.wt.ocr.ml.ImageIO.bitmapToFloatNHWC(original, INPUT, INPUT);

        OrtImageClassifier clf = new OrtImageClassifier(ModelManager.cls(), INPUT);
        float pSensitive = clf.predict(fb);

        boolean isSensitive = pSensitive >= 0.55f; // tune this

        if (isSensitive) {
            // optional: confirm + get mask
            fb.rewind();
            OrtSegModel seg = new OrtSegModel(ModelManager.seg(), INPUT);
            float[][] mask = seg.run(fb);
            float maxProb = OrtSegModel.maxProb(mask);

            if (maxProb >= 0.40f) { // tune this
                // you can blur preview or just notify
                showNotification(); // your existing method with PendingIntent to MainActivity
            } else {
                // borderline – still notify if you want
                showNotification();
            }
        }
    } catch (Exception ex) {
        android.util.Log.e("PrivLens", "ML error", ex);
    }
});
