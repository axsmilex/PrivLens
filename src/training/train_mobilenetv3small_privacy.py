#!/usr/bin/env python3
# train_mobilenetv3small_privacy.py

import os, math, argparse, json, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224
BATCH = 32
EPOCHS = 12
LR = 3e-4
SEED = 42

# ---------------------------
# Utilities
# ---------------------------

def set_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

def read_csv(path):
    df = pd.read_csv(path)
    assert {'path','label'}.issubset(df.columns), "CSV must contain 'path' and 'label'."
    return df

def decode_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)  # works for PNG too
    return img

def preprocess_train(img):
    # Resize first to keep aspect ratio variability
    img = tf.image.resize(img, [IMG_SIZE+16, IMG_SIZE+16])

    # Document-ish augmentations
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, 0.9, 1.1)
    img = tf.image.random_saturation(img, 0.9, 1.1)
    img = tf.image.random_hue(img, 0.02)

    # random crop back to 224
    img = tf.image.random_crop(img, [IMG_SIZE, IMG_SIZE, 3])

    # occasional small rotation/perspective-like effect (approx w/ transpose/flip/rot 90)
    choice = tf.random.uniform([], 0, 4, dtype=tf.int32)
    img = tf.cond(choice==1, lambda: tf.image.flip_left_right(img), lambda: img)
    img = tf.cond(choice==2, lambda: tf.image.flip_up_down(img),   lambda: img)
    img = tf.cond(choice==3, lambda: tf.image.rot90(img),          lambda: img)

    # Mild gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=5.0)
    img = tf.clip_by_value(img + noise, 0.0, 255.0)

    # Normalize 0..1
    img = img / 255.0
    return img

def preprocess_eval(img):
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img

def build_ds(df, training):
    paths = df['path'].values
    labels = df['label'].astype('float32').values

    ds_paths = tf.data.Dataset.from_tensor_slices(paths)
    ds_imgs  = ds_paths.map(decode_img, num_parallel_calls=AUTOTUNE)
    ds_imgs  = ds_imgs.map(preprocess_train if training else preprocess_eval, num_parallel_calls=AUTOTUNE)

    ds_labels = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((ds_imgs, ds_labels))
    if training:
        ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(AUTOTUNE)
    return ds

def compute_class_weights(df):
    # pos_weight for BCE if using logits; for Keras metrics we can also pass class_weight
    counts = df['label'].value_counts().to_dict()
    n0, n1 = counts.get(0,1), counts.get(1,1)
    total = n0 + n1
    w0 = total/(2.0*n0)
    w1 = total/(2.0*n1)
    return {0: w0, 1: w1}

# ---------------------------
# Model
# ---------------------------

def build_model():
    # Pretrained MobileNetV3Small
    base = keras.applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    # Unfreeze most layers (fine-tune)
    for l in base.layers:
        l.trainable = True

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    # NOTE: we normalized 0..1 already; MobileNetV3Small in TF usually expects 0..1
    x = base(x, training=True)
    x = layers.Dropout(0.2)(x)
    # Binary head
    outputs = layers.Dense(1, activation="sigmoid", name="pSensitive")(x)
    model = keras.Model(inputs, outputs)
    return model

def auroc(y_true, y_pred):
    return tf.metrics.AUC(curve='ROC', from_logits=False, name="auroc")(y_true, y_pred)

# ---------------------------
# Export to TFLite
# ---------------------------

def export_tflite(model, export_dir, rep_ds=None):
    os.makedirs(export_dir, exist_ok=True)

    # FP32
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = conv.convert()
    open(os.path.join(export_dir, "sensitive_image_v1_fp32.tflite"), "wb").write(tflite_fp32)

    # FP16
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite_fp16 = conv.convert()
    open(os.path.join(export_dir, "sensitive_image_v1_fp16.tflite"), "wb").write(tflite_fp16)

    # INT8 (full integer quant)
    if rep_ds is not None:
        def representative_gen():
            for img_batch, _ in rep_ds.take(100):  # ~100 batches for calibration
                for img in img_batch:  # each img: [224,224,3] float32 0..1
                    yield [tf.expand_dims(tf.cast(img, tf.float32), 0)]
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = representative_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type  = tf.uint8
        conv.inference_output_type = tf.uint8
        tflite_int8 = conv.convert()
        open(os.path.join(export_dir, "sensitive_image_v1_int8.tflite"), "wb").write(tflite_int8)

# ---------------------------
# Main
# ---------------------------

def main(args):
    set_seeds()

    train_df = read_csv(args.train_csv)
    val_df   = read_csv(args.val_csv)
    test_df  = read_csv(args.test_csv)

    train_ds = build_ds(train_df, training=True)
    val_ds   = build_ds(val_df,   training=False)
    test_ds  = build_ds(test_df,  training=False)

    class_weights = compute_class_weights(train_df)

    model = build_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="prec"),
            keras.metrics.Recall(name="rec"),
            keras.metrics.AUC(curve='ROC', name="auroc")
        ],
    )

    ckpt_dir = os.path.join(args.out, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    cbs = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(ckpt_dir, "best.keras"),
            monitor="val_auroc", mode="max",
            save_best_only=True, save_weights_only=False
        ),
        keras.callbacks.EarlyStopping(monitor="val_auroc", mode="max", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1)
    ]

    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=cbs
    )

    # Evaluate on test
    test_metrics = model.evaluate(test_ds, return_dict=True)
    print("TEST:", test_metrics)

    # Save Keras + TFLite
    os.makedirs(args.out, exist_ok=True)
    model.save(os.path.join(args.out, "sensitive_image_v1.keras"))

    # TFLite export; use a subset of train_ds for calibration
    export_tflite(model, export_dir=os.path.join(args.out, "export"), rep_ds=train_ds)

    # Pick a threshold from val set for best F1 (optional)
    y_true, y_prob = [], []
    for xb, yb in val_ds:
        p = model.predict(xb, verbose=0).ravel()
        y_prob.extend(p.tolist()); y_true.extend(yb.numpy().tolist())
    y_true = np.array(y_true); y_prob = np.array(y_prob)

    def f1_at(t):
        yhat = (y_prob >= t).astype(int)
        tp = ((yhat==1) & (y_true==1)).sum()
        fp = ((yhat==1) & (y_true==0)).sum()
        fn = ((yhat==0) & (y_true==1)).sum()
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        return f1, prec, rec

    ts = np.linspace(0.2, 0.9, 71)
    scores = [f1_at(t) for t in ts]
    best_idx = int(np.argmax([s[0] for s in scores]))
    best_t, (best_f1, best_prec, best_rec) = float(ts[best_idx]), scores[best_idx]
    with open(os.path.join(args.out, "val_threshold.json"), "w") as f:
        json.dump({"threshold": best_t, "F1": best_f1, "precision": best_prec, "recall": best_rec}, f, indent=2)
    print("Best threshold on val:", best_t, "F1:", best_f1, "P:", best_prec, "R:", best_rec)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--val_csv",   required=True)
    p.add_argument("--test_csv",  required=True)
    p.add_argument("--out", default="./runs/mnv3s_privacy")
    args = p.parse_args()
    main(args)
