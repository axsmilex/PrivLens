# train_mobilenetv3small_privacy_v2.py
import os, json, argparse, pathlib
import numpy as np
import pandas as pd
import tensorflow as tf
import shutil
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

AUTOTUNE = tf.data.AUTOTUNE

# ------------------------
# IO & splits
# ------------------------
def read_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    # normalize headers -> "path","label"
    if "path" not in df.columns or "label" not in df.columns:
        canon = {c.lower().strip(): c for c in df.columns}
        path_col = canon.get("path") or canon.get("image_path") or canon.get("img") or canon.get("file") or canon.get("filename")
        label_col = canon.get("label") or canon.get("is_sensitive") or canon.get("ground_truth") or canon.get("class")
        if path_col is None or label_col is None:
            raise ValueError(f"CSV must have columns path,label (or compatible). Found: {list(df.columns)}")
        df = df.rename(columns={path_col: "path", label_col: "label"})
    # absolute/normalized paths; int labels
    df["path"] = df["path"].apply(lambda p: str(pathlib.Path(p)))
    df["label"] = df["label"].astype(int)
    return df

def print_split_stats(name, df):
    n = len(df); pos = int(df["label"].sum()); neg = n - pos
    print(f"[{name}] n={n}  pos={pos}  neg={neg}  pos_ratio={pos/max(1,n):.3f}")

# ------------------------
# Dataset
# ------------------------
def decode_img(path, img_size):
    img_b = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_b, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)        # [0,1]
    img = tf.image.resize(img, [img_size, img_size])
    return img

def make_dataset(df, img_size, batch, train=False, shuffle=True, aug=None):
    paths = df["path"].values
    labels = df["label"].values.astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle and train:
        ds = ds.shuffle(min(8192, len(paths)), reshuffle_each_iteration=True)

    def _map(p, y):
        img = decode_img(p, img_size)
        if train and aug is not None:
            img = aug(img, training=True)
        return img, y

    ds = ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch).prefetch(AUTOTUNE)
    return ds

# ------------------------
# Model
# ------------------------
def build_model(img_size=224, dropout=0.2):
    # Use MobileNetV3Small with imagenet weights; add preprocessing
    preprocess = tf.keras.applications.mobilenet_v3.preprocess_input
    inp = tf.keras.Input(shape=(img_size, img_size, 3))
    x = layers.Lambda(preprocess, name="preprocess")(inp)
    base = tf.keras.applications.MobileNetV3Small(
        include_top=False, input_tensor=x, weights="imagenet"
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="sensitive_prob")(x)
    model = tf.keras.Model(inp, out, name="mobilenetv3_privacy")
    return model, base

def set_trainable_tail(backbone, unfreeze_ratio: float):
    # Unfreeze last N% of the backbone layers
    total = len(backbone.layers)
    cut = int(total * (1.0 - unfreeze_ratio))
    for i, l in enumerate(backbone.layers):
        l.trainable = (i >= cut)

# ------------------------
# Loss & exports
# ------------------------
def focal_loss(gamma=2.0, alpha=0.25):
    def _loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        w = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)
        return -tf.reduce_mean(w * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return _loss

def export_tflite_all(model, out_dir, rep_ds=None, img_size=224):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Build a concrete function (freezes the graph; avoids trackable traversal)
    @tf.function(input_signature=[
        tf.TensorSpec([None, img_size, img_size, 3], tf.float32, name="input")
    ])
    def serve(x):
        y = model(x, training=False)
        return {"sensitive_prob": y}

    concrete = serve.get_concrete_function()

    # 2) FP32 TFLite
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    tflite_fp32 = conv.convert()
    fp32_path = os.path.join(out_dir, "mobilenetv3_privacy_fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)

    # 3) Dynamic-range INT8 (float input/output)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dr = conv.convert()
    dr_path = os.path.join(out_dir, "mobilenetv3_privacy_int8_dynamic.tflite")
    with open(dr_path, "wb") as f:
        f.write(tflite_dr)

    # 4) Full INT8 (uint8 in/out) — optional; skip if kernels unsupported
    int8_path = None
    try:
        conv = tf.lite.TFLiteConverter.from_concrete_functions([concrete])
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        if rep_ds is not None:
            def rep_gen():
                # a few dozen samples is fine
                for x, _ in rep_ds.unbatch().take(100):
                    # ensure float32 input for rep samples
                    yield [tf.cast(tf.expand_dims(x, 0), tf.float32)]
            conv.representative_dataset = rep_gen
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        conv.inference_input_type = tf.uint8
        conv.inference_output_type = tf.uint8
        tflite_int8 = conv.convert()
        int8_path = os.path.join(out_dir, "mobilenetv3_privacy_int8_full.tflite")
        with open(int8_path, "wb") as f:
            f.write(tflite_int8)
    except Exception as e:
        print("[export] Full INT8 quantization not produced (OK):", e)

    return fp32_path, dr_path, int8_path



# ------------------------
# Args / main
# ------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv",   required=True)
    ap.add_argument("--test_csv",  required=True)
    ap.add_argument("--out",       required=True)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--epochs_head", type=int, default=6)
    ap.add_argument("--epochs_ft",   type=int, default=18)
    ap.add_argument("--ft_ratio",    type=float, default=0.5, help="unfreeze last N%% of backbone")
    ap.add_argument("--use_focal",   action="store_true", help="use focal loss instead of BCE+class_weights")
    return ap.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Light, text-friendly augmentations (no rotate API needed)
    AUG = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),     # ~±5°
        layers.RandomContrast(0.1),
        layers.RandomZoom(0.05),
    ], name="augment")

    # --- Load splits
    train_df = read_csv(args.train_csv)
    val_df   = read_csv(args.val_csv)
    test_df  = read_csv(args.test_csv)

    print_split_stats("train", train_df)
    print_split_stats("val",   val_df)
    print_split_stats("test",  test_df)

    # --- Datasets
    train_ds = make_dataset(train_df, args.img_size, args.batch, train=True, aug=AUG)
    val_ds   = make_dataset(val_df,   args.img_size, args.batch, train=False, shuffle=False)
    test_ds  = make_dataset(test_df,  args.img_size, args.batch, train=False, shuffle=False)

    # --- Model build
    model, backbone = build_model(img_size=args.img_size)

    # Phase 1: train head only
    backbone.trainable = False

    loss_fn = focal_loss() if args.use_focal else tf.keras.losses.BinaryCrossentropy()
    metrics = [tf.keras.metrics.AUC(name="auroc"), "accuracy",
               tf.keras.metrics.Precision(name="prec"),
               tf.keras.metrics.Recall(name="rec")]
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss=loss_fn, metrics=metrics)

    class_weight = None
    if not args.use_focal:
        y = train_df["label"].values
        cw = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y)
        class_weight = {0: float(cw[0]), 1: float(cw[1])}
        print("Class weights:", class_weight)

    ckpt_dir = os.path.join(args.out, "ckpt_v2"); os.makedirs(ckpt_dir, exist_ok=True)
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best_head.keras"),
                                           monitor="val_auroc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auroc", mode="max", patience=2, factor=0.5, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_auroc", mode="max", patience=4, restore_best_weights=True),
    ]
    print("\n=== Phase 1: head-only training ===")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_head,
              class_weight=class_weight, callbacks=cbs, verbose=2)

    # Phase 2: unfreeze last N% and fine-tune with lower LR (same model)
    print("\n=== Phase 2: fine-tune backbone ===")
    backbone.trainable = True
    set_trainable_tail(backbone, args.ft_ratio)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss_fn, metrics=metrics)

    cbs2 = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(ckpt_dir, "best_finetune.keras"),
                                           monitor="val_auroc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auroc", mode="max", patience=2, factor=0.5, min_lr=5e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_auroc", mode="max", patience=5, restore_best_weights=True),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs_ft,
              class_weight=None if args.use_focal else class_weight, callbacks=cbs2, verbose=2)

    # --- Eval
    print("\n=== Final validation & test ===")
    val_metrics  = model.evaluate(val_ds,  return_dict=True, verbose=0)
    test_metrics = model.evaluate(test_ds, return_dict=True, verbose=0)
    print("VAL :", val_metrics)
    print("TEST:", test_metrics)

    # --- Threshold calibration (maximize F1 on val)
    y_val = []; p_val = []
    for xb, yb in val_ds:
        y_val.append(yb.numpy())
        p_val.append(model.predict(xb, verbose=0).squeeze())
    y_val = np.concatenate(y_val).astype(int)
    p_val = np.concatenate(p_val)
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    f1 = (2*prec*rec)/(prec+rec+1e-9)
    best_idx = int(np.nanargmax(f1[:-1])) if len(f1) > 1 else 0
    best_t = float(thr[best_idx]) if len(thr) else 0.5
    print(f"Calibrated threshold (max F1 on val): {best_t:.3f}  |  val_AUROC={roc_auc_score(y_val, p_val):.3f}  val_F1={np.nanmax(f1):.3f}")

    with open(os.path.join(args.out, "val_threshold.json"), "w") as f:
        json.dump({
            "threshold": best_t,
            "val_auroc": float(roc_auc_score(y_val, p_val)),
            "val_f1": float(np.nanmax(f1)),
            "val_prec_at_best": float(prec[best_idx]),
            "val_rec_at_best": float(rec[best_idx]),
        }, f, indent=2)

    # --- Save Keras + TFLite
    keras_path = os.path.join(args.out, "sensitive_image_v2.keras")
    model.save(keras_path)
    print("Saved Keras:", keras_path)

    export_dir = os.path.join(args.out, "export_v2"); os.makedirs(export_dir, exist_ok=True)
    fp32, dr, int8 = export_tflite_all(model, export_dir, rep_ds=train_ds)
    print("TFLite FP32:         ", fp32)
    print("TFLite INT8 dynamic: ", dr)
    if int8:
        print("TFLite INT8 full:    ", int8)

if __name__ == "__main__":
    main()
