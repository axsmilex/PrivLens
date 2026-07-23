# quick_keras_eval.py
import argparse, numpy as np, pandas as pd
from PIL import Image
import tensorflow as tf

def preprocess(img_path, size):
    img = Image.open(img_path).convert("RGB").resize((size, size))
    x = np.asarray(img, dtype=np.float32)
    x = x / 127.5 - 1.0          # MobileNetV3 normalization to [-1, 1]
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to .keras model")
    ap.add_argument("--csv", required=True, help="CSV with columns path,label")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    print("Loading model…")
    model = tf.keras.models.load_model(args.model)

    print("Reading CSV…")
    df = pd.read_csv(args.csv)
    paths = df["path"].tolist()
    labels = df["label"].astype(int).values

    print(f"Loading {len(paths)} images…")
    xs = np.stack([preprocess(p, args.size) for p in paths], axis=0)

    print("Running inference…")
    probs = model.predict(xs, batch_size=args.batch, verbose=0).reshape(-1)
    preds = (probs >= args.threshold).astype(int)

    acc = (preds == labels).mean()
    from sklearn.metrics import roc_auc_score, classification_report
    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")

    print(f"\nAccuracy: {acc:.4f} | AUROC: {auroc:.4f} | threshold={args.threshold}")
    print("\nClassification report:")
    print(classification_report(labels, preds, target_names=["non_sensitive","sensitive"]))

if __name__ == "__main__":
    main()
