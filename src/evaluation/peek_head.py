import os, sys, pandas as pd, numpy as np, tensorflow as tf
from PIL import Image

MODEL = r"C:\RIT\GA\BIV-Priv-Seg\artifacts\sensitive_image_v1.keras"
CSV   = r"C:\RIT\GA\BIV-Priv-Seg\splits\val.csv"
# If your CSV uses relative names like "9.jpeg", set this to the images folder:
BASE  = r"C:\RIT\GA\BIV-Priv-Seg\BIV-Priv-Seg"
IMG_SIZE = 224

CAND_IMG_COLS = ["image_path","filename","file","path","img","image"]

def find_img_col(df):
    for c in CAND_IMG_COLS:
        if c in df.columns:
            return c
    raise SystemExit(f"No image column found. Tried: {CAND_IMG_COLS}\nColumns: {list(df.columns)}")

def to_path(x):
    x = str(x)
    # if absolute already:
    if os.path.isabs(x):
        return x
    return os.path.join(BASE, x)

def load_img(p, size=224):
    im = Image.open(p).convert("RGB").resize((size, size))
    x = np.asarray(im, dtype=np.float32)/255.0
    return x[None,...]

m = tf.keras.models.load_model(MODEL, compile=False)
print("Model output shape:", m.output_shape)  # should be (None, 1) for sigmoid

df = pd.read_csv(CSV)
img_col = find_img_col(df)
paths = [to_path(p) for p in df[img_col].tolist()]

# Take first 5 that exist:
keep = [p for p in paths if os.path.exists(p)][:5]
if not keep:
    raise SystemExit("No existing image paths found after joining with BASE.")

for p in keep:
    x = load_img(p, IMG_SIZE)
    y = m.predict(x, verbose=0).squeeze().item()  # sigmoid prob of "sensitive"
    print(os.path.basename(p), "prob_sensitive=", round(y, 3))
