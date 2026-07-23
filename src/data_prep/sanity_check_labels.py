# peek_head.py
import tensorflow as tf, numpy as np, pandas as pd
from PIL import Image

model_path = r"C:\RIT\GA\BIV-Priv-Seg\artifacts\sensitive_image_v1.keras"
csv_path   = r"C:\RIT\GA\BIV-Priv-Seg\splits\val.csv"
img_col    = "image_path"  # or "filename"
label_col  = "is_sensitive"

m = tf.keras.models.load_model(model_path, compile=False)
print("Output shape:", m.output_shape)

df = pd.read_csv(csv_path).head(5)
def load(img, size=224):
    x = np.asarray(Image.open(img).convert("RGB").resize((size,size))) / 255.0
    return x[None,...].astype("float32")

for p in df[img_col].tolist():
    x = load(p)
    y = m.predict(x, verbose=0)
    print(os.path.basename(p), y, y.shape)
