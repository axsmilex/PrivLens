import os

ROOT = r"."
IMG_DIR = os.path.join(ROOT, "query_images")
OUT = os.path.join(ROOT, "splits", "test.txt")

os.makedirs(os.path.dirname(OUT), exist_ok=True)

imgs = []
for fn in os.listdir(IMG_DIR):
    low = fn.lower()
    if low.endswith((".jpg", ".jpeg", ".png")):
        imgs.append(f"query_images/{fn}")  # relative path for YOLO

imgs.sort()

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(imgs))

print("Wrote", OUT, "with", len(imgs), "images")
