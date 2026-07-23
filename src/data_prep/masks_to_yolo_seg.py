#!/usr/bin/env python3
"""
Convert decoded masks (from masks/<split>/...) into YOLOv8-Seg labels.

Inputs:
  decoded_index_{train2017,val2017,test2017}.csv
  masks/<split>/*__<idx>__<attr>.png

Outputs:
  labels/<split>/<image_id>.txt
"""

import csv
import json
from pathlib import Path
import re
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_BY_SPLIT = {
    "train2017": ROOT / "decoded_index_train2017.csv",
    "val2017":   ROOT / "decoded_index_val2017.csv",
    "test2017":  ROOT / "decoded_index_test2017.csv",
}
MASKS_DIR = ROOT / "masks"
LABELS_DIR = ROOT / "labels"

# ======= EDIT THIS: attr_id -> class index =======
ATTR_TO_CLASS = {
    "a105_face_all": 0,                  # face
    "a108_license_plate_all": 1,         # license plate
    "a31_passport": 2, "a32_drivers_license": 2, "a33_student_id": 2, # id doc
    "a30_credit_card": 3,
    "a49_phone": 4,                       # phone / screen content
    "a35_mail": 5, "a37_receipt": 5, "a38_ticket": 5,
    "a106_address_current_all": 6, "a107_address_home_all": 6,
    "a111_name_all": 7,
    "a90_email": 8, "a85_username": 8,
    "a110_nudity_all": 9,
    # everything else -> optional: comment out or map to a class
}

def largest_polygon_from_mask(mask, eps_frac=0.002, min_points=8):
    """Return the largest exterior polygon as a list of (x, y) float points."""
    h, w = mask.shape
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    # pick largest component
    c = max(cnts, key=cv2.contourArea)
    eps = eps_frac * (h + w)
    c = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
    if len(c) < min_points:
        return []
    # clip to image bounds
    c[:, 0] = np.clip(c[:, 0], 0, w - 1)
    c[:, 1] = np.clip(c[:, 1], 0, h - 1)
    return [(float(x), float(y)) for x, y in c]

def write_label_line(fp, cls_id, poly, w, h):
    # normalize to [0,1]
    nums = []
    for x, y in poly:
        nums += [x / w, y / h]
    s = str(cls_id) + " " + " ".join(f"{v:.6f}" for v in nums)
    fp.write(s + "\n")

def process_split(split):
    csv_path = CSV_BY_SPLIT[split]
    out_dir = LABELS_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        # group rows by image_id
        by_image = {}
        for r in rdr:
            img = r["image_id"]
            by_image.setdefault(img, []).append(r)

    n_img, n_inst, n_kept = 0, 0, 0
    for image_id, rows in by_image.items():
        # assume all rows have same W/H for a given image_id
        w = int(rows[0]["image_width"]); h = int(rows[0]["image_height"])
        out_txt = out_dir / f"{image_id}.txt"
        wrote_any = False

        with open(out_txt, "w", encoding="utf-8") as fp:
            for r in rows:
                n_inst += 1
                attr = r["attr_id"]
                # parse attr id from filename if needed
                if attr not in ATTR_TO_CLASS:
                    # try to recover from mask filename "...__<attr>.png"
                    m = re.search(r"__([a-z0-9_]+)\.png$", r["mask_path"], flags=re.I)
                    if m and m.group(1) in ATTR_TO_CLASS:
                        attr = m.group(1)
                    else:
                        continue  # skip unlabeled attr

                cls_id = ATTR_TO_CLASS[attr]
                mask_path = Path(r["mask_path"])
                mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                mask = (mask > 127).astype(np.uint8) * 255

                poly = largest_polygon_from_mask(mask)
                if not poly:
                    continue
                write_label_line(fp, cls_id, poly, w, h)
                wrote_any = True
                n_kept += 1

        if not wrote_any:
            out_txt.unlink(missing_ok=True)
        n_img += 1

    print(f"[{split}] images={n_img}, instances={n_inst}, kept={n_kept}, labels dir={out_dir}")

def main(): 
    for split in ["train2017", "val2017", "test2017"]:
        process_split(split)

if __name__ == "__main__":
    main()
