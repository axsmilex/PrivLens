#!/usr/bin/env python3
"""
Decode COCO-style RLE segmentations from your dataset and save masks/polygons.

Inputs:
  annotations/
    instances_train2017.json
    instances_val2017.json
    instances_test2017.json

Outputs:
  masks/<split>/<image_id>__<idx>__<attr>.png     (binary 0/255)
  decoded_index_<split>.csv                       (one row per instance)
  polygons_<split>.json (optional, if --save-polygons)

Usage:
  python tools/decode_coco_rle_all.py                     # default: all splits
  python tools/decode_coco_rle_all.py --splits train2017  # just one
  python tools/decode_coco_rle_all.py --save-polygons --save-bboxes

Notes:
  - Your JSON is "COCO-like": per-image records under data["annotations"][image_id]["attributes"].
  - Each attribute has: "segmentation": {"counts": <RLE string>, "size": [H, W]}
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pycocotools import mask as maskUtils
import imageio.v2 as imageio

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

ROOT = Path(__file__).resolve().parents[1]          # .../Yolov8n
ANN_DIR = ROOT / "annotations"
MASK_DIR = ROOT / "masks"

DEFAULT_SPLITS = ["train2017", "val2017", "test2017"]


def decode_rle(rle_obj):
    """
    rle_obj: {"counts": <str or list>, "size": [H, W]}
    Returns binary mask (H,W) uint8 with values 0/1.
    """
    counts = rle_obj.get("counts")
    size = rle_obj.get("size")  # [H, W]
    if isinstance(counts, str):
        # pycocotools expects 'counts' bytes for compressed RLE
        rle = {"counts": counts.encode("utf-8"), "size": list(size)}
    else:
        # uncompressed RLE (rare here); let frPyObjects normalize it
        rle = maskUtils.frPyObjects(rle_obj, size[0], size[1])
    m = maskUtils.decode(rle)
    # decode can return (H,W,1); squeeze and cast to uint8 (0/1)
    if m.ndim == 3:
        m = m[..., 0]
    return (m > 0).astype(np.uint8)


def mask_to_polygons(mask, min_points=4, eps_frac=0.002):
    """
    Convert binary mask (0/1) -> list of polygons (each is [x1,y1,x2,y2,...]).
    Requires OpenCV. If cv2 is not available, returns [].
    """
    if not _HAS_CV2:
        return []
    # OpenCV expects 0/255
    cnts, _ = cv2.findContours((mask * 255).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    polys = []
    eps = eps_frac * (h + w)
    for c in cnts:
        if len(c) < min_points:
            continue
        c = cv2.approxPolyDP(c, eps, True)
        poly = c.reshape(-1, 2).astype(float)
        # clip to image bounds
        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        polys.append(poly.reshape(-1).tolist())
    return polys


def bbox_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    return [int(x0), int(y0), int(x1 - x0 + 1), int(y1 - y0 + 1)]  # [x,y,w,h]


def process_split(split, save_polygons=False, save_bboxes=False):
    in_json = ANN_DIR / f"instances_{split}.json"
    out_masks = MASK_DIR / split
    out_masks.mkdir(parents=True, exist_ok=True)
    out_csv = ROOT / f"decoded_index_{split}.csv"

    with open(in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    ann_map = data["annotations"]  # {image_id: {..., "attributes":[...]}}
    rows = []
    polygons_dump = []

    total_inst = 0
    saved_inst = 0

    for image_id, rec in ann_map.items():
        attrs = rec.get("attributes", [])
        H = rec.get("image_height")
        W = rec.get("image_width")
        if not H or not W:
            sz = rec.get("size")
            if isinstance(sz, list) and len(sz) == 2:
                H, W = int(sz[0]), int(sz[1])

        for k, a in enumerate(attrs):
            total_inst += 1
            attr = a.get("attr_id", "unknown")
            seg = a.get("segmentation")
            if not seg or "counts" not in seg or "size" not in seg:
                continue

            mask = decode_rle(seg)
            # Save instance mask as 0/255 PNG
            mask_path = out_masks / f"{image_id}__{k:03d}__{attr}.png"
            imageio.imwrite(mask_path.as_posix(), (mask * 255).astype(np.uint8))
            saved_inst += 1

            # bbox: prefer source bbox, else derive from mask
            bbox = a.get("bbox")
            if (not bbox) and save_bboxes:
                bbox = bbox_from_mask(mask)

            poly_rec = None
            if save_polygons:
                polys = mask_to_polygons(mask)
                poly_rec = {
                    "image_id": image_id,
                    "instance_idx": k,
                    "attr_id": attr,
                    "polygons": polys
                }
                polygons_dump.append(poly_rec)

            rows.append([
                image_id, k, attr,
                W, H,
                mask_path.as_posix(),
                json.dumps(bbox) if bbox is not None else "",
                len(poly_rec["polygons"]) if poly_rec else 0
            ])

    # CSV index
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "instance_idx", "attr_id",
                    "image_width", "image_height",
                    "mask_path", "bbox_xywh", "num_polygons"])
        w.writerows(rows)

    # Optional polygons JSON
    if save_polygons:
        poly_json = ROOT / f"polygons_{split}.json"
        with open(poly_json, "w", encoding="utf-8") as f:
            json.dump(polygons_dump, f)
        print(f"[{split}] polygons → {poly_json}")

    print(f"[{split}] decoded {saved_inst}/{total_inst} instances → {out_masks}")
    print(f"[{split}] index CSV → {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=DEFAULT_SPLITS,
                    help="Any of: train2017 val2017 test2017")
    ap.add_argument("--save-polygons", action="store_true",
                    help="Also extract polygon contours (needs OpenCV).")
    ap.add_argument("--save-bboxes", action="store_true",
                    help="Add bboxes (uses source bbox if present, or mask-derived).")
    args = ap.parse_args()

    for s in args.splits:
        process_split(s, save_polygons=args.save_polygons, save_bboxes=args.save_bboxes)


if __name__ == "__main__":
    main()
