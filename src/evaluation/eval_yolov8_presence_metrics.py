# eval_yolov8_presence_metrics.py
"""
Evaluate YOLOv8 (detect/segment) as a binary *presence* classifier at the image level.

- If --data and --split are given, it reads the split images from your dataset YAML.
  Ground truth presence is inferred from the matching labels/<split>/*.txt files.
- If --source is given (custom images folder), preds.csv is still written.
  If you ALSO pass --labels_root that mirrors your images layout, metrics are computed too.
- Outputs:
    <outdir>/preds.csv   (image, n_detections, confidences, gt_has_object, pred_has_object)
    <outdir>/metrics.csv (images, TP, FP, FN, TN, accuracy, precision, recall, f1)

Usage examples (Windows PowerShell):

# Evaluate on test split defined in your YAML
py .\eval_yolov8_presence_metrics.py `
  --model  "C:\...\runs\segment\priv_seg_bin\weights\best.pt" `
  --data   "C:\...\Yolov8n\yolo_multi.yaml" `
  --split  test --task segment --imgsz 640 --conf 0.25 --device cpu `
  --outdir "C:\...\Yolov8n\results_best_cpu"

# Predict on your own images (no labels). If you point --labels_root, metrics are computed.
py .\eval_yolov8_presence_metrics.py `
  --model  "C:\...\runs_gpu\yolov8n-seg.pt" `
  --source "C:\path\to\my_images" `
  --task segment --imgsz 640 --conf 0.25 --device 0 `
  --outdir "C:\...\Yolov8n\results_gpu_on_my_images"
"""
import argparse
import csv
import os
import sys
from glob import glob

import numpy as np
import yaml
from ultralytics import YOLO


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_images_from_dir(d):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return sorted([p for p in glob(os.path.join(d, "**", "*"), recursive=True)
                   if os.path.splitext(p)[1].lower() in exts])


def infer_split_dirs_from_yaml(data_yaml, split):
    """
    Returns (images_dir, labels_dir) for the given split by reading your dataset yaml.
    We assume the yaml uses fields like:
        path: C:/.../Yolov8n
        train: images/train2017
        val:   images/val2017
        test:  images/test2017
    and that labels live parallel to images (labels/<splitdir>).
    """
    base = data_yaml.get("path", "")
    rel = data_yaml.get(split, None)
    if rel is None:
        raise ValueError(f"Split '{split}' not found in data yaml.")

    # Normalize and build absolute image directory
    images_dir = os.path.join(base, rel) if base else rel
    images_dir = os.path.normpath(images_dir)

    # Guess labels directory by replacing "images" with "labels"
    # Example: images/train2017 -> labels/train2017
    parts = images_dir.replace("\\", "/").split("/")
    if "images" in parts:
        parts[parts.index("images")] = "labels"
        labels_dir = os.path.normpath("/".join(parts))
    else:
        # Fallback: sibling "labels" next to images folder name
        parent = os.path.dirname(images_dir)
        leaf = os.path.basename(images_dir)
        labels_dir = os.path.join(parent, "labels", leaf)
        labels_dir = os.path.normpath(labels_dir)

    return images_dir, labels_dir


def label_file_for_image(image_path, labels_root):
    """
    Convert an image path to its corresponding YOLO .txt label path under labels_root,
    preserving the leaf filename (with .txt extension).
    """
    stem = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    return os.path.join(labels_root, stem)


def has_gt_object(label_path):
    """Return 1 if label file exists with at least one non-empty line, else 0."""
    if not os.path.exists(label_path):
        return 0
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return 1
    except Exception:
        return 0
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLOv8 .pt")
    ap.add_argument("--data", help="Path to dataset YAML")
    ap.add_argument("--split", choices=["train", "val", "test"], help="Dataset split")
    ap.add_argument("--source", help="Custom images folder (overrides --data/--split)")
    ap.add_argument("--labels_root", help="Optional labels root for --source to compute metrics")
    ap.add_argument("--task", choices=["detect", "segment"], default="segment")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    preds_csv = os.path.join(args.outdir, "preds.csv")
    metrics_csv = os.path.join(args.outdir, "metrics.csv")

    # Resolve image list and (optional) labels root
    labels_root = None
    if args.source:
        images_dir = args.source
        images = list_images_from_dir(images_dir)
        labels_root = args.labels_root  # may be None
    else:
        if not (args.data and args.split):
            print("Either provide --source OR both --data and --split.", file=sys.stderr)
            sys.exit(2)
        data_yaml = read_yaml(args.data)
        images_dir, labels_root = infer_split_dirs_from_yaml(data_yaml, args.split)
        images = list_images_from_dir(images_dir)

    if not images:
        print("No images found to evaluate.", file=sys.stderr)
        sys.exit(1)

    # Load model once
    model = YOLO(args.model, task=args.task)

    # Stream predictions one-by-one to avoid high memory use
    # Note: Ultralytics will read images internally; we just pass paths in a list.
    # We call model.predict repeatedly on small lists to reduce memory pressure.
    rows = []  # for preds.csv
    TP = FP = FN = TN = 0

    for img_path in images:
        results = model.predict(
            source=[img_path],
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            stream=False,  # we pass a single image list; stream not needed
            verbose=False
        )

        # For segmentation: results[0].masks, detection: results[0].boxes
        r = results[0]
        if args.task == "segment" and r.masks is not None:
            n_det = len(r.masks)
            confs = [float(c) for c in (r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else [])]
        else:
            # detect or seg w/o masks -> use boxes
            n_det = int(r.boxes.shape[0]) if r.boxes is not None else 0
            confs = [float(c) for c in (r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else [])]

        pred_has_object = 1 if n_det > 0 else 0

        # Ground truth presence (only if labels_root is provided)
        if labels_root:
            lab_path = label_file_for_image(img_path, labels_root)
            gt_has = has_gt_object(lab_path)
        else:
            gt_has = ""

        rows.append({
            "image": img_path,
            "n_detections": n_det,
            "confidences": ";".join(f"{c:.3f}" for c in confs),
            "gt_has_object": gt_has,
            "pred_has_object": pred_has_object
        })

        if labels_root:
            if gt_has == 1 and pred_has_object == 1:
                TP += 1
            elif gt_has == 0 and pred_has_object == 1:
                FP += 1
            elif gt_has == 1 and pred_has_object == 0:
                FN += 1
            elif gt_has == 0 and pred_has_object == 0:
                TN += 1

    # Write preds.csv
    with open(preds_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "n_detections", "confidences", "gt_has_object", "pred_has_object"])
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # If we have labels_root, compute and write metrics.csv
    if labels_root:
        images_n = TP + FP + FN + TN
        eps = 1e-9
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        accuracy = (TP + TN) / (images_n + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["split", os.path.basename(images_dir) if not args.source else "custom"])
            w.writerow(["images", images_n])
            w.writerow(["TP", TP])
            w.writerow(["FP", FP])
            w.writerow(["FN", FN])
            w.writerow(["TN", TN])
            w.writerow(["accuracy", f"{accuracy:.6f}"])
            w.writerow(["precision", f"{precision:.6f}"])
            w.writerow(["recall", f"{recall:.6f}"])
            w.writerow(["f1", f"{f1:.6f}"])

        print(f"Saved metrics to: {metrics_csv}")
    else:
        print("No labels_root provided → metrics not computed (preds.csv still saved).")

    print(f"Saved predictions to: {preds_csv}")


if __name__ == "__main__":
    main()