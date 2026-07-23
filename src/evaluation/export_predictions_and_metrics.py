#!/usr/bin/env python3
"""
export_predictions_and_metrics.py

- Predicts on ALL images in --source and writes predictions_test.csv
- Computes image-level metrics against YOLO TXT labels in --labels and writes metrics_test.csv
- (Optional) also runs Ultralytics .val() if --data/--split are provided and appends mAP metrics

Examples (PowerShell):
  py C:\RIT\GA\BIV-Priv-Seg\Yolov8n\export_predictions_and_metrics.py `
    --model "C:\RIT\GA\BIV-Priv-Seg\runs\priv_seg_bin\weights\best.pt" `
    --source "C:\RIT\GA\BIV-Priv-Seg\Yolov8n\images\test2017" `
    --labels "C:\RIT\GA\BIV-Priv-Seg\Yolov8n\labels\test2017" `
    --imgsz 640 --conf 0.25 --task segment `
    --out_dir "C:\RIT\GA\BIV-Priv-Seg\Yolov8n\results_best_cpu"

(Optional mAP via Ultralytics validator):
  ... --data "C:\RIT\GA\BIV-Priv-Seg\Yolov8n\yolo_multi.yaml" --split test
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root: Path) -> List[Path]:
    files = []
    for ext in IMG_EXTS:
        files.extend(root.glob(f"*{ext}"))
    # Also scan nested dirs (common for some datasets)
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            files.append(p)
    # unique + stable order
    files = sorted({p.resolve() for p in files})
    return files


def has_gt_label(labels_root: Path, img_path: Path) -> bool:
    """A GT-positive image is one that has a non-empty label txt next to it (same stem) under labels_root."""
    lbl = labels_root / (img_path.stem + ".txt")
    if not lbl.exists():
        return False
    try:
        txt = lbl.read_text(encoding="utf-8").strip()
    except Exception:
        return False
    return len(txt) > 0


def count_predictions(result, task: str) -> Tuple[int, List[float]]:
    """Return (n_dets, confidences[]) for detect/segment result."""
    if task == "detect":
        if result.boxes is None:
            return 0, []
        conf = result.boxes.conf
        if conf is None:
            return 0, []
        confs = [float(c) for c in conf.cpu().numpy().tolist()]
        return len(confs), confs
    else:  # segment
        # In Ultralytics, result.masks is present when any segmentation was produced
        if result.masks is None:
            return 0, []
        # confidence is stored in boxes.conf even for segment models
        if result.boxes is None or result.boxes.conf is None:
            return 0, []
        confs = [float(c) for c in result.boxes.conf.cpu().numpy().tolist()]
        return len(confs), confs


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser("Export per-image predictions and image-level metrics")
    ap.add_argument("--model", required=True, type=str, help="Path to .pt weights")
    ap.add_argument("--source", required=True, type=str, help="Folder of test images")
    ap.add_argument("--labels", required=True, type=str, help="Folder of matching YOLO TXT labels")
    ap.add_argument("--task", choices=["detect", "segment"], default="segment")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--device", default="cpu", help="'cpu' or CUDA index (e.g. 0)")
    ap.add_argument("--out_dir", required=True, type=str, help="Output folder for CSV(s)")

    # Optional: run Ultralytics validator too (to append mAP metrics)
    ap.add_argument("--data", type=str, default=None, help="Dataset YAML (optional)")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])

    args = ap.parse_args()

    model = YOLO(args.model)

    source_dir = Path(args.source).resolve()
    labels_dir = Path(args.labels).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(source_dir)
    if not images:
        raise SystemExit(f"No images found under: {source_dir}")

    # ------- Inference stream over all images -------
        # ------- Inference over all images (robust, one-by-one) -------
    pred_rows = []
    TP = FP = FN = TN = 0
    bad_files = []

    for img_path in images:
        try:
            res_list = model.predict(
                source=str(img_path),
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                task=args.task,
                verbose=False,
                stream=False,   # <-- IMPORTANT: single image, not a stream list
                save=False,
            )
            res = res_list[0]
        except Exception as e:
            bad_files.append((str(img_path), str(e)))
            continue

        gt_has = 1 if has_gt_label(labels_dir, img_path) else 0
        n_det, confs = count_predictions(res, args.task)
        pred_has = 1 if n_det > 0 else 0

        pred_rows.append(
            {
                "image": str(img_path),
                "n_detections": n_det,
                "confidences": ";".join(f"{c:.3f}" for c in confs) if confs else "",
                "gt_has_object": gt_has,
                "pred_has_object": pred_has,
            }
        )

        if gt_has == 1 and pred_has == 1:
            TP += 1
        elif gt_has == 0 and pred_has == 1:
            FP += 1
        elif gt_has == 1 and pred_has == 0:
            FN += 1
        else:
            TN += 1

    # (optional) dump any files we skipped so you can inspect them
    if bad_files:
        with open(out_dir / "skipped_images.txt", "w", encoding="utf-8") as f:
            for p, err in bad_files:
                f.write(f"{p}\t{err}\n")
        print(f"[WARN] Skipped {len(bad_files)} image(s). See: {out_dir / 'skipped_images.txt'}")

    # ------- Image-level metrics -------
    accuracy = safe_div(TP + TN, TP + FP + FN + TN)
    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

    metrics_rows = [
        {"metric": "split", "value": "test"},
        {"metric": "images", "value": len(images)},
        {"metric": "TP", "value": TP},
        {"metric": "FP", "value": FP},
        {"metric": "FN", "value": FN},
        {"metric": "TN", "value": TN},
        {"metric": "accuracy", "value": accuracy},
        {"metric": "precision", "value": precision},
        {"metric": "recall", "value": recall},
        {"metric": "f1", "value": f1},
    ]

    # ------- Optional: validator (mAP) -------
    if args.data:
        print("[INFO] Running Ultralytics validator for mAP…")
        val_res = model.val(
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            task=args.task,
            verbose=False,
        )
        # Ultralytics returns a 'metrics' object; we’ll try to serialize common fields if present
        try:
            m = val_res.results_dict  # new style
        except Exception:
            # fallback to older attributes
            m = {}
            for k in [
                "metrics/precision(B)",
                "metrics/recall(B)",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "metrics/precision(M)",
                "metrics/recall(M)",
                "metrics/mAP50(M)",
                "metrics/mAP50-95(M)",
                "fitness",
            ]:
                try:
                    m[k] = float(getattr(val_res, k.replace("/", "_").replace("(", "_").replace(")", "_"), 0.0))
                except Exception:
                    pass

        # Append any available metrics (guarded)
        for k, v in m.items():
            metrics_rows.append({"metric": str(k), "value": float(v)})

        # Also drop a JSON snapshot of raw val metrics
        with open(out_dir / "metrics_val_raw.json", "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)

    # ------- Write CSVs -------
    write_csv(out_dir / "predictions_test.csv", pred_rows,
              ["image", "n_detections", "confidences", "gt_has_object", "pred_has_object"])
    write_csv(out_dir / "metrics_test.csv", metrics_rows, ["metric", "value"])

    print("\n[OK] Wrote:")
    print(f"  {out_dir / 'predictions_test.csv'}")
    print(f"  {out_dir / 'metrics_test.csv'}")
    if args.data:
        print(f"  {out_dir / 'metrics_val_raw.json'}")


if __name__ == "__main__":
    main()
