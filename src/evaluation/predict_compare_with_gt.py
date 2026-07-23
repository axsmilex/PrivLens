#!/usr/bin/env python3
"""
predict_compare_with_gt.py

Run a YOLOv8 segmentation or detection model on a folder of images, then compare
presence/absence predictions with ground truth contained in an Excel sheet.

Ground truth format (flexible):
- Column B contains filenames (with or without extension).
- Column L contains ground-truth label (True/False or 1/0) meaning: True => contains sensitive info.
- If your sheet uses different columns or has headers, you can override with --filename_col and --label_col
  by giving either the Excel column letter (e.g., "B", "L") or the exact header name.

Outputs:
- <outcsv>: per-image rows with predictions and ground truth
- <metrics_csv>: a tiny CSV with TP/FP/FN/TN, accuracy, precision, recall, F1

Example:
python predict_compare_with_gt.py \
  --model "C:/path/to/best.pt" \
  --images "C:/path/to/my_images" \
  --excel  "C:/path/to/Copy of test_result_new.xlsx" \
  --sheet  "Sheet1" \
  --filename_col B --label_col L \
  --task segment --conf 0.25 --imgsz 640 --device cpu \
  --outcsv "C:/path/to/results_compare.csv"
"""

import argparse
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

def letter_to_index(col: str) -> int:
    col = col.strip().upper()
    total = 0
    for ch in col:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"Invalid column letter: {col}")
        total = total * 26 + (ord(ch) - ord('A') + 1)
    return total - 1

def pick_column(df: pd.DataFrame, spec: str) -> str:
    for c in df.columns:
        if str(c).strip().lower() == str(spec).strip().lower():
            return c
    try:
        idx = letter_to_index(spec)
        return df.columns[idx]
    except Exception:
        pass
    raise KeyError(f"Could not resolve column '{spec}' by name or letter. Available: {list(df.columns)}")

def coerce_bool(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "y", "t"}:
        return 1
    if s in {"0", "false", "no", "n", "f"}:
        return 0
    try:
        v = float(s)
        if v == 1.0:
            return 1
        if v == 0.0:
            return 0
    except Exception:
        pass
    return None

def basename_no_ext(p: str) -> str:
    b = os.path.basename(p)
    b = re.sub(r'\s+', '', b)
    name, _ = os.path.splitext(b)
    return name.lower()

def load_ground_truth(excel_path: str, sheet: Optional[str], filename_col: str, label_col: str) -> dict:
    df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
    df = df.dropna(axis=1, how="all")
    file_col = pick_column(df, filename_col)
    gt_col   = pick_column(df, label_col)

    mapping = {}
    for _, row in df.iterrows():
        fname_cell = row.get(file_col, None)
        if pd.isna(fname_cell):
            continue
        key = basename_no_ext(str(fname_cell))
        label = coerce_bool(row.get(gt_col, None))
        if label is None:
            continue
        mapping[key] = label
    return mapping

def compute_metrics(tp, fp, fn, tn) -> dict:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "images": total,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def run_predictions(args) -> pd.DataFrame:
    from ultralytics import YOLO

    model = YOLO(args.model)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    img_paths = [str(p) for p in Path(args.images).rglob("*") if p.suffix.lower() in exts]
    img_paths.sort()

    rows = []
    results_gen = model.predict(
        source=img_paths,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    )
    for img_path, res in zip(img_paths, results_gen):
        n_det = 0
        confs = []
        if getattr(res, "boxes", None) is not None and res.boxes is not None and len(res.boxes) > 0:
            confs = res.boxes.conf.detach().cpu().numpy().tolist()
            n_det = sum(c >= args.conf - 1e-9 for c in confs)
        pred_has_object = 1 if n_det > 0 else 0

        rows.append({
            "image": img_path,
            "n_detections": int(n_det),
            "confidences": ";".join(f"{c:.3f}" for c in confs) if confs else "",
            "pred_has_object": pred_has_object,
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--excel", required=True)
    ap.add_argument("--sheet", default=None)
    ap.add_argument("--filename_col", default="B")
    ap.add_argument("--label_col", default="L")
    ap.add_argument("--task", default="segment", choices=["segment", "detect"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--outcsv", required=True)
    ap.add_argument("--metrics_csv", default=None)
    args = ap.parse_args()

    gt_map = load_ground_truth(args.excel, args.sheet, args.filename_col, args.label_col)
    pred_df = run_predictions(args)

    pred_df["key"] = pred_df["image"].apply(basename_no_ext)
    gt_series = pred_df["key"].map(gt_map)
    pred_df["gt_has_object"] = gt_series.fillna(-1).astype(int)

    pred_df["TP"] = ((pred_df["pred_has_object"] == 1) & (pred_df["gt_has_object"] == 1)).astype(int)
    pred_df["FP"] = ((pred_df["pred_has_object"] == 1) & (pred_df["gt_has_object"] == 0)).astype(int)
    pred_df["FN"] = ((pred_df["pred_has_object"] == 0) & (pred_df["gt_has_object"] == 1)).astype(int)
    pred_df["TN"] = ((pred_df["pred_has_object"] == 0) & (pred_df["gt_has_object"] == 0)).astype(int)
    pred_df["gt_known"] = (pred_df["gt_has_object"] != -1).astype(int)

    known = pred_df[pred_df["gt_known"] == 1]
    tp = int(known["TP"].sum())
    fp = int(known["FP"].sum())
    fn = int(known["FN"].sum())
    tn = int(known["TN"].sum())
    metrics = compute_metrics(tp, fp, fn, tn)

    out_path = Path(args.outcsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_path, index=False)

    if args.metrics_csv:
        m = pd.DataFrame([metrics])
        m.to_csv(args.metrics_csv, index=False)

    print("Saved:", out_path)
    print("Known GT images:", len(known), " / total:", len(pred_df))
    print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
    print("Accuracy:", f"{metrics['accuracy']:.4f}",
          "Precision:", f"{metrics['precision']:.4f}",
          "Recall:", f"{metrics['recall']:.4f}",
          "F1:", f"{metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
