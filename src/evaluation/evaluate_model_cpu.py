#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU evaluation for YOLOv8 (detect/segment) + optional GT merge.
- Works with models like yolov8n-seg.pt (segmentation) or yolov8n.pt (detection)
- Predicts presence = (any detection above conf threshold)
- Outputs:
  - predictions.csv  (image, n_detections, confidences, pred_has_object)
  - merged_with_gt.csv (adds GT + TP/FP/TN/FN)
  - metrics.csv (TP/FP/FN/TN + accuracy/precision/recall/f1)
"""
import argparse, os, csv, pathlib
from collections import defaultdict

import pandas as pd
from ultralytics import YOLO

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    folder = pathlib.Path(folder)
    return sorted([str(p) for p in folder.rglob("*") if p.suffix.lower() in exts])

def compute_conf_list(res):
    # supports detect/segment: both have .boxes with .conf
    confs = []
    if hasattr(res, "boxes") and res.boxes is not None and len(res.boxes) > 0:
        try:
            confs = res.boxes.conf.detach().cpu().numpy().tolist()
        except Exception:
            confs = [float(c) for c in res.boxes.conf]
    return confs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to .pt (e.g., yolov8n-seg.pt)")
    ap.add_argument("--images", required=True, help="Folder with test images")
    ap.add_argument("--outdir", required=True, help="Output folder for CSVs")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")  # force CPU
    ap.add_argument("--gt_xlsx", default="", help="Optional Excel with GT")
    ap.add_argument("--gt_sheet", default=0, help="Excel sheet name or index")
    ap.add_argument("--gt_filename_col", default="B", help="Column with filename (e.g., 'B')")
    ap.add_argument("--gt_label_col", default="L", help="Column with ground-truth True/False (e.g., 'L')")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load model on CPU
    model = YOLO(args.weights)
    model.to(args.device)

    # Gather images
    images = list_images(args.images)
    if not images:
        raise SystemExit(f"No images found in: {args.images}")

    # Predict in a memory-friendly streaming way
    preds_rows = []
    for img_path, res in zip(images, model.predict(source=images,
                                                   stream=True,
                                                   device=args.device,
                                                   imgsz=args.imgsz,
                                                   conf=args.conf,
                                                   verbose=False)):
        confs = compute_conf_list(res)
        n_det = len(confs)
        preds_rows.append({
            "image": img_path,
            "n_detections": n_det,
            "confidences": ";".join(f"{c:.3f}" for c in confs),
            "pred_has_object": int(n_det > 0)
        })

    # Save predictions
    pred_csv = os.path.join(args.outdir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image","n_detections","confidences","pred_has_object"])
        w.writeheader()
        for r in preds_rows:
            w.writerow(r)

    print(f"[OK] Wrote predictions: {pred_csv}")

    # If GT excel provided, merge and compute metrics
    if args.gt_xlsx:
        # read Excel as plain table and address columns by letter
        df_raw = pd.read_excel(args.gt_xlsx, sheet_name=args.gt_sheet, header=None)
        # Column letters -> indices
        def col_idx(letter):
            letter = letter.upper()
            idx = 0
            for ch in letter:
                idx = idx * 26 + (ord(ch) - ord('A') + 1)
            return idx - 1

        fn_idx = col_idx(args.gt_filename_col)
        gt_idx = col_idx(args.gt_label_col)

        gt_df = df_raw.iloc[:, [fn_idx, gt_idx]].copy()
        gt_df.columns = ["filename", "gt_has_object"]
        # Normalize to string stem for join
        gt_df["filename"] = gt_df["filename"].astype(str).str.strip()
        # Coerce GT to bool/int
        gt_df["gt_has_object"] = gt_df["gt_has_object"].astype(str).str.strip().str.lower().isin(["true","1","yes","y"])

        pred_df = pd.DataFrame(preds_rows)
        pred_df["filename"] = pred_df["image"].apply(lambda p: pathlib.Path(p).name)

        merged = pred_df.merge(gt_df, on="filename", how="left")

        # Confusion flags
        merged["TP"] = ((merged["pred_has_object"]==1) & (merged["gt_has_object"]==True)).astype(int)
        merged["FP"] = ((merged["pred_has_object"]==1) & (merged["gt_has_object"]==False)).astype(int)
        merged["FN"] = ((merged["pred_has_object"]==0) & (merged["gt_has_object"]==True)).astype(int)
        merged["TN"] = ((merged["pred_has_object"]==0) & (merged["gt_has_object"]==False)).astype(int)

        # Metrics
        TP = merged["TP"].sum()
        FP = merged["FP"].sum()
        FN = merged["FN"].sum()
        TN = merged["TN"].sum()
        total = TP+FP+FN+TN

        accuracy  = (TP+TN)/total if total else 0.0
        precision = TP/(TP+FP) if (TP+FP)>0 else 0.0
        recall    = TP/(TP+FN) if (TP+FN)>0 else 0.0
        f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0

        merged_csv = os.path.join(args.outdir, "merged_with_gt.csv")
        merged.to_csv(merged_csv, index=False, encoding="utf-8")
        print(f"[OK] Wrote merged results: {merged_csv}")

        metrics_csv = os.path.join(args.outdir, "metrics.csv")
        pd.DataFrame([
            {"metric":"images","value":len(merged)},
            {"metric":"TP","value":TP},
            {"metric":"FP","value":FP},
            {"metric":"FN","value":FN},
            {"metric":"TN","value":TN},
            {"metric":"accuracy","value":accuracy},
            {"metric":"precision","value":precision},
            {"metric":"recall","value":recall},
            {"metric":"f1","value":f1},
        ]).to_csv(metrics_csv, index=False)
        print(f"[OK] Wrote metrics: {metrics_csv}")

if __name__ == "__main__":
    main()
