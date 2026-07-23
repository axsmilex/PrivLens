#!/usr/bin/env python3
"""
train_yolov8_set.py
Train YOLOv8 (detect/segment) and export Android-friendly artifacts.

Examples (Windows):
  # Segmentation (recommended for privacy cues)
  py train_yolov8_set.py --data C:/RIT/GA/BIV-Priv-Seg/Yolov8n/biv_priv_seg.yaml ^
     --task segment --epochs 100 --imgsz 640 --batch 8 --device cpu --name priv_seg_y8n_v1

  # Detection (boxes only)
  py train_yolov8_set.py --data C:/RIT/GA/BIV-Priv-Seg/Yolov8n/biv_priv_seg.yaml ^
     --task detect --epochs 100 --imgsz 640 --batch 8 --device cpu --name priv_det_y8n_v1
"""

import argparse
import sys
from pathlib import Path
import yaml

from ultralytics import YOLO


# ------------------------- CLI ------------------------- #
def parse_args():
    p = argparse.ArgumentParser("Train YOLOv8 and export for Android")
    p.add_argument("--data", type=str, required=True, help="Path to dataset YAML")
    p.add_argument("--task", choices=["detect", "segment"], default="segment",
                   help="Train detection or segmentation (default: segment)")
    p.add_argument("--model", type=str, default=None,
                   help="Override model checkpoint (e.g., yolov8n.pt, yolov8n-seg.pt)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640, help="Image size (multiple of 32)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu", help="'cpu' or CUDA index like '0'")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--project", type=str, default="runs")
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--patience", type=int, default=100, help="Early-stop patience")
    p.add_argument("--int8", action="store_true", help="Export INT8 TFLite (PTQ)")
    p.add_argument("--resume", action="store_true", help="Resume last training in run dir")
    return p.parse_args()


# ---------------------- utilities ---------------------- #
def read_names_from_yaml(yaml_path: Path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    names = y.get("names")
    if isinstance(names, dict):
        # Sort by numeric key order if dict
        names = [v for k, v in sorted(names.items(), key=lambda kv: int(kv[0]))]
    elif isinstance(names, list):
        names = [str(n) for n in names]
    else:
        raise ValueError("YAML must define 'names' as list or {id:name} dict")
    return names


def write_labels_txt(names, save_dir: Path):
    labels_txt = save_dir / "labels.txt"
    labels_txt.write_text("\n".join([str(n).strip() for n in names]) + "\n", encoding="utf-8")
    return labels_txt


def suggest_default_model(task: str) -> str:
    return "yolov8n.pt" if task == "detect" else "yolov8n-seg.pt"


# ------------------------ main ------------------------- #
def main():
    args = parse_args()

    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[ERROR] Data YAML not found: {data_yaml}", file=sys.stderr)
        sys.exit(1)

    # pick default model if none provided
    if not args.model:
        args.model = suggest_default_model(args.task)

    print(f"[INFO] Task={args.task} | Model={args.model} | Data={data_yaml}")

    # Train
    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,          # use 'cpu' on machines without CUDA
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        verbose=True,
    )

    # Resolve save_dir and best weights
    save_dir = Path(getattr(model.trainer, "save_dir", getattr(results, "save_dir", ".")))
    weights_dir = save_dir / "weights"
    best_pt = (weights_dir / "best.pt") if (weights_dir / "best.pt").exists() else (weights_dir / "last.pt")
    print(f"[INFO] Best weights: {best_pt}")

    # Re-load best and export
    model_best = YOLO(str(best_pt))

    # TorchScript
    ts_path = model_best.export(format="torchscript", optimize=True, imgsz=args.imgsz)
    print(f"[OK] TorchScript: {ts_path}")

    # NCNN
    ncnn_dir = model_best.export(format="ncnn", imgsz=args.imgsz)
    print(f"[OK] NCNN: {ncnn_dir}  (contains *.param and *.bin)")

    # TFLite (with optional INT8)
    tflite_path = model_best.export(
        format="tflite",
        imgsz=args.imgsz,
        int8=args.int8,
        data=str(data_yaml),   # representative dataset for INT8 PTQ
    )
    print(f"[OK] TFLite: {tflite_path} (int8={args.int8})")

    # labels.txt for Android assets
    names = read_names_from_yaml(data_yaml)
    labels_txt = write_labels_txt(names, save_dir)
    print(f"[OK] labels.txt: {labels_txt}")

    # Final summary
    print("\n=== Android artifacts ready ===")
    print(f"Run dir: {save_dir}")
    print(f"- PyTorch (TorchScript): {ts_path}")
    print(f"- NCNN: {ncnn_dir}")
    print(f"- TFLite: {tflite_path}")
    print(f"- Labels: {labels_txt}")
    print("================================\n")


if __name__ == "__main__":
    main()
