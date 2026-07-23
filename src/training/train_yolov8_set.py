#!/usr/bin/env python3
"""
GPU-optimized YOLOv8 train/export for multi-class segmentation or detection.
- Auto-selects GPU if available
- Sensible defaults for stability + speed (AMP, EMA, cosine LR optional)
- Android-friendly exports: TorchScript, NCNN, ONNX
"""

import argparse, sys, yaml, torch, math
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser("Train YOLOv8 (GPU) and export")
    p.add_argument("--data",   type=str, required=True, help="dataset YAML")
    p.add_argument("--task",   choices=["detect","segment"], default="segment")
    p.add_argument("--model",  type=str, default=None, help="e.g. yolov8n-seg.pt")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz",  type=int, default=640)
    p.add_argument("--batch",  type=int, default=16, help="per-device batch")
    p.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', '0', '0,1'")
    p.add_argument("--workers",type=int, default=8)
    p.add_argument("--project",type=str, default="runs")
    p.add_argument("--name",   type=str, default="priv_seg_multi")
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cache",  choices=["ram","disk","False"], default="False")
    p.add_argument("--coslr",  action="store_true", help="use cosine LR schedule")
    p.add_argument("--accum",  type=int, default=1, help="grad accumulation steps")
    p.add_argument("--freeze", type=int, default=0, help="freeze first N layers")
    p.add_argument("--val_split", type=str, default="val")
    return p.parse_args()

def suggest_default_model(task):
    return "yolov8n.pt" if task=="detect" else "yolov8n-seg.pt"

def resolve_device(d):
    if d=="auto":
        return "0" if torch.cuda.is_available() else "cpu"
    return d

def read_names(yaml_path: Path):
    y = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = y["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda k:int(k))]
    return [str(n) for n in names]

if __name__=="__main__":
    args = parse_args()
    data_yaml = Path(args.data).resolve()
    if not data_yaml.exists():
        print(f"[ERROR] Data YAML not found: {data_yaml}", file=sys.stderr); sys.exit(1)

    if not args.model:
        args.model = suggest_default_model(args.task)
    args.device = resolve_device(args.device)

    print(f"[INFO] Task={args.task} | Model={args.model} | Data={data_yaml} | Device={args.device}")

    cache_opt = False if args.cache=="False" else args.cache
    model = YOLO(args.model)

    # Scale LR by batch*accum relative to nominal 64 bs
    nominal_bs = 64
    bs_effective = max(1, args.batch) * max(1, args.accum)
    lr0 = 0.01 * (bs_effective / nominal_bs)

    # ----- TRAIN -----
    results = model.train(
        data=str(data_yaml),
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,       # '0' for GPU0; '0,1' multi-GPU
        workers=args.workers,
        project=args.project,
        name=args.name,
        patience=args.patience,
        resume=args.resume,
        single_cls=False,         # << multi-class
        amp=True,                 # mixed precision (fast on GPU)
        cos_lr=args.coslr,
        lr0=lr0,
        cache=cache_opt,          # 'ram' or 'disk' or False
        deterministic=False,
        optimizer="auto",
        close_mosaic=10,
        mosaic=1.0,
        copy_paste=0.0,
        mixup=0.0,
        dropout=0.0,
        box=7.5, cls=0.5, dfl=1.5,
        # accumulate=args.accum,    # gradient accumulation
        freeze=args.freeze if args.freeze>0 else None,
        verbose=True,
    )

    # Where results saved
    save_dir = Path(getattr(model.trainer, "save_dir", getattr(results, "save_dir", ".")))
    weights_dir = save_dir / "weights"
    best_pt = weights_dir / ("best.pt" if (weights_dir / "best.pt").exists() else "last.pt")
    print(f"[INFO] Best weights: {best_pt}")

    # ----- VALIDATE on chosen split -----
    model_best = YOLO(str(best_pt))
    model_best.val(split=args.val_split, imgsz=args.imgsz, device=args.device)

    # ----- EXPORTS (pick what you use in Android) -----
    try:
        ts = model_best.export(format="torchscript", imgsz=args.imgsz, optimize=True, device=args.device)
        print(f"[OK] TorchScript: {ts}")
    except Exception as e:
        print("[WARN] TorchScript export:", e)

    try:
        ncnn = model_best.export(format="ncnn", imgsz=args.imgsz, device=args.device)
        print(f"[OK] NCNN dir: {ncnn}")
    except Exception as e:
        print("[WARN] NCNN export:", e)

    try:
        onnx = model_best.export(format="onnx", imgsz=args.imgsz, device=args.device, opset=12, dynamic=True)
        print(f"[OK] ONNX: {onnx}")
    except Exception as e:
        print("[WARN] ONNX export:", e)

    # labels.txt for Android assets
    labels_txt = save_dir / "labels.txt"
    labels_txt.write_text("\n".join(read_names(data_yaml))+"\n", encoding="utf-8")
    print(f"[OK] labels.txt: {labels_txt}")

    print("\n=== Artifacts ready ===")
    print(f"Run dir: {save_dir}")
    print(f"- Weights: {best_pt}")
    print(f"- TorchScript/NCNN/ONNX exported (see logs above)")
    print("=======================\n")

# to run the code:

# py file_path\train_yolov8_set.py --data  C:\RIT\GA\BIV-Priv-Seg\Yolov8n\yolo_multi.yaml 
# --task segment --model yolov8n-seg.pt --epochs 100 --imgsz 640 --batch 16 --device 0 
# --name priv_seg_multi --coslr
