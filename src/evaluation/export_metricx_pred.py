import argparse, csv, json, yaml
from pathlib import Path
from ultralytics import YOLO

def load_split_dirs(data_yaml: Path, split: str):
    with open(data_yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    base = Path(y.get("path", "."))
    split_key = {"train":"train", "val":"val", "test":"test"}[split]
    img_rel = y[split_key]  # e.g., "images/test2017"
    # labels folder is sibling of images folder
    img_dir = (base / img_rel).resolve()
    lbl_dir = (base / "labels" / Path(img_rel).name).resolve()
    return img_dir, lbl_dir

def has_gt_object(lbl_path: Path) -> bool:
    # YOLO-seg label: each line starts with class_id followed by 2n normalized coords.
    try:
        if not lbl_path.exists(): return False
        txt = lbl_path.read_text().strip()
        return any(line.strip() and line.split()[0].isdigit() for line in txt.splitlines())
    except Exception:
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to .pt (yolov8n-seg, etc.)")
    ap.add_argument("--data",  required=True, help="Path to biv_priv_seg.yaml")
    ap.add_argument("--split", default="test", choices=["train","val","test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pred_csv = outdir / f"predictions_{args.split}.csv"
    metrics_csv = outdir / f"metrics_{args.split}.csv"
    metrics_json = outdir / f"metrics_{args.split}.json"

    # Resolve split dirs from your YAML
    img_dir, lbl_dir = load_split_dirs(Path(args.data), args.split)

    # Load model
    model = YOLO(args.model)

    # --- 1) STREAM predictions and write a compact CSV ---
    rows = []
    tp=fp=fn=tn=0
    # use folder (faster & robust)
    for r in model.predict(source=str(img_dir), task="segment", imgsz=args.imgsz,
                           conf=args.conf, stream=True, verbose=False, save=False):
        img_path = Path(r.path)
        n = 0 if r.boxes is None else len(r.boxes)
        confs = [] if r.boxes is None else [float(c) for c in r.boxes.conf.cpu().numpy().tolist()]
        # map image → label
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt_present = has_gt_object(lbl_path)
        pred_present = n > 0

        if pred_present and gt_present: tp += 1
        elif pred_present and not gt_present: fp += 1
        elif (not pred_present) and gt_present: fn += 1
        else: tn += 1

        rows.append({
            "image": str(img_path),
            "n_detections": n,
            "confidences": ";".join(f"{c:.3f}" for c in confs),
            "gt_has_object": int(gt_present),
            "pred_has_object": int(pred_present)
        })

    # write predictions csv
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["image","n_detections","confidences","gt_has_object","pred_has_object"])
        w.writeheader(); w.writerows(rows)

    # simple presence-based metrics
    total = tp+fp+fn+tn
    precision = tp / (tp+fp) if (tp+fp) else 0.0
    recall    = tp / (tp+fn) if (tp+fn) else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) else 0.0
    accuracy  = (tp+tn) / total if total else 0.0
    simple_metrics = {
        "split": args.split, "images": total,
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1
    }

    # --- 2) Official YOLO validation to get mAP for segmentation ---
    val_res = model.val(data=args.data, split=args.split, imgsz=args.imgsz, save_json=True, task="segment", verbose=False)
    # results_dict works across versions; contains map50, map, etc.
    official = getattr(val_res, "results_dict", {})
    # dump both
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({"simple": simple_metrics, "yolo_official": official}, f, indent=2)

    # also a tiny CSV summary for quick glance
    with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric","value"])
        for k,v in simple_metrics.items():
            w.writerow([k,v])
        for k,v in official.items():
            w.writerow([k,v])

    print(f"[OK] Wrote predictions -> {pred_csv}")
    print(f"[OK] Wrote metrics (CSV) -> {metrics_csv}")
    print(f"[OK] Wrote metrics (JSON) -> {metrics_json}")

if __name__ == "__main__":
    main()
