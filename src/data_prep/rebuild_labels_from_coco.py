
from __future__ import annotations
import argparse, json, sys, yaml, pathlib, re
from collections import Counter, defaultdict

SPLITS = ["train2017","val2017","test2017"]

def find_json(ann_root: pathlib.Path, split: str) -> pathlib.Path | None:
    # support both: train2017.json and instances_train2017.json
    cands = [
        ann_root / f"{split}.json",
        ann_root / f"instances_{split}.json",
    ]
    for p in cands:
        if p.exists():
            return p
    return None

def load_yaml_names(yaml_path: pathlib.Path) -> list[str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = data.get("names", None)
    if not isinstance(names, list) or not names:
        raise ValueError(f"'names' list not found in YAML: {yaml_path}")
    # normalize: strip spaces
    return [str(n).strip() for n in names]

def norm_poly(poly_xy: list[int|float], w: int, h: int) -> list[str]:
    """Normalize polygon [x1,y1,...] to strings with 6 decimals."""
    out = []
    it = iter(poly_xy)
    for x, y in zip(it, it):
        nx = float(x) / float(w)
        ny = float(y) / float(h)
        # clamp just in case
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        out.append(f"{nx:.6f}")
        out.append(f"{ny:.6f}")
    return out

def main():
    ap = argparse.ArgumentParser(description="Rebuild YOLOv8 segmentation labels from COCO-like privacy annotations")
    ap.add_argument("--images_root", required=True, type=pathlib.Path)
    ap.add_argument("--annotations_root", required=True, type=pathlib.Path)
    ap.add_argument("--out_labels_root", required=True, type=pathlib.Path)
    ap.add_argument("--class_mode", choices=["single","multi"], default="single",
                    help="single: all instances → class 0; multi: use attr_id → class mapping")
    ap.add_argument("--names_yaml", type=pathlib.Path,
                    help="When --class_mode multi, read class order from YAML 'names'")
    ap.add_argument("--splits", nargs="*", default=SPLITS, help="Which splits to build")
    args = ap.parse_args()

    ann_root = args.annotations_root
    out_root = args.out_labels_root
    out_root.mkdir(parents=True, exist_ok=True)

    # Build attr_id -> class_id mapping (multi only)
    attr_to_cid: dict[str,int] = {}
    known_attrs: set[str] = set()

    if args.class_mode == "multi":
        if not args.names_yaml:
            print("[ERROR] --names_yaml is required for multi-class mode.", file=sys.stderr)
            sys.exit(2)
        names = load_yaml_names(args.names_yaml)
        attr_to_cid = {name: i for i, name in enumerate(names)}
        # Some people keep names like 'a38_ticket'; accept exact match only.
        # You can add aliases here if needed.

    total_imgs = 0
    total_lines = 0
    per_split_counts = {}
    per_class_counts = Counter()
    missing_attr = Counter()

    for split in args.splits:
        ann_path = find_json(ann_root, split)
        if not ann_path:
            print(f"[WARN] No annotations file for split '{split}' under {ann_root}", file=sys.stderr)
            continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        # Allow two shapes: {annotations: {...}} or direct dict of images
        annot = data.get("annotations", data)

        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        n_imgs = 0
        n_lines = 0

        for img_id, rec in annot.items():
            w = rec.get("image_width") or rec.get("width")
            h = rec.get("image_height") or rec.get("height")
            if not (w and h):
                # skip malformed
                continue

            attrs = rec.get("attributes", [])
            lines = []

            for a in attrs:
                attr_id = a.get("attr_id")
                polys = a.get("polygons") or []

                # Choose class id
                if args.class_mode == "single":
                    cid = 0
                else:
                    if attr_id not in attr_to_cid:
                        missing_attr[attr_id] += 1
                        continue
                    cid = attr_to_cid[attr_id]
                    known_attrs.add(attr_id)

                # Each polygon becomes one line
                for poly in polys:
                    if not poly or len(poly) < 6:
                        continue  # need at least 3 points
                    norm = norm_poly(poly, w, h)
                    lines.append(" ".join([str(cid)] + norm))
                    per_class_counts[cid] += 1

            # Write (even if empty → YOLO expects no file for empty image, so skip)
            if lines:
                (out_dir / f"{img_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
                n_lines += len(lines)
                n_imgs += 1
            # else: leave no .txt to indicate no objects in this image

        per_split_counts[split] = (n_imgs, n_lines)
        total_imgs += n_imgs
        total_lines += n_lines
        print(f"[BUILD] {split}: images_with_labels={n_imgs}, label_lines={n_lines}, out={out_dir}")

    if args.class_mode == "multi":
        # Sanity report
        print("\n[CLASS SUMMARY]")
        print("class_id  count")
        for cid in sorted(per_class_counts):
            print(f"{cid:7d}  {per_class_counts[cid]}")
        if missing_attr:
            print("\n[WARNING] Attributes in JSON that did NOT match any YAML names (skipped):")
            for k, v in missing_attr.most_common():
                print(f"  {k}  -> {v} instances")

        # Optional: warn if some YAML names never appeared
        unused = [n for n in attr_to_cid.keys() if n not in known_attrs]
        if unused:
            print("\n[NOTE] YAML names never seen in data:")
            for n in unused:
                print(" ", n)

    print(f"\n[OK] Done. Total images with labels: {total_imgs}, total polygons written: {total_lines}")
    print(f"[OK] Labels at: {out_root}")
    return 0

if __name__ == "__main__":
    sys.exit(main())