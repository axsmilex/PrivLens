#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn an Excel/CSV label file for BIV-Priv-Seg into train/val/test CSVs.

It tries to find labels in one of:
  - is_sensitive (preferred)  | Marked_Sensitive | ground_truth | result (TP/FN -> 1, TN/FP -> 0)

It tries to find filenames in one of:
  - filename | Filename_orig (uses basename) | id

It resolves each image under --base-dir, trying .jpeg/.jpg/.png if needed,
then performs a stratified split and writes train/val/test CSVs with columns:
  path,label
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _bool_from_any(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return int(x != 0)
    s = str(x).strip().lower()
    if s in {"true", "yes", "y", "t", "1"}:  return 1
    if s in {"false", "no", "n", "f", "0"}:  return 0
    # if we only have a "result" column: TP/FN means GT positive, TN/FP means GT negative
    if s in {"tp", "fn"}: return 1
    if s in {"tn", "fp"}: return 0
    return np.nan


def extract_filename(row) -> Optional[str]:
    for col in ["filename", "Filename", "file", "File"]:
        if col in row and pd.notna(row[col]):
            c = str(row[col]).strip()
            if c:
                return c
    if "Filename_orig" in row and pd.notna(row["Filename_orig"]):
        return os.path.basename(str(row["Filename_orig"]))
    if "id" in row and pd.notna(row["id"]):
        return f"{int(row['id'])}"
    return None


def resolve_local_path(base_dir: Path, name: str) -> Optional[Path]:
    p = base_dir / name
    if p.exists():  # exact match
        return p
    stem = Path(name).stem
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        cand = base_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    # also search recursively if needed
    for ext in [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]:
        matches = list(base_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def choose_label_column(df: pd.DataFrame) -> str:
    for col in ["is_sensitive", "Marked_Sensitive", "ground_truth", "result"]:
        if col in df.columns:
            return col
    raise ValueError("No label column found (expected is_sensitive / Marked_Sensitive / ground_truth / result).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="Path to Excel (.xlsx) or CSV with annotations")
    ap.add_argument("--sheet-name", default=None, help="Optional sheet name for Excel")
    ap.add_argument("--base-dir", required=True, help="Folder containing all images")
    ap.add_argument("--out-dir", required=True, help="Output dir for train/val/test CSVs")
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels_path = Path(args.labels)
    base_dir    = Path(args.base_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    if labels_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(labels_path, sheet_name=args.sheet_name)
    else:
        df = pd.read_csv(labels_path)

    # normalize column names
    df.columns = [str(c).strip().replace("\n", "_").replace(" ", "_") for c in df.columns]

    # choose label col & normalize to 0/1
    label_col = choose_label_column(df)
    df["label"] = df[label_col].map(_bool_from_any).astype("float")
    df = df[~df["label"].isna()].copy()
    df["label"] = df["label"].astype(int)

    # resolve local paths
    filenames, paths = [], []
    for _, row in df.iterrows():
        name = extract_filename(row)
        if not name:
            filenames.append(None); paths.append(None); continue
        local = resolve_local_path(base_dir, name)
        filenames.append(local.name if local else None)
        paths.append(str(local) if local else None)

    df["resolved_filename"] = filenames
    df["path"] = paths
    df = df[~df["path"].isna()].copy()

    slim = df[["path", "label"]].drop_duplicates()

    # stratified split
    train_val, test = train_test_split(
        slim, test_size=args.test_size, stratify=slim["label"], random_state=args.seed
    )
    val_rel = args.val_size / (1.0 - args.test_size)
    train, val = train_test_split(
        train_val, test_size=val_rel, stratify=train_val["label"], random_state=args.seed
    )

    # write CSVs
    (out_dir / "train.csv").write_text(train.to_csv(index=False), encoding="utf-8")
    (out_dir / "val.csv").write_text(val.to_csv(index=False), encoding="utf-8")
    (out_dir / "test.csv").write_text(test.to_csv(index=False), encoding="utf-8")

    # print quick stats
    for name, d in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        n = len(d); pos = int(d["label"].sum()); neg = n - pos
        print(f"{name}: n={n}, pos={pos} ({pos/n:.1%}), neg={neg} ({neg/n:.1%})")

    print("Wrote CSVs to:", out_dir)


if __name__ == "__main__":
    main()
