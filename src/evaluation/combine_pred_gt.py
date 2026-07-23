import argparse, re
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

def extract_id_from_path(p: str):
    if not isinstance(p, str):
        return None
    name = Path(p).name
    m = re.search(r'(\d+)(?=\.[A-Za-z]+$)', name)  # e.g., 123 in 123.jpg/.jpeg
    return int(m.group(1)) if m else None

def infer_gt_sheet(xls: pd.ExcelFile):
    for s in xls.sheet_names:
        df = xls.parse(s, header=0)
        cols = {str(c).strip().lower() for c in df.columns}
        if "ground_truth" in cols:
            return s
    return xls.sheet_names[0]

def to_int_id(series):
    """Coerce to pandas nullable Int64 (avoids float issues)."""
    return pd.to_numeric(series, errors="coerce").astype("Int64")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred_sheet", default=None)
    ap.add_argument("--gt_sheet", default=None)
    ap.add_argument("--out_csv", default="pred_vs_gt.csv")
    ap.add_argument("--metrics_csv", default="pred_vs_gt_metrics.csv")
    args = ap.parse_args()

    # --- load predictions
    pred_path = Path(args.pred)
    if pred_path.suffix.lower() in [".xlsx", ".xls"]:
        pred_df = pd.read_excel(pred_path, sheet_name=args.pred_sheet)
    else:
        pred_df = pd.read_csv(pred_path)

    pred_df.columns = [c.strip().lower() for c in pred_df.columns]
    if "image" not in pred_df.columns or "pred_has_object" not in pred_df.columns:
        raise ValueError("Predictions must have 'image' and 'pred_has_object' columns.")

    pred_df["file_id"] = pred_df["image"].apply(extract_id_from_path)
    pred_df["file_id"] = to_int_id(pred_df["file_id"])

    # --- load ground truth
    xls = pd.ExcelFile(args.gt)
    gt_sheet = args.gt_sheet or infer_gt_sheet(xls)
    gt_df = xls.parse(gt_sheet)
    gt_df.columns = [c.strip().lower() for c in gt_df.columns]

    # pick id column
    id_col = None
    for cand in ["filenar", "fileid", "id", "filename", "filename_orig"]:
        if cand in gt_df.columns:
            id_col = cand
            break
    if id_col is None:
        for c in gt_df.columns:
            if gt_df[c].astype(str).str.contains(r"\.(jpg|jpeg|png)$", case=False, na=False).any():
                id_col = c
                break
    if id_col is None:
        raise ValueError("Could not find an id/filename column in GT (e.g., 'filenar').")

    if id_col == "filenar":
        gt_df["file_id"] = to_int_id(gt_df["filenar"])
    else:
        gt_df["file_id"] = gt_df[id_col].astype(str).apply(extract_id_from_path)
        gt_df["file_id"] = to_int_id(gt_df["file_id"])

    if "ground_truth" not in gt_df.columns:
        raise ValueError("Ground-truth sheet must contain a 'ground_truth' column.")

    # coerce ground_truth to 0/1
    gt_df["ground_truth_bin"] = (
        gt_df["ground_truth"].astype(str).str.strip().str.lower()
        .map({"true": 1, "1": 1, "yes": 1, "false": 0, "0": 0, "no": 0})
    )
    if gt_df["ground_truth_bin"].isna().any():
        gt_df["ground_truth_bin"] = pd.to_numeric(gt_df["ground_truth"], errors="coerce").fillna(0).astype(int)

    # drop rows with missing ids before merge
    pred_df = pred_df.dropna(subset=["file_id"])
    gt_df = gt_df.dropna(subset=["file_id"])

    # ensure same dtype for join
    pred_df["file_id"] = pred_df["file_id"].astype("Int64")
    gt_df["file_id"] = gt_df["file_id"].astype("Int64")

    keep_pred = [c for c in ["image","n_detections","confidence","pred_has_object","file_id"] if c in pred_df.columns]
    merged = pd.merge(pred_df[keep_pred],
                      gt_df[["file_id","ground_truth_bin"]],
                      on="file_id", how="inner", validate="m:1").rename(
                          columns={"ground_truth_bin":"ground_truth"})

    merged["pred_has_object"] = pd.to_numeric(merged["pred_has_object"], errors="coerce").fillna(0).astype(int)
    merged["ground_truth"] = pd.to_numeric(merged["ground_truth"], errors="coerce").fillna(0).astype(int)
    merged["match"] = (merged["pred_has_object"] == merged["ground_truth"]).astype(int)

    # metrics
    y_true = merged["ground_truth"].to_numpy()
    y_pred = merged["pred_has_object"].to_numpy()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

    metrics_df = pd.DataFrame([
        ["images", len(merged)],
        ["TP", int(tp)], ["FP", int(fp)], ["FN", int(fn)], ["TN", int(tn)],
        ["accuracy", acc], ["precision", prec], ["recall", rec], ["f1", f1]
    ], columns=["metric","value"])

    merged.to_csv(args.out_csv, index=False)
    metrics_df.to_csv(args.metrics_csv, index=False)

    # helpful counts
    pred_only = pred_df[~pred_df["file_id"].isin(merged["file_id"])]
    gt_only   = gt_df[~gt_df["file_id"].isin(merged["file_id"])]

    print(f"\nSaved merged: {args.out_csv}")
    print(f"Saved metrics: {args.metrics_csv}\n")
    print(metrics_df.to_string(index=False))
    if len(pred_only):
        print(f"\nNote: {len(pred_only)} prediction rows had ids not found in GT (skipped).")
    if len(gt_only):
        print(f"Note: {len(gt_only)} GT rows had ids not found in predictions (skipped).")

if __name__ == "__main__":
    main()
