# remap_all_to_zero.py
import glob, os

BASE = r"C:/RIT/GA/BIV-Priv-Seg/Yolov8n"
splits = ["train2017","val2017","test2017"]
for s in splits:
    for p in glob.glob(os.path.join(BASE, "labels", s, "*.txt")):
        lines_out = []
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                parts = line.split()
                parts[0] = "0"                    # force class id to 0
                lines_out.append(" ".join(parts))
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(lines_out) + ("\n" if lines_out else ""))
print("Done. All label class ids set to 0.")
