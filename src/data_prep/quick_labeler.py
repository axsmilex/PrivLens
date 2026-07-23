# quick_labeler.py
import os, csv, sys
from PIL import Image
import matplotlib.pyplot as plt

folder = sys.argv[1]  # path to your BIV-Priv-Seg images folder
outcsv = sys.argv[2]  # e.g., labels.csv

files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
files.sort()

labels = {}
if os.path.exists(outcsv):
    with open(outcsv, 'r') as f:
        for row in csv.reader(f):
            if row and row[0] != 'filename':
                labels[row[0]] = row[1]

plt.ion()
for i,f in enumerate(files):
    if f in labels: 
        continue
    img = Image.open(os.path.join(folder,f)).convert('RGB')
    plt.imshow(img); plt.title(f"[{i+1}/{len(files)}] {f}\nPress '1' for PRIVATE, '0' for NOT PRIVATE, 's' skip")
    plt.axis('off'); plt.show(); plt.pause(0.001)
    key = input("Label: ")
    if key not in ('0','1','s'):
        key = 's'
    if key != 's':
        labels[f] = key
    plt.clf()

with open(outcsv, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['filename','label'])
    for f in files:
        if f in labels:
            w.writerow([f, labels[f]])
