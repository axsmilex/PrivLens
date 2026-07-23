import glob, os
base=r"C:/RIT/GA/BIV-Priv-Seg/Yolov8n"
for split in ["train2017","val2017","test2017"]:
    n_imgs=len(glob.glob(os.path.join(base,"images",split,"*.jpg")))
    n_lbls=len(glob.glob(os.path.join(base,"labels",split,"*.txt")))
    print(split, n_imgs, "images,", n_lbls, "labels")
