#!/usr/bin/env python3
import os, csv, math, random, argparse
from typing import Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

# ---- tiny helpers ----
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(2,3)) + self.eps
        den = probs.sum(dim=(2,3)) + targets.sum(dim=(2,3)) + self.eps
        dice = 1.0 - (num / den)
        return dice.mean()

def dice_score(logits, targets, thr=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    inter = (preds * targets).sum().item()
    union = preds.sum().item() + targets.sum().item()
    if union == 0:
        return 1.0
    return (2.0 * inter) / union

# ---- dataset ----
class PrivSegDataset(Dataset):
    def __init__(self, root, split_csv, img_size=320, aug=True):
        self.root = root
        self.records = []
        with open(split_csv, "r") as f:
            rdr = csv.reader(f)
            header = next(rdr)
            for row in rdr:
                self.records.append((row[0], int(row[1])))
        self.img_size = img_size
        self.aug = aug
        # torchvision only (no extra deps)
        self.t_train = T.Compose([
            T.Resize((img_size, img_size)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05) if aug else T.Lambda(lambda x:x),
            T.ToTensor()
        ])
        self.t_eval = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self): return len(self.records)

    def _read_mask(self, p):
        if not os.path.exists(p):  # allow missing masks (label=0)
            return Image.fromarray(np.zeros((self.img_size, self.img_size), dtype=np.uint8))
        m = Image.open(p).convert("L")
        m = m.resize((self.img_size, self.img_size), Image.NEAREST)
        return m

    def __getitem__(self, idx):
        fn, label = self.records[idx]
        ip = os.path.join(self.root, "images", fn)
        mp = os.path.join(self.root, "masks", os.path.splitext(fn)[0] + ".png")

        im = Image.open(ip).convert("RGB")
        mask = self._read_mask(mp)

        if self.aug:
            x = self.t_train(im)
        else:
            x = self.t_eval(im)

        m = torch.from_numpy(np.array(mask, dtype=np.float32) / 255.0)[None, ...]  # [1,H,W]
        y = torch.tensor([float(label)], dtype=torch.float32)  # binary

        return x, m, y

# ---- models ----
class MobileUNet(nn.Module):
    # Minimal UNet-ish head on top of MobileNetV3-Small features
    def __init__(self):
        super().__init__()
        backbone = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.stem = nn.Sequential(backbone.features[:3])   # 24c @ /2
        self.mid  = nn.Sequential(backbone.features[3:8])  # 48c @ /4
        self.deep = nn.Sequential(backbone.features[8:])   # 576c @ /8
        self.up1 = nn.ConvTranspose2d(576, 128, 2, 2)
        self.up2 = nn.ConvTranspose2d(128+48, 64, 2, 2)
        self.up3 = nn.ConvTranspose2d(64+24, 32, 2, 2)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s = self.stem(x)
        m = self.mid(s)
        d = self.deep(m)
        u1 = torch.relu(self.up1(d))              # /4
        u2 = torch.relu(self.up2(torch.cat([u1, m], 1))) # /2
        u3 = torch.relu(self.up3(torch.cat([u2, s], 1))) # /1
        return self.out(u3)                       # [B,1,H,W]

class MobileClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        base = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1")
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(576, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        f = self.backbone(x)
        f = self.pool(f).flatten(1)
        return self.fc(f)  # logits

# ---- train step ----
def train_one_epoch(cls, seg, opt, dl, dev, ce, dice):
    cls.train(); seg.train()
    tot, n = 0.0, 0
    for x, m, y in dl:
        x, m, y = x.to(dev), m.to(dev), y.to(dev)
        opt.zero_grad()
        logit_c = cls(x)
        logit_s = seg(x)
        loss_c = ce(logit_c.view(-1), y.view(-1))
        loss_s = dice(logit_s, m) + nn.functional.binary_cross_entropy_with_logits(logit_s, m)
        loss = loss_c + loss_s
        loss.backward()
        opt.step()
        tot += loss.item()
        n += 1
    return tot / max(1,n)

@torch.no_grad()
def evaluate(cls, seg, dl, dev):
    cls.eval(); seg.eval()
    from sklearn.metrics import roc_auc_score
    ys, ps = [], []
    dices = []
    for x,m,y in dl:
        x,m,y = x.to(dev), m.to(dev), y.to(dev)
        lc = cls(x).view(-1)
        ls = seg(x)
        ys.extend(y.detach().cpu().numpy().tolist())
        ps.extend(torch.sigmoid(lc).detach().cpu().numpy().tolist())
        dices.append(dice_score(ls, m))
    try:
        auc = roc_auc_score(np.array(ys), np.array(ps))
    except Exception:
        auc = float("nan")
    return float(np.mean(dices)), float(auc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)       # data/
    ap.add_argument("--train_csv", default="data/split_train.csv")
    ap.add_argument("--val_csv",   default="data/split_val.csv")
    ap.add_argument("--img_size", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--out", default="artifacts")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tr = PrivSegDataset(args.root, args.train_csv, args.img_size, aug=True)
    va = PrivSegDataset(args.root, args.val_csv,   args.img_size, aug=False)
    dl_tr = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=4, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cls = MobileClassifier().to(dev)
    seg = MobileUNet().to(dev)

    opt = torch.optim.AdamW(list(cls.parameters())+list(seg.parameters()), lr=args.lr, weight_decay=1e-4)
    ce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    best = 0.0
    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(cls, seg, opt, dl_tr, dev, ce, dice)
        d, auc = evaluate(cls, seg, dl_va, dev)
        print("epoch:", ep, "train_loss:", round(tr_loss,4), "val_dice:", round(d,4), "val_auc:", round(auc,4))
        score = d + auc
        if score > best:
            best = score
            torch.save(cls.state_dict(), os.path.join(args.out, "cls.pt"))
            torch.save(seg.state_dict(), os.path.join(args.out, "seg.pt"))

    # export ONNX
    cls.eval(); seg.eval()
    dummy = torch.randn(1,3,args.img_size,args.img_size, device=dev)
    torch.onnx.export(cls, dummy, os.path.join(args.out,"privlens_cls.onnx"),
        input_names=["input"], output_names=["logits"], opset_version=17)
    torch.onnx.export(seg, dummy, os.path.join(args.out,"privlens_seg.onnx"),
        input_names=["input"], output_names=["mask_logits"], opset_version=17)

if __name__ == "__main__":
    main()
