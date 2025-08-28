# train_eval.py
import os, json, random, time
from pathlib import Path
from collections import defaultdict

import torch, torchvision
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# -----------------------------------------------------------------
# 1. CONFIG
# -----------------------------------------------------------------
DATA_DIR   = Path("dataset")
BATCH_SIZE = 32
N_EPOCHS   = 20
LR         = 3e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_P  = Path("weights"); WEIGHTS_P.mkdir(exist_ok=True)

random.seed(42); torch.manual_seed(42)

# -----------------------------------------------------------------
# 2. HELPERS TO SCAN THE HIERARCHICAL DATASET
# -----------------------------------------------------------------
def scan_dataset(split):
    """
    Walk through dataset/<split> and build:
      • samples: list[(image_path, bin_lbl, roast_lbl, defect_lbl)]
      • mappings for roast and defect names → ids
    """
    base = DATA_DIR / split
    roast_map, defect_map = {}, {}
    roast_cnt, defect_cnt = 0, 0
    samples = []

    for cls in ("normal", "defect"):
        root = base / cls
        for sub in sorted(os.listdir(root)):
            subdir = root / sub
            if not subdir.is_dir(): continue
            for img_f in subdir.glob("*.*"):
                bin_lbl   = 0 if cls == "normal" else 1
                roast_lbl = defect_lbl = -1
                if cls == "normal":
                    if sub not in roast_map:
                        roast_map[sub] = roast_cnt; roast_cnt += 1
                    roast_lbl = roast_map[sub]
                else:
                    if sub not in defect_map:
                        defect_map[sub] = defect_cnt; defect_cnt += 1
                    defect_lbl = defect_map[sub]
                samples.append((img_f, bin_lbl, roast_lbl, defect_lbl))
    return samples, roast_map, defect_map

# -----------------------------------------------------------------
# 3. DATASET/LOADER
# -----------------------------------------------------------------
train_tf = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.ColorJitter(.2,.2,.2,.1),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

val_tf = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([.485,.456,.406],[.229,.224,.225])
])

class CoffeeBeanDS(Dataset):
    def __init__(self, samples, tfm):
        self.samples = samples; self.tfm = tfm
    def __len__(self):  return len(self.samples)
    def __getitem__(self, i):
        p, bin_lbl, roast_lbl, defect_lbl = self.samples[i]
        img = self.tfm(Image.open(p).convert("RGB"))
        return img, torch.tensor(bin_lbl), torch.tensor(roast_lbl), torch.tensor(defect_lbl)

def make_loaders():
    train_s, roast_map, defect_map = scan_dataset("train")
    test_s , *_                   = scan_dataset("test")

    train_ds = CoffeeBeanDS(train_s, train_tf)
    test_ds  = CoffeeBeanDS(test_s , val_tf)

    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  num_workers=4, pin_memory=True)
    test_dl  = DataLoader(test_ds , BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    return train_dl, test_dl, roast_map, defect_map

# -----------------------------------------------------------------
# 4. MODEL with three heads
# -----------------------------------------------------------------
class MultiHeadNet(nn.Module):
    def __init__(self, n_roast, n_defect):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT").features
        self.pool     = nn.AdaptiveAvgPool2d(1)
        feat_dim      = 1280   # MobileNetV2 last chan.
        self.head_bin = nn.Linear(feat_dim, 2)          # Normal / Defect
        self.head_roa = nn.Linear(feat_dim, n_roast)    # roast levels
        self.head_def = nn.Linear(feat_dim, n_defect)   # defect types
    def forward(self, x):
        f = self.pool(self.backbone(x)).flatten(1)
        return self.head_bin(f), self.head_roa(f), self.head_def(f)

# -----------------------------------------------------------------
# 5. LOSS that ignores irrelevant heads
# -----------------------------------------------------------------
def loss_fn(outputs, targets):
    out_bin, out_roa, out_def = outputs
    y_bin , y_roa , y_def     = targets
    loss_b = nn.CrossEntropyLoss()(out_bin, y_bin)

    # mask: only compute roast loss when y_bin==0; defect when y_bin==1
    mask_roa = (y_roa >= 0)
    mask_def = (y_def >= 0)

    loss_r = (nn.CrossEntropyLoss()(out_roa[mask_roa], y_roa[mask_roa])
              if mask_roa.any() else torch.tensor(0., device=out_bin.device))
    loss_d = (nn.CrossEntropyLoss()(out_def[mask_def], y_def[mask_def])
              if mask_def.any() else torch.tensor(0., device=out_bin.device))
    return loss_b + loss_r + loss_d

# -----------------------------------------------------------------
# 6. TRAIN / EVAL
# -----------------------------------------------------------------
def train_one_epoch(model, dl, optimiser):
    model.train(); tot=0
    for x, yb, yr, yd in dl:
        x, yb, yr, yd = x.to(DEVICE), yb.to(DEVICE), yr.to(DEVICE), yd.to(DEVICE)
        optimiser.zero_grad()
        loss = loss_fn(model(x), (yb, yr, yd))
        loss.backward(); optimiser.step()
        tot += loss.item()*x.size(0)
    return tot/len(dl.dataset)

@torch.no_grad()
def evaluate(model, dl):
    model.eval()
    y_true_bin, y_pred_bin, y_proba_bin = [], [], []
    for x, yb, _, _ in dl:
        x = x.to(DEVICE)
        logits_bin = model(x)[0]
        probs      = torch.softmax(logits_bin,1)
        preds      = probs.argmax(1).cpu()
        y_true_bin.extend(yb.numpy()); y_pred_bin.extend(preds.numpy()); y_proba_bin.extend(probs[:,1].cpu().numpy())

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true_bin, y_pred_bin, average="binary")
    try:
        auc = roc_auc_score(y_true_bin, y_proba_bin)
    except ValueError:
        auc = float('nan')
    return acc, prec, rec, f1, auc

def main():
    train_dl, test_dl, roast_map, defect_map = make_loaders()
    model = MultiHeadNet(len(roast_map), len(defect_map)).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

    best_acc = 0.0
    for epoch in range(1, N_EPOCHS+1):
        t0=time.time()
        tr_loss = train_one_epoch(model, train_dl, optimizer)
        scheduler.step()

        acc, prec, rec, f1, auc = evaluate(model, test_dl)
        print(f"Ep{epoch:02d}  loss {tr_loss:.3f}  "
              f"Acc {acc:.3f} Prec {prec:.3f} Rec {rec:.3f} F1 {f1:.3f} AUC {auc:.3f}  "
              f"{time.time()-t0:.1f}s")

        if acc > best_acc:
            best_acc = acc
            torch.save({
                "state_dict": model.state_dict(),
                "roast_map": roast_map,
                "defect_map": defect_map
            }, WEIGHTS_P / "best_model.pt")
    print(f"\nBest binary accuracy: {best_acc:.3%}")

if __name__ == "__main__":
    main()