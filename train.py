"""Train a chest X-ray classifier and save models/model.pt.

Expected folder layout (class names = folder names):
    DATA/train/NORMAL, DATA/train/PNEUMONIA     (or BACTERIA / VIRUS / NORMAL if labels are verified)
    DATA/test/...                               (held out, never used for tuning)
    DATA/val/...                                (optional; otherwise 15% of train is used)

Usage:  python train.py --data path/to/chest_xray --arch densenet121 --epochs 8
NOTE: Without patient IDs, a random val split from train may leak patients across splits.
"""
import argparse, json, random
from datetime import date
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from backend.inference import build_model, MEAN, STD, SIZE

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--arch", default="densenet121", choices=["densenet121", "resnet18", "efficientnet_b0"])
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
root = Path(a.data)

norm = transforms.Normalize(MEAN, STD)
ev_tf = transforms.Compose([transforms.Resize((SIZE, SIZE)), transforms.ToTensor(), norm])
# No horizontal flips: anatomy is not left/right symmetric (heart position).
tr_tf = transforms.Compose([transforms.Resize((SIZE, SIZE)),
                            transforms.RandomAffine(7, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                            transforms.ColorJitter(0.1, 0.1), transforms.ToTensor(), norm])
rgb = lambda path: datasets.folder.default_loader(path)

full_tr = datasets.ImageFolder(root / "train", tr_tf, loader=rgb)
classes = full_tr.classes
test_ds = datasets.ImageFolder(root / "test", ev_tf, loader=rgb)
assert test_ds.classes == classes, "train/test class folders differ"

if (root / "val").exists() and len(datasets.ImageFolder(root / "val")) >= 100:
    train_ds, val_ds = full_tr, datasets.ImageFolder(root / "val", ev_tf, loader=rgb)
else:
    idx = np.random.permutation(len(full_tr)); k = int(0.15 * len(idx))
    full_ev = datasets.ImageFolder(root / "train", ev_tf, loader=rgb)
    train_ds, val_ds = Subset(full_tr, idx[k:]), Subset(full_ev, idx[:k])
    print("[WARN] Using random 15% of train as validation (possible patient leakage).")

targets = np.array(full_tr.targets)
tr_idx = train_ds.indices if isinstance(train_ds, Subset) else np.arange(len(train_ds))
counts = np.bincount(targets[tr_idx], minlength=len(classes))
weights = torch.tensor(counts.sum() / (len(classes) * counts), dtype=torch.float32).to(dev)
print("classes:", classes, "train counts:", counts.tolist())

mk = lambda ds, sh: DataLoader(ds, batch_size=a.batch, shuffle=sh, num_workers=0)
tr_dl, va_dl, te_dl = mk(train_ds, True), mk(val_ds, False), mk(test_ds, False)

net = build_model(a.arch, len(classes), pretrained=True).to(dev)
loss_fn = nn.CrossEntropyLoss(weight=weights)
opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)


def logits_of(dl):
    net.eval(); L, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            L.append(net(x.to(dev)).cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y)


def report(L, Y, T=1.0):
    pred = (L / T).argmax(1).numpy(); y = Y.numpy(); n = len(classes)
    cm = np.zeros((n, n), int)
    for t, q_ in zip(y, pred): cm[t, q_] += 1
    per = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp; tn = cm.sum() - tp - fn - fp
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        per[c] = {"precision": round(prec, 4), "sensitivity": round(rec, 4),
                  "specificity": round(tn / max(tn + fp, 1), 4),
                  "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "support": int(cm[i].sum())}
    return {"accuracy": round(float((pred == y).mean()), 4), "per_class": per,
            "confusion_matrix": cm.tolist(), "n": int(len(y))}


best, best_state = -1, None
for ep in range(a.epochs):
    net.train(); tot = 0
    for x, y in tr_dl:
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad(); loss = loss_fn(net(x), y); loss.backward(); opt.step(); tot += loss.item()
    L, Y = logits_of(va_dl); r = report(L, Y)
    bal = np.mean([v["sensitivity"] for v in r["per_class"].values()])  # balanced accuracy
    print(f"epoch {ep+1}/{a.epochs} loss {tot/len(tr_dl):.4f} val_acc {r['accuracy']:.4f} val_bal_acc {bal:.4f}")
    if bal > best:
        best = bal; best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}

net.load_state_dict(best_state)
# Temperature scaling on validation logits (calibration)
L, Y = logits_of(va_dl)
logT = torch.zeros(1, requires_grad=True)
o = torch.optim.LBFGS([logT], lr=0.1, max_iter=50)
def closure():
    o.zero_grad(); l = nn.functional.cross_entropy(L / logT.exp(), Y); l.backward(); return l
o.step(closure)
T = float(logT.exp().item()); print(f"temperature: {T:.3f}")

Lt, Yt = logits_of(te_dl)
metrics = {"test": report(Lt, Yt, T), "val_balanced_accuracy": round(float(best), 4),
           "train_counts": dict(zip(classes, counts.tolist()))}
print(json.dumps(metrics["test"], indent=2))

out = Path("models/model.pt"); out.parent.mkdir(exist_ok=True)
torch.save({"arch": a.arch, "classes": classes, "state_dict": best_state, "temperature": T,
            "metrics": metrics, "version": f"{a.arch}-{date.today().isoformat()}"}, out)
print("saved", out)
