"""Train the sign classifier.

Run from project root:   python src/train.py
Saves: checkpoints/best.pt, reports/training_curves.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.sign_model import build_model
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from transforms import augment_sequence


class AugmentedDataset(torch.utils.data.Dataset):
    """Training dataset with on-the-fly stochastic augmentation.

    Each epoch the model sees a different perturbed variant of every sample,
    which massively expands the effective training data for small datasets.
    """
    def __init__(self, X, y, augment_prob=0.9, seed=0):
        self.X = X
        self.y = y
        self.augment_prob = augment_prob
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = self.X[i]
        if self.rng.random() < self.augment_prob:
            x = augment_sequence(x, rng=self.rng)
        return torch.from_numpy(x), torch.tensor(self.y[i]).long()


def make_loader(X, y, batch_size, shuffle):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long()),
        batch_size=batch_size, shuffle=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    proc = Path(cfg["paths"]["processed_dir"])
    data = np.load(proc / "dataset.npz")
    import json
    labels = json.load(open(proc / "labels.json"))

    train_dl = DataLoader(
        AugmentedDataset(data["X_train"], data["y_train"], augment_prob=0.5),
        batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_dl = make_loader(data["X_val"], data["y_val"], cfg["training"]["batch_size"], False)

    model = build_model(cfg, len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(),
                           lr=cfg["training"]["learning_rate"],
                           weight_decay=cfg["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    ckpt_dir = Path(cfg["paths"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(cfg["paths"]["reports_dir"])
    rep_dir.mkdir(parents=True, exist_ok=True)

    best, patience = 0.0, 0
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        tl = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(xb)

        model.eval()
        vl, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                vl += loss_fn(out, yb).item() * len(xb)
                correct += (out.argmax(1) == yb).sum().item()
                n += len(xb)
        tl /= len(train_dl.dataset)
        vl /= len(val_dl.dataset)
        acc = correct / n
        sched.step(vl)
        hist["train_loss"].append(tl)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(acc)
        print(f"epoch {epoch:3d}  train_loss={tl:.4f}  val_loss={vl:.4f}  val_acc={acc:.4f}", flush=True)

        if acc > best:
            best, patience = acc, 0
            torch.save({"model_state": model.state_dict(),
                        "model_type": cfg["model"]["type"],
                        "labels": labels}, ckpt_dir / "best.pt")
            print(f"  -> new best val_acc={best:.4f}, checkpoint saved")
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("early stopping triggered")
                break

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["train_loss"], label="train")
    ax[0].plot(hist["val_loss"], label="val")
    ax[0].set_title("loss"); ax[0].legend(); ax[0].set_xlabel("epoch")
    ax[1].plot(hist["val_acc"], color="green")
    ax[1].set_title("validation accuracy"); ax[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(rep_dir / "training_curves.png", dpi=120)
    print(f"done — best val_acc={best:.4f} (curves: reports/training_curves.png)")


if __name__ == "__main__":
    main()
