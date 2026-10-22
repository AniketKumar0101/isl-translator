"""Evaluate the trained model on the held-out test set.

Run from project root:   python src/evaluate.py
Saves: reports/confusion_matrix.png ; prints a full classification report.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.sign_model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = np.load(Path(cfg["paths"]["processed_dir"]) / "dataset.npz")
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt",
                      map_location=device, weights_only=False)
    labels = ckpt["labels"]

    model = build_model(cfg, len(labels), model_type=ckpt["model_type"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X = torch.from_numpy(data["X_test"]).to(device)
    with torch.no_grad():
        pred = model(X).argmax(1).cpu().numpy()
    y = data["y_test"]

    print(classification_report(y, pred, target_names=labels, digits=4))

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (test set)")
    fig.colorbar(im)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    out = Path(cfg["paths"]["reports_dir"]) / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
