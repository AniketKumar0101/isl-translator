"""Normalise landmark sequences and build stratified train/val/test splits.

Reads:  data/landmarks/<label>/*.npy
Writes: data/processed/dataset.npz   (X_train, y_train, X_val, y_val, X_test, y_test)
        data/processed/labels.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transforms import normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    rng = np.random.RandomState(cfg["data"]["random_seed"])

    lm_dir = Path(cfg["paths"]["landmarks_dir"])
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)

    labels = cfg["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}

    X, y = [], []
    for label in labels:
        files = sorted((lm_dir / label).glob("*.npy"))
        if not files:
            print(f"WARNING: no samples for label '{label}' in {lm_dir/label}")
        for f in files:
            X.append(normalise(np.load(f)))
            y.append(label_to_id[label])
    if not X:
        raise SystemExit(f"No landmark files found under {lm_dir}/ — run generate_synthetic.py or extract_landmarks.py first.")
    X = np.stack(X)
    y = np.array(y)
    print(f"loaded {len(X)} samples of shape {X.shape[1:]}, {len(labels)} classes")

    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    vs, ts = cfg["data"]["val_split"], cfg["data"]["test_split"]
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=vs / (1 - ts), random_state=42, stratify=y_tv)

    np.savez_compressed(proc / "dataset.npz",
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        X_test=X_test, y_test=y_test)
    json.dump(labels, open(proc / "labels.json", "w"), indent=2)
    for name, split in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name}: {len(split)} samples")


if __name__ == "__main__":
    main()
