"""Build train/val/test .npz using INCLUDE's official splits (not a random split).

Reads:
    data/landmarks/<label>/*.npy          (from extract_landmarks.py)
    data/split_map.json                   ("<label>||<stem>" -> "train"|"val"|"test")
Writes:
    data/processed/dataset.npz            (X_train, y_train, X_val, y_val, X_test, y_test)
    data/processed/labels.json

Any landmark file with no entry in the split map is reported and EXCLUDED
(it must not silently land in a random split).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from transforms import normalise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    root = Path(__file__).resolve().parent.parent.parent

    lm_dir = root / cfg["paths"]["landmarks_dir"]
    proc = root / cfg["paths"]["processed_dir"]
    proc.mkdir(parents=True, exist_ok=True)

    labels = cfg["labels"]
    label_to_id = {label: i for i, label in enumerate(labels)}
    split_map = json.loads(open(root / "data" / "split_map.json").read())

    buckets = {"train": [], "val": [], "test": []}
    y_buckets = {"train": [], "val": [], "test": []}
    missing, per_label = [], {}
    for label in labels:
        files = sorted((lm_dir / label).glob("*.npy"))
        per_label[label] = {"train": 0, "val": 0, "test": 0, "missing": 0}
        if not files:
            print(f"WARNING: no landmark files for label '{label}'")
        for f in files:
            key = f"{label}||{f.stem}"
            split = split_map.get(key)
            if split is None:
                missing.append(str(f))
                per_label[label]["missing"] += 1
                continue
            buckets[split].append(normalise(np.load(f)))
            y_buckets[split].append(label_to_id[label])
            per_label[label][split] += 1

    if missing:
        print(f"WARNING: {len(missing)} landmark files had NO official-split mapping "
              f"(excluded from dataset):")
        for m in missing[:20]:
            print("   ", m)
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")

    out = {}
    for split in ("train", "val", "test"):
        X = np.stack(buckets[split]) if buckets[split] else np.zeros((0, 30, 126), np.float32)
        y = np.array(y_buckets[split], dtype=np.int64)
        out[f"X_{split}"] = X
        out[f"y_{split}"] = y
        print(f"{split}: {len(y)} samples, shape {X.shape}")
    np.savez_compressed(proc / "dataset.npz", **out)
    json.dump(labels, open(proc / "labels.json", "w"), indent=2)

    print("\nper-label split counts:")
    for label in labels:
        c = per_label[label]
        print(f"  {label:15s} train={c['train']:3d} val={c['val']:2d} "
              f"test={c['test']:3d} missing={c['missing']}")


if __name__ == "__main__":
    main()
