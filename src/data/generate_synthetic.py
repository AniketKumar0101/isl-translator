"""Generate synthetic landmark data so the FULL pipeline runs with zero real videos.

Each class gets a distinctive motion signature (class-specific frequencies and
phases), so a model can genuinely learn to separate the classes. Output mimics
extract_landmarks.py exactly:

    data/landmarks/<label>/synth_000.npy ...

Run as:  cd src/data && python generate_synthetic.py --samples-per-class 60
"""
import argparse
from pathlib import Path

import numpy as np
import yaml


def make_sequence(class_id, seq_len, seed):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 1, seq_len)
    seq = np.zeros((seq_len, 2, 21, 3), dtype=np.float64)
    for j in range(21):  # per landmark
        f1, f2 = 1.0 + class_id * 0.35, 0.7 + class_id * 0.22
        p1, p2 = class_id * 0.9 + j * 0.12, class_id * 0.5 + j * 0.07
        x = 0.5 + 0.18 * np.sin(2 * np.pi * f1 * t + p1)
        y = 0.5 + 0.14 * np.cos(2 * np.pi * f2 * t + p2)
        z = 0.04 * np.sin(2 * np.pi * (f1 + f2) / 2 * t + p1)
        seq[:, 0, j, 0] = x + rng.normal(0, 0.015, seq_len)
        seq[:, 0, j, 1] = y + rng.normal(0, 0.015, seq_len)
        seq[:, 0, j, 2] = z + rng.normal(0, 0.008, seq_len)
        if class_id % 2 == 0:  # even classes use both hands (mirrored)
            seq[:, 1, j, 0] = 1 - x + rng.normal(0, 0.015, seq_len)
            seq[:, 1, j, 1] = y + rng.normal(0, 0.015, seq_len)
            seq[:, 1, j, 2] = z + rng.normal(0, 0.008, seq_len)
    return seq.reshape(seq_len, -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--samples-per-class", type=int, default=60)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["landmarks_dir"])
    seq_len = cfg["data"]["sequence_length"]
    for c, label in enumerate(cfg["labels"]):
        d = out / label
        d.mkdir(parents=True, exist_ok=True)
        for s in range(args.samples_per_class):
            np.save(d / f"synth_{s:03d}.npy", make_sequence(c, seq_len, seed=1000 * c + s))
        print(f"{label}: {args.samples_per_class} synthetic samples -> {d}")


if __name__ == "__main__":
    main()
