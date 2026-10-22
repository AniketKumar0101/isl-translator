"""Shared transforms for landmark sequences.

A sequence has shape (T, H*L*3) where H = number of hands, L = landmarks
per hand. The key normalisation makes every sample translation- and
scale-invariant *per hand*: subtract the wrist, divide by hand size
(wrist -> middle-finger MCP distance). This is what lets the model focus
on the shape of the gesture instead of where the hand sits in the frame.
"""
import numpy as np


def normalise(seq, num_hands=2, landmarks_per_hand=21):
    seq = np.asarray(seq, dtype=np.float64).reshape(-1, num_hands, landmarks_per_hand, 3)
    out = np.zeros_like(seq)
    for h in range(num_hands):
        hand = seq[:, h]                       # (T, 21, 3)
        wrist = hand[:, :1, :]                 # (T, 1, 3)
        scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1).reshape(-1, 1, 1) + 1e-6
        out[:, h] = (hand - wrist) / scale     # all-zero hand stays all-zero
    return out.reshape(seq.shape[0], -1).astype(np.float32)


def augment_sequence(seq, num_hands=2, landmarks_per_hand=21, rng=None):
    """GENTLE stochastic augmentation for a NORMALISED (T, 126) sequence.

    Mild perturbations only — strong augmentation destroyed the signal on
    this small dataset (val collapsed 33% -> 18%). Zero-hands stay zero.
    """
    if rng is None:
        rng = np.random.default_rng()
    T = seq.shape[0]
    x = np.asarray(seq, dtype=np.float64).reshape(T, num_hands, landmarks_per_hand, 3).copy()
    hand_present = (np.abs(x).sum(axis=(2, 3)) > 1e-9)[..., None, None]  # (T,H,1,1)

    # 1. gentle scale (signing size variation)
    x = x * rng.uniform(0.95, 1.05) * hand_present

    # 2. gentle in-plane rotation (camera tilt)
    ang = np.deg2rad(rng.uniform(-8, 8))
    c, sn = np.cos(ang), np.sin(ang)
    xy = x[..., :2].copy()
    x[..., 0] = xy[..., 0] * c - xy[..., 1] * sn
    x[..., 1] = xy[..., 0] * sn + xy[..., 1] * c

    # 3. tiny joint noise (detection jitter), only on present hands
    x = x + rng.normal(0, 0.005, size=x.shape) * hand_present

    return x.reshape(T, -1).astype(np.float32)
