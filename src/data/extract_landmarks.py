"""Extract MediaPipe hand-landmark sequences from sign-language videos.

Expected layout:
    data/raw/<label>/*.mp4

Output (one fixed-length sequence per video):
    data/landmarks/<label>/<video_stem>.npy      # shape (sequence_length, 126)

126 = 2 hands x 21 landmarks x 3 coords. Missing hands are zero-filled.

Uses the MediaPipe Tasks API (HandLandmarker, VIDEO running mode) —
mp.solutions was removed in mediapipe>=1.0.
"""
import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yaml

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS = 3
FEATURES_PER_FRAME = NUM_HANDS * LANDMARKS_PER_HAND * COORDS  # 126
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"


def init_landmarker():
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(opts)


def result_to_vector(hand_landmarks):
    """Convert HandLandmarker result to a 126-dim vector (zeros if no hands)."""
    vec = np.zeros(FEATURES_PER_FRAME, dtype=np.float32)
    if hand_landmarks:
        # sort left-to-right by wrist x so hand order is stable across frames
        ordered = sorted(hand_landmarks, key=lambda h: h[0].x)
        for i, hand_lm in enumerate(ordered[:NUM_HANDS]):
            base = i * LANDMARKS_PER_HAND * COORDS
            for j, lm in enumerate(hand_lm):
                vec[base + j * 3] = lm.x
                vec[base + j * 3 + 1] = lm.y
                vec[base + j * 3 + 2] = lm.z
    return vec


def resample(frames, n):
    """Resample a (T, 126) array to exactly n frames."""
    if len(frames) == n:
        return frames
    if len(frames) == 0:
        return np.zeros((n, FEATURES_PER_FRAME), dtype=np.float32)
    if len(frames) > n:  # uniform subsample
        idx = np.linspace(0, len(frames) - 1, n).astype(int)
        return frames[idx]
    # pad short clips by repeating the last frame
    pad = np.repeat(frames[-1:], n - len(frames), axis=0)
    return np.concatenate([frames, pad], axis=0)


def video_to_sequence(video_path, sequence_length):
    landmarker = init_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ms_per_frame = 1000.0 / fps
    frames = []
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(idx * ms_per_frame)
            result = landmarker.detect_for_video(mp_image, ts_ms)
            frames.append(result_to_vector(result.hand_landmarks))
            idx += 1
    finally:
        cap.release()
        landmarker.close()
    arr = np.stack(frames) if frames else np.zeros((0, FEATURES_PER_FRAME), np.float32)
    return resample(arr, sequence_length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    raw = Path(cfg["paths"]["raw_dir"])
    out = Path(cfg["paths"]["landmarks_dir"])
    seq_len = cfg["data"]["sequence_length"]

    videos = []
    for ext in ("*.mp4", "*.mov", "*.avi", "*.MP4", "*.MOV", "*.AVI"):
        videos.extend(sorted(raw.rglob(ext)))
    # de-dup in case of case-insensitive filesystems
    videos = sorted(set(videos))
    if not videos:
        print(f"No videos found under {raw}/ — expected data/raw/<label>/*.mp4")
        return
    for v in videos:
        label = v.parent.name
        dest = out / label / (v.stem + ".npy")
        if dest.exists():
            print(f"skip (exists): {dest}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.save(dest, video_to_sequence(v, seq_len))
        print(f"saved: {dest}")


if __name__ == "__main__":
    main()
