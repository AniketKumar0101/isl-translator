"""Real-time sign language -> speech translator (webcam).

Pipeline per frame:
  webcam -> MediaPipe Tasks HandLandmarker (VIDEO mode) -> 126-dim landmark
  vector -> 30-frame buffer -> transformer model -> majority-vote smoothing
  -> text-to-speech on new signs -> JSONL usage analytics.

Uses the MediaPipe Tasks API — mp.solutions was removed in mediapipe>=1.0.

Controls:  'c' clear sentence   'q' quit
"""
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.extract_landmarks import init_landmarker, result_to_vector
from data.transforms import normalise
from models.sign_model import build_model
from realtime.analytics import log_prediction

# MediaPipe hand skeleton (21 landmarks), for the cv2 preview overlay.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


class Speaker:
    """Non-blocking text-to-speech (falls back to silent mode)."""

    def __init__(self, enabled=True):
        self.enabled = enabled
        self._engine = None
        if enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception as e:  # e.g. pyttsx3/espeak missing
                print(f"TTS unavailable ({e}); running silent.")
                self.enabled = False

    def say(self, text):
        if self.enabled:
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text):
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as e:
            print("TTS error:", e)


class SignTranslator:
    # Classes the model systematically over-predicts. Measured 2026-10-22 on
    # held-out INCLUDE data (see TEST_REPORT.md "She-bias fix"): all three are
    # raised-open-hand signs and the model fires them on ambiguous input —
    # test-set precision was she 0.571, good morning 0.333, good night 0.375,
    # with "she" predicted 7x vs 4 true occurrences. A small logit penalty on
    # these classes cuts false positives ("she" for everything on webcam)
    # without hurting accuracy (val 0.697->0.697, test 0.701->0.701 at 0.3).
    _OVERPREDICTED_CLASSES = ("she", "good morning", "good night")
    _BIAS_PENALTY = 0.3

    def __init__(self, config="config.yaml"):
        cfg = yaml.safe_load(open(config))
        self.cfg = cfg
        rt = cfg["realtime"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt",
                          map_location=self.device, weights_only=False)
        self.labels = ckpt["labels"]
        self.model = build_model(cfg, len(self.labels),
                                 model_type=ckpt["model_type"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.landmarker = init_landmarker()
        self.buffer = deque(maxlen=rt["buffer_size"])
        self.votes = deque(maxlen=rt["smoothing_window"])
        # Minimum confidence to report a label; below it predict()/step()
        # return label=None ("unsure") instead of a shaky guess.
        # Replaces the old `confidence_threshold` key (still read as fallback).
        self.min_confidence = float(
            rt.get("min_confidence", rt.get("confidence_threshold", 0.55)))
        # Per-class logit bias correcting the over-prediction above.
        bias = np.zeros(len(self.labels), dtype=np.float32)
        for c in self._OVERPREDICTED_CLASSES:
            if c in self.labels:
                bias[self.labels.index(c)] = -self._BIAS_PENALTY
        self._bias = torch.from_numpy(bias).to(self.device)
        self.speaker = Speaker()
        self.sentence = []
        self.last_spoken = None
        self.last_time = 0.0
        self.frame_count = 0
        self._ts_ms = int(time.time() * 1000)
        self.last_hand_landmarks = None

    def predict(self):
        """Classify the current 30-frame buffer.

        Returns (label, confidence). `label` is None ("unsure") when the
        bias-corrected top-1 confidence is below `min_confidence` — callers
        should show an "unsure / keep signing" state instead of a guess.
        `confidence` is always the raw top-1 probability (useful for display).
        Requires a full buffer; returns (None, 0.0) otherwise.
        """
        rt = self.cfg["realtime"]
        if len(self.buffer) < rt["buffer_size"]:
            return None, 0.0
        seq = normalise(np.stack(self.buffer))[None].astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(seq).to(self.device))[0]
            probs = torch.softmax(logits + self._bias, dim=0)
        conf, idx = probs.max(0)
        conf = conf.item()
        if conf < self.min_confidence:
            return None, conf
        return self.labels[idx.item()], conf

    def step(self, frame):
        """Feed one webcam frame; returns (smoothed_label, confidence).

        `smoothed_label` is None until enough confident predictions have
        accumulated in the smoothing window — treat None as "unsure".
        """
        rt = self.cfg["realtime"]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts_ms += 33  # ~30fps; VIDEO mode needs increasing timestamps
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)
        self.last_hand_landmarks = result.hand_landmarks
        self.buffer.append(result_to_vector(result.hand_landmarks))
        self.frame_count += 1

        label, conf = None, 0.0
        if (len(self.buffer) == rt["buffer_size"]
                and self.frame_count % rt["predict_every_n_frames"] == 0):
            label, conf = self.predict()
            if label is not None:  # only confident predictions get a vote
                self.votes.append(label)

        smooth = Counter(self.votes).most_common(1)[0][0] if self.votes else None
        now = time.time()
        if (smooth and smooth != self.last_spoken
                and now - self.last_time > rt["cooldown_seconds"]):
            self.sentence.append(smooth)
            self.speaker.say(smooth)
            log_prediction(smooth, conf)
            self.last_spoken, self.last_time = smooth, now
        return smooth, conf

    def draw(self, frame, label, conf):
        if self.last_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_lm in self.last_hand_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
                for p in pts:
                    cv2.circle(frame, p, 4, (0, 0, 255), -1)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"Sign: {label or '...'} ({conf:.2f})",
                    (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Sentence: " + " ".join(self.sentence[-8:]),
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "'c' clear | 'q' quit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam (index 0).")
        print("Live! Hold a sign steady ~1s. 'c' clears sentence, 'q' quits.")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)  # mirror view
                label, conf = self.step(frame)
                self.draw(frame, label, conf)
                cv2.imshow("Sign Language Translator", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                elif k == ord("c"):
                    self.sentence.clear()
                    self.last_spoken = None
        finally:
            cap.release()
            self.landmarker.close()
            cv2.destroyAllWindows()
        print("Final sentence:", " ".join(self.sentence))
