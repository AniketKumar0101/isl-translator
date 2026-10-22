# 🤟 ISL Translator — Indian Sign Language to Speech

Real-time Indian Sign Language recognition that turns signs into text and speech. Built for the **Code for a Billion** hackathon (Health track).

**Try it:** upload a short video of someone signing → get the predicted sign, confidence, and one-tap speech output.

## The problem

Millions of deaf and hard-of-hearing Indians are locked out of everyday spoken communication — at clinics, pharmacies, government offices. Professional ISL interpreters are scarce and expensive. A free, open-source translator that runs in a browser meaningfully expands who can be understood.

## What it does

- 🎥 **Video upload demo** (`streamlit run app.py`) — upload a clip, see the top-3 predictions with confidence bars, hear it spoken aloud
- 📹 **Live webcam mode** (`python run_realtime.py`) — continuous recognition with smoothing and voice output
- 📊 **Anonymous usage analytics** — prediction counts only, no images or personal data, to measure real-world impact

## Model card

| | |
|---|---|
| Architecture | Transformer encoder (2 layers, 128-dim) |
| Input | 30-frame sequences of MediaPipe hand landmarks (2 hands × 21 points × 3D = 126 features/frame) |
| Vocabulary | **17 ISL signs** — greetings, courtesy phrases, pronouns |
| Test accuracy | **70.1%** on a held-out test set (47/67; chance = 5.9%) |
| Training data | 358 real ISL videos from the INCLUDE dataset (CC-BY-4.0) |

**Honest numbers:** the test set has zero video overlap with training data. Per-sign recall is 100% for *hello, thank you, how are you, alright, pleased, she, it* — and 0% for *good evening*, which the model confuses with the other *good-X* greetings. Confidence is always shown so users see uncertainty.

**Vocabulary:** hello · thank you · good morning · good afternoon · good evening · good night · how are you · alright · pleased · i · you · you all · he · she · we · they · it

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py        # video-upload demo (opens in browser)
python run_realtime.py      # live webcam mode
```

To retrain on your own data: put videos in `data/raw/<sign>/`, then

```bash
python src/data/extract_landmarks.py
python src/data/build_official_splits.py
python src/train.py
python src/evaluate.py
```

## How it works

1. **Pose extraction** — MediaPipe HandLandmarker finds 21 3D landmarks per hand. Classifying landmarks instead of raw pixels makes the model fast and robust to lighting and backgrounds.
2. **Normalisation** — each hand is centred on its wrist and scaled by hand size, so the model learns *gesture shape*, not hand position.
3. **Sequence model** — a transformer reads 30-frame windows and predicts one of 17 signs.
4. **Speech** — the prediction is spoken via the browser's speech synthesis.

## Project structure

```
app.py / run_realtime.py       # demo entry points
config.yaml                    # hyper-parameters, paths, vocabulary
src/data/extract_landmarks.py  # video → landmark .npy (MediaPipe Tasks API)
src/data/build_official_splits.py  # train/val/test splits
src/models/sign_model.py       # LSTM / Transformer classifiers
src/train.py / src/evaluate.py # training loop, metrics
src/realtime/                  # webcam inference, smoothing, TTS, analytics
checkpoints/best.pt            # trained 17-class transformer (included)
models/hand_landmarker.task    # MediaPipe hand model (included, 7.8 MB)
```

## Data

Trained on the [INCLUDE dataset](https://zenodo.org/records/4010759) (AI4Bharat, CC-BY-4.0). Raw videos are **not** included in this repo — download them from Zenodo; see `DATASET_ATTRIBUTION.md`. The trained checkpoint and extracted-landmark pipeline are included so the demo works out of the box.

## Built with

Developed in **AgentFoundry**, the official IDE of Code for a Billion. Open-sourced under Apache-2.0 for the hackathon — free for any NGO, school, or government body to adopt.
