"""Streamlit demo: upload a sign video -> get prediction + speech.

Run from project root:   streamlit run app.py
Requires a trained model: checkpoints/best.pt  (17-class ISL transformer)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from data.extract_landmarks import video_to_sequence
from data.transforms import normalise
from models.sign_model import build_model
from realtime.analytics import log_prediction

st.set_page_config(page_title="Sign Language Translator", page_icon="🤟")
st.title("🤟 Sign Language → Speech Translator")
st.caption("Indian Sign Language · 17 signs · trained on real ISL video")


@st.cache_resource
def load_model():
    cfg = yaml.safe_load(open("config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(Path(cfg["paths"]["checkpoint_dir"]) / "best.pt",
                      map_location=device, weights_only=False)
    model = build_model(cfg, len(ckpt["labels"]),
                        model_type=ckpt["model_type"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return cfg, device, model, ckpt["labels"], ckpt["model_type"]


try:
    cfg, device, model, labels, model_type = load_model()
except FileNotFoundError:
    st.error("No trained model found. Train one first: `python src/train.py`")
    st.stop()

seq_len = cfg["data"]["sequence_length"]

with st.expander("📖 Signs I understand", expanded=False):
    st.write(", ".join(f"`{l}`" for l in labels))

tab1, tab2 = st.tabs(["🎥 Video prediction", "ℹ️ About"])

with tab1:
    f = st.file_uploader("Upload a sign video", type=["mp4", "mov", "avi"])
    if f:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(f.read())
        tmp.close()
        with st.spinner("Extracting hand landmarks and predicting..."):
            seq = normalise(video_to_sequence(tmp.name, seq_len))[None].astype(np.float32)
            with torch.no_grad():
                probs = torch.softmax(
                    model(torch.from_numpy(seq).to(device)), dim=1)[0].cpu().numpy()
        top = np.argsort(probs)[::-1][:3]
        pred, conf = labels[top[0]], float(probs[top[0]])
        log_prediction(pred, conf)  # anonymous usage analytics (JSONL)

        st.markdown(f"# 🗣️ {pred}")
        st.metric("Confidence", f"{conf:.1%}")
        if st.button("🔊 Speak it"):
            safe = pred.replace("\\", "\\\\").replace("'", "\\'")
            components.html(
                f"<script>speechSynthesis.speak(new SpeechSynthesisUtterance('{safe}'));</script>",
                height=0,
            )
        st.subheader("Top 3 guesses")
        st.bar_chart({labels[i]: float(probs[i]) for i in top})
        st.video(f)
        st.caption("Tip: the live webcam demo with voice output runs via `python run_realtime.py`")

with tab2:
    st.markdown(f"""
**How it works**

1. **Capture** — video frames from an uploaded clip (or live webcam via `run_realtime.py`).
2. **Pose extraction** — MediaPipe HandLandmarker finds 21 3D landmarks per hand
   (2 hands × 21 × 3 = 126 features/frame). Classifying landmarks instead of
   raw pixels makes the model fast and robust to lighting/backgrounds.
3. **Normalisation** — each hand is centred on its wrist and scaled by hand
   size, so the model learns *gesture shape*, not hand position.
4. **Sequence model** — a {model_type} reads 30-frame windows and predicts one
   of {len(labels)} ISL signs.
5. **Speech** — the predicted sign is spoken aloud in your browser.

**Model card**

- Architecture: {model_type} encoder, trained on the INCLUDE Indian Sign
  Language dataset (greetings + pronouns, CC-BY-4.0, see DATASET_ATTRIBUTION.md)
- Test accuracy: **70.1%** on INCLUDE's official signer-independent test split
  (47/67; chance = 5.9%)
- Vocabulary: {len(labels)} signs — greetings, courtesy phrases and pronouns

**Project structure**

```
config.yaml                    # all hyper-parameters & paths
src/data/extract_landmarks.py  # video -> landmark .npy (MediaPipe Tasks API)
src/data/build_official_splits.py  # INCLUDE official train/val/test splits
src/models/sign_model.py       # LSTM / Transformer classifiers
src/train.py / src/evaluate.py # training loop, metrics, confusion matrix
src/realtime/inference.py      # webcam app with smoothing + TTS + analytics
src/realtime/analytics.py      # anonymous usage logging (JSONL)
run_realtime.py / app.py       # entry points
```
""")
