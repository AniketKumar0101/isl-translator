"""Streamlit demo: live webcam sign translation + upload-a-video prediction.

Run from project root:   streamlit run app.py
Requires a trained model: checkpoints/best.pt  (17-class ISL transformer)
"""
import sys
import tempfile
from pathlib import Path

import cv2
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
from realtime.inference import SignTranslator

try:
    import av
    from streamlit_webrtc import (
        RTCConfiguration,
        VideoProcessorBase,
        WebRtcMode,
        webrtc_streamer,
    )
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False

# Display threshold for the UI: below this we show "unsure" instead of a guess.
# If SignTranslator exposes min_confidence (bias-correction work), prefer it —
# read fresh at processor init so either version of inference.py works.
DISPLAY_CONF_THRESHOLD = 0.55
# Anti-latch: a persistent-but-shaky label stays "unsure" below this confidence.
STEADY_CONF_FLOOR = 0.75
LATCH_FRAMES = 4

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


def speak_js(text):
    safe = text.replace("\\", "\\\\").replace("'", "\\'")
    components.html(
        f"<script>speechSynthesis.speak(new SpeechSynthesisUtterance('{safe}'));</script>",
        height=0,
    )


class SignProcessor(VideoProcessorBase):
    """Per-session live processor: webcam frame -> step() -> gated overlay.

    Display gating (the "she"-at-42% fix): a sign is only shown/announced when
    its confidence clears the threshold; otherwise the frame shows "unsure".
    A shaky label that persists without ever getting confident stays unsure
    instead of latching onto the display.
    """

    def __init__(self) -> None:
        self.translator = SignTranslator()
        self.threshold = (
            getattr(self.translator, "min_confidence", None)
            or DISPLAY_CONF_THRESHOLD
        )
        self.display_label = None
        self.display_conf = 0.0
        self._last_label = None
        self._same_count = 0

    def _gate(self, label, conf):
        if label is None or conf < self.threshold:
            self._last_label, self._same_count = None, 0
            return None, conf
        if label == self._last_label:
            self._same_count += 1
        else:
            self._last_label, self._same_count = label, 0
        if self._same_count >= LATCH_FRAMES and conf < STEADY_CONF_FLOOR:
            return None, conf  # persistent shaky guess -> stay unsure
        return label, conf

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        img = frame.to_ndarray(format="bgr24")
        label, conf = self.translator.step(img)  # step() owns the Tasks-API clock
        shown, shown_conf = self._gate(label, conf)
        self.display_label, self.display_conf = shown, shown_conf
        self.translator.draw(img, shown, shown_conf)
        h, w = img.shape[:2]
        # Cover the desktop-app hint line, draw a big verdict banner instead.
        cv2.rectangle(img, (0, h - 52), (w, h), (0, 0, 0), -1)
        if shown is None:
            cv2.putText(img, "Unsure - hold the sign steady", (15, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
        else:
            cv2.putText(img, f"{shown} ({shown_conf:.0%})", (15, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def render_live():
    if not WEBRTC_OK:
        st.error("Live mode needs `streamlit-webrtc` — install it with "
                 "`pip install -r requirements.txt` and rerun.")
        return
    st.write("Hold a sign steady in front of your camera — guesses appear live "
             "on the video. Low-confidence guesses show as *unsure* instead "
             "of a shaky label.")
    ctx = webrtc_streamer(
        key="isl-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_processor_factory=SignProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    proc = None
    try:
        proc = ctx.video_processor if ctx else None
    except Exception:
        proc = None

    label = proc.display_label if proc else None
    sentence = list(proc.translator.sentence) if proc else []

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔊 Speak last sign", key="live_speak"):
            if label:
                speak_js(label)
            else:
                st.info("No confident sign yet — hold a sign steady in front "
                        "of the camera.")
    with col2:
        if st.button("Clear sentence", key="live_clear"):
            if proc:
                proc.translator.sentence.clear()
                proc.translator.last_spoken = None
    st.write("**Sentence:** " + (" ".join(sentence) if sentence else "—"))


def render_upload():
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

        # Threshold gate (same treatment as live mode): never present a
        # shaky guess as the answer.
        if conf >= DISPLAY_CONF_THRESHOLD:
            st.markdown(f"# 🗣️ {pred}")
            st.metric("Confidence", f"{conf:.1%}")
            if st.button("🔊 Speak it"):
                speak_js(pred)
        else:
            st.warning("🤔 Unsure — confidence too low. Try a clearer video, "
                       "or hold the sign steady and centered.")
        st.subheader("Top 3 guesses")
        st.bar_chart({labels[i]: float(probs[i]) for i in top})
        st.video(f)
        st.caption("Tip: the live webcam demo with voice output runs via `python run_realtime.py`")


def render_about():
    st.markdown(f"""
**How it works**

1. **Capture** — live webcam frames (Live mode) or an uploaded clip.
2. **Pose extraction** — MediaPipe HandLandmarker finds 21 3D landmarks per hand
   (2 hands × 21 × 3 = 126 features/frame). Classifying landmarks instead of
   raw pixels makes the model fast and robust to lighting/backgrounds.
3. **Normalisation** — each hand is centred on its wrist and scaled by hand
   size, so the model learns *gesture shape*, not hand position.
4. **Sequence model** — a {model_type} reads 30-frame windows and predicts one
   of {len(labels)} ISL signs. Guesses below {DISPLAY_CONF_THRESHOLD:.0%} confidence
   are shown as *unsure* rather than a shaky label.
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


with st.expander("📖 Signs I understand", expanded=False):
    st.write(", ".join(f"`{l}`" for l in labels))

mode = st.radio("Mode", ("🔴 Live (realtime)", "📤 Upload video"),
                horizontal=True, label_visibility="collapsed")

if mode.startswith("🔴"):
    render_live()
    with st.expander("ℹ️ About", expanded=False):
        render_about()
else:
    tab1, tab2 = st.tabs(["🎥 Video prediction", "ℹ️ About"])
    with tab1:
        render_upload()
    with tab2:
        render_about()
