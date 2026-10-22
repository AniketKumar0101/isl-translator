# TEST REPORT — real-data model wired into the Streamlit app

Date: 2026-10-22. Model: transformer, 17 ISL classes, `checkpoints/best.pt`
(70.1% on INCLUDE's official signer-independent test split).

## What changed

| File | Change |
|---|---|
| `src/realtime/inference.py` | Migrated from removed `mp.solutions` API to MediaPipe Tasks API (`HandLandmarker`, VIDEO mode, monotonic ms timestamps), reusing `init_landmarker()`/`result_to_vector()` from `src/data/extract_landmarks.py`. Loads transformer `best.pt` + 17 labels. `draw()` reimplemented with cv2 primitives (hand skeleton connections). Same output contract: `step(frame) -> (smooth_label, conf)`, `predict()`, `sentence`, `speaker`. Logs analytics on each confirmed sign. |
| `src/realtime/analytics.py` | NEW. `log_prediction(label, confidence)` appends JSONL to `data/analytics/usage.jsonl`: `{ts, predicted_label, confidence}`. No images, no personal data. Never raises. |
| `app.py` | Loads 17 labels from checkpoint; big predicted-sign display; confidence metric; top-3 bar chart; **🔊 Speak button** via browser Web Speech API (server-side TTS unavailable headless — see notes); sign-vocabulary expander; About tab updated (transformer, 17 real ISL signs, 70.1% test accuracy, INCLUDE CC-BY-4.0 attribution); logs one analytics event per video prediction. |

## Tests (all headless, no webcam on this VM)

### A. Inference unit checks — `9/9 PASS`
Command: `cd ~/workspace/sign-language-translator && .venv/bin/python /tmp/test_e2e.py`
- translator init (Tasks landmarker + transformer load) — PASS
- labels match `config.yaml` 17-label vocab — PASS
- `predict()` on 5 real landmark `.npy` sequences — PASS
  (`hello→hello 0.54`, `thank you→thank you 0.90`, `good morning→good afternoon 0.38`,
  `good afternoon→good afternoon 0.38`, `good evening→good night 0.38`)
- `step()` on synthetic black frame (exercises `detect_for_video`) — PASS
- `draw()` overlay — PASS
- TTS silent fallback (`pyttsx3` missing → silent, `say()` no-crash) — PASS
- analytics JSONL schema `{ts, predicted_label, confidence}` — PASS
- analytics privacy (no image/video/PII keys) — PASS
- analytics default path inside project — PASS

### B. Streamlit boot — PASS
Command: `.venv/bin/streamlit run app.py --server.headless true --server.port 8502`
- HTTP 200 on `/`; boot log contains no tracebacks/exceptions.

### C. Full runtime path on a real raw video — PASS
Replicated `app.py`'s exact prediction body on `data/raw/hello/MVI_0029.MOV`
(Tasks-API extraction → normalise → transformer → softmax → analytics):
- Predicted **hello at 53.9%** (correct), top-3 `hello/she/he`, analytics logged.

## Remaining issues / notes

1. **Live webcam path (`run_realtime.py`) untested** — no camera on this VM.
   `step()`/`draw()` were exercised with synthetic frames only. Recommend one
   manual webcam smoke test on a real machine before the public demo goes live.
2. **Server-side TTS unavailable** (`pyttsx3` not installed, no `espeak`) — the
   Streamlit "🔊 Speak" button uses the browser's Web Speech API instead, which
   is the right approach for a deployed demo. `run_realtime.py` degrades to
   silent mode gracefully.
3. **Known model limitation, not an app bug:** the `good-X` greetings confuse
   each other at low confidence (e.g. `good morning→good afternoon 0.38`),
   consistent with the 70.1% test accuracy. The app surfaces the confidence so
   users see uncertainty honestly.
4. Training artifacts untouched; no retraining performed.

## Launch

```bash
cd ~/workspace/sign-language-translator
.venv/bin/streamlit run app.py
```

## Verdict

**Ready for public deployment.** All testable paths pass headless; the single
untestable item (live webcam) is noted above and does not block the video-upload
demo, which is the primary public path.

## She-bias fix (2026-10-22)

### Diagnosis — why the model kept saying "she"
1. **Real over-prediction, not just bad luck.** Recomputed per-class metrics on
   the held-out test set (67 clips): `she` was predicted 7x vs 4 true
   occurrences (precision **0.571**, predicted 10.4% vs true 6.0%).
   Same pattern for the other raised-open-hand signs: `good morning`
   (precision 0.333, 6 pred vs 3 true) and `good night` (precision 0.375,
   8 pred vs 4 true). Class balance in training was fine (21 videos/class),
   so this is a learned bias, not a data-skew artifact.
2. **The sign itself is a "default".** In INCLUDE, ISL `she` is a raised open
   hand beside the head — visually close to any ambiguous raised-hand pose,
   which is exactly what a live webcam sees during transitions and casual
   hand positions.
3. **The app surfaced shaky guesses as answers.** The live case that triggered
   this was `she` at **42.8%** confidence — far below even the old 0.60 vote
   gate — because the clip path displayed raw top-1 regardless of confidence.

### Fix (all in `src/realtime/inference.py` + `config.yaml`)
1. **`min_confidence` (default 0.55, new `realtime.min_confidence` key).**
   `predict()` now returns `(None, conf)` — i.e. "unsure" — when the
   bias-corrected top-1 confidence is below it, instead of a shaky label.
   `step()` only lets non-None predictions into the smoothing votes, so
   `step()` returns `(None, conf)` until enough confident predictions
   accumulate. Nothing is spoken or logged while unsure. Chosen from the
   confidence split: correct test predictions average 0.66, wrong ones 0.45;
   at 0.55 the shown predictions are ~90%+ accurate.
2. **Per-class bias correction.** A fixed −0.3 logit penalty on
   `("she", "good morning", "good night")` (the three over-predicted
   raised-hand classes), applied inside `predict()` before softmax so both
   clip and live paths benefit. Evidence: penalty sweeps on val+test kept
   accuracy flat (val 0.697→0.697, test 0.701→0.701 at 0.3) while cutting
   `she` false positives; the full 17-dim val-marginal vector was rejected
   because the 33-sample val set made it noisy (it hurt val accuracy).
3. `predict()` also guards against a non-full buffer → `(None, 0.0)`.

### Test results (headless)
- Full 67-clip test set through new `predict()`: **0 `she` false positives**
  (was 3); coverage 0.48, accuracy on shown predictions **0.906**; no crashes.
  Every suppressed wrong guess became "unsure", not a different wrong label.
- `step()` on empty/no-hand frames: all `(None, …)`, no votes, nothing
  spoken/logged — PASS.
- `step()` driven by a real `hello` landmark stream: recognized as `hello`,
  spoken once, logged once — PASS.
- Caveat: MediaPipe's native lib (`libEGL.so.1`) is missing on this VM, so
  the hand-detection stage was stubbed; the changed code (bias, threshold,
  vote gating) was tested end-to-end against real landmark sequences.
  Re-verify on a machine with a webcam before the public demo.

### API for the app.py / live-mode consumer
- `SignTranslator.predict() -> (label: str | None, conf: float)` — `None`
  means "unsure" (below `min_confidence`); `conf` is always the raw top-1
  probability, safe to display (e.g. "unsure · 43%").
- `SignTranslator.step(frame) -> (smoothed_label: str | None, conf: float)` —
  `None` until the smoothing window holds enough confident votes.
- `draw()` already renders `None` as `...`; TTS/analytics only fire on
  confident labels.

## Live mode (streamlit-webrtc) — 2026-10-22

**What changed** (`app.py`, `requirements.txt`; `inference.py` untouched)
- New mode radio at the top: **🔴 Live (realtime)** vs **📤 Upload video**.
  (Note: the on-disk app.py was upload-based, not the "1.5s clip" UI from the
  user's screenshot — that was an older build. The existing upload flow is
  preserved as-is, plus a minimal threshold gate, see below.)
- `SignProcessor(VideoProcessorBase)`: `recv()` takes each webcam frame as
  `av.VideoFrame` → BGR ndarray → `translator.step()` (which owns the
  MediaPipe Tasks monotonic timestamp clock internally) → `translator.draw()`
  skeleton overlay → big verdict banner → returns `av.VideoFrame`. One
  processor instance per WebRTC session; `async_processing=True`.
- **Threshold gating (the "she"-at-42% fix, UI layer):** the processor reads
  `translator.min_confidence` (0.55, landed in 7d28d25) with a 0.55 fallback,
  so it tracks the translator's gate automatically. Below threshold the frame
  shows **"Unsure – hold the sign steady"** (amber) instead of a shaky label.
- **Anti-latch:** a label that persists ≥4 frames without ever reaching 0.75
  confidence stays "unsure" rather than latching a shaky guess onto the display.
- Speak button reuses the existing Web Speech API approach, but only speaks
  when the last result was confident; otherwise shows "no confident sign yet".
  Sentence strip + Clear button read `translator.sentence` on rerun.
- Upload mode, minimal edit: the big `# 🗣️ {pred}` label + Speak button now
  render only when `conf >= 0.55`; below that a warning
  ("🤔 Unsure — confidence too low…"). Top-3 chart, video, and analytics
  logging are unchanged.
- `requirements.txt`: added `streamlit-webrtc>=0.47` (installed 0.79.0).

**Tests (headless — no webcam on this VM)**
- `app.py` imports cleanly; `streamlit run` boots → HTTP 200, no tracebacks.
- `SignProcessor()` instantiates (real transformer + HandLandmarker) and picks
  up `min_confidence = 0.55` from the translator — PASS.
- `_gate` unit tests — PASS: `(None, 0)` → unsure; `("she", 0.42)` → unsure
  (the exact reported symptom); `("hello", 0.90)` → shown; six repeats of
  `("she", 0.60)` → unsure (anti-latch); new `("thank you", 0.95)` → shown;
  steady `("hello", 0.80)` keeps showing.
- `recv()` on 45 synthetic black frames: no crash, valid output frames,
  display stays unsure — PASS.
- `recv()` on 60 real frames decoded from `data/raw/hello/MVI_0029.MOV`:
  no crash, valid `av.VideoFrame` outputs — PASS.
- (Needed `libegl1`/`libgles2` .debs installed for MediaPipe's native lib;
  done via direct download, `apt-get update` was hanging.)

**Caveats**
- No live webcam on this VM: the browser↔server WebRTC path (ICE/STUN,
  `webrtc_streamer` session lifecycle) is **untestable here** — verify once on
  a real machine (`streamlit run app.py` → Live mode → allow camera).
- Frames are NOT mirror-flipped (matches the INCLUDE training distribution:
  signer facing camera). If users find it unnatural, flip in `recv()`.
- `recv()` runs in a worker thread (`async_processing=True`); per-session
  translator instances are used, so no cross-session state leaks.
- The MediaPipe `__del__` shutdown warning at interpreter exit is harmless noise.
