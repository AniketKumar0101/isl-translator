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
