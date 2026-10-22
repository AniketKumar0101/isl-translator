"""Privacy-respecting usage analytics for the translator demo.

Each prediction event appends one JSONL line:
    {"ts": <unix time>, "predicted_label": <str>, "confidence": <float>}

No images, no video, no personal data — only anonymous prediction counts.
This feeds the "proven impact" evidence for the hackathon submission.
"""
import json
import time
from pathlib import Path


def analytics_path():
    p = Path(__file__).resolve().parents[2] / "data" / "analytics" / "usage.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log_prediction(label, confidence, path=None):
    """Append one prediction event. Never raises (analytics must not break the app)."""
    try:
        p = Path(path) if path else analytics_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": time.time(),
            "predicted_label": str(label),
            "confidence": round(float(confidence), 4),
        }
        with open(p, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:  # analytics is best-effort
        print(f"analytics log failed: {e}")
