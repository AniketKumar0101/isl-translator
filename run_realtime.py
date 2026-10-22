"""Entry point for the live webcam demo.

Run from project root:   python run_realtime.py
Requires a trained model: checkpoints/best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from realtime.inference import SignTranslator

if __name__ == "__main__":
    SignTranslator().run()
