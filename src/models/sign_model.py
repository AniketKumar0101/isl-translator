"""Sequence classifiers for sign-language landmark data.

Input: (batch, T=30, 126) landmark sequences.
Two architectures are provided — pick via `model.type` in config.yaml:
  - "lstm":        bidirectional LSTM, strong baseline, trains fast
  - "transformer": Transformer encoder, usually a few % better with more data
"""
import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=126, hidden_size=128, num_layers=2,
                 num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=126, d_model=128, nhead=4, num_layers=2,
                 num_classes=10, dropout=0.3, seq_len=30):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x):
        x = self.proj(x) + self.pos[:, :x.size(1), :]
        x = self.encoder(x)
        return self.head(x.mean(dim=1))  # mean-pool over time


def build_model(cfg, num_classes, model_type=None):
    m = cfg["model"]
    kind = model_type or m["type"]
    if kind == "lstm":
        return LSTMClassifier(
            hidden_size=m["hidden_size"], num_layers=m["num_layers"],
            num_classes=num_classes, dropout=m["dropout"])
    if kind == "transformer":
        return TransformerClassifier(
            d_model=m["hidden_size"], num_layers=m["num_layers"],
            num_classes=num_classes, dropout=m["dropout"],
            seq_len=cfg["data"]["sequence_length"])
    raise ValueError(f"unknown model.type: {kind!r} (expected 'lstm' or 'transformer')")
