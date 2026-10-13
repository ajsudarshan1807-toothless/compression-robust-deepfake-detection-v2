"""
Video temporal sequence models for deepfake detection.
Analyzes frame sequences across time to detect micro-flickering, temporal discontinuity,
and blending boundary anomalies across frames.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TemporalPoolingModel(nn.Module):
    """Baseline (a): Average or max pooling of frame features."""

    def __init__(self, in_features: int = 1280, pooling_type: str = "mean"):
        super().__init__()
        self.pooling_type = pooling_type
        self.head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is (B, T, D)
        if self.pooling_type == "mean":
            pooled = torch.mean(x, dim=1)
        else:
            pooled, _ = torch.max(x, dim=1)

        # Frame-level scores
        frame_scores = self.head(x.view(-1, x.size(-1))).view(x.size(0), x.size(1))
        video_score = self.head(pooled)
        return video_score, frame_scores


class TemporalLSTMModel(nn.Module):
    """Model (b): 1-layer Bidirectional LSTM over sequence of frame feature vectors."""

    def __init__(self, in_features: int = 1280, hidden_size: int = 256):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x is (B, T, D)
        lstm_out, (hn, _) = self.lstm(x)
        # Concatenate forward and backward final hidden states
        final_h = torch.cat((hn[0], hn[1]), dim=1)
        video_score = self.head(final_h)

        # Per-frame logits for timeline seek bar
        frame_scores = self.frame_head(x.view(-1, x.size(-1))).view(x.size(0), x.size(1))
        return video_score, frame_scores


class TemporalTransformerModel(nn.Module):
    """Model (c): Small 2-layer, 4-head Temporal Transformer Encoder."""

    def __init__(self, in_features: int = 1280, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        proj_x = self.proj(x)
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x_with_cls = torch.cat((cls_tokens, proj_x), dim=1)
        x_pos = self.pos_encoder(x_with_cls)
        encoded = self.transformer(x_pos)

        # CLS output for video verdict
        cls_out = encoded[:, 0, :]
        video_score = self.head(cls_out)

        frame_scores = self.frame_head(x.view(-1, d)).view(b, t)
        return video_score, frame_scores
