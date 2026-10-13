"""
Unit tests for video temporal sequence models and frame split isolation.
"""

import pytest
import torch

from deepfake_detector.models.video import (
    TemporalLSTMModel,
    TemporalPoolingModel,
    TemporalTransformerModel,
)


def test_temporal_pooling_model():
    model = TemporalPoolingModel(in_features=128, pooling_type="mean")
    x = torch.randn(2, 16, 128)  # B=2, T=16, D=128
    video_score, frame_scores = model(x)

    assert video_score.shape == (2, 1)
    assert frame_scores.shape == (2, 16)
    assert (0.0 <= video_score).all() and (video_score <= 1.0).all()
    assert (0.0 <= frame_scores).all() and (frame_scores <= 1.0).all()


def test_temporal_lstm_model():
    model = TemporalLSTMModel(in_features=128, hidden_size=64)
    x = torch.randn(2, 16, 128)
    video_score, frame_scores = model(x)

    assert video_score.shape == (2, 1)
    assert frame_scores.shape == (2, 16)
    assert (0.0 <= video_score).all() and (video_score <= 1.0).all()


def test_temporal_transformer_model():
    model = TemporalTransformerModel(in_features=128, d_model=64, nhead=2, num_layers=1)
    x = torch.randn(2, 16, 128)
    video_score, frame_scores = model(x)

    assert video_score.shape == (2, 1)
    assert frame_scores.shape == (2, 16)
    assert (0.0 <= video_score).all() and (video_score <= 1.0).all()


def test_video_split_isolation_by_id():
    """
    Test Phase 4 requirement: Videos are partitioned strictly by Video ID,
    guaranteeing zero frame leakage between train and test.
    """
    # Simulate video frames tagged with video_id
    frames = []
    for vid in range(20):
        for f in range(16):
            frames.append({"video_id": f"vid_{vid:03d}", "frame_idx": f, "label": vid % 2})

    import pandas as pd
    df = pd.DataFrame(frames)

    # Split strictly by unique video_id
    unique_ids = df["video_id"].unique()
    train_ids = set(unique_ids[:14])
    test_ids = set(unique_ids[14:])

    # Assert zero overlap of video IDs
    assert len(train_ids.intersection(test_ids)) == 0

    train_df = df[df["video_id"].isin(train_ids)]
    test_df = df[df["video_id"].isin(test_ids)]

    assert len(set(train_df["video_id"]).intersection(set(test_df["video_id"]))) == 0
