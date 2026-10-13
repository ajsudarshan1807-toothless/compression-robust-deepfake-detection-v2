"""
Unit tests for model architectures and tensor forward passes.
"""

import pytest
import torch

from deepfake_detector.models.freq_branch import FrequencyBranch, extract_haar_dwt_features
from deepfake_detector.models.fusion import FusionModel
from deepfake_detector.models.spatial_branch import SpatialBranch


def test_spatial_branch_forward():
    model = SpatialBranch(num_classes=1, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 1)
    feats = model(x, return_features=True)
    assert feats.shape == (2, 1280)


def test_extract_haar_dwt_features():
    x = torch.randn(2, 3, 64, 64)
    dwt = extract_haar_dwt_features(x)
    # 3 channels * 3 subbands (LH, HL, HH) = 9 channels, H//4 = 16
    assert dwt.shape == (2, 9, 16, 16)


def test_frequency_branch_forward():
    model = FrequencyBranch(in_channels=9, num_classes=1)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1)
    feats = model(x, return_features=True)
    assert feats.shape == (2, 128)


def test_fusion_model_forward():
    model = FusionModel(pretrained_spatial=False)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 1)
    fused = model(x, return_features=True)
    assert fused.shape == (2, 1408)
