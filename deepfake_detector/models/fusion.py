"""
Fusion models combining spatial and frequency domain feature representations.
Supports:
- Baseline FusionModel (EfficientNet-B0 + DWT CNN) for Models C and D
- FusionModelV2 (CLIP Vision + DWT CNN) for Phase 3
"""

from __future__ import annotations

import torch
import torch.nn as nn

from deepfake_detector.models.freq_branch import FrequencyBranch
from deepfake_detector.models.spatial_branch import SpatialBranch


class FusionModel(nn.Module):
    """Dual-branch spatial + frequency fusion network (Models C and D)."""

    def __init__(self, pretrained_spatial: bool = True):
        super().__init__()
        self.spatial = SpatialBranch(pretrained=pretrained_spatial)
        self.frequency = FrequencyBranch()

        fusion_dim = self.spatial.feature_dim + self.frequency.feature_dim  # 1280 + 128 = 1408
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        spatial_feats = self.spatial(x, return_features=True)
        freq_feats = self.frequency(x, return_features=True)
        fused = torch.cat((spatial_feats, freq_feats), dim=1)

        if return_features:
            return fused
        return self.fusion_head(fused)


class FusionModelV2(nn.Module):
    """
    CLIP Vision + Frequency Branch fusion model.
    Eliminates semantic human face bias and provides high robustness to WhatsApp compression.
    """

    def __init__(self, pretrained_spatial: bool = False, freeze_clip: bool = True):
        super().__init__()
        from deepfake_detector.models.clip_branch import ClipBranch
        self.spatial = ClipBranch(pretrained=pretrained_spatial, freeze=freeze_clip)
        self.frequency = FrequencyBranch()

        fusion_dim = self.spatial.feature_dim + self.frequency.feature_dim  # 768 + 128 = 896
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        spatial_feats = self.spatial(x, return_features=True)
        freq_feats = self.frequency(x, return_features=True)
        fused = torch.cat((spatial_feats, freq_feats), dim=1)
        if return_features:
            return fused
        return self.fusion_head(fused)

