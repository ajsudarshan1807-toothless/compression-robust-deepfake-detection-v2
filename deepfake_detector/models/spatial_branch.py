"""
Spatial branch architecture using EfficientNet-B0.
Extracts 1280-dimensional feature representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class SpatialBranch(nn.Module):
    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = efficientnet_b0(weights=weights)
        else:
            self.backbone = efficientnet_b0(weights=None)

        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.feature_dim = in_features

        self.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        features = self.backbone(x)
        if return_features:
            return features
        return self.head(features)
