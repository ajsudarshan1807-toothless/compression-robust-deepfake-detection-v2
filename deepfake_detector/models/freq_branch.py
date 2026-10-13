"""
Frequency branch architecture using 2-level Haar Discrete Wavelet Transform (DWT) and CNN.
Extracts 128-dimensional high-frequency anomaly features.
"""

from __future__ import annotations

import numpy as np
import pywt
import torch
import torch.nn as nn


def extract_haar_dwt_features(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extract 2-level Haar DWT high-frequency subbands.
    Input: (B, C, H, W)
    Output: (B, 9, H//4, W//4)
    """
    device = image_tensor.device
    np_img = image_tensor.detach().cpu().numpy()
    b, c, _, _ = np_img.shape

    batch_subbands = []
    for i in range(b):
        channel_subbands = []
        for j in range(c):
            coeffs = pywt.wavedec2(np_img[i, j], "haar", level=2)
            lh2, hl2, hh2 = coeffs[1]
            channel_subbands.extend([lh2, hl2, hh2])
        batch_subbands.append(np.stack(channel_subbands, axis=0))

    stacked = np.stack(batch_subbands, axis=0)
    return torch.tensor(stacked, dtype=torch.float32, device=device)


class FrequencyBranch(nn.Module):
    def __init__(self, in_channels: int = 9, num_classes: int = 1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.feature_dim = 128
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        x_dwt = extract_haar_dwt_features(x)
        features = self.cnn(x_dwt)
        features = features.view(features.size(0), -1)

        if return_features:
            return features
        return self.head(features)
