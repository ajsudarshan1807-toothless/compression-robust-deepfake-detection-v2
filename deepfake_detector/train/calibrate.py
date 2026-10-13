"""
Temperature Scaling Post-Hoc Calibration for Deepfake Detection.
Calibrates model logits to minimize Expected Calibration Error (ECE) and NLL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from deepfake_detector.eval.metrics import compute_ece


class TemperatureScaler(nn.Module):
    """
    Learns a single temperature parameter T > 0 such that calibrated_prob = Sigmoid(logit / T).
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = self.temperature.clamp(min=0.01)
        return torch.sigmoid(logits / temp)

    def fit(self, val_logits: torch.Tensor, val_targets: torch.Tensor, max_iter: int = 50) -> float:
        """
        Fit temperature parameter using L-BFGS on validation logits and targets.
        """
        criterion = nn.BCELoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        val_logits = val_logits.detach()
        val_targets = val_targets.detach().float()

        def eval_loss():
            optimizer.zero_grad()
            temp = self.temperature.clamp(min=0.01)
            probs = torch.sigmoid(val_logits / temp)
            loss = criterion(probs, val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


def calibrate_predictions(
    probs: np.ndarray,
    targets: np.ndarray,
    temperature: float | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """
    Given raw probabilities in (0, 1) and true labels in {0, 1}:
    Calculates uncalibrated ECE, fits/applies temperature scaling, and calculates calibrated ECE.
    """
    probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
    logits = np.log(probs / (1.0 - probs))

    t_logits = torch.tensor(logits, dtype=torch.float32)
    t_targets = torch.tensor(targets, dtype=torch.float32)

    scaler = TemperatureScaler()
    if temperature is None:
        best_temp = scaler.fit(t_logits, t_targets)
    else:
        scaler.temperature.data.fill_(temperature)
        best_temp = temperature

    with torch.no_grad():
        calibrated_probs = scaler(t_logits).numpy()

    ece_before = compute_ece(targets, probs)
    ece_after = compute_ece(targets, calibrated_probs)

    return calibrated_probs, best_temp, ece_before, ece_after
