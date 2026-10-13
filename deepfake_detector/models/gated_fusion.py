"""
Learned Gated Fusion for Dual-Stream (Global Background + Local Face) detection.
Replaces the heuristic max(global, face) rule (Defect D9).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GatedDualStreamFusion(nn.Module):
    """
    Learned fusion gate.
    Input features: [score_global, score_face, face_conf, face_area_ratio]
    Output: Fused probability in [0, 1]
    """

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        score_global: float | torch.Tensor,
        score_face: float | torch.Tensor | None,
        face_conf: float = 0.0,
        face_area_ratio: float = 0.0,
    ) -> float:
        # Fall back gracefully to global if no face is detected
        if score_face is None or face_conf < 0.5:
            if isinstance(score_global, torch.Tensor):
                return float(score_global.item())
            return float(score_global)

        # Convert to tensor for MLP
        sg = float(score_global.item() if isinstance(score_global, torch.Tensor) else score_global)
        sf = float(score_face.item() if isinstance(score_face, torch.Tensor) else score_face)

        feats = torch.tensor([[sg, sf, face_conf, face_area_ratio]], dtype=torch.float32)
        with torch.no_grad():
            fused = self.mlp(feats).item()
        return float(fused)
