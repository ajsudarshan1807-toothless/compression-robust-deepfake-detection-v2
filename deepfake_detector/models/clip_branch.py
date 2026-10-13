"""
CLIP Vision Transformer branch with optional PEFT LoRA adapter injection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPVisionModel


class ClipBranch(nn.Module):
    """
    CLIP Vision Backbone (OpenAI ViT-B/32).
    Supports frozen extraction, partial unfreezing, or LoRA adapter injection.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        if pretrained:
            self.model = CLIPVisionModel.from_pretrained(model_name)
        else:
            config = CLIPVisionConfig.from_pretrained(model_name)
            self.model = CLIPVisionModel(config)

        self.feature_dim = 768

        if use_lora:
            from peft import LoraConfig, get_peft_model

            # Target linear query and value projections in ViT attention
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
            )
            self.model = get_peft_model(self.model, peft_config)
        elif freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            # Unfreeze last N layers if desired
            unfreeze_layers = 4
            for param in self.model.parameters():
                param.requires_grad = False
            for layer in self.model.encoder.layers[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor, return_features: bool = True) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        features = outputs.pooler_output
        return features


class ClipDetector(nn.Module):
    """
    Unified CLIP Detector architecture implementing:
    - 3A: Frozen CLIP backbone + Linear probe (linear_probe)
    - 3B: CLIP with PEFT LoRA attention adapter fine-tuning (lora)
    - 3C: Dual-stream CLIP + Trained Haar DWT Frequency branch (two_stream)
    """

    def __init__(
        self,
        mode: str = "lora",
        pretrained: bool = True,
        lora_r: int = 16,
        model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.mode = mode

        if mode == "linear_probe":
            self.clip_branch = ClipBranch(
                pretrained=pretrained,
                freeze=True,
                use_lora=False,
                model_name=model_name,
            )
            self.head = nn.Sequential(
                nn.Linear(768, 1),
                nn.Sigmoid(),
            )
        elif mode == "lora":
            self.clip_branch = ClipBranch(
                pretrained=pretrained,
                freeze=False,
                use_lora=True,
                lora_r=lora_r,
                model_name=model_name,
            )
            self.head = nn.Sequential(
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        elif mode == "two_stream":
            from deepfake_detector.models.freq_branch import FrequencyBranch

            self.clip_branch = ClipBranch(
                pretrained=pretrained,
                freeze=False,
                use_lora=True,
                lora_r=lora_r,
                model_name=model_name,
            )
            self.freq_branch = FrequencyBranch(in_channels=9)
            self.head = nn.Sequential(
                nn.Linear(768 + 128, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'linear_probe', 'lora', 'two_stream'.")

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        clip_feats = self.clip_branch(x)

        if self.mode == "two_stream":
            freq_feats = self.freq_branch(x, return_features=True)
            features = torch.cat([clip_feats, freq_feats], dim=1)
        else:
            features = clip_feats

        if return_features:
            return features

        return self.head(features)

