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
