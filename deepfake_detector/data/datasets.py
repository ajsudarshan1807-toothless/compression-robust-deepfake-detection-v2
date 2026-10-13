"""
Dataset classes and dataset rebuilding pipelines.
Applies on-the-fly degradation prior to resizing, preventing artificial resampling artifacts.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class DeepfakeImageDataset(Dataset):
    """
    Standard PyTorch dataset for deepfake detection.
    Degradation transforms are executed dynamically on the native resolution PIL image
    before geometric resizing and normalization.
    """

    def __init__(
        self,
        csv_file: str | Path,
        transform: Callable | None = None,
        degradation: Callable | None = None,
        test_qf: int | None = None,
    ):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.degradation = degradation
        self.test_qf = test_qf

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        img_path = Path(row["filename"])
        label = int(row["label"])

        image = Image.open(img_path).convert("RGB")

        # Fixed test-time QF condition if specified
        if self.test_qf is not None:
            from deepfake_detector.data.degradations import jpeg_compress

            image = jpeg_compress(image, quality=self.test_qf)
        # Training-time stochastic degradation
        elif self.degradation is not None:
            image = self.degradation(image)

        # Standard transformation (Resize -> ToTensor -> Normalize)
        if self.transform is not None:
            img_tensor = self.transform(image)
        else:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(image)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return img_tensor, label_tensor


class FaceCropper:
    """
    Facial detector and extractor using MTCNN with customizable margin expansion.
    """

    def __init__(self, margin_fraction: float = 0.20, device: str = "cpu"):
        from facenet_pytorch import MTCNN

        self.margin_fraction = margin_fraction
        self.detector = MTCNN(keep_all=False, select_largest=True, device=device)

    def crop_face(self, image: Image.Image) -> tuple[Image.Image | None, list[int] | None]:
        """Detect face and return (cropped_pil_image, [x1, y1, x2, y2]) or (None, None)."""
        boxes, probs = self.detector.detect(image)
        if boxes is None or len(boxes) == 0 or probs[0] is None or probs[0] < 0.80:
            return None, None

        box = boxes[0]
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1

        margin_w = w * self.margin_fraction
        margin_h = h * self.margin_fraction

        img_w, img_h = image.size
        crop_x1 = max(0, int(x1 - margin_w))
        crop_y1 = max(0, int(y1 - margin_h))
        crop_x2 = min(img_w, int(x2 + margin_w))
        crop_y2 = min(img_h, int(y2 + margin_h))

        cropped = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        return cropped, [crop_x1, crop_y1, crop_x2, crop_y2]


def build_default_transform(image_size: int = 224, is_clip: bool = True) -> transforms.Compose:
    """Return standard evaluation / training transform pipeline."""
    if is_clip:
        # OpenAI CLIP normalization
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    else:
        # ImageNet normalization (for EfficientNet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
