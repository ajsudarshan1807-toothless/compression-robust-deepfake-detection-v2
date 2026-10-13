"""
Shared test fixtures for deepfake detector tests.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="session")
def fixture_image_dir(tmp_path_factory) -> Path:
    """Create a temporary directory containing 16 test images (8 real, 8 fake)."""
    img_dir = tmp_path_factory.mktemp("test_images")
    (img_dir / "real").mkdir(exist_ok=True)
    (img_dir / "fake").mkdir(exist_ok=True)

    rng = np.random.RandomState(42)

    for i in range(8):
        # Generate synthetic real image (smooth gradient + slight noise)
        arr = np.linspace(0, 255, 224, dtype=np.uint8)
        grid = np.tile(arr, (224, 1))
        noise = rng.normal(0, 5, (224, 224)).astype(np.int16)
        channel = np.clip(grid.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        real_img = np.stack([channel, channel, channel], axis=-1)
        Image.fromarray(real_img).save(img_dir / "real" / f"real_{i:02d}.jpg", quality=95)

        # Generate synthetic fake image (high-frequency checkerboard / noise artifacts)
        checker = (rng.rand(224, 224, 3) * 255).astype(np.uint8)
        Image.fromarray(checker).save(img_dir / "fake" / f"fake_{i:02d}.jpg", quality=95)

    return img_dir
