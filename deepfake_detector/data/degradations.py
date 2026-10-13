"""
Compression and degradation augmentation transforms.
Includes single and double JPEG compression, Gaussian blur, resizing, noise,
and simulated social-media (WhatsApp-style) degradation.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    """Apply JPEG compression at a specific quality factor (1-100)."""
    quality = max(1, min(100, int(quality)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    compressed = Image.open(buf)
    return compressed.convert("RGB")


def double_jpeg_compress(image: Image.Image, q1: int, q2: int) -> Image.Image:
    """Apply two consecutive stages of JPEG compression (common on social media re-uploads)."""
    first_pass = jpeg_compress(image, q1)
    return jpeg_compress(first_pass, q2)


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    """Apply Gaussian blur with given radius."""
    if radius <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def down_upscale(image: Image.Image, scale_factor: float) -> Image.Image:
    """Downscale then upscale image to simulate low-bandwidth resolution degradation."""
    w, h = image.size
    new_w = max(16, int(w * scale_factor))
    new_h = max(16, int(h * scale_factor))
    downsampled = image.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
    return downsampled.resize((w, h), resample=Image.Resampling.BICUBIC)


def add_gaussian_noise(image: Image.Image, std: float = 8.0) -> Image.Image:
    """Add additive Gaussian noise."""
    arr = np.array(image, dtype=np.float32)
    noise = np.random.normal(0, std, arr.shape)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)


def whatsapp_sim_degrade(image: Image.Image) -> Image.Image:
    """
    Simulate WhatsApp image transmission degradation profile:
    Downscale large images (max dim ~1280 or ~800), add subtle blur, and re-compress with QF ~65-75.
    """
    w, h = image.size
    max_dim = max(w, h)
    if max_dim > 1024:
        scale = 1024.0 / max_dim
        image = image.resize((int(w * scale), int(h * scale)), resample=Image.Resampling.LANCZOS)

    # Re-compress with WhatsApp-typical JPEG quality (~70)
    return jpeg_compress(image, quality=random.randint(65, 75))


class OnTheFlyDegradation:
    """
    Stochastic on-the-fly degradation pipeline evaluated per sample in Dataset.__getitem__.
    Prevents static feature memorization and ensures continuous robustness learning.
    """

    def __init__(
        self,
        jpeg_prob: float = 0.7,
        jpeg_min_qf: int = 15,
        jpeg_max_qf: int = 95,
        double_jpeg_prob: float = 0.3,
        clean_prob: float = 0.15,
        downscale_prob: float = 0.3,
        blur_prob: float = 0.25,
    ):
        self.jpeg_prob = jpeg_prob
        self.jpeg_min_qf = jpeg_min_qf
        self.jpeg_max_qf = jpeg_max_qf
        self.double_jpeg_prob = double_jpeg_prob
        self.clean_prob = clean_prob
        self.downscale_prob = downscale_prob
        self.blur_prob = blur_prob

    def __call__(self, img: Image.Image) -> Image.Image:
        # Pass-through clean image without corruption
        if random.random() < self.clean_prob:
            return img

        out = img

        # Downscale-upscale degradation
        if random.random() < self.downscale_prob:
            scale = random.uniform(0.3, 0.85)
            out = down_upscale(out, scale)

        # Mild blur
        if random.random() < self.blur_prob:
            radius = random.uniform(0.5, 1.5)
            out = gaussian_blur(out, radius)

        # JPEG compression or Double JPEG
        if random.random() < self.jpeg_prob:
            if random.random() < self.double_jpeg_prob:
                q1 = random.randint(self.jpeg_min_qf + 10, self.jpeg_max_qf)
                q2 = random.randint(self.jpeg_min_qf, q1)
                out = double_jpeg_compress(out, q1, q2)
            else:
                q = random.randint(self.jpeg_min_qf, self.jpeg_max_qf)
                out = jpeg_compress(out, q)

        return out
