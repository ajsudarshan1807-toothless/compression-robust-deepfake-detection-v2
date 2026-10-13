"""
Unit tests for image degradations and compression augmentation.
"""

from PIL import Image

from deepfake_detector.data.degradations import (
    OnTheFlyDegradation,
    add_gaussian_noise,
    double_jpeg_compress,
    down_upscale,
    gaussian_blur,
    jpeg_compress,
    whatsapp_sim_degrade,
)


def test_jpeg_compress():
    img = Image.new("RGB", (224, 224), color=(100, 150, 200))
    out = jpeg_compress(img, quality=50)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_double_jpeg_compress():
    img = Image.new("RGB", (224, 224), color=(120, 80, 40))
    out = double_jpeg_compress(img, q1=85, q2=40)
    assert out.size == (224, 224)


def test_down_upscale():
    img = Image.new("RGB", (200, 200), color=(50, 50, 50))
    out = down_upscale(img, scale_factor=0.5)
    assert out.size == (200, 200)


def test_gaussian_blur():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    out = gaussian_blur(img, radius=1.0)
    assert out.size == (100, 100)


def test_add_noise():
    img = Image.new("RGB", (64, 64), color=(128, 128, 128))
    out = add_gaussian_noise(img, std=5.0)
    assert out.size == (64, 64)


def test_whatsapp_sim():
    img = Image.new("RGB", (1500, 1200), color=(80, 100, 120))
    out = whatsapp_sim_degrade(img)
    assert max(out.size) <= 1024


def test_on_the_fly_pipeline():
    pipeline = OnTheFlyDegradation(jpeg_prob=1.0, clean_prob=0.0)
    img = Image.new("RGB", (224, 224), color=(150, 150, 150))
    out = pipeline(img)
    assert out.size == (224, 224)
    assert out.mode == "RGB"
