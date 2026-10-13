"""
Unit tests validating synthetic fixture image generation.
"""

from pathlib import Path

from PIL import Image


def test_fixture_images(fixture_image_dir: Path):
    real_files = list((fixture_image_dir / "real").glob("*.jpg"))
    fake_files = list((fixture_image_dir / "fake").glob("*.jpg"))

    assert len(real_files) == 8
    assert len(fake_files) == 8

    for p in real_files + fake_files:
        with Image.open(p) as img:
            assert img.mode == "RGB"
            assert img.size == (224, 224)
