"""
Unit tests for core utilities.
"""

import pytest
import torch

from deepfake_detector.utils import get_device, load_config, safe_load_checkpoint, set_seed


def test_set_seed_reproducibility():
    set_seed(42)
    t1 = torch.rand(5)
    set_seed(42)
    t2 = torch.rand(5)
    assert torch.equal(t1, t2)


def test_get_device():
    dev = get_device("cpu")
    assert dev.type == "cpu"
    auto_dev = get_device("auto")
    assert auto_dev.type in {"cpu", "cuda", "mps"}


def test_load_config():
    cfg = load_config("configs/base.yaml")
    assert "paths" in cfg
    assert "model" in cfg
    assert cfg["model"]["image_size"] == 224


def test_safe_load_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        safe_load_checkpoint("checkpoints/non_existent_model.pt")
