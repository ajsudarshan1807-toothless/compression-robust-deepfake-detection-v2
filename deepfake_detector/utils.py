"""
Core utilities for Compression-Robust Deepfake Detection.
Handles reproducible seeding, device management, and secure checkpoint loading.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """Set seeds globally for python, numpy, torch, and CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """Worker init function for PyTorch DataLoader to ensure reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(device_str: str = "auto") -> torch.device:
    """
    Resolve device dynamically.
    If 'auto', prioritize CUDA -> MPS -> CPU.
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def safe_load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    """Load checkpoint weights with weights_only=True to prevent arbitrary code execution."""
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path_obj}")
    return torch.load(path_obj, map_location=map_location, weights_only=True)


def load_config(config_path: str | Path = "configs/base.yaml") -> dict[str, Any]:
    """Load YAML config file."""
    path_obj = Path(config_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")
    with open(path_obj, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
