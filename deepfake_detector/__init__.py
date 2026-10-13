"""
Compression-Robust Deepfake Detection Package.
"""

from deepfake_detector.utils import get_device, load_config, safe_load_checkpoint, set_seed

__version__ = "0.1.0"
__all__ = ["__version__", "get_device", "load_config", "safe_load_checkpoint", "set_seed"]
