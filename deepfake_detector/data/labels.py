"""
Unified source of truth for classification labels.
Canonical standard:
    REAL = 0
    FAKE = 1
"""

from __future__ import annotations

from typing import Any

REAL: int = 0
FAKE: int = 1

LABEL_NAMES: dict[int, str] = {
    REAL: "real",
    FAKE: "fake",
}


def get_label_name(label: int) -> str:
    """Return human-readable label name."""
    return LABEL_NAMES.get(label, "unknown")


def get_canonical_label_from_name(name: str) -> int:
    """Normalize class string to canonical integer label (0=Real, 1=Fake)."""
    normalized = name.strip().lower()
    if normalized in {"real", "realart", "authentic", "pristine", "original"}:
        return REAL
    elif normalized in {"fake", "aiartdata", "synthetic", "ai", "deepfake", "manipulated"}:
        return FAKE
    else:
        raise ValueError(f"Unrecognized class name: '{name}'. Cannot determine Real vs Fake.")


def build_hf_label_map(class_names: list[str]) -> dict[int, int]:
    """
    Build explicit dictionary mapping from raw HuggingFace ClassLabel integer index
    to canonical label (0=Real, 1=Fake) based on ClassLabel.names at runtime.
    """
    mapping = {}
    for idx, name in enumerate(class_names):
        canonical = get_canonical_label_from_name(name)
        mapping[idx] = canonical
    return mapping


def resolve_hf_feature_mapping(dataset_features: Any) -> dict[int, int]:
    """
    Extract label names from HuggingFace dataset features object or feature dict
    and return mapping {hf_raw_index: canonical_int}.
    """
    label_feature = None
    if hasattr(dataset_features, "get"):
        label_feature = dataset_features.get("label")
    elif hasattr(dataset_features, "features"):
        label_feature = dataset_features.features.get("label")

    if label_feature is None:
        raise ValueError("Could not find 'label' feature in dataset features.")

    if hasattr(label_feature, "names"):
        names = label_feature.names
    else:
        raise TypeError(f"Label feature is not a ClassLabel with .names: {label_feature}")

    return build_hf_label_map(names)
