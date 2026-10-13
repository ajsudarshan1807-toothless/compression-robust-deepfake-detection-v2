"""
Unit and integration tests for dataset splitting, deduplication, and data integrity.
"""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from deepfake_detector.data.splits import (
    compute_phash,
    compute_sha1,
    create_stratified_splits,
    deduplicate_records,
)


def test_deduplicate_records(tmp_path: Path):
    import numpy as np

    # Create distinct textured images
    rng = np.random.RandomState(42)
    arr1 = rng.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    arr2 = arr1.copy()  # Exact copy
    arr3 = (np.sin(np.linspace(0, 10, 100))[:, None, None] * 127 + 128).astype(np.uint8)
    arr3 = np.tile(arr3, (1, 100, 3))  # Completely different gradient pattern

    img1 = Image.fromarray(arr1)
    img2 = Image.fromarray(arr2)
    img3 = Image.fromarray(arr3)

    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    p3 = tmp_path / "img3.png"

    img1.save(p1)
    img2.save(p2)
    img3.save(p3)

    records = [
        {
            "filename": str(p1),
            "sha1": compute_sha1(p1.read_bytes()),
            "phash": compute_phash(img1),
            "label": 0,
        },
        {
            "filename": str(p2),
            "sha1": compute_sha1(p2.read_bytes()),
            "phash": compute_phash(img2),
            "label": 0,
        },
        {
            "filename": str(p3),
            "sha1": compute_sha1(p3.read_bytes()),
            "phash": compute_phash(img3),
            "label": 1,
        },
    ]

    deduped, removed = deduplicate_records(records, max_hamming_dist=6)
    assert removed == 1
    assert len(deduped) == 2


def test_stratified_splits_balance():
    data = []
    for i in range(100):
        data.append(
            {
                "filename": f"img_real_{i}.jpg",
                "label": 0,
                "sha1": f"sha_real_{i}",
                "phash": f"ph_{i}",
            }
        )
        data.append(
            {
                "filename": f"img_fake_{i}.jpg",
                "label": 1,
                "sha1": f"sha_fake_{i}",
                "phash": f"ph_f_{i}",
            }
        )

    df = pd.DataFrame(data)
    train_df, val_df, test_df = create_stratified_splits(
        df, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42
    )

    assert len(train_df) == 140
    assert len(val_df) == 30
    assert len(test_df) == 30

    # Check class balance
    assert (train_df["label"] == 0).sum() == 70
    assert (train_df["label"] == 1).sum() == 70

    # Ensure no split overlap by sha1
    train_sha = set(train_df["sha1"])
    val_sha = set(val_df["sha1"])
    test_sha = set(test_df["sha1"])

    assert len(train_sha.intersection(val_sha)) == 0
    assert len(train_sha.intersection(test_sha)) == 0
    assert len(val_sha.intersection(test_sha)) == 0


def test_rebuilt_splits_disjoint_and_valid():
    """Verify that committed/generated splits have no overlap and decode properly."""
    split_dir = Path("data/splits")
    if not (split_dir / "train.csv").exists():
        pytest.skip("Splits not generated yet.")

    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    # Check no overlap by filename or sha1
    train_paths = set(train_df["filename"])
    val_paths = set(val_df["filename"])
    test_paths = set(test_df["filename"])

    assert len(train_paths.intersection(val_paths)) == 0
    assert len(train_paths.intersection(test_paths)) == 0
    assert len(val_paths.intersection(test_paths)) == 0

    if "sha1" in train_df.columns:
        train_sha = set(train_df["sha1"])
        val_sha = set(val_df["sha1"])
        test_sha = set(test_df["sha1"])
        assert len(train_sha.intersection(val_sha)) == 0
        assert len(train_sha.intersection(test_sha)) == 0
        assert len(val_sha.intersection(test_sha)) == 0

    # Check labels in {0, 1}
    for df in (train_df, val_df, test_df):
        assert set(df["label"].unique()).issubset({0, 1})

    # Sample images decode properly to RGB
    for sample_path in list(train_paths)[:5]:
        with Image.open(sample_path) as img:
            assert img.mode == "RGB"
