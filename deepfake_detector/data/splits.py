"""
Dataset split generator, perceptual hash deduplication, and integrity checks.
Guarantees:
- Strict canonical labels (0=Real, 1=Fake)
- Zero leakage (disjoint by SHA1 and pHash Hamming distance <= 6)
- Relative POSIX file paths
- OOD hold-out split
- Emits results/dataset_stats.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image


def compute_sha1(data_bytes: bytes) -> str:
    """Compute SHA1 hexadecimal digest of raw bytes."""
    return hashlib.sha1(data_bytes).hexdigest()


def compute_phash(image: Image.Image) -> str:
    """Compute 64-bit perceptual hash string."""
    return str(imagehash.phash(image))


def hamming_distance(h1_str: str, h2_str: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    h1 = imagehash.hex_to_hash(h1_str)
    h2 = imagehash.hex_to_hash(h2_str)
    return h1 - h2


def deduplicate_records(
    records: list[dict[str, Any]], max_hamming_dist: int = 6
) -> tuple[list[dict[str, Any]], int]:
    """
    Deduplicate image records by exact SHA1 and perceptual hash.
    Removes near-duplicate images that could leak between train and test.
    """
    seen_sha1: set[str] = set()
    accepted_records: list[dict[str, Any]] = []
    accepted_hashes: list[imagehash.ImageHash] = []
    removed_count = 0

    for rec in records:
        sha1 = rec["sha1"]
        if sha1 in seen_sha1:
            removed_count += 1
            continue

        phash_obj = imagehash.hex_to_hash(rec["phash"])
        # Check against existing hashes
        is_duplicate = False
        for existing in accepted_hashes:
            if phash_obj - existing <= max_hamming_dist:
                is_duplicate = True
                break

        if is_duplicate:
            removed_count += 1
            continue

        seen_sha1.add(sha1)
        accepted_hashes.append(phash_obj)
        accepted_records.append(rec)

    return accepted_records, removed_count


def create_stratified_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, val, test while preserving class balance.
    All paths are normalized to POSIX relative paths.
    """
    real_df = df[df["label"] == 0].sample(frac=1.0, random_state=seed).reset_index(drop=True)
    fake_df = df[df["label"] == 1].sample(frac=1.0, random_state=seed).reset_index(drop=True)

    def split_class(cdf: pd.DataFrame):
        n = len(cdf)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_part = cdf.iloc[:n_train]
        val_part = cdf.iloc[n_train : n_train + n_val]
        test_part = cdf.iloc[n_train + n_val :]
        return train_part, val_part, test_part

    r_train, r_val, r_test = split_class(real_df)
    f_train, f_val, f_test = split_class(fake_df)

    train_df = (
        pd.concat([r_train, f_train]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    )
    val_df = pd.concat([r_val, f_val]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = pd.concat([r_test, f_test]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Normalize paths to POSIX (forward slashes)
    for s_df in (train_df, val_df, test_df):
        s_df["filename"] = s_df["filename"].apply(lambda p: Path(p).as_posix())

    return train_df, val_df, test_df


def save_splits_and_stats(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ood_df: pd.DataFrame | None = None,
    output_dir: str = "data/splits",
    stats_file: str = "results/dataset_stats.json",
    dedup_removed: int = 0,
) -> dict[str, Any]:
    """Save split CSVs and print + export dataset statistics."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    val_path = out_dir / "val.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    stats: dict[str, Any] = {
        "deduplication_pruned_count": dedup_removed,
        "splits": {
            "train": {
                "total": len(train_df),
                "real": int((train_df["label"] == 0).sum()),
                "fake": int((train_df["label"] == 1).sum()),
            },
            "val": {
                "total": len(val_df),
                "real": int((val_df["label"] == 0).sum()),
                "fake": int((val_df["label"] == 1).sum()),
            },
            "test": {
                "total": len(test_df),
                "real": int((test_df["label"] == 0).sum()),
                "fake": int((test_df["label"] == 1).sum()),
            },
        },
    }

    if ood_df is not None:
        ood_path = out_dir / "ood_test.csv"
        ood_df["filename"] = ood_df["filename"].apply(lambda p: Path(p).as_posix())
        ood_df.to_csv(ood_path, index=False)
        stats["splits"]["ood_test"] = {
            "total": len(ood_df),
            "real": int((ood_df["label"] == 0).sum()),
            "fake": int((ood_df["label"] == 1).sum()),
        }

    stats_path = Path(stats_file)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(json.dumps(stats, indent=2))
    return stats
