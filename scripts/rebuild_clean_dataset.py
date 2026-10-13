"""
Rebuild the dataset properly with verified canonical labels:
- REAL = 0 (from train_chunk_5.parquet, RealArt)
- FAKE = 1 (from train_chunk_0.parquet, AiArtData)
- Retain native resolution where possible
- Deduplicate with SHA1 and pHash (Hamming <= 6)
- Generate stratified splits with relative POSIX paths
- Export results/dataset_stats.json
"""

import io
from pathlib import Path
import uuid
from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm

from deepfake_detector.data.labels import FAKE, REAL
from deepfake_detector.data.splits import (
    compute_phash,
    compute_sha1,
    create_stratified_splits,
    deduplicate_records,
    save_splits_and_stats,
)


def rebuild(
    target_samples_per_class: int = 1500,
    raw_chunk_0: str = "data/raw/train_chunk_0.parquet",
    raw_chunk_5: str = "data/raw/train_chunk_5.parquet",
    out_dir_real: str = "data/processed/real",
    out_dir_fake: str = "data/processed/fake",
):
    path_real_dir = Path(out_dir_real)
    path_fake_dir = Path(out_dir_fake)
    path_real_dir.mkdir(parents=True, exist_ok=True)
    path_fake_dir.mkdir(parents=True, exist_ok=True)

    records = []

    print(f"Loading REAL images from {raw_chunk_5} (RealArt, raw label 1 -> canonical REAL=0)...")
    t5 = pq.read_table(raw_chunk_5)
    df5 = t5.to_pandas()
    real_count = 0

    for _, row in tqdm(df5.iterrows(), total=min(len(df5), target_samples_per_class * 2), desc="Extracting Real"):
        if real_count >= target_samples_per_class:
            break
        try:
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            sha1 = compute_sha1(img_bytes)
            phash = compute_phash(img)

            filename = f"real_{uuid.uuid4().hex[:16]}.jpg"
            save_path = path_real_dir / filename
            img.save(save_path, format="JPEG", quality=95)

            records.append({
                "filename": save_path.as_posix(),
                "label": REAL, # Canonical 0
                "source": "Hemg_RealArt",
                "sha1": sha1,
                "phash": phash,
            })
            real_count += 1
        except Exception:
            continue

    print(f"Loading FAKE images from {raw_chunk_0} (AiArtData, raw label 0 -> canonical FAKE=1)...")
    t0 = pq.read_table(raw_chunk_0)
    df0 = t0.to_pandas()
    fake_count = 0

    for _, row in tqdm(df0.iterrows(), total=min(len(df0), target_samples_per_class * 2), desc="Extracting Fake"):
        if fake_count >= target_samples_per_class:
            break
        try:
            img_bytes = row["image"]["bytes"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            sha1 = compute_sha1(img_bytes)
            phash = compute_phash(img)

            filename = f"fake_{uuid.uuid4().hex[:16]}.jpg"
            save_path = path_fake_dir / filename
            img.save(save_path, format="JPEG", quality=95)

            records.append({
                "filename": save_path.as_posix(),
                "label": FAKE, # Canonical 1
                "source": "Hemg_AiArtData",
                "sha1": sha1,
                "phash": phash,
            })
            fake_count += 1
        except Exception:
            continue

    print(f"Extracted {real_count} real and {fake_count} fake images. Deduplicating...")
    deduped_records, removed_count = deduplicate_records(records, max_hamming_dist=6)
    print(f"Deduplication removed {removed_count} near-duplicate samples. Remaining: {len(deduped_records)}")

    import pandas as pd
    df = pd.DataFrame(deduped_records)

    # Hold out 10% for OOD test
    ood_df = df.sample(frac=0.10, random_state=42).reset_index(drop=True)
    main_df = df.drop(ood_df.index).reset_index(drop=True)

    train_df, val_df, test_df = create_stratified_splits(main_df, train_frac=0.70, val_frac=0.15, test_frac=0.15, seed=42)

    stats = save_splits_and_stats(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        ood_df=ood_df,
        output_dir="data/splits",
        stats_file="results/dataset_stats.json",
        dedup_removed=removed_count,
    )
    print("Rebuild complete!")
    return stats


if __name__ == "__main__":
    rebuild(target_samples_per_class=1200)
