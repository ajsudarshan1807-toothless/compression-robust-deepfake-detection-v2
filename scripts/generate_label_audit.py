"""
Generate visual label audit grids for human inspection.
Dumps 24 random samples per class per dataset into results/label_audit/*.png.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from datasets import load_dataset
from deepfake_detector.data.labels import resolve_hf_feature_mapping, get_label_name


def make_grid(images: list[Image.Image], cols: int = 6, thumb_size: int = 150) -> Image.Image:
    """Combine images into a montage grid."""
    rows = (len(images) + cols - 1) // cols
    grid_img = Image.new("RGB", (cols * thumb_size, rows * thumb_size), color=(20, 20, 25))

    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        thumb = img.resize((thumb_size, thumb_size))
        grid_img.paste(thumb, (c * thumb_size, r * thumb_size))

    return grid_img


def run_audit_dump(output_dir: str = "results/label_audit", num_samples: int = 24):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Auditing Hemg/AI-Generated-vs-Real-Images-Datasets...")
    ds_hemg = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train")
    mapping_hemg = resolve_hf_feature_mapping(ds_hemg.features)
    print(f"Hemg raw features: {ds_hemg.features['label'].names} -> Mapping: {mapping_hemg}")

    for raw_label, canonical_label in mapping_hemg.items():
        class_name = ds_hemg.features["label"].names[raw_label]
        canonical_name = get_label_name(canonical_label)
        filtered = [x["image"].convert("RGB") for x in ds_hemg if x["label"] == raw_label][:num_samples]
        grid = make_grid(filtered)
        save_file = out_path / f"hemg_raw_{raw_label}_{class_name}_canonical_{canonical_name}.png"
        grid.save(save_file)
        print(f"Saved audit grid: {save_file}")

    print("\nAuditing Thien0103/DeepFake_Extracted_Face_Images...")
    ds_thien = load_dataset("Thien0103/DeepFake_Extracted_Face_Images", split="train")
    mapping_thien = resolve_hf_feature_mapping(ds_thien.features)
    print(f"Thien raw features: {ds_thien.features['label'].names} -> Mapping: {mapping_thien}")

    for raw_label, canonical_label in mapping_thien.items():
        class_name = ds_thien.features["label"].names[raw_label]
        canonical_name = get_label_name(canonical_label)
        filtered = [x["image"].convert("RGB") for x in ds_thien if x["label"] == raw_label][:num_samples]
        grid = make_grid(filtered)
        save_file = out_path / f"thien_raw_{raw_label}_{class_name}_canonical_{canonical_name}.png"
        grid.save(save_file)
        print(f"Saved audit grid: {save_file}")


if __name__ == "__main__":
    run_audit_dump()
