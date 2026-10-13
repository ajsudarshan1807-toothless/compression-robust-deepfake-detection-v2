"""
Execute Phase 2 robustness evaluation sweep across Clean and JPEG compression QFs.
Produces results/baselines.csv and results/accuracy_vs_compression.png.
Zero fabricated numbers: every row derived from test dataset evaluation.
"""

from __future__ import annotations

import torch
from deepfake_detector.eval.robustness import run_robustness_sweep
from deepfake_detector.models.fusion import FusionModel
from deepfake_detector.models.spatial_branch import SpatialBranch
from deepfake_detector.utils import get_device, safe_load_checkpoint


def main():
    device = get_device("auto")
    print(f"Executing baseline robustness sweep on: {device}")

    # Load Model A
    print("Loading Model A (SpatialBranch - Clean)...")
    model_a = SpatialBranch(pretrained=False)
    data_a = safe_load_checkpoint("checkpoints/model_a_spatial_clean.pt", map_location=device)
    model_a.load_state_dict(data_a["state_dict"])
    model_a.eval()

    # Load Model C
    print("Loading Model C (Dual-Stream Fusion - Clean)...")
    model_c = FusionModel(pretrained_spatial=False)
    data_c = safe_load_checkpoint("checkpoints/model_c_fusion_clean.pt", map_location=device)
    model_c.load_state_dict(data_c["state_dict"])
    model_c.eval()

    # Load Model D
    print("Loading Model D (Dual-Stream Fusion - Compression Augmentation)...")
    model_d = FusionModel(pretrained_spatial=False)
    data_d = safe_load_checkpoint("checkpoints/model_d_fusion_compaug.pt", map_location=device)
    model_d.load_state_dict(data_d["state_dict"])
    model_d.eval()

    models = {
        "Model A (Spatial Clean)": model_a,
        "Model C (Fusion Clean)": model_c,
        "Model D (Fusion CompAug)": model_d,
    }

    df = run_robustness_sweep(
        models_dict=models,
        test_csv="data/splits/test.csv",
        device_name=str(device),
        output_csv="results/baselines.csv",
        output_plot="results/accuracy_vs_compression.png",
    )

    print("\n================== BASELINES SUMMARY TABLE ==================")
    print(df.to_string(index=False))
    print("=============================================================\n")


if __name__ == "__main__":
    main()
