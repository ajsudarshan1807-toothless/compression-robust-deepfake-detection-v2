"""
Phase 3 Benchmark & Ablation Evaluation.
Compares Model 3A (Frozen CLIP + Linear Probe) vs Model 3B (CLIP + PEFT LoRA with CompAug)
across Clean, QF90, QF70, QF50, and QF30 on both Held-out Test and OOD Test sets.
Computes Delta AUC (Clean -> QF30) with 95% bootstrap confidence intervals.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from deepfake_detector.data.datasets import DeepfakeImageDataset, build_default_transform
from deepfake_detector.eval.metrics import compute_binary_metrics, compute_bootstrap_ci
from deepfake_detector.models.clip_branch import ClipDetector
from deepfake_detector.utils import get_device, safe_load_checkpoint


def eval_model(
    model: torch.nn.Module,
    split_csv: str,
    device: torch.device,
    qf: int | None = None,
    batch_size: int = 32,
) -> tuple[dict[str, float], tuple[float, float, float]]:
    model.eval()
    transform = build_default_transform(image_size=224, is_clip=True)
    dataset = DeepfakeImageDataset(split_csv, transform=transform, test_qf=qf)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = outputs.squeeze(1).cpu().numpy().tolist()
            targs = targets.squeeze(1).cpu().numpy().tolist()
            all_probs.extend(probs)
            all_targets.extend(targs)

    metrics = compute_binary_metrics(all_targets, all_probs)
    ci = compute_bootstrap_ci(all_targets, all_probs, metric_func="auc_roc", num_resamples=1000)
    return metrics, ci


def run_clip_evaluation(
    device_name: str = "auto",
    output_csv: str = "results/clip_benchmark.csv",
    output_plot: str = "results/clip_robustness.png",
):
    device = get_device(device_name)
    print(f"Running CLIP Benchmark on {device}...")

    # Load Model 3A
    print("Loading Model 3A (CLIP Linear Probe)...")
    model_3a = ClipDetector(mode="linear_probe", pretrained=False)
    data_3a = safe_load_checkpoint("checkpoints/model_3a_clip_probe.pt", map_location=device)
    model_3a.load_state_dict(data_3a["state_dict"])
    model_3a = model_3a.to(device)
    model_3a.eval()

    # Load Model 3B
    print("Loading Model 3B (CLIP PEFT LoRA)...")
    model_3b = ClipDetector(mode="lora", pretrained=False, lora_r=16)
    data_3b = safe_load_checkpoint("checkpoints/model_3b_clip_lora.pt", map_location=device)
    model_3b.load_state_dict(data_3b["state_dict"])
    model_3b = model_3b.to(device)
    model_3b.eval()

    models = {
        "Model 3A (CLIP Probe)": model_3a,
        "Model 3B (CLIP LoRA)": model_3b,
    }

    eval_splits = {
        "Held-Out Test": "data/splits/test.csv",
        "OOD Generator Test": "data/splits/ood_test.csv",
    }

    conditions = {
        "Clean": None,
        "QF90": 90,
        "QF70": 70,
        "QF50": 50,
        "QF30": 30,
    }

    rows = []
    delta_results = {}

    for split_name, split_path in eval_splits.items():
        print(f"\n================ EVALUATION ON: {split_name} ================")
        clean_aucs = {}

        for cond_name, qf in conditions.items():
            print(f"\n--- Condition: {cond_name} ---")
            for m_name, model in models.items():
                metrics, ci = eval_model(model, split_path, device, qf=qf)
                auc_point, ci_low, ci_high = ci

                if cond_name == "Clean":
                    clean_aucs[m_name] = auc_point

                row = {
                    "Split": split_name,
                    "Model": m_name,
                    "Compression": cond_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1": metrics["f1"],
                    "AUC-ROC": metrics["auc_roc"],
                    "AUC_CI_Lower": ci_low,
                    "AUC_CI_Upper": ci_high,
                    "ECE": metrics["ece"],
                }
                rows.append(row)
                print(f"  [{m_name}] Acc: {metrics['accuracy']:.4f} | AUC: {metrics['auc_roc']:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}]) | ECE: {metrics['ece']:.4f}")

                if cond_name == "QF30":
                    delta_auc = clean_aucs[m_name] - auc_point
                    delta_results[(split_name, m_name)] = delta_auc

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved benchmark results to {output_csv}")

    # Print Gate 3 Delta-AUC comparison
    print("\n================== GATE 3 DELTA-AUC COMPARISON ==================")
    for (split_name, m_name), delta in delta_results.items():
        print(f"{split_name} | {m_name} -> Delta AUC (Clean -> QF30): {delta:+.4f}")
    print("=================================================================\n")

    # Plot Robustness Curves
    cond_labels = list(conditions.keys())
    x = range(len(cond_labels))
    plt.figure(figsize=(10, 5))

    for m_name in models.keys():
        subset = df[(df["Model"] == m_name) & (df["Split"] == "Held-Out Test")]
        plt.plot(x, subset["Accuracy"], marker="o", label=f"{m_name} (Held-out)", linewidth=2)
        subset_ood = df[(df["Model"] == m_name) & (df["Split"] == "OOD Generator Test")]
        plt.plot(x, subset_ood["Accuracy"], marker="s", linestyle="--", label=f"{m_name} (OOD)", linewidth=2)

    plt.xticks(x, cond_labels, fontsize=11)
    plt.xlabel("Compression Condition", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("CLIP Detector Robustness: Model 3A (Probe) vs Model 3B (LoRA CompAug)", fontsize=13)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"Saved robustness plot to {output_plot}")

    return df


if __name__ == "__main__":
    run_clip_evaluation()
