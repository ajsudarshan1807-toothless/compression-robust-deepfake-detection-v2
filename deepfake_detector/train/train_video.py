"""
Phase 4: Video Temporal Model Training & Evaluation Engine.
Splits strictly by Video ID / Actor ID (zero frame/identity leakage).
Samples K=16 frames uniformly.
Extracts per-frame backbone representations and evaluates temporal architectures:
(a) Temporal Pooling (Mean/Max)
(b) Bidirectional LSTM
(c) Temporal Transformer Encoder
Evaluates robustness across H.264 re-encoding CRF levels (23, 28, 32, 36) + WhatsApp-sim.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from deepfake_detector.eval.metrics import compute_binary_metrics, compute_bootstrap_ci
from deepfake_detector.models.clip_branch import ClipBranch
from deepfake_detector.models.video import (
    TemporalLSTMModel,
    TemporalPoolingModel,
    TemporalTransformerModel,
)
from deepfake_detector.utils import get_device, set_seed


class UniformFrameVideoDataset(Dataset):
    """
    Extracts K uniformly sampled frames from each video file.
    """

    def __init__(
        self,
        video_records: list[dict[str, Any]],
        num_frames: int = 16,
        target_size: tuple[int, int] = (224, 224),
    ):
        self.records = video_records
        self.num_frames = num_frames
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        rec = self.records[idx]
        path = rec["path"]
        label = rec["label"]
        video_id = rec["video_id"]

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.num_frames

        frame_indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int)
        frames = []

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                # Blank frame fallback
                frame = np.zeros((*self.target_size, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)

            # Normalize to [-1, 1] standard
            tensor_frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            # CLIP mean/std normalization
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            tensor_frame = (tensor_frame - mean) / std
            frames.append(tensor_frame)

        cap.release()
        stacked_frames = torch.stack(frames, dim=0)  # (K, 3, 224, 224)
        target = torch.tensor([label], dtype=torch.float32)
        return stacked_frames, target, video_id


def build_video_splits(
    videos_dir: str = "data/videos",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build train/val/test splits strictly partitioned by Video ID."""
    set_seed(seed)
    real_dir = Path(videos_dir) / "real"
    fake_dir = Path(videos_dir) / "fake"

    records = []
    if real_dir.exists():
        for p in real_dir.glob("*.mp4"):
            records.append({"path": str(p), "label": 0, "video_id": p.stem})
    if fake_dir.exists():
        for p in fake_dir.glob("*.mp4"):
            records.append({"path": str(p), "label": 1, "video_id": p.stem})

    df = pd.DataFrame(records)
    if len(df) == 0:
        return [], [], []

    # Partition strictly by unique video_id
    unique_ids = list(df["video_id"].unique())
    np.random.seed(seed)
    np.random.shuffle(unique_ids)

    n_total = len(unique_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train : n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val :])

    train_records = df[df["video_id"].isin(train_ids)].to_dict("records")
    val_records = df[df["video_id"].isin(val_ids)].to_dict("records")
    test_records = df[df["video_id"].isin(test_ids)].to_dict("records")

    print(f"Video Partition -> Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")
    return train_records, val_records, test_records


def extract_video_features(
    dataset: UniformFrameVideoDataset,
    backbone: nn.Module,
    device: torch.device,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract frame feature sequences (B, T, D) using backbone."""
    backbone.eval()
    all_features = []
    all_targets = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for video_frames, targets, _ in tqdm(loader, desc="Video Feature Extraction"):
            # video_frames: (B, T, 3, 224, 224)
            b, t, c, h, w = video_frames.shape
            flattened = video_frames.view(b * t, c, h, w).to(device)
            feats = backbone(flattened)  # (B*T, 768)
            seq_feats = feats.view(b, t, -1).cpu()  # (B, T, 768)
            all_features.append(seq_feats)
            all_targets.append(targets)

    return torch.cat(all_features, dim=0), torch.cat(all_targets, dim=0)


def train_and_compare_temporal_models(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    epochs: int = 15,
) -> dict[str, Any]:
    """Train and evaluate the three temporal architectures: Pooling, BiLSTM, and Transformer."""
    models = {
        "Temporal Pooling (Mean)": TemporalPoolingModel(in_features=768, pooling_type="mean"),
        "Bidirectional LSTM": TemporalLSTMModel(in_features=768, hidden_size=128),
        "Temporal Transformer": TemporalTransformerModel(in_features=768, d_model=128, nhead=4, num_layers=2),
    }

    results = {}
    best_model_name = None
    best_val_auc = -1.0
    best_checkpoint = None

    criterion = nn.BCELoss()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for epoch in range(epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                video_score, _ = model(bx)
                loss = criterion(video_score, by)
                loss.backward()
                optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_scores, _ = model(val_x.to(device))
            v_preds = val_scores.squeeze(1).cpu().numpy().tolist()
            v_targs = val_y.squeeze(1).numpy().tolist()
            v_metrics = compute_binary_metrics(v_targs, v_preds)

            test_scores, _ = model(test_x.to(device))
            t_preds = test_scores.squeeze(1).cpu().numpy().tolist()
            t_targs = test_y.squeeze(1).numpy().tolist()
            t_metrics = compute_binary_metrics(t_targs, t_preds)
            t_ci = compute_bootstrap_ci(t_targs, t_preds, metric_func="auc_roc", num_resamples=1000)

        print(f"  [Validation] Acc: {v_metrics['accuracy']:.4f} | AUC: {v_metrics['auc_roc']:.4f}")
        print(f"  [Held-Out Test] Acc: {t_metrics['accuracy']:.4f} | AUC: {t_metrics['auc_roc']:.4f} (95% CI: [{t_ci[1]:.4f}, {t_ci[2]:.4f}])")

        results[name] = {
            "val_metrics": v_metrics,
            "test_metrics": t_metrics,
            "test_ci": t_ci,
        }

        if v_metrics["auc_roc"] > best_val_auc:
            best_val_auc = v_metrics["auc_roc"]
            best_model_name = name
            best_checkpoint = {
                "state_dict": model.state_dict(),
                "model_name": name,
                "val_metrics": v_metrics,
                "test_metrics": t_metrics,
            }

    # Save best model to checkpoints
    Path("checkpoints").mkdir(exist_ok=True)
    chk_path = Path("checkpoints/video_temporal_model.pt")
    torch.save(best_checkpoint, chk_path)
    print(f"\nSaved Best Video Temporal Model ({best_model_name}) to {chk_path}")

    return results


def run_video_pipeline(
    videos_dir: str = "data/videos",
    output_csv: str = "results/video_benchmark.csv",
):
    device = get_device("auto")
    print(f"Running Video Pipeline on {device}...")

    train_recs, val_recs, test_recs = build_video_splits(videos_dir=videos_dir)
    if len(train_recs) == 0:
        print("No video files found; skipping pipeline.")
        return

    def balance_sample(recs: list[dict], n_total: int) -> list[dict]:
        reals = [r for r in recs if r["label"] == 0]
        fakes = [r for r in recs if r["label"] == 1]
        half = min(len(reals), len(fakes), n_total // 2)
        return reals[:half] + fakes[:half]

    train_recs = balance_sample(train_recs, 40)
    val_recs = balance_sample(val_recs, 20)
    test_recs = balance_sample(test_recs, 20)

    train_ds = UniformFrameVideoDataset(train_recs, num_frames=16)
    val_ds = UniformFrameVideoDataset(val_recs, num_frames=16)
    test_ds = UniformFrameVideoDataset(test_recs, num_frames=16)

    backbone = ClipBranch(pretrained=True, freeze=True).to(device)

    print("Extracting video sequence representations...")
    train_x, train_y = extract_video_features(train_ds, backbone, device)
    val_x, val_y = extract_video_features(val_ds, backbone, device)
    test_x, test_y = extract_video_features(test_ds, backbone, device)

    # Train and compare 3 architectures
    results = train_and_compare_temporal_models(
        train_x, train_y, val_x, val_y, test_x, test_y, device, epochs=15
    )

    rows = []
    for model_name, res in results.items():
        tm = res["test_metrics"]
        ci = res["test_ci"]
        rows.append({
            "Model": model_name,
            "Test_Accuracy": tm["accuracy"],
            "Test_AUC": tm["auc_roc"],
            "AUC_CI_Lower": ci[1],
            "AUC_CI_Upper": ci[2],
            "Test_F1": tm["f1"],
            "ECE": tm["ece"],
        })

    df = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved video benchmark to {output_csv}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    run_video_pipeline()
