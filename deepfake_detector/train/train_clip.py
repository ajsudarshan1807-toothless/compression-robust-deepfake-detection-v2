"""
Phase 3: CLIP + LoRA Training and Temperature Calibration Engine.
Trains:
- 3A: Frozen CLIP ViT-B/32 + Linear Probe (cached features)
- 3B: CLIP + PEFT LoRA with On-The-Fly Stochastic Compression Augmentation
- 3C: Two-stream CLIP + Trained Frequency Branch
Fits post-hoc temperature scaling on validation set to minimize ECE.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from deepfake_detector.data.datasets import DeepfakeImageDataset, build_default_transform
from deepfake_detector.data.degradations import OnTheFlyDegradation
from deepfake_detector.eval.metrics import compute_binary_metrics, compute_ece
from deepfake_detector.models.clip_branch import ClipBranch, ClipDetector
from deepfake_detector.train.calibrate import calibrate_predictions
from deepfake_detector.utils import get_device, set_seed


def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown_commit"


def train_clip_probe_cached(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    device_name: str = "auto",
    epochs: int = 10,
    lr: float = 1e-3,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
) -> dict[str, Any]:
    """Train 3A (Linear Probe) by caching frozen CLIP features for ultra-fast training."""
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n--- Training Model 3A (CLIP Linear Probe) on {device} ---")

    transform = build_default_transform(image_size=224, is_clip=True)
    train_ds = DeepfakeImageDataset(train_csv, transform=transform)
    val_ds = DeepfakeImageDataset(val_csv, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print("Extracting frozen CLIP features for training set...")
    backbone = ClipBranch(pretrained=True, freeze=True).to(device)
    backbone.eval()

    def extract_all(loader: DataLoader):
        all_feats = []
        all_labels = []
        with torch.no_grad():
            for imgs, lbls in tqdm(loader, desc="Feature extraction"):
                imgs = imgs.to(device)
                feats = backbone(imgs)
                all_feats.append(feats.cpu())
                all_labels.append(lbls)
        return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)

    train_x, train_y = extract_all(train_loader)
    val_x, val_y = extract_all(val_loader)

    cached_train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    head = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid()).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_metrics = {}

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for bx, by in cached_train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = head(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        head.eval()
        with torch.no_grad():
            val_out = head(val_x.to(device))
            val_loss = criterion(val_out, val_y.to(device)).item()
            preds = val_out.squeeze(1).cpu().numpy().tolist()
            targs = val_y.squeeze(1).numpy().tolist()

        metrics = compute_binary_metrics(targs, preds)
        print(f"Epoch {epoch+1}/{epochs} -> Train Loss: {total_loss/len(train_x):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val AUC: {metrics['auc_roc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics

    # Assemble full Model 3A and calibrate
    full_model = ClipDetector(mode="linear_probe", pretrained=True).to(device)
    full_model.head.load_state_dict(head.state_dict())

    # Fit temperature scaling
    _, temp, ece_before, ece_after = calibrate_predictions(
        np.array(preds), np.array(targs)
    )
    best_metrics["temperature"] = temp
    best_metrics["ece_raw"] = ece_before
    best_metrics["ece_calibrated"] = ece_after
    print(f"  [Calibration] Fitted Temp T = {temp:.4f} | ECE: {ece_before:.4f} -> {ece_after:.4f}")

    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    save_path = chk_dir / "model_3a_clip_probe.pt"
    torch.save({
        "state_dict": full_model.state_dict(),
        "model_name": "model_3a_clip_probe",
        "mode": "linear_probe",
        "metrics": best_metrics,
        "temperature": temp,
        "git_commit": get_git_commit_hash(),
        "seed": seed,
    }, save_path)
    print(f"Saved Model 3A to {save_path}")
    return best_metrics


def train_clip_model(
    mode: str = "lora",
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    device_name: str = "auto",
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 16,
    use_degradation: bool = True,
    checkpoint_dir: str = "checkpoints",
    seed: int = 42,
) -> dict[str, Any]:
    """Train Model 3B (LoRA) or 3C (Two-stream) end-to-end with degradation augmentation."""
    set_seed(seed)
    device = get_device(device_name)
    model_name = f"model_3b_clip_{mode}" if mode == "lora" else "model_3c_clip_two_stream"
    print(f"\n--- Training {model_name} (Mode: {mode}, Aug: {use_degradation}) on {device} ---")

    transform = build_default_transform(image_size=224, is_clip=True)
    degradation = OnTheFlyDegradation(jpeg_prob=0.7, clean_prob=0.15) if use_degradation else None

    train_ds = DeepfakeImageDataset(train_csv, transform=transform, degradation=degradation)
    val_ds = DeepfakeImageDataset(val_csv, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ClipDetector(mode=mode, pretrained=True, lora_r=16).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_metrics = {}

    chk_dir = Path(checkpoint_dir)
    chk_dir.mkdir(parents=True, exist_ok=True)
    save_path = chk_dir / f"{model_name}.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for inputs, targets in train_pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(targets)
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * len(targets)
                all_probs.extend(outputs.squeeze(1).cpu().numpy().tolist())
                all_targets.extend(targets.squeeze(1).cpu().numpy().tolist())

        val_loss = val_loss / len(val_loader.dataset)
        metrics = compute_binary_metrics(all_targets, all_probs)

        print(
            f"Epoch {epoch+1}/{epochs} -> "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {metrics['accuracy']:.4f} | "
            f"Val AUC: {metrics['auc_roc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics
            best_metrics["epoch"] = epoch + 1
            best_metrics["val_loss"] = val_loss

            # Fit post-hoc temperature scaling
            _, temp, ece_before, ece_after = calibrate_predictions(
                np.array(all_probs), np.array(all_targets)
            )
            best_metrics["temperature"] = temp
            best_metrics["ece_raw"] = ece_before
            best_metrics["ece_calibrated"] = ece_after

            torch.save({
                "state_dict": model.state_dict(),
                "model_name": model_name,
                "mode": mode,
                "metrics": best_metrics,
                "temperature": temp,
                "git_commit": get_git_commit_hash(),
                "seed": seed,
            }, save_path)
            print(f"  [*] Saved checkpoint to {save_path} (T = {temp:.4f}, ECE: {ece_before:.4f} -> {ece_after:.4f})")

    return best_metrics


if __name__ == "__main__":
    train_clip_probe_cached()
    train_clip_model(mode="lora", epochs=2)
