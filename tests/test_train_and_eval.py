"""
Unit tests for metrics calculation, bootstrap confidence intervals, and sanity gates.
"""

import numpy as np
import pytest
import torch

from deepfake_detector.eval.metrics import compute_binary_metrics, compute_bootstrap_ci, compute_ece
from deepfake_detector.models.spatial_branch import SpatialBranch
from deepfake_detector.train.train_image import (
    run_overfit_one_batch_test,
    run_shuffled_label_control_test,
)


def test_compute_binary_metrics():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]
    metrics = compute_binary_metrics(y_true, y_prob)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auc_roc"] == 1.0


def test_compute_ece():
    y_true = np.array([0, 1])
    y_prob = np.array([0.0, 1.0])
    ece = compute_ece(y_true, y_prob, num_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-5)


def test_bootstrap_ci():
    y_true = [0] * 50 + [1] * 50
    y_prob = [0.1] * 50 + [0.9] * 50
    point, lower, upper = compute_bootstrap_ci(y_true, y_prob, num_resamples=100)

    assert lower <= point <= upper
    assert point == 1.0


def test_overfit_one_batch_gate():
    model = SpatialBranch(num_classes=1, pretrained=False)
    device = torch.device("cpu")
    passed = run_overfit_one_batch_test(model, device, batch_size=8)
    assert passed is True


def test_shuffled_label_control_gate():
    model = SpatialBranch(num_classes=1, pretrained=False)
    device = torch.device("cpu")
    auc = run_shuffled_label_control_test(model, device, batch_size=8, num_samples=32)
    assert 0.30 <= auc <= 0.70
