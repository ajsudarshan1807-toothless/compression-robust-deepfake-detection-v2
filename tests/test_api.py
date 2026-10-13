"""
Unit and API contract tests for FastAPI inference service.
"""

import io
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from apps.api.main import app

client = TestClient(app)


def test_healthz_endpoint():
    response = client.get("/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "model_version" in data


def test_model_info_endpoint():
    response = client.get("/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "thresholds" in data
    assert data["thresholds"]["real_max"] == 0.35
    assert data["thresholds"]["fake_min"] == 0.65


def test_analyze_image_endpoint():
    # Create sample image
    img = Image.new("RGB", (224, 224), color=(120, 140, 160))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/v1/analyze/image",
        files={"file": ("sample.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "p_fake_calibrated" in data
    assert data["state"] in {"REAL", "UNCERTAIN", "FAKE"}
    assert "timings_ms" in data
    assert "caveat" in data


def test_analyze_invalid_image_error():
    # Pass corrupted bytes
    response = client.post(
        "/v1/analyze/image",
        files={"file": ("corrupted.jpg", b"invalid_binary_stream", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "Could not decode" in response.json()["detail"]


def test_robustness_sweep_endpoint():
    img = Image.new("RGB", (100, 100), color=(50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/v1/robustness",
        files={"file": ("sample.jpg", buf, "image/jpeg")},
    )
    assert response.status_code == 200
    curve = response.json()
    assert isinstance(curve, list)
    assert len(curve) > 0
    assert "qf" in curve[0]
    assert "p_fake" in curve[0]
