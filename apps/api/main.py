"""
FastAPI Inference Service for Compression-Robust Deepfake Detection.
Exposes REST endpoints for image analysis, robustness curves, and health status.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from deepfake_detector.infer.predictor import Predictor

app = FastAPI(
    title="Deepfake Detection API",
    description="Dual-stream spatial and frequency deepfake detection service with compression robustness.",
    version="1.0.0",
)

# CORS middleware for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy global predictor instance
predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global predictor
    if predictor is None:
        chk_path = Path("checkpoints/model_a_spatial_clean.pt")
        predictor = Predictor(checkpoint_path=chk_path if chk_path.exists() else None)
    return predictor


@app.get("/healthz")
def health_check() -> dict[str, str]:
    pred = get_predictor()
    return {
        "status": "healthy",
        "device": str(pred.device),
        "model_version": pred.model_version,
    }


@app.get("/v1/model-info")
def model_info() -> dict[str, Any]:
    pred = get_predictor()
    return {
        "model_version": pred.model_version,
        "device": str(pred.device),
        "temperature": pred.temperature,
        "thresholds": {
            "real_max": pred.threshold_real,
            "fake_min": pred.threshold_fake,
        },
        "description": "Dual-stream Spatial + Frequency ensemble with calibrated uncertainty intervals.",
    }


@app.post("/v1/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    generate_heatmap: bool = Query(True),
    generate_spectrum: bool = Query(True),
) -> dict[str, Any]:
    # Check max file size (25 MB)
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit of 25 MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file. Could not decode into RGB image.")

    pred = get_predictor()
    result = pred.analyze_image(
        image=image,
        generate_heatmap=generate_heatmap,
        generate_spectrum=generate_spectrum,
    )

    from dataclasses import asdict
    return asdict(result)


@app.post("/v1/robustness")
async def robustness_sweep(
    file: UploadFile = File(...),
) -> list[dict[str, Any]]:
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit of 25 MB.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    pred = get_predictor()
    return pred.sweep_robustness(image)
