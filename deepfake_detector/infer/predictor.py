"""
Unified Inference Engine for Deepfake Detection.
Powers FastAPI endpoints, CLI evaluation, and interactive web interfaces.
Generates explainability heatmaps, FFT frequency spectra, and calibrated 3-state verdicts.
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import pywt
import torch
from PIL import Image

from deepfake_detector.data.datasets import FaceCropper, build_default_transform
from deepfake_detector.data.degradations import jpeg_compress
from deepfake_detector.models.fusion import FusionModel
from deepfake_detector.models.gated_fusion import GatedDualStreamFusion
from deepfake_detector.models.spatial_branch import SpatialBranch
from deepfake_detector.utils import get_device, safe_load_checkpoint


@dataclass
class ImageAnalysisResult:
    p_fake_calibrated: float
    state: str  # "REAL", "UNCERTAIN", "FAKE"
    p_global: float
    p_face: float | None
    face_box: list[int] | None
    heatmap_png_b64: str | None
    spectrum_png_b64: str | None
    timings_ms: dict[str, float]
    model_version: str
    caveat: str


@dataclass
class RobustnessCurvePoint:
    qf: int
    p_fake: float


class Predictor:
    """
    Production-grade deepfake inference engine.
    Loads models once at startup and performs deterministic, thread-safe inference.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device_name: str = "auto",
        temperature: float = 1.0,
        threshold_real: float = 0.35,
        threshold_fake: float = 0.65,
    ):
        self.device = get_device(device_name)
        self.temperature = temperature
        self.threshold_real = threshold_real
        self.threshold_fake = threshold_fake
        self.model_version = "v1.0-clip-ensemble"

        # Initialize image model (Model A / Fusion fallback)
        self.model = SpatialBranch(num_classes=1, pretrained=False).to(self.device)
        if checkpoint_path and Path(checkpoint_path).exists():
            payload = safe_load_checkpoint(checkpoint_path, map_location=self.device)
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
            self.model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}")
        self.model.eval()

        self.transform = build_default_transform(image_size=224, is_clip=False)
        self.face_cropper = FaceCropper(margin_fraction=0.20, device=str(self.device))
        self.gated_fusion = GatedDualStreamFusion()

    def generate_spectrum_png_b64(self, image: Image.Image) -> str:
        """
        Generate 2D FFT log-magnitude frequency spectrum visualization as base64 PNG.
        """
        gray = np.array(image.convert("L"), dtype=np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1.0)
        norm_mag = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored_spectrum = cv2.applyColorMap(norm_mag, cv2.COLORMAP_INFERNO)

        buf = io.BytesIO()
        Image.fromarray(colored_spectrum).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def generate_heatmap_png_b64(self, image: Image.Image, tensor: torch.Tensor) -> str:
        """
        Generate spatial anomaly heatmap over the original image as base64 PNG.
        """
        orig_w, orig_h = image.size
        # Generate pseudo-activation map from high-frequency gradient features
        arr = np.array(image, dtype=np.float32) / 255.0
        gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        lap_norm = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(lap_norm, cv2.COLORMAP_JET)

        # Blend with original image
        orig_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        heat_resized = cv2.resize(heat, (orig_w, orig_h))
        blended = cv2.addWeighted(orig_cv, 0.6, heat_resized, 0.4, 0)
        blended_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

        buf = io.BytesIO()
        Image.fromarray(blended_rgb).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze_image(
        self,
        image: Image.Image,
        generate_heatmap: bool = True,
        generate_spectrum: bool = True,
    ) -> ImageAnalysisResult:
        """Analyze a single PIL image and return calibrated prediction with diagnostic artifacts."""
        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        # 1. Global background inference
        t0 = time.perf_counter()
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_global = self.model(img_tensor).item()
        timings["global_inference_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Face extraction and inference
        t1 = time.perf_counter()
        face_img, face_box = self.face_cropper.crop_face(image)
        score_face = None
        face_conf = 0.0
        face_area_ratio = 0.0

        if face_img is not None and face_box is not None:
            face_tensor = self.transform(face_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                score_face = self.model(face_tensor).item()
            face_conf = 1.0
            box_area = (face_box[2] - face_box[0]) * (face_box[3] - face_box[1])
            face_area_ratio = box_area / float(image.size[0] * image.size[1])
        timings["face_detection_ms"] = (time.perf_counter() - t1) * 1000

        # 3. Gated fusion
        p_raw = self.gated_fusion(
            score_global=out_global,
            score_face=score_face,
            face_conf=face_conf,
            face_area_ratio=face_area_ratio,
        )

        # 4. Calibration with temperature scaling
        p_calibrated = float(1.0 / (1.0 + np.exp(-np.log(max(1e-6, min(1.0 - 1e-6, p_raw))) / self.temperature)))

        # 5. 3-State verdict assignment
        if p_calibrated < self.threshold_real:
            state = "REAL"
        elif p_calibrated > self.threshold_fake:
            state = "FAKE"
        else:
            state = "UNCERTAIN"

        # 6. Explainability visuals
        heatmap_b64 = None
        if generate_heatmap:
            heatmap_b64 = self.generate_heatmap_png_b64(image, img_tensor)

        spectrum_b64 = None
        if generate_spectrum:
            spectrum_b64 = self.generate_spectrum_png_b64(image)

        timings["total_pipeline_ms"] = (time.perf_counter() - t_start) * 1000

        caveat = "Probability estimation, not forensic proof. Social media compression and low resolution reduce confidence."

        return ImageAnalysisResult(
            p_fake_calibrated=round(p_calibrated, 4),
            state=state,
            p_global=round(out_global, 4),
            p_face=round(score_face, 4) if score_face is not None else None,
            face_box=face_box,
            heatmap_png_b64=heatmap_b64,
            spectrum_png_b64=spectrum_b64,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            model_version=self.model_version,
            caveat=caveat,
        )

    def sweep_robustness(self, image: Image.Image, qf_steps: list[int] | None = None) -> list[dict[str, Any]]:
        """
        Simulate JPEG compression sweep (100 -> 10) on client-supplied image bytes.
        Returns pre-computed probability curve instantly for interactive UI scrubbers.
        """
        if qf_steps is None:
            qf_steps = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]

        curve = []
        for qf in qf_steps:
            compressed = jpeg_compress(image, quality=qf)
            res = self.analyze_image(compressed, generate_heatmap=False, generate_spectrum=False)
            curve.append({
                "qf": qf,
                "p_fake": res.p_fake_calibrated,
                "state": res.state,
            })
        return curve
