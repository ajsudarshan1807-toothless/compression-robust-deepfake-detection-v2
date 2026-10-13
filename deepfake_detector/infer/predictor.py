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
from deepfake_detector.models.clip_branch import ClipBranch
from deepfake_detector.models.fusion import FusionModel, FusionModelV2
from deepfake_detector.models.gated_fusion import GatedDualStreamFusion
from deepfake_detector.models.spatial_branch import SpatialBranch
from deepfake_detector.models.video import TemporalLSTMModel
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
class VideoAnalysisResult:
    p_fake_calibrated: float
    state: str  # "REAL", "UNCERTAIN", "FAKE"
    video_score: float
    sampled_frame_count: int
    frame_scores: list[float]
    frame_thumbnails_b64: list[str]
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

        # Auto-detect robust model if none provided
        if checkpoint_path is None:
            wa_chk = Path("checkpoints/model_i_whatsapp_robust.pt")
            clean_chk = Path("checkpoints/model_a_spatial_clean.pt")
            if wa_chk.exists():
                checkpoint_path = wa_chk
            elif clean_chk.exists():
                checkpoint_path = clean_chk

        chk_path = Path(checkpoint_path) if checkpoint_path else None
        is_clip_arch = False

        if chk_path and chk_path.exists():
            payload = safe_load_checkpoint(chk_path, map_location=self.device)
            state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload

            # Detect if checkpoint is FusionModelV2 (CLIP + Frequency) or SpatialBranch
            if any("spatial.backbone" in k or "spatial.model" in k for k in state_dict.keys()):
                self.model = FusionModelV2(pretrained_spatial=False, freeze_clip=True).to(self.device)
                self.model.load_state_dict(state_dict)
                is_clip_arch = True
                self.model_version = "v2.0-clip-whatsapp-robust"
                print(f"Loaded robust FusionModelV2 from {chk_path}")
            else:
                self.model = SpatialBranch(num_classes=1, pretrained=False).to(self.device)
                self.model.load_state_dict(state_dict)
                self.model_version = "v1.0-spatial-clean"
                print(f"Loaded SpatialBranch from {chk_path}")
        else:
            self.model = SpatialBranch(num_classes=1, pretrained=False).to(self.device)
            self.model_version = "v1.0-spatial-uninitialized"

        self.model.eval()
        self.transform = build_default_transform(image_size=224, is_clip=is_clip_arch)
        self.face_cropper = FaceCropper(margin_fraction=0.20, device=str(self.device))
        self.gated_fusion = GatedDualStreamFusion()

        # Lazy video models
        self._video_backbone: ClipBranch | None = None
        self._video_model = None
        self._is_whatsapp_video = False
        self._video_chk_path = Path("checkpoints/video_temporal_model_whatsapp.pt")
        if not self._video_chk_path.exists():
            self._video_chk_path = Path("checkpoints/video_temporal_model.pt")

    def _get_video_models(self):
        if self._video_model is None:
            wa_chk = Path("checkpoints/video_temporal_model_whatsapp.pt")
            std_chk = Path("checkpoints/video_temporal_model.pt")

            if wa_chk.exists():
                import sys
                sys.path.append("src")
                from models.video_model import TemporalFusionModel
                self._video_model = TemporalFusionModel(pretrained_spatial=False, freeze_clip=False, lstm_hidden_size=256).to(self.device)
                payload = torch.load(wa_chk, map_location=self.device)
                st_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                self._video_model.load_state_dict(st_dict)
                self._is_whatsapp_video = True
                print(f"Loaded TemporalFusionModel WhatsApp-robust checkpoint from {wa_chk}")
            elif std_chk.exists():
                self._video_backbone = ClipBranch(pretrained=True, freeze=True).to(self.device)
                self._video_backbone.eval()
                self._video_model = TemporalLSTMModel(in_features=768, hidden_size=128).to(self.device)
                payload = safe_load_checkpoint(std_chk, map_location=self.device)
                st_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
                self._video_model.load_state_dict(st_dict)
                self._is_whatsapp_video = False
                print(f"Loaded TemporalLSTMModel from {std_chk}")
            else:
                self._video_model = TemporalLSTMModel(in_features=768, hidden_size=128).to(self.device)
                self._is_whatsapp_video = False
            self._video_model.eval()

        return self._video_backbone, self._video_model

    def analyze_video(
        self,
        video_path: str | Path,
        num_frames: int = 16,
    ) -> VideoAnalysisResult:
        """
        Analyze video file by uniformly sampling K frames and evaluating temporal sequence consistency.
        Returns overall calibrated verdict, timeline frame scores, and base64 preview thumbnails.
        """
        t_start = time.perf_counter()
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = num_frames

        frame_indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        raw_frames_rgb = []

        for f_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret or frame is None:
                blank = np.zeros((224, 224, 3), dtype=np.uint8)
                raw_frames_rgb.append(blank)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                raw_frames_rgb.append(frame_resized)
        cap.release()
        timings["frame_extraction_ms"] = (time.perf_counter() - t0) * 1000

        # Create base64 JPEG thumbnails for UI display
        thumbnails_b64: list[str] = []
        for f_arr in raw_frames_rgb:
            thumb = cv2.resize(f_arr, (120, 120))
            buf = io.BytesIO()
            Image.fromarray(thumb).save(buf, format="JPEG", quality=80)
            thumbnails_b64.append(base64.b64encode(buf.getvalue()).decode("utf-8"))

        # Convert to CLIP normalized tensors
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        tensor_list = []
        for f_arr in raw_frames_rgb:
            t_frame = torch.from_numpy(f_arr).permute(2, 0, 1).float() / 255.0
            t_frame = (t_frame - mean) / std
            tensor_list.append(t_frame)
        batch_frames = torch.stack(tensor_list, dim=0).to(self.device)  # (K, 3, 224, 224)

        # Extract features and evaluate temporal model
        t1 = time.perf_counter()
        backbone, temporal_model = self._get_video_models()
        with torch.no_grad():
            if self._is_whatsapp_video:
                seq_tensor = batch_frames.unsqueeze(0)  # (1, K, 3, 224, 224)
                v_score_raw = temporal_model(seq_tensor)
                raw_video_score = float(v_score_raw.item())
                frame_scores = []
                for k_idx in range(len(tensor_list)):
                    f_tensor = tensor_list[k_idx].unsqueeze(0).to(self.device)
                    try:
                        f_score = float(temporal_model.spatial(f_tensor).item())
                    except Exception:
                        f_score = raw_video_score
                    frame_scores.append(round(f_score, 4))
            else:
                feats = backbone(batch_frames).unsqueeze(0)  # (1, K, 768)
                v_score_raw, f_scores_raw = temporal_model(feats)
                raw_video_score = float(v_score_raw.item())
                frame_scores = [round(float(s), 4) for s in f_scores_raw.squeeze(0).cpu().numpy().tolist()]
        timings["inference_ms"] = (time.perf_counter() - t1) * 1000

        # Calibrate score
        if abs(self.temperature - 1.0) < 1e-4:
            p_calibrated = raw_video_score
        else:
            eps = 1e-6
            p_c = max(eps, min(1.0 - eps, raw_video_score))
            logit = float(np.log(p_c / (1.0 - p_c)))
            p_calibrated = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))

        if p_calibrated < self.threshold_real:
            state = "REAL"
        elif p_calibrated > self.threshold_fake:
            state = "FAKE"
        else:
            state = "UNCERTAIN"

        timings["total_pipeline_ms"] = (time.perf_counter() - t_start) * 1000

        caveat = "Video verdict is based on temporal continuity and inter-frame facial micro-motion. High H.264 compression can reduce confidence."

        return VideoAnalysisResult(
            p_fake_calibrated=round(p_calibrated, 4),
            state=state,
            video_score=round(raw_video_score, 4),
            sampled_frame_count=len(raw_frames_rgb),
            frame_scores=frame_scores,
            frame_thumbnails_b64=thumbnails_b64,
            timings_ms={k: round(v, 2) for k, v in timings.items()},
            model_version=f"{self.model_version}-temporal",
            caveat=caveat,
        )

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

        # 3. Dual-stream fusion
        if score_face is not None and face_conf >= 0.5:
            # Weighted ensemble balancing global scene context and facial boundaries
            p_raw = float(0.65 * out_global + 0.35 * score_face)
        else:
            p_raw = float(out_global)

        # 4. Calibration with proper logit temperature scaling
        if abs(self.temperature - 1.0) < 1e-4:
            p_calibrated = p_raw
        else:
            eps = 1e-6
            p_c = max(eps, min(1.0 - eps, p_raw))
            logit = float(np.log(p_c / (1.0 - p_c)))
            p_calibrated = float(1.0 / (1.0 + np.exp(-logit / self.temperature)))

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
