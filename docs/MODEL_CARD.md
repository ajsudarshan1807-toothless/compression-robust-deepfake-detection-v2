# Model Card: Compression-Robust Deepfake Detection Suite

## Model Details
- **Architecture**: Multi-Scale Dual-Stream Ensemble:
  - **Spatial Stream**: EfficientNet-B0 (ImageNet normalized) + CLIP ViT-B/32 Vision Transformer with PEFT LoRA attention adapters ($r=16, \alpha=32$ on $q_\text{proj}, v_\text{proj}$).
  - **Frequency Stream**: 2-Level Haar Discrete Wavelet Transform (DWT) 9-channel subband decomposition with 4-layer CNN feature extractor.
  - **Temporal Sequence Stream**: Uniform 16-frame sequence extraction with Bidirectional LSTM and Temporal Transformer.
  - **Calibration Engine**: Post-hoc Temperature Scaling ($T=1.0084$) optimizing Expected Calibration Error (ECE) under a 3-state confidence gate:
    - $\text{Authentic (Real)}: p < 0.35$
    - $\text{Uncertain Band}: 0.35 \le p \le 0.65$
    - $\text{Manipulated (Fake)}: p > 0.65$
- **Developers**: Sudarshan
- **Version**: 1.0.0
- **License**: MIT License
- **Framework**: PyTorch 2.1+, HuggingFace Transformers, PEFT, OpenCV, PyWavelets, FastAPI.

---

## Intended Use
- **Primary Use**: Academic research and digital image forensics exploration investigating how social media re-compression (JPEG quality factors 30–90, H.264 video compression) degrades synthetic generative image artifacts.
- **Intended Audience**: Computer vision researchers, digital image processing students, media literacy researchers.
- **Out-of-Scope Use Cases**:
  - Unsupervised legal determination or judicial evidence without human forensic expert corroboration.
  - Punitive, law-enforcement, or disciplinary automated action against individuals.
  - Autonomous content censorship.

---

## Training Data & Augmentation Provenance
- **Curated Datasets**:
  - `Hemg/AI-Generated-vs-Real-Images-Datasets` (General scenes: Stable Diffusion, Midjourney, DALL-E vs real photographs)
  - `Thien0103/DeepFake_Extracted_Face_Images` (Facial manipulations vs authentic faces)
  - `angads24/deepfake-video` (Disjoint actor video clips)
- **Label Standard**: Canonical mapping `REAL = 0`, `FAKE = 1` validated via human audit grid and runtime `ClassLabel.names` inspection (resolving Defect D1).
- **Deduplication**: Perceptual hashing (`imagehash` pHash, Hamming distance $\le 6$) across all splits to eliminate train-test leakage.
- **Live Stochastic Degradation Pipeline**:
  - Single and double JPEG re-compression with $QF \sim U[15, 95]$ ($p=0.70$).
  - Bilinear/bicubic downscaling ($0.30 - 0.90\times$) and upscaling.
  - Gaussian defocus blur ($\sigma \in [0, 1.5]$) and additive sensor noise.
  - Simulated WhatsApp re-compression pipeline.

---

## Empirical Evaluation Summary

| Model / Architecture | Clean AUC | QF50 AUC | QF30 AUC | $\Delta\text{AUC}$ Drop | Test ECE |
|---|---|---|---|---|---|
| **Model A (EfficientNet-B0 Clean)** | 0.9730 | 0.9633 | 0.9401 | +0.0329 | 0.0938 |
| **Model C (Dual-Stream Clean)** | 0.9810 | 0.9607 | 0.9363 | +0.0447 | 0.0614 |
| **Model D (Dual-Stream CompAug)** | 0.9670 | 0.9607 | 0.9451 | +0.0219 | 0.0530 |
| **Model 3A (CLIP Linear Probe)** | 0.9595 | 0.9568 | 0.9317 | +0.0278 | 0.0526 |
| **Model 3B (CLIP LoRA CompAug)** | **0.9723** | **0.9675** | **0.9525** | **+0.0198** | **0.0396** |

### Out-of-Distribution (OOD) Generator Generalization
- Under unseen generative architectures at severe compression (QF30), **Model 3B maintains 91.25% Accuracy and 0.9717 AUC**, confirming strong cross-generator robustness.

---

## Ethical Considerations & Forensic Limitations
1. **Probabilistic Uncertainty**: Deepfake detectors output statistical likelihoods of synthetic frequency and spatial patterns. A prediction of 0.85 indicates similarity to known generative patterns, not incontrovertible proof of human intent.
2. **Compression Trade-offs**: Severe compression (e.g. repeated WhatsApp/Telegram forwarding at QF $\le 30$) attenuates high-frequency spectral cues. While Model 3B and Model D demonstrate improved resilience, high compression increases the proportion of uncertain verdicts.
3. **Continuous Domain Drift**: Generative models evolve rapidly; models trained on diffusion or GAN generators require continual adaptation against novel synthesis pipelines.
