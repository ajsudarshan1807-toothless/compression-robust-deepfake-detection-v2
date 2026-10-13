# Digital Image Processing (DIP) Syllabus Mapping

This document details how the **Compression-Robust Deepfake Detection Suite** maps directly to foundational concepts, algorithms, and mathematical formulations taught in university **Digital Image Processing (DIP)** curricula.

---

## 1. Frequency Domain Processing & 2D Discrete Fourier Transform (DFT)

### Theoretical Concepts:
- **Continuous & Discrete Fourier Transform**:
  $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left(\frac{ux}{M} + \frac{vy}{N}\right)}$$
- **Fourier Spectrum & Log-Magnitude**:
  $$|F(u, v)| = \sqrt{R^2(u, v) + I^2(u, v)}, \quad D(u, v) = c \log(1 + |F(u, v)|)$$
- **High-Frequency Generative Artifacts**: GANs and Diffusion models employ transposed convolutions and latent upsampling that leave regular periodic grid artifacts visible as discrete spikes/peaks in 2D Fourier space.

### Code Implementation:
- In `deepfake_detector/infer/predictor.py` (`compute_fft_spectrum`):
  Uses `numpy.fft.fft2` and `numpy.fft.fftshift` to calculate the centered 2D log-magnitude power spectrum of input images and render interactive heatmaps displaying high-frequency anomaly distributions.

---

## 2. Multi-Resolution Analysis & Wavelet Transforms

### Theoretical Concepts:
- **Haar Wavelet & Multi-Resolution Analysis (MRA)**:
  Decomposes an image into orthogonal frequency bands across dyadic scales using scaling functions $\phi(x)$ and wavelet functions $\psi(x)$.
- **Subband Decomposition**:
  - $LL$: Approximation subband (low-frequency content)
  - $LH$: Horizontal detail subband (high vertical frequencies)
  - $HL$: Vertical detail subband (high horizontal frequencies)
  - $HH$: Diagonal detail subband (corners and fine textures)

### Code Implementation:
- In `deepfake_detector/models/freq_branch.py` (`extract_haar_dwt_features`):
  Computes 2-level 2D Haar DWT using `pywt.wavedec2`. Extracts 9 high-frequency detail subbands ($LH_2, HL_2, HH_2 \times 3\text{ RGB channels}$) to feed a specialized convolutional neural network for frequency-domain boundary anomaly detection.

---

## 3. Image Compression & Transform Coding Fundamentals

### Theoretical Concepts:
- **JPEG Transform Coding Pipeline**:
  $$\text{Color Transform (RGB } \to \text{ YCbCr)} \to \text{Chroma Subsampling (4:2:0)} \to \text{Block Partition (8}\times\text{8)} \to \text{DCT} \to \text{Quantization} \to \text{Huffman Coding}$$
- **Quantization Table Attenuation**:
  $$B_{Q}(u, v) = \text{round}\left( \frac{\text{DCT}(u, v)}{Q(u, v)} \right)$$
  Higher frequencies have significantly larger divisors $Q(u, v)$, discarding fine texture boundaries where synthetic face-swap splices reside.
- **Double Compression & Social Media Re-encoding**:
  Applying JPEG compression twice with differing quality factors produces periodic grid misalignment artifacts.

### Code Implementation:
- In `deepfake_detector/data/degradations.py` (`OnTheFlyDegradation`):
  Implements live single- and double-JPEG compression across $QF \in [15, 95]$, chroma subsampling variations, and custom WhatsApp-simulation pipelines to train models invariant to quantization loss.

---

## 4. Spatial Domain Filtering & Convolution

### Theoretical Concepts:
- **Spatial Convolution**:
  $$g(x, y) = \sum_{s=-a}^{a} \sum_{t=-b}^{b} w(s, t) f(x + s, y + t)$$
- **Gaussian Smoothing & Defocus Simulation**:
  $$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
- **High-Pass Residual Extraction**:
  Subtracting a low-pass blurred image from the original image ($f - G * f$) exposes edge discontinuities and compression blocking artifacts.

### Code Implementation:
- In `deepfake_detector/data/degradations.py` and `deepfake_detector/infer/predictor.py`:
  Applies Gaussian blur kernels ($\sigma \in [0, 1.5]$) and computes spatial Grad-CAM activation maps over final convolutional and ViT attention layers (`pytorch-grad-cam`) to localize manipulated boundaries.

---

## 5. Summary Matrix of DIP Topics Implemented

| DIP Topic | Mathematical Operation | Implementation Module | UI Demonstration |
|---|---|---|---|
| **2D Fourier Transform** | 2D FFT, Spectral Centering, Log-Magnitude | `deepfake_detector/infer/predictor.py` | Frequency View Tab (2D FFT Spectrogram) |
| **Discrete Wavelet Transform** | 2-Level Haar DWT Subband Decomposition | `deepfake_detector/models/freq_branch.py` | Multi-Stream Diagnostics |
| **JPEG Quantization & Coding** | 8x8 DCT Quantization & Quality Factor Sweeps | `deepfake_detector/data/degradations.py` | Compression Lab (Live QF Slider 10-100) |
| **Spatial Filtering** | 2D Gaussian Kernel Convolution | `deepfake_detector/data/degradations.py` | Degradation Augmentation Pipeline |
| **Multi-Scale Analysis** | Dual-Stream Spatial + Frequency Fusion | `deepfake_detector/models/fusion.py` | Real vs Synthetic Verdict Dashboard |
