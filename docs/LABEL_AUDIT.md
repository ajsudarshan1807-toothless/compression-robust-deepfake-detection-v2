# Label Audit Sign-Off (Human-in-the-Loop & Empirical Verification)

**Date:** 2026-10-13  
**Status:** Signed Off  
**Auditor:** DIP Project Engineering Team  
**Scope:** Dataset label mapping validation for `Hemg/AI-Generated-vs-Real-Images-Datasets` and `Thien0103/DeepFake_Extracted_Face_Images`.

---

## 1. Canonical Standard
All models, evaluation scripts, and APIs in this repository adhere strictly to:
- **`REAL = 0`** (Pristine, Authentic, Camera Capture, Original)
- **`FAKE = 1`** (AI-Generated, Deepfake, Synthetic, Manipulated)

---

## 2. Dataset Analysis & Audit Verdict

### Dataset A: `Hemg/AI-Generated-vs-Real-Images-Datasets`
- **HuggingFace Metadata Feature:** `ClassLabel(names=['AiArtData', 'RealArt'])`
- **Index 0 (`AiArtData`):**
  - *Visual Characteristics:* Synthetic digital illustrations, diffusion model outputs, rendering artifacts, unnatural textures.
  - *Audit Verdict:* **AI-Generated (FAKE)**
  - *Canonical Mapping:* `0 -> 1 (FAKE)`
- **Index 1 (`RealArt`):**
  - *Visual Characteristics:* Authentic real-world artworks, natural paintings, photographic captures.
  - *Audit Verdict:* **Authentic (REAL)**
  - *Canonical Mapping:* `1 -> 0 (REAL)`
- **Legacy Defect (D1) Confirmed:**
  - `src/data_prep.py` incorrectly treated `row['label'] == 0` as Real and saved to `data/processed/real/` with `label=0`.
  - This inverted the entire training dataset for Models A, C, and D.

---

### Dataset B: `Thien0103/DeepFake_Extracted_Face_Images`
- **HuggingFace Metadata Feature:** `ClassLabel(names=['fake', 'real'])`
- **Index 0 (`fake`):**
  - *Visual Characteristics:* Extracted facial regions containing facial manipulation, warping, boundary blending, and deepfake artifacts.
  - *Audit Verdict:* **Deepfake Face (FAKE)**
  - *Canonical Mapping:* `0 -> 1 (FAKE)`
- **Index 1 (`real`):**
  - *Visual Characteristics:* Unmanipulated natural human faces.
  - *Audit Verdict:* **Authentic Face (REAL)**
  - *Canonical Mapping:* `1 -> 0 (REAL)`

---

## 3. Automated Guardrail (Pytest Integration)

To prevent regression or accidental inversion:
- `tests/test_labels.py` programmatically asserts the canonical mapping for both datasets.
- The pipeline uses `deepfake_detector.data.labels.resolve_hf_feature_mapping()` at runtime rather than hardcoding numeric constants.

**Sign-off:** Approved.
