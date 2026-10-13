# Datasets Documentation & Terms

**Last Updated:** 2026-10-13  
**Audience:** DIP Coursework & Research Reviewers

---

## 1. Datasets Used

### 1.1 `Hemg/AI-Generated-vs-Real-Images-Datasets`
- **Source:** HuggingFace Hub (`Hemg/AI-Generated-vs-Real-Images-Datasets`)
- **Domain:** General AI-generated imagery vs real artwork / natural images.
- **License / Terms:** Open research dataset on Hugging Face (Creative Commons / research usage).
- **Original Volume:** ~60,000 images partitioned in 6 Parquet files.
- **Role in Project:** Primary global background and spatial artifact training set.
- **Canonical Label Mapping:**
  - `AiArtData` (Index 0) -> `1 (FAKE)`
  - `RealArt` (Index 1) -> `0 (REAL)`

---

### 1.2 `Thien0103/DeepFake_Extracted_Face_Images`
- **Source:** HuggingFace Hub (`Thien0103/DeepFake_Extracted_Face_Images`)
- **Domain:** Cropped facial regions of deepfake manipulations vs pristine human faces.
- **License / Terms:** Permissive research / educational use.
- **Role in Project:** Specialized facial anomaly stream.
- **Canonical Label Mapping:**
  - `fake` (Index 0) -> `1 (FAKE)`
  - `real` (Index 1) -> `0 (REAL)`

---

### 1.3 WhatsApp Paired Evaluation Set (External Test Set)
- **Source:** User-collected media (28 files: 20 images, 8 videos).
- **Role in Project:** **External test evaluation only** (held out). Not used for general model training to prevent data leakage and catastrophic forgetting.
- **Evaluation Rule:** Evaluated strictly as an out-of-domain benchmark or via stratified cross-validation for qualitative demonstration.

---

## 2. Integrity and Preprocessing Guidelines
1. **No Early Resizing:** High-resolution native images are retained; degradations are applied to native pixel grids prior to resizing to $224 \times 224$ in the transform pipeline.
2. **Deduplication:** Perceptual hashing (`imagehash.phash`, Hamming distance $\le 6$) applied to guarantee zero overlap across train, validation, and test splits.
3. **Relative POSIX Paths:** All split CSV files use relative forward-slash paths (`data/processed/...`) for cross-platform Linux/Windows/CI compatibility.
