# Defect Confirmation Report (D1–D11)

**Date:** 2026-10-13  
**Branch:** `phase-0-structure`  
**Evaluation Scope:** Codebase audit and empirical defect confirmation for `compression-robust-deepfake-detection` according to `AGENT_PLAN.md` §1.1.

---

## Summary of Findings

| ID | Stated Defect | Status | Key Evidence |
|---|---|---|---|
| **D1** | Probable label inversion | **CONFIRMED** | `src/data_prep.py` mapped HF label 0 (`AiArtData` = fake) to Real (`label=0`) and 1 (`RealArt` = real) to Fake (`label=1`). Conversely, `extract_features.py` inverted labels with `label = 1 if item['label'] == 0 else 0`. Model Final AUC in `results/ablation_table.csv` is 0.0105 (classic signature of inverted evaluation). |
| **D2** | Committed results show no learning | **CONFIRMED** | In `results/ablation_table.csv`, Models A, C, and D have accuracies that are exact integer multiples of 1/30 (e.g., 14/30=0.467, 13/30=0.433, 16/30=0.533), demonstrating an evaluation over only 30 images with near-chance AUC (0.36–0.52). |
| **D3** | DWT/frequency branch never trained in CLIP pipeline | **CONFIRMED** | In `src/models/fusion_model.py`, `FrequencyBranch` is randomly initialized. In `extract_features.py`, features are extracted once with `model.eval()`, and only the MLP `fusion_head` is trained on pre-cached tensors. The frequency weights remain fixed random projections. |
| **D4** | On-the-fly compression augmentation is static | **CONFIRMED** | In `train_clip_final.py`, `RandomJPEGCompress` is applied once during initial feature extraction into RAM. Training then proceeds over static `TensorDataset` vectors across all epochs; the classifier never sees stochastic augmentations per epoch. |
| **D5** | "100% accuracy on WhatsApp" is train-on-test | **CONFIRMED** | `train_clip_finetune.py` (lines 22–33) directly ingests the 28 user-supplied files from `C:\Users\sudarshan\.gemini\...` and overfits on them for 5 epochs. `test_new_fakes.py` then tests on those exact same files. `project_summary.md` acknowledges catastrophic forgetting on clean/general data as a consequence. |
| **D6** | README claims contradict code implementation | **CONFIRMED** | README claims MTCNN face cropping in `data_prep.py` (code explicitly prints "Extracting full images (no face cropping)"); README claims `train.py` trains Models A–D (code trains only Model D); README/summary claims CLIP ViT-L/14 with last 6 layers unfrozen (code uses `openai/clip-vit-base-patch32` with `freeze=True` by default). |
| **D7** | Demo cannot run from a fresh clone | **CONFIRMED** | `app.py` attempts to load `checkpoints/model_i_whatsapp_robust.pt` and `video_temporal_model_whatsapp.pt`. `checkpoints/*.pt` is gitignored. No committed training script saves files with these filenames (`train_clip_finetune.py` saves `model_h_clip_robust.pt`, `train_video.py` saves `video_temporal_model.pt`). |
| **D8** | Portability issues (hard-coded paths, backslashes, missing packages) | **CONFIRMED** | Hard-coded Windows user paths in `train_clip_finetune.py` and `test_new_fakes.py`; `data/splits/train.csv` contains Windows backslashes (`data/processed/fake\7c67...jpg`); `requirements.txt` lacks `transformers`, `pyarrow`, `requests`, `huggingface_hub`. |
| **D9** | Biased inference rule (`max(global, face)`) | **CONFIRMED** | `app.py` line 265 executes `confidence_fake = max(score_global, score_face)` with an uncalibrated 0.5 threshold and no uncertain zone, heavily inflating the false positive rate on uncorrupted images. |
| **D10** | Repository hygiene & security defects | **CONFIRMED** | ~15 scratch scripts in root directory; no pytest test suite; `torch.load` calls lack `weights_only=True`; unseeded runs; brittle `sys.path.append('src')` imports throughout. |
| **D11** | Data integrity & shortcut learning risks | **CONFIRMED** | Raw images were resized to 224x224 before degradation in `src/data_prep.py`, destroying native camera sensor artifacts and JPEG grid structure prior to synthetic compression experiments. |

---

## Detailed Evidence & Verification Logs

### D1: Label Inversion
1. Inspection of HuggingFace dataset features:
   - `Hemg/AI-Generated-vs-Real-Images-Datasets`:
     `ClassLabel(names=['AiArtData', 'RealArt'])` -> Index `0` is `'AiArtData'` (Fake), Index `1` is `'RealArt'` (Real).
   - `Thien0103/DeepFake_Extracted_Face_Images`:
     `ClassLabel(names=['fake', 'real'])` -> Index `0` is `'fake'`, Index `1` is `'real'`.
2. Inspection of `src/data_prep.py`:
   - Line 59: `if row['label'] != 0: continue` -> saved to `data/processed/real` with record `'label': 0`.
   - Line 77: `if row['label'] != 1: continue` -> saved to `data/processed/fake` with record `'label': 1`.
   - **Verdict:** Inverted! Index 0 (`AiArtData` = fake) was labeled as 0 (Real).
3. Inspection of `results/ablation_table.csv`:
   - `Model Final, Clean`: Accuracy = `0.0567`, AUC-ROC = `0.0105`.
   - Inverting the classification decision yields Accuracy ~`94.33%`, AUC ~`0.9895`.

### D2: Committed Results Show No Learning
1. In `results/ablation_table.csv`:
   - Model A (Clean): Accuracy = 0.4666666666666667 (14 / 30)
   - Model C (Clean): Accuracy = 0.4333333333333333 (13 / 30)
   - Model D (Clean): Accuracy = 0.4333333333333333 (13 / 30)
   - Model A (QF70): Accuracy = 0.5666666666666667 (17 / 30)
   - Model C (QF30): Accuracy = 0.6000000000000000 (18 / 30)
   - Every single entry is an exact multiple of $1/30$, indicating that evaluation was executed on a trivial 30-sample slice rather than the 1,500-sample test set.

### D3: DWT / Frequency Branch is Never Trained
1. In `src/models/fusion_model.py`:
   - Line 45: `self.frequency = FrequencyBranch()` is initialized without pre-trained weights.
2. In `extract_features.py`:
   - Line 66–67: `model = FusionModelV2(pretrained_spatial=True, freeze_clip=True); model.eval()`
   - Lines 76–78: Spatial and frequency features are extracted once in inference mode and saved to disk.
   - Frequency CNN parameters are never updated via backpropagation.

### D4: Static "On-the-fly" Compression Augmentation
1. In `train_clip_final.py`:
   - Lines 40, 98–108: `RandomJPEGCompress` is evaluated once when loading images to create `features_tensor`.
   - Lines 122–129: The training loop operates on `TensorDataset(train_features, train_labels)`. No augmentations are generated across training epochs.

### D5: Train-on-Test Evaluation for WhatsApp Dataset
1. In `train_clip_finetune.py`:
   - Lines 22–33: Directly loads images from user directory `C:\Users\sudarshan\.gemini\antigravity-ide\brain\cf955e02-7d14-4b84-aa7e-1ddd56ed693b`.
2. In `test_new_fakes.py`:
   - Lines 30–45: Evaluates `model_h_clip_robust.pt` on the identical file paths.
3. In `project_summary.md`:
   - Section 5 documents catastrophic forgetting on the general dataset due to hyper-specialization on this 28-file set.

### D6: README Discrepancies
1. Face crops: README §1 claims MTCNN face cropping in `src/data_prep.py`. Line 44 of `src/data_prep.py` states `"Extracting full images (no face cropping for AI Generation detection)..."`.
2. Models trained: README §3 claims `src/train.py` trains Models A, B, C, D. `src/train.py` lines 130–133 only invokes `train_model` for Model D.
3. Backbone model: `project_summary.md` claims CLIP ViT-L/14; `src/models/clip_branch.py` line 11 instantiates `openai/clip-vit-base-patch32`.

### D7: Missing Checkpoints on Fresh Clone
1. `app.py` lines 176 and 185 load:
   - `checkpoints/model_i_whatsapp_robust.pt`
   - `checkpoints/video_temporal_model_whatsapp.pt`
2. Neither filename is produced by any training script (`train_video.py` produces `video_temporal_model.pt`; `train_clip_finetune.py` produces `model_h_clip_robust.pt`).
3. `.gitignore` lines 36–43 excludes all checkpoints (`checkpoints/*`, `*.pt`).

### D8: Portability & System Dependencies
1. Hard-coded local Windows absolute paths in `train_clip_finetune.py` and `test_new_fakes.py`.
2. Unnormalized file paths with Windows backslashes in `data/splits/train.csv`, `val.csv`, `test.csv`.
3. Incomplete dependencies in `requirements.txt` (omitted `transformers`, `pyarrow`, `requests`, `huggingface_hub`).
