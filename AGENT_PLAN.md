# AGENT_PLAN.md — Compression-Robust Deepfake Detection: Improvement Plan

Audience: an AI coding agent (Claude Code / Cursor / Copilot-agent / etc.) working inside the repo
`ajsudarshan1807-toothless/compression-robust-deepfake-detection` (DIP semester-5 project).
Written 2026-10-13 from a full read of the repo. Nothing in the repo was executed when this was written,
so every "bug" below is *read from code* and must be confirmed by you (Phase 1) before fixing.

---------------------------------------------------------------------------------------------------
## 0. Operating rules (read first, obey throughout)

1. **Work one phase at a time.** Each phase ends with a GATE. Do not start the next phase until the gate passes. If a gate fails, stop and report why.
2. **Never fabricate numbers.** Every metric in any README/report/figure must come from code you actually ran. Save raw outputs (CSV/JSON) under `results/` with the git commit hash and seed.
3. **Tests first** where practical (pytest). Create a failing test that demonstrates each bug, then fix it.
4. **Never train and evaluate on the same data.** The 28-file WhatsApp set is an *external test set*, not a training set (see Phase 5 for how to split it honestly).
5. **Work on a branch per phase** (`phase-1-data`, `phase-2-baselines`, ...). Small commits, clear messages.
6. **License discipline.** When adapting code from a public repo, keep its license + attribution in `THIRD_PARTY.md`, and rewrite to this repo's style. No blind copy-paste.
7. **Do not commit** weights, datasets, or videos. Provide scripts that download them (HF hub, `gdown`, etc.).
8. Prefer **small, verifiable steps** over big rewrites. Keep the CPU path working (author has no GPU in the repo notes; the owner now has an RTX 4060 8 GB — support both, auto-select device).

---------------------------------------------------------------------------------------------------
## 1. Ground truth: what the repo is today

Pipeline evolution (all in one flat repo):
`EfficientNet-B0 + Haar-DWT CNN (planned Models A/B/C/D)` → `HF AI-vs-Real dataset (5k+5k, 224x224, no face crop)` → `CLIP ViT-B/32 (768-d, frozen) + DWT-CNN (128-d) -> MLP head on cached features` → `WhatsApp fine-tune on 28 files` → `video: 10 frames -> CLIP+DWT features -> LSTM(256)` → `Streamlit app "Deepfake Shield"`.

### 1.1 Defects to confirm and fix (ordered by impact)

| # | Defect (evidence) | Why it matters |
|---|---|---|
| D1 | **Probable label inversion.** `src/data_prep.py` maps HF label 0 → *real* (label 0) and 1 → *fake*. Scripts `train_clip_*.py`, `extract_features.py` map `label==0 → 1 (fake)`; a comment in `train_clip_combined.py` says "0 = AiArtData (Fake), 1 = RealArt (Real)". `results/ablation_table.csv` "Model Final" rows show accuracy ≈ 5–8 % with AUC ≈ 0.01–0.02 — far *below* chance, the classic signature of inverted labels. | Every downstream number is suspect. |
| D2 | **Committed results show no learning.** Models A/C/D sit at 40–60 % accuracy, AUC ≈ 0.4–0.6 (chance). All accuracies are multiples of 1/30 → they came from a 30-image run, not the 1,500-image `test.csv`. | No valid evidence the system works. |
| D3 | **The DWT/frequency branch is never trained in the CLIP pipeline.** `extract_features.py`/`train_clip_final.py` call `model.frequency(...)` with random init and train only `fusion_head` on the cached 896-d vectors. Same in `video_model.py`/`train_video.py` (only LSTM + head train). The "128 frequency dims" are a fixed random projection. | The project's headline idea (frequency artefacts) is not actually being tested. |
| D4 | **"On-the-fly compression augmentation" isn't on-the-fly for the CLIP route.** Compression is sampled once while caching features; the head never sees fresh augmentations per epoch. | Robustness claim unsupported. |
| D5 | **"100 % accuracy on WhatsApp" is train-on-test.** Fine-tuned on the same 28 files it is scored on (20 images, 8 videos). `project_summary.md` itself reports catastrophic forgetting on the general set. | Do not present as a result. |
| D6 | **README ≠ code.** README says MTCNN face crops in `data_prep.py` (not done); says `train.py` trains A–D (trains only Model D); `evaluate.py` loads A and C checkpoints nothing produces; summary says CLIP ViT-L/14 (code uses `openai/clip-vit-base-patch32`); "last 6 layers unfrozen" only when `freeze_clip=False`. | Unreproducible. |
| D7 | **Demo can't run from a fresh clone.** `app.py` loads `checkpoints/model_i_whatsapp_robust.pt` and `video_temporal_model_whatsapp.pt` that no committed script saves (`train_video.py` saves `video_temporal_model.pt`; finetune saves `model_h_clip_robust.pt`). Checkpoints are gitignored. | Can't demo. |
| D8 | **Portability.** Hard-coded `C:\Users\sudarshan\.gemini\...` paths (`train_clip_finetune.py`, `test_new_fakes.py`); split CSVs contain Windows backslashes (`data/processed/fake\xxx.jpg`); `requirements.txt` lacks `transformers`, `pyarrow`, `requests`, `huggingface_hub`; README run command is Windows-only. | Breaks on Linux/Mac/CI. |
| D9 | **Inference rule biases toward "fake".** `app.py` uses `max(score_global, score_face)`; threshold fixed 0.5; no calibration; no "uncertain" band. | High false-positive rate on real photos. |
| D10 | **Hygiene.** ~15 `test_*.py`/`search_*.py` scratch scripts (not tests), no real tests, `torch.load` without `weights_only=True`, no seeds, no config system, `sys.path.append('src')` hacks, in-place feature mutation `feats[:,768:]=0` with a *per-batch* (not per-sample) coin flip. | Fragile, unreviewable. |
| D11 | **Data validity.** Images are uuid-named HF samples with no identity/source metadata; near-duplicate leakage between train/val/test is possible; 224x224 resize destroys native resolution/JPEG grid, so "compression" experiments start from already-resampled data. | Inflated or meaningless robustness numbers. |

### 1.2 What is genuinely good (keep)
- CLIP backbone choice for generalisation (consistent with the literature, see §5).
- Face + global dual-view idea.
- JPEG QF sweep protocol (90/70/50/30) and the `jpeg_compress` helper.
- Honest limitation notes in `project_summary.md`.
- The Streamlit app's *feature set* (compression slider, threshold slider, image + video tabs).

---------------------------------------------------------------------------------------------------
## 2. Target outcome and success metrics

Define these in `docs/EVAL_PROTOCOL.md` **before** training anything (Phase 0), then never change them after seeing results.

**Primary claim to test:** "A CLIP-based detector trained with realistic compression augmentation loses less accuracy at QF30–QF50 / WhatsApp-style recompression than the same detector trained on clean data."

| Metric | Report |
|---|---|
| Accuracy, AUC-ROC, AP, F1 | at each condition: Clean, QF90, QF70, QF50, QF30, WhatsApp-sim (resize+JPEG), and (video) H.264 CRF 23/28/32 |
| Robustness drop | ΔAUC = AUC(clean) − AUC(QF30), with 95 % bootstrap CI (≥1,000 resamples) |
| Calibration | ECE + reliability diagram, before/after temperature scaling |
| Generalisation | leave-one-source-out (train on dataset X, test on Y) |
| Ablations | CLIP-only vs CLIP+frozen-random-DWT vs CLIP+trained-DWT vs EfficientNet+DWT; with/without compression aug |
| Seeds | ≥3 seeds, report mean ± std |

**Minimum bar to call the project "working":** clean AUC ≥ 0.90 on a *held-out, deduplicated* test split, and ΔAUC(clean→QF30) for the augmented model smaller than for the clean-trained model with non-overlapping CIs. If this is not met, report it honestly as a finding (the original agent_instructions.md already demands this).

---------------------------------------------------------------------------------------------------
## 3. The plan — phases, steps, gates

### PHASE 0 — Repo restructure and reproducibility (½–1 day)
Branch: `phase-0-structure`

1. Create target layout (move, don't rewrite logic yet):
   ```
   deepfake_detector/            # python package (pyproject.toml, `pip install -e .`)
     data/    (datasets.py, labels.py, splits.py, degradations.py)
     models/  (clip_branch.py, freq_branch.py, fusion.py, video.py)
     train/   (train_image.py, train_video.py, calibrate.py)
     eval/    (evaluate.py, robustness.py, metrics.py, plots.py)
     infer/   (predictor.py  <- single place for image+video inference used by CLI, API and UI)
   configs/   (base.yaml, clip_lora.yaml, ablations/*.yaml)   # OmegaConf/Hydra or plain YAML + dataclasses
   apps/      (api/  FastAPI ; web/  frontend ; streamlit_app.py legacy fallback)
   tests/     scripts/   docs/   results/   checkpoints/ (gitignored)
   scratch/   (move every test_*.py, search_*.py, download_*.py here or delete; keep real downloaders in scripts/)
   ```
2. Replace `requirements.txt` with `pyproject.toml` + `requirements.lock` (pin versions: torch, torchvision, transformers, peft, open_clip_torch, opencv-python-headless, pywavelets, albumentations or kornia, scikit-learn, pandas, pyarrow, fastapi, uvicorn, python-multipart, streamlit, pytorch-grad-cam, matplotlib, plotly, pytest, ruff, mypy-lite). Verify a clean venv install on Python 3.11/3.12.
3. Add `configs/base.yaml` (paths, seeds, batch sizes, device auto-select `cuda` → `mps` → `cpu`). Remove every absolute/Windows path (`pathlib.Path`, repo-relative only).
4. Add `Makefile` (or `justfile`): `make setup data train eval app test lint`.
5. Add `ruff` + `pytest` + pre-commit; add GitHub Actions CI (lint + CPU unit tests on a 16-image fixture).
6. Add global `set_seed()` (python, numpy, torch, cuda, dataloader workers).
7. Security: `torch.load(..., weights_only=True)`; `.gitignore` check for weights/data.

**GATE 0:** fresh clone → `make setup && make test` passes on Linux CI.

---

### PHASE 1 — Fix data and labels (1–2 days) ← most important phase
Branch: `phase-1-data`

1. **Write one source of truth for labels** in `data/labels.py`: `REAL=0, FAKE=1` and an explicit per-dataset mapping dict with the *HF `ClassLabel.names`* read at runtime (e.g. `ds.features['label'].names`) — never hard-code "0 means X".
2. **Label audit test (human-in-the-loop):** script dumps a grid of 24 random images per class per dataset to `results/label_audit/*.png`. A human (the student) eyeballs it once and records the verdict in `docs/LABEL_AUDIT.md`. Add a pytest that asserts the mapping dict equals the recorded verdict. This directly tests D1.
3. **Confirm D1 empirically:** load the existing split CSVs; use a CLIP zero-shot or the old checkpoint if available; report whether labels are inverted. Record result.
4. **Rebuild the dataset properly:**
   - Primary: `Hemg/AI-Generated-vs-Real-Images-Datasets` (general images) + `Thien0103/DeepFake_Extracted_Face_Images` (faces) — both already used. Verify each dataset's license/terms on its HF card and record in `docs/DATASETS.md`.
   - Keep **original resolution** where possible; do the 224 resize *inside the model transform*, after degradation (so JPEG grids are applied to native pixels, not to a 224x224 resample).
   - **Deduplicate** with perceptual hash (`imagehash` pHash, Hamming ≤ 6) across *all* splits; log removed counts.
   - **Split by source/generator/identity where metadata exists**, otherwise by pHash cluster. Fixed seed. Store as CSV with **relative POSIX paths** + `source`, `label`, `sha1` columns.
   - Hold out **one whole generator/dataset as an OOD test set** (never used for tuning).
5. Add a **face-crop dataset variant** with MTCNN (or RetinaFace via insightface — see §5) with 20 % margin, skipping/logging no-face images (this is what the README claims but code doesn't do).
6. Add `tests/test_data.py`: no overlap between splits (by path, sha1, pHash), class balance within ±5 %, labels ∈ {0,1}, image decodes to RGB.

**GATE 1:** `docs/LABEL_AUDIT.md` signed off; tests green; `results/dataset_stats.json` printed from code.

---

### PHASE 2 — Honest baselines (1 day)
Branch: `phase-2-baselines`

1. Re-implement **Model A** (EfficientNet-B0), **B** (DWT-CNN alone), **C** (fusion, clean), **D** (fusion, compression aug) exactly as `agent_instructions.md` specifies — on the *fixed* data, with a real train loop (AMP on GPU, early stopping on val loss, patience 3, best-checkpoint saved).
2. **Sanity gates inside the training script** (fail loudly):
   - Overfit-one-batch test (loss → ~0 on 32 images in <100 steps).
   - Shuffled-label control: train with shuffled labels, val AUC must be ≈0.5 (proves no leakage/bug).
3. Evaluate with `eval/robustness.py` (§ Phase 5 protocol) and write `results/baselines.csv` + `results/accuracy_vs_compression.png`.
4. Expected: Model A should clearly beat chance on clean data. If A < 0.90 val accuracy, debug training (lr, normalisation — **EfficientNet uses ImageNet mean/std, CLIP uses its own**, mixing them is a common bug) before moving on.

**GATE 2:** Model A clean AUC ≥ 0.90; shuffled-label control ≈ 0.5; tables generated by code.

---

### PHASE 3 — Make the model actually good (3–5 days)
Branch: `phase-3-model`

Three approaches (compare, then pick; do not skip the comparison):

| Option | Idea | Pros | Cons / risk |
|---|---|---|---|
| **3A. Frozen CLIP + linear/MLP probe** (UnivFD-style) | Cached CLIP features, tiny head | Fast on CPU, strong cross-generator generalisation in the literature | Cannot adapt to compression artefacts; augmentation must be re-extracted each epoch (slow) |
| **3B. CLIP + LoRA on last N blocks (recommended)** | Add LoRA adapters (`peft`) to attention q/v of the last 6 blocks of the vision tower; train adapters + head with per-epoch on-the-fly augmentation | Learns compression-invariant cues without catastrophic forgetting of CLIP's semantics; ~1 % trainable params; fits 8 GB VRAM with AMP + batch 32 | Needs GPU for sensible speed; must verify PEFT target module names for HF `CLIPVisionModel` (expected `q_proj`, `v_proj` — verify, don't assume) |
| **3C. Two-stream with a *trained* frequency branch** | CLIP (3B) + DWT/FFT/SRM-residual CNN trained jointly; learned gating instead of `max(global,face)` | Tests the project's actual DIP hypothesis; great for the report | More moving parts; frequency cues are exactly what compression destroys — may give small gains (report honestly) |

**Recommended path:** build 3A as the cheap baseline → 3B as the main model → 3C as an ablation (CLIP+trained-DWT vs CLIP only). Steps:

1. **Fix D3:** train the frequency branch (unfreeze it, or pretrain it standalone = Model B) — or remove it from claims.
2. **Fix D4: true on-the-fly degradation** in the `Dataset.__getitem__` (so every epoch sees new random degradations). Build `data/degradations.py`:
   - JPEG QF ∈ U[15,95] (p=0.7), **double-JPEG** (compress twice with different QF — social media does this), random chroma subsampling.
   - Down/up-scale (0.3–0.9) with random interpolation, Gaussian blur σ∈[0,1.5], Gaussian/Poisson noise, optional H.264 single-frame re-encode via ffmpeg/cv2 for video-frame training.
   - Keep a clean pass-through probability (~15 %).
   - Fit WhatsApp-sim parameters **empirically**: send ~20 original images/videos through real WhatsApp, compare with originals, and record resulting resolution/size/QF (estimate QF from quantisation tables via PIL `.quantization`). Commonly WhatsApp downsizes large images and re-encodes JPEG (and re-encodes video to H.264 at reduced bitrate) — **exact values are a guess until you measure them**.
3. **Fix D9 – fuse face + global properly:** a learned gate (small MLP on [s_global, s_face, face_conf, face_area_ratio]) trained on val; fall back to global when no face. Drop plain `max`.
4. **Calibration:** temperature scaling on the val set (`calibrate.py`); report ECE; define three output states: `REAL (p<0.35)`, `UNCERTAIN`, `FAKE (p>0.65)` thresholds chosen on val to hit target precision (don't hand-tune on test).
5. **Regularise for forgetting:** mix 70 % general+face data with at most 10 % WhatsApp-like data; use low LR (1e-5 for LoRA params w/ cosine), weight decay 1e-4, label smoothing 0.05; early-stop on a *mixed* val set.
6. **Model zoo hygiene:** every checkpoint saved with `{config.yaml, git_hash, metrics.json, label_map}`; loader refuses a checkpoint without them. One canonical name per model (`image_clip_lora_v1.pt`, `video_lstm_v1.pt`).
7. Log with TensorBoard or Weights & Biases (offline mode OK).

**GATE 3:** On the held-out + OOD sets, model 3B beats 3A in ΔAUC(clean→QF30) with non-overlapping bootstrap CIs, **or** the report documents that it did not.

---

### PHASE 4 — Video pipeline (2–3 days)
Branch: `phase-4-video`

1. **Split by video ID** (and by source identity if known), never by frame.
2. Sample **K=16 frames uniformly + face-tracked crops** (detect on every 4th frame, track/interpolate; crop 20 % margin). Reject clips with <50 % face frames, log them.
3. Per-frame backbone = the Phase-3 image model (frozen or LoRA). Temporal head options (compare): (a) mean/max pooling of frame logits [baseline], (b) 1-layer BiLSTM (current idea), (c) small temporal Transformer (2 layers, 4 heads). Report all three; ship the best on val.
4. **Training augmentation for video:** re-encode clips with ffmpeg H.264 (CRF 23–36, optional downscale to 480/720p) *before* frame extraction, including **WhatsApp-sim**. Cache the re-encoded copies (deterministic seed) so training is not ffmpeg-bound.
5. Evaluate on CRF 23/28/32/36 + WhatsApp-sim; same bootstrap protocol.
6. Output **per-frame scores** + temporal attention weights for the UI timeline (Phase 6).
7. Dataset options (check licences/access): the 400-clip `angads24/deepfake-video` used now (too small — treat as smoke-test), plus FaceForensics++ (c23/c40 already compressed — very relevant), Celeb-DF v2, DFDC — several need a request/approval form; start those requests on day 1 (see §5).

**GATE 4:** video model reports AUC with CI on a video-ID-disjoint test set at all re-encode levels; no frame leakage test passes (`tests/test_video_split.py`).

---

### PHASE 5 — Evaluation protocol, ablations, and the report numbers (1–2 days)
Branch: `phase-5-eval`

1. `eval/robustness.py`: for each model × each condition → AUC/AP/Acc/F1 + ECE, bootstrap CI. Conditions applied **only at test time, fresh per level, with fixed seeds**.
2. **WhatsApp external set (the user's 28 files):** treat as *test only*. If you want to use it for adaptation, do **k-fold by file** (e.g. 4-fold stratified), report mean ± std of held-out folds, and state loudly that n=28 can only support qualitative claims. Better: collect more (aim ≥200 images) — take originals, send through WhatsApp yourself, save received copies; this "paired original/received" set is also what you use to fit the degradation parameters in Phase 3 step 2 (use *different* images for fitting vs testing).
3. **Leave-one-source-out** generalisation table.
4. **Ablation table** (rows = models, columns = conditions): A, B, C, D, 3A, 3B, 3B+trained-DWT, 3B without aug.
5. Figures (matplotlib, vector PDF + PNG): accuracy-vs-QF line chart with CI bands; ROC curves; reliability diagram; Grad-CAM gallery (TP/FP/FN); frequency-spectrum comparison of real vs fake (mean log-FFT magnitude and mean DWT sub-band energy) — these are *the* DIP-course figures.
6. `results/RESULTS.md` is **auto-generated** from the CSV/JSON files.

**GATE 5:** one command, `make eval`, regenerates every table/figure in the report from checkpoints.

---

### PHASE 6 — UI/UX overhaul (3–5 days; can start in parallel after Phase 3 gives a stable `Predictor` API)
Branch: `phase-6-ui`

#### 6.0 What is wrong with the current UI (from `app.py`)
- 270 lines of CSS injected via `st.markdown(unsafe_allow_html=True)`; leftover login-form CSS (`.stTextInput`) that nothing uses; hides Streamlit's header/footer.
- Models loaded at import; a single `st.spinner` wraps *everything*; no progress feedback; whole page reruns on every slider move (re-runs CLIP each time).
- Single number + coloured banner; no explanation of *why*. Overstated copy ("Splicing detected…"). Fixed 0.5 threshold; `max()` fusion.
- No heatmap, no frequency view, no per-frame video timeline, no export, no history, no error handling for bad/huge files, no mobile layout.

#### 6.1 Stack decision (compare, then choose)

| Option | What | Pros | Cons | Verdict |
|---|---|---|---|---|
| UI-1 | **Polish Streamlit** (custom theme via `.streamlit/config.toml`, `st.fragment`, `st.cache_resource`, `st.session_state`, Plotly) | 1 day; one language; easiest for students | Limited interactivity (no click-to-seek, no overlay-slider without hacks); reruns | Use as **fallback/demo-in-class** |
| UI-2 | **Gradio Blocks** | Very quick, nice built-in image/video components & annotated images, hosts free on HF Spaces | Opinionated look, harder to get custom design | Good for a public Spaces demo |
| UI-3 | **FastAPI inference service + React (Vite) + Tailwind + shadcn/ui + Recharts + Framer Motion** | Best-looking, fully custom, real components (overlay slider, video timeline), clean API reusable by CLI/mobile; testable (Playwright) | More code; two languages | **Recommended primary UI** if ≥1 week is available; the AI agent writes most of it |

Recommendation: **UI-3 as the headline deliverable, UI-1 kept as a thin fallback** that calls the same `Predictor`. (Decision driver: the owner asked for "the best" UI and an AI agent is doing the work; the API boundary also forces the clean `infer/predictor.py` that Phase 0 needs.)

#### 6.2 Architecture
```
apps/api (FastAPI)           POST /v1/analyze/image   multipart -> JSON
                             POST /v1/analyze/video   multipart -> job_id ; GET /v1/jobs/{id} (SSE or polling)
                             POST /v1/robustness      image + [QF list] -> prob-vs-QF curve
                             GET  /v1/model-info      name, version, git hash, thresholds, training-data summary
                             GET  /healthz
infer/predictor.py           load_models() once at startup; analyze_image(), analyze_video(); returns dataclasses:
                             {p_fake_calibrated, state(REAL|UNCERTAIN|FAKE), p_global, p_face, face_box,
                              heatmap_png_b64, spectrum_png_b64, per_frame:[{t,p,face_box}], timings_ms, model_version}
apps/web (Vite+React+TS)     pages: Analyze (image/video), Robustness Lab, Batch, History, About/Model Card
```
Security/robustness for the API: max upload size (e.g. 25 MB image / 100 MB video), MIME + magic-byte sniffing, decode with PIL `verify()` and re-open, strip EXIF, temp files cleaned, per-request timeout, CORS limited to the web origin, no persistence of uploads unless the user opts in to History, rate limit (slowapi). Run inference in a threadpool/worker so the event loop isn't blocked.

#### 6.3 Feature list (build in this priority order)
**P0 — must have**
1. Drag-and-drop + paste + file picker; instant preview; clear file-type/size errors.
2. **Verdict card:** calibrated probability as a gauge/bar, 3-state label (Real / Uncertain / Fake), one-line plain-English caveat ("Probability, not proof. Compression and resizing lower reliability."). Colour is never the only signal (icons + text; WCAG AA contrast).
3. **Explainability overlay:** Grad-CAM (or attention-rollout for ViT) heatmap over the image with an opacity slider and a before/after toggle; face bounding box drawn on the image; global-vs-face score chips.
4. **Compression Lab (the project's showcase):** a QF slider (10–100) that re-compresses *client-supplied bytes on the server* and shows (a) original vs compressed side-by-side with a zoomable diff slider, (b) a live line chart of P(fake) vs QF — computed once as a batch sweep (QF 100→10 in steps of 10) and drawn instantly; slider scrub just moves a cursor on the curve (no re-inference per tick — fixes the rerun problem). Include a "WhatsApp-sim" preset.
5. **Video:** thumbnail strip of sampled frames, per-frame P(fake) timeline chart (click a point → seeks the video player / highlights frame), overall verdict, temporal-attention shading.
6. Skeleton loaders, progress (SSE) for video, cancel button, error toasts, empty states.
7. Responsive (360 px phone → 1440 px desktop), keyboard accessible, light/dark theme.

**P1 — should have**
8. **Frequency view tab (DIP tie-in):** log-magnitude FFT spectrum of the image and of the compressed version; Haar-DWT sub-band (LH/HL/HH) thumbnails with energy bars — shows *why* compression hides cues. Generated server-side with numpy/pywt.
9. Threshold control in an "Advanced" drawer (default from `model-info`), shows precision/recall trade-off from the val curve.
10. **Batch mode:** upload up to 20 images → sortable table (thumbnail, p_fake, state, face found) → CSV/JSON export.
11. **Report export:** one-click PDF/HTML report (image, verdict, heatmap, QF curve, model version, disclaimer).
12. About/Model Card page: training data, metrics table (auto-pulled from `results/*.json`), known limitations, ethics note.

**P2 — nice to have**
13. History (IndexedDB, local only); compare two images side-by-side; shareable permalink (opt-in).
14. i18n (EN/TA); PWA install; webcam capture.
15. "Try samples" gallery from `test_images/` so a viva examiner can demo instantly.

#### 6.4 Design direction
Keep the dark "glass" idea but make it *functional*: neutral dark surface (#0B1020-ish), one accent colour, red/amber/green reserved strictly for verdict states, 8-px spacing grid, a single type scale (Inter), `prefers-reduced-motion` respected, glass blur only on the verdict card (not every panel — it hurts legibility and performance). Components via **shadcn/ui** (Radix primitives → accessible by default), charts via **Recharts** or **visx**, motion via Framer Motion. Provide a Figma-free design token file (`tokens.css`). (If the owner wants a mockup first, the Figma/Lovable connectors available in the owner's Claude workspace can generate one — optional.)

#### 6.5 UI acceptance criteria (testable)
- Lighthouse (desktop): Performance ≥ 85, Accessibility ≥ 95, Best-practices ≥ 95.
- Playwright E2E: upload sample real & fake image → verdict visible < 5 s on GPU / < 20 s on CPU; QF scrub does **not** trigger new network calls; invalid file shows an error; video timeline click seeks.
- `axe-core` run: 0 serious/critical issues.
- Works at 360 px width with no horizontal scroll.
- API contract tests (pytest + `schemathesis` or plain `httpx`) incl. oversize file, corrupt image, wrong MIME → 4xx not 5xx.
- Heatmap sanity test: heatmap shape == image shape; values in [0,1]; deterministic for a fixed seed.

**GATE 6:** `make app` starts API+web; E2E suite green; screenshots of every screen saved in `docs/screens/`; a 90-second screen recording for the viva.

---

### PHASE 7 — Docs, packaging, and the DIP report (2 days)
1. Rewrite `README.md` so every command works from a fresh clone; include an architecture diagram (Mermaid), data-flow of the inference path, the results table (auto-included), and a **Known Limitations** section (small data, JPEG/WhatsApp-sim vs real, face-only video, no adversarial robustness, not forensic proof).
2. `Dockerfile` (CPU) + `docker-compose.yml` (api + web); optional CUDA image.
3. `MODEL_CARD.md` + `DATASHEET.md` (intended use, limits, biases, licences).
4. Map project to DIP syllabus in the report: sampling/resizing/interpolation (down-up scaling), JPEG = colour-space conversion + chroma subsampling + 8x8 DCT + quantisation (include the pipeline figure and the QF→quant-table plot), wavelet (Haar) decomposition, FFT spectra of real vs generated images, spatial filtering (blur), noise models, image-quality metrics (PSNR/SSIM vs QF — easy and impressive to add), morphological/connected-component post-processing of heatmaps if desired.
5. Ethics paragraph: dual-use, false-positive harm, not for judging individuals, consent for images used in demos.

**GATE 7:** a reviewer who has never seen the repo reproduces Phase 5 numbers by following README top-to-bottom.

---------------------------------------------------------------------------------------------------
## 4. Execution order and time budget (≈ 3 weeks part-time; compress as needed)

| Order | Phase | Parallel? | Depends on |
|---|---|---|---|
| 1 | 0 structure | – | – |
| 2 | 1 data/labels | – | 0 |
| 3 | 2 baselines | – | 1 |
| 4 | 3 model | UI can start on a *mock* `Predictor` (returns canned JSON) | 2 |
| 5 | 4 video | with 6 | 3 |
| 6 | 6 UI | with 4/5 once API schema frozen | 0 + mock |
| 7 | 5 eval | after 3, 4 | 3,4 |
| 8 | 7 docs/report | last | all |

**Start dataset-access requests (FaceForensics++, Celeb-DF, DFDC) on day 1** — approval can take days.

---------------------------------------------------------------------------------------------------
## 5. Public GitHub repos / resources worth using

Verified = fetched from the GitHub page during this session (stars/licence as shown then). "Not re-verified" = recalled from general knowledge; the agent must check licence/maintenance before adopting.

| Repo | Use it for | Licence / state (verified unless noted) | Verdict |
|---|---|---|---|
| [SCLBD/DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | Reference training/eval harness; 36 detectors (5 naive, 20 spatial, 3 frequency, 8 video); preprocessing + FF++/Celeb-DF/DFDC protocols; compare your model against Xception/SBI/frequency detectors | ~1.1k★, **CC BY-NC 4.0** (non-commercial OK for coursework); latest commit date not visible, README says actively updated | **Adapt ideas + protocol; baseline numbers.** Don't vendor the whole thing. |
| [WisconsinAIVision/UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) | The canonical "frozen CLIP ViT-L/14 + linear probe" recipe; training/eval scripts; justifies your 3A baseline | 378★, **MIT** | **Adapt** (read train/validate for the recipe; rewrite). Needs ~72 GB ProGAN data for the paper setting — don't replicate that; use for method only. |
| [jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | Grad-CAM/ScoreCAM/EigenCAM heatmaps incl. ViT (needs a `reshape_transform` that drops the CLS token; see its `utils/reshape_transforms.py`) | ~13k★, **MIT** | **Adopt** as a dependency for the UI heatmap. Verify CLIP-ViT target layer (`...encoder.layers[-1].layer_norm1`) works by a unit test. |
| [huggingface/peft](https://github.com/huggingface/peft) | LoRA adapters for the CLIP vision tower | ~21.6k★, **Apache-2.0** | **Adopt.** Whether `CLIPVisionModel` accepts `target_modules=["q_proj","v_proj"]` was *not* confirmed from the page — test it. |
| [mapooon/SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages) (CVPR'22 Oral) | Idea: synthesise fakes from real images (self-blending) — a way to get *unlimited labelled fakes* for the face-swap type | 275★; research-purpose licence (commercial needs agreement); latest changelog Feb 2023; heavy setup (dlib 81-pt landmarks, FF++) | **Idea only / optional stretch**; don't depend on it. |
| [albumentations-team/albumentations](https://github.com/albumentations-team/albumentations) | `ImageCompression`, `Downscale`, `Defocus`, noise transforms for degradation pipeline | MIT but **archived Jul 2025**; successor *AlbumentationsX* is AGPL-3.0/commercial | **Caution.** Safer: write your own `degradations.py` (PIL/OpenCV, ~80 lines) or use `torchvision.transforms.v2` / `kornia`. |
| [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) — *not re-verified* | Alternative CLIP weights (ViT-L/14 LAION/DataComp, SigLIP); HF `transformers` CLIP is fine too | MIT (recalled) | **Optional** — try ViT-L/14 only if the 4060's 8 GB allows (use AMP + LoRA). |
| [deepinsight/insightface](https://github.com/deepinsight/insightface) — *not re-verified* | Better face detector/landmarks (RetinaFace/SCRFD) than MTCNN | code MIT but **pretrained models are non-commercial** (recalled) | **Optional** swap for `facenet-pytorch` MTCNN; check licence. |
| [timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch) — *not re-verified* | Current MTCNN dependency | MIT (recalled); low maintenance | Keep for now; abstract behind a `FaceDetector` interface so it can be swapped. |
| [gradio-app/gradio](https://github.com/gradio-app/gradio), [streamlit/streamlit](https://github.com/streamlit/streamlit) — *not re-verified* | Fallback demo UIs | Apache-2.0 (recalled) | Fallbacks only. |
| FastAPI, Vite, React, Tailwind, shadcn/ui, Recharts, Framer Motion, Playwright — *not re-verified* | Primary UI stack | permissive (MIT/Apache, recalled) | Adopt. |

Papers worth citing / skimming (the agent should open each to confirm claims; I only saw titles in a search):
- Ojha et al., *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (CVPR 2023) — arXiv 2302.10174 (UnivFD).
- *C2P-CLIP: Injecting Category Common Prompt in CLIP to Enhance Generalization in Deepfake Detection* — arXiv 2408.09647 (I did not verify a GitHub repo for it; check the paper's page).
- *DeeCLIP: A Robust and Generalizable Transformer-Based Framework for Detecting AI-Generated Images* — arXiv 2504.19876 (reported as robustness-focused — read the abstract to see whether it tests compression).
- Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images* (CVPR 2022).
- DeepfakeBench paper (Yan et al., NeurIPS D&B 2023) — for evaluation protocol and the compression-sensitivity of detectors (its README shows e.g. Xception AUC 0.96 → 0.83 from c23 → c40).

Datasets (check licence/access on each card; several are gated):
- Already used: `Hemg/AI-Generated-vs-Real-Images-Datasets`, `Thien0103/DeepFake_Extracted_Face_Images`, `angads24/deepfake-video` (HF).
- Add for credibility: FaceForensics++ (c23/c40 variants are natively compressed — ideal for this topic), Celeb-DF v2, DFDC (approval forms), GenImage (AI-image generators), plus the owner's own WhatsApp paired set.

---------------------------------------------------------------------------------------------------
## 6. Self-critique: weakest points, failure modes, what could still go wrong

1. **Data is the ceiling.** 10k general + face images with unknown provenance won't generalise to new generators or real WhatsApp media. Mitigation: OOD hold-out, paired WhatsApp set, honest claims. *Risk remains high.*
2. **Frequency cues vs compression is a contradiction.** JPEG/H.264 remove exactly the high-frequency traces the DWT branch targets; the trained-DWT ablation may show no gain. That's a legitimate finding — report it; do not tune until it "wins".
3. **Dataset shortcut learning.** Detectors often latch onto format/resolution/JPEG-quality differences between real and fake classes (real = JPEG, fake = PNG, etc.). Check class-conditional file size, resolution, EXIF and JPEG quantisation tables in Phase 1; if they differ systematically, equalise (re-encode both classes identically) *before* training. This could silently explain a high clean accuracy.
4. **WhatsApp-sim fidelity.** Simulated degradations ≠ the real pipeline; WhatsApp behaviour changes with versions/OS. Report results on real paired data separately from simulated.
5. **Calibration drift** under new compression levels — ECE measured only on val distribution. Show reliability per condition.
6. **Small-n statistics.** n=28 can't support "100 %" or any percentage claim; use CIs and say "qualitative".
7. **LoRA on 8 GB:** ViT-L/14 + AMP may OOM at batch 32 → use gradient accumulation / ViT-B first; ViT-L as an optional upgrade.
8. **UI scope creep.** P0 first; P2 only if time remains. A reliable P0 beats a half-working P1.
9. **Misuse/ethics.** A detector's false positive can harm a real person; surface uncertainty in the UI and keep the disclaimer visible.
10. **Unverified items in this plan:** all repo stars/licences marked "recalled"; PEFT target modules for CLIP; exact WhatsApp compression parameters; which of the three papers' repos exist. Treat as hypotheses to check.

---------------------------------------------------------------------------------------------------
## 7. Definition of Done (checklist)

- [ ] Labels audited and documented; label tests green; D1–D11 each closed or explicitly waived in `docs/KNOWN_ISSUES.md`
- [ ] Deduped, leak-free splits + OOD hold-out; shortcut check (resolution/format/quant-table) done
- [ ] Models A/B/C/D reproduced + LoRA-CLIP + ablations; shuffled-label control ≈ 0.5
- [ ] True on-the-fly degradation, WhatsApp-sim fitted from paired real data
- [ ] Calibrated 3-state output; learned face/global fusion
- [ ] Video model split by video ID with H.264 re-encode robustness curves
- [ ] `make eval` regenerates all tables/figures with CIs; ≥3 seeds
- [ ] FastAPI + React UI with P0 features, E2E + a11y tests green; Streamlit fallback working
- [ ] README reproducible from fresh clone; Dockerfile; model card; DIP-syllabus mapping; ethics section
- [ ] No fabricated numbers anywhere; every figure traceable to a results file + commit hash

---------------------------------------------------------------------------------------------------
## 8. Copy-paste kickoff prompt for the agent

> You are working in this repo. Read `AGENT_PLAN.md` fully. Start with Phase 0, then Phase 1. Follow the operating rules in §0: one phase at a time, tests first, never fabricate numbers, stop at each GATE and summarise evidence (command output, file paths) before continuing. First action: create branch `phase-0-structure`, run the existing scripts only to confirm defects D1–D8 (do not fix yet), and write the confirmed/refuted result for each defect into `docs/DEFECT_CONFIRMATION.md`. Ask me before downloading any gated dataset or spending money on cloud compute.
