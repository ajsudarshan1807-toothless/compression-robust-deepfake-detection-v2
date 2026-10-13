# Evaluation Protocol and Target Outcomes

**Version:** 1.0  
**Status:** Frozen (Pre-experimentation baseline)  
**Date:** 2026-10-13  

---

## 1. Primary Hypothesis

> "A deepfake detector trained with realistic compression augmentation loses less accuracy and discriminability at QF30–QF50 / WhatsApp-style recompression than the same detector architecture trained on clean, uncompressed data."

---

## 2. Experimental Conditions

The system is evaluated under standardized, reproducible degradation conditions at test time:

1. **Clean**: Unmodified test images (native resolution rescaled by standard evaluation transform).
2. **JPEG QF90**: Mild social media compression (PIL JPEG quality 90).
3. **JPEG QF70**: Moderate web compression (PIL JPEG quality 70).
4. **JPEG QF50**: Aggressive compression (PIL JPEG quality 50).
5. **JPEG QF30**: Severe degradation (PIL JPEG quality 30).
6. **WhatsApp-sim**: Resizing + empirical JPEG recompression profile simulating WhatsApp image pipelines.
7. **Video Re-encoding (H.264)**: Constant Rate Factor (CRF) 23, 28, 32, 36.

---

## 3. Core Evaluation Metrics

For each model and condition, the following metrics are reported:

| Metric | Formulation / Purpose | Target / Criterion |
|---|---|---|
| **Accuracy** | Fraction of correct classifications at threshold $\tau = 0.5$ (or calibrated threshold) | Baseline > 0.90 clean |
| **AUC-ROC** | Area Under Receiver Operating Characteristic Curve (threshold-independent discriminability) | $\ge 0.90$ clean |
| **Average Precision (AP)** | Area under the precision-recall curve | Monitored |
| **F1 Score** | Harmonic mean of precision and recall | Monitored |
| **Robustness Drop ($\Delta\text{AUC}$)** | $\Delta\text{AUC} = \text{AUC}(\text{Clean}) - \text{AUC}(\text{QF30})$ | Smaller drop for augmented model with non-overlapping 95% bootstrap CIs |
| **Expected Calibration Error (ECE)** | Difference between model confidence and empirical accuracy across 10 bins | Evaluated before and after temperature scaling |
| **Generalization** | Leave-one-source-out cross-dataset testing | Report cross-generator drop |

---

## 4. Confidence Intervals & Statistical Rigor

- **Bootstrap Resampling**: All evaluation metrics must report 95% confidence intervals derived from $\ge 1,000$ bootstrap resamples.
- **Seeds**: Minimum of 3 random seeds ($42, 1337, 2026$) for training; report mean $\pm$ standard deviation.
- **Zero Data Leakage**: Evaluation splits are disjoint by image hash (`imagehash` pHash, Hamming distance $\le 6$) and generator identity.

---

## 5. Minimum Acceptance Gate (Gate 0 / Gate 2 / Gate 3)

1. Clean test AUC $\ge 0.90$ on a held-out, deduplicated test split.
2. $\Delta\text{AUC}(\text{Clean} \to \text{QF30})$ for the augmented model is statistically significantly smaller than for the clean-trained model (non-overlapping bootstrap CIs), or documented transparently as a finding if refuted.
3. Calibration yields bounded uncertainty intervals:
   - **Authentic (Real)**: $p < 0.35$
   - **Uncertain**: $0.35 \le p \le 0.65$
   - **AI-Generated (Fake)**: $p > 0.65$
