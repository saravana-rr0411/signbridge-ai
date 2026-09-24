# SignBridge AI — V6 10-Sign Model Evaluation Report

**Date:** 2026-09-24 23:37:21  
**Architecture:** 2-layer Bidirectional GRU (Hidden Dim = 64, Masked Mean+Max Pooling, FC64, 10 Classes)  
**Input Features:** 30 frames × 168 features (Strictly matching the proven V3 production pipeline)  
**Vocabulary (10 Signs):** HELLO, HELP, YES, NO, PLEASE, THANK_YOU, DOCTOR, PAIN, SICK, WHERE  
**Excluded Class:** `BATHROOM` (Permanently excluded)  
**Test Set Scope:** Completely frozen signer-independent test split ($N=74$, 8 unseen signers; including the exact 39 test samples from V3).

---

## 1. Executive Summary

| Metric | V6 10-Sign Model | Baseline / Reference | Status |
| :--- | :---: | :---: | :---: |
| **Best Training Epoch** | **47** | 1–120 | Converged |
| **Best Validation Loss** | **0.3147** | — | Minimized |
| **Best Validation Accuracy** | **93.85%** | — | Signer-Independent |
| **Frozen Test Accuracy (Overall N=74)** | **87.84%** | Baseline | Full 10-Class Test |
| **Macro F1 Score** | **85.09%** | — | Balanced Across 10 Classes |
| **Weighted F1 Score** | **89.21%** | — | Weighted by Class Support |
| **Mean Test Confidence** | **96.89%** | $\ge 70.0\%$ | ✅ **EXCEEDS THRESHOLD** |
| **Predictions $\ge 70\%$ Conf** | **94.59%** | $\ge 70.0\%$ | ✅ **DOMINANT CONFIDENCE** |
| **Inference Latency (Mean)** | **0.76 ms** | $< 10.0$ ms | ✅ **ULTRA-FAST (1320.6 FPS)** |
| **Inference Latency (P95)** | **0.78 ms** | $< 15.0$ ms | ✅ **HIGH RESPONSIVENESS** |

---

## 2. Per-Class Test Performance (Frozen Test Set, N=74)

| Class | Category | Support | Precision | Recall | F1-Score | Mean Confidence | Confidence $\ge 70\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELLO** | V3 Baseline | 4 | 100.0% | 100.0% | **100.0%** | 100.0% | 100.0% |
| **HELP** | V3 Baseline | 8 | 100.0% | 100.0% | **100.0%** | 99.9% | 100.0% |
| **YES** | V3 Baseline | 8 | 88.9% | 100.0% | **94.1%** | 99.6% | 100.0% |
| **NO** | V3 Baseline | 10 | 100.0% | 80.0% | **88.9%** | 90.0% | 80.0% |
| **PLEASE** | V3 Baseline | 8 | 100.0% | 62.5% | **76.9%** | 95.8% | 87.5% |
| **THANK_YOU** | V3 Baseline | 1 | 20.0% | 100.0% | **33.3%** | 100.0% | 100.0% |
| **DOCTOR** | New 4 Sign | 10 | 100.0% | 70.0% | **82.3%** | 97.7% | 100.0% |
| **PAIN** | New 4 Sign | 6 | 100.0% | 100.0% | **100.0%** | 94.4% | 83.3% |
| **SICK** | New 4 Sign | 11 | 100.0% | 90.9% | **95.2%** | 99.6% | 100.0% |
| **WHERE** | New 4 Sign | 8 | 66.7% | 100.0% | **80.0%** | 96.2% | 100.0% |


---

## 3. Confusion Matrix & Pairwise Error Analysis

![V6 Confusion Matrix](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v6_10_sign_confusion_matrix.png)

### Specifically Investigated Pairs:

| Confusion Pair | Error Count on Frozen Test Set |
| :--- | :---: |
| `NO -> YES` | 1 sample(s) |
| `NO -> WHERE` | 1 sample(s) |
| `WHERE -> NO` | 0 sample(s) |
| `WHERE -> SICK` | 0 sample(s) |
| `PLEASE -> THANK_YOU` | 3 sample(s) |
| `SICK -> THANK_YOU` | 1 sample(s) |
| `DOCTOR -> OTHER` | 3 sample(s) |
| `PAIN -> OTHER` | 0 sample(s) |


---

## 4. Direct Apples-to-Apples Comparison against Production V3 (N=39)

Both models evaluated on the **exact same 39 test samples** belonging to the six production signs (`HELLO`, `HELP`, `YES`, `NO`, `PLEASE`, `THANK_YOU`):

| Model | Evaluated Vocabulary | Six-Sign Accuracy (N=39) | Six-Sign Macro F1 (N=39) |
| :--- | :---: | :---: | :---: |
| **Production V3 Six-Sign Model** | 6 Classes | **92.31%** | **86.74%** |
| **Experimental V6 Ten-Sign Model** | 10 Classes | **87.18%** | **83.32%** |

### Performance on the 4 Newly Added Signs (`DOCTOR`, `PAIN`, `SICK`, `WHERE`, N=35):
- **V6 Four-Sign Accuracy:** **88.57%**
- **V6 Four-Sign Macro F1:** **90.45%**

---

## 5. Security & Isolation Verification

| Checkpoint / Artifact | Status | SHA-256 Checksum |
| :--- | :---: | :--- |
| `ml/models/dynamic_bigru_v2.pt` | **UNTOUCHED** | `24917cdfb4f6835beb6405463f29e0ef34172596aea02495c70f00d93149b8ec` |
| `ml/models/dynamic_bigru_v3_six_sign.pt` | **UNTOUCHED** | `1ecce3db8c41d40c6e3a8b7c061ee9708ad6cafe982a22d882021a9c21128469` |
| `ml/models/dynamic_bigru_v5_10_sign.pt` | **UNTOUCHED** | `404c202c80f9ce298ee54aaa085f594c75ffcf8b065ec0334723c3d580b44f37` |
| `ml/models/dynamic_bigru_v6_10_sign.pt` | **NEW V6 CHECKPOINT** | `a98ff9f86a3dead52a9be87d139afde4d5d7a50508000284e7ce6db460ce12c0` |
| Production FastAPI Inference | **UNTOUCHED** | Serving V3 Six-Sign Model |
| Website UI / WebRTC | **UNTOUCHED** | Production state preserved |
| 70% Confidence Threshold | **UNTOUCHED** | Preserved |

---

## 6. Live Webcam Validation Results (50 Controlled Attempts)

Script: [`validate_v6_10_sign_live_webcam.py`](file:///Users/saravanarajaram0411/CLG/KPR/ml/scripts/validate_v6_10_sign_live_webcam.py)  
Validation Log: [`v6_10_sign_live_validation.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v6_10_sign_live_validation.json)

- **Total Controlled Attempts:** 50 (5 attempts per sign across all 10 signs)
- **Correct Accepted Predictions (Match & $\ge 70\%$):** 45 / 50
- **Overall Empirical Accuracy:** **90.00%**
- **Accepted-Only Accuracy:** **91.84%** (45 / 49)
- **Acceptance Rate:** **98.00%** (49 / 50 accepted at $\ge 70\%$ threshold)
- **Mean Prediction Confidence:** **98.14%**
- **Mean CPU Latency:** **0.80 ms** | **P95 Latency:** **0.84 ms**

### Per-Sign Live Performance Breakdown:

| Sign | Attempts | Correct (Accepted) | Empirical Acc | Accepted-Only Acc | Avg Confidence | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **HELLO** | 5 | 5 | **100.0%** | 100.0% | 99.95% | Highly Stable (100% Consistent) |
| **HELP** | 5 | 5 | **100.0%** | 100.0% | 99.92% | Highly Stable (100% Consistent) |
| **YES** | 5 | 5 | **100.0%** | 100.0% | 99.77% | Highly Stable (100% Consistent) |
| **NO** | 5 | 4 | **80.0%** | 80.0% | 95.43% | 1 confusion to YES at fast tempo |
| **PLEASE** | 5 | 3 | **60.0%** | 60.0% | 99.98% | 2 confusions to THANK_YOU |
| **THANK_YOU** | 5 | 5 | **100.0%** | 100.0% | 99.98% | Highly Stable (100% Consistent) |
| **DOCTOR** | 5 | 5 | **100.0%** | 100.0% | 99.27% | Highly Stable (100% Consistent) |
| **PAIN** | 5 | 4 | **80.0%** | 100.0% | 93.44% | 1 low-confidence reject (67.6%), 0 false high-conf |
| **SICK** | 5 | 4 | **80.0%** | 80.0% | 99.67% | 1 confusion to THANK_YOU |
| **WHERE** | 5 | 5 | **100.0%** | 100.0% | 93.99% | Highly Stable (100% Consistent, 0 confusion to NO) |

---

## 7. Production Suitability Assessment

### Key Findings:
1. **Performance on Six-Sign Subset:** V6 achieves **87.18% accuracy / 83.32% macro F1** on the exact 39 six-sign test samples, compared to **92.31% accuracy / 86.74% macro F1** for the production V3 six-sign model. The expansion to 10 classes produces a slight -5.13% trade-off on the original 6 signs due to head competition.
2. **Performance of the 4 New Hospital Signs:** V6 excels on the 4 new signs, achieving **88.57% test accuracy** and **90.45% macro F1** offline, and **90.0% empirical accuracy** live:
   - `PAIN`: 100% test recall, 100% precision; 0 false positives.
   - `SICK`: 90.9% test recall, 100% precision; live 80% (major improvement over V5 after pruning stomach-cropped clips).
   - `WHERE`: 100% test recall, 80.0% F1; live 100% consistency (0 confusion to `NO` or `SICK`).
   - `DOCTOR`: 70.0% test recall, 100% precision; live 100% consistency.
3. **Weakest Signs & Critical Confusions:**
   - `PLEASE` (62.5% test recall, 60% live): Confuses with `THANK_YOU` due to chest-to-chin hand trajectory similarity.
   - `THANK_YOU` (support $N=1$ in test set): High recall (100%) but lower precision (20.0%) as it attracts false positives from `PLEASE`.
   - `NO` (80% test recall, 80% live): 1 sample confused with `YES` under extreme tempo variation.
4. **Production Deployment Recommendation:**
   - **Recommendation:** **RETAIN V3 IN PRODUCTION FOR NOW; KEEP V6 AS EXPERIMENTAL CANDIDATE.**
   - While V6 represents a major leap over V5 (+13.72% gain, 0 WHERE->NO collapse, 100% PAIN precision), activating it in production would reduce the 6-sign baseline accuracy from 92.31% to 87.18%.
   - V6 remains securely stored in `ml/models/dynamic_bigru_v6_10_sign.pt` as an isolated experimental asset.

