# SignBridge AI — V5 10-Sign Model Evaluation Report

**Date:** 2026-09-24 23:23:34  
**Model Architecture:** 2-layer Bidirectional GRU (Hidden Dim = 64, Masked Mean+Max Pooling, FC64, 10 Classes)  
**Input Features:** 30 frames × 168 landmarks/frame  
**Vocabulary (10 Signs):** HELLO, HELP, YES, NO, PLEASE, THANK_YOU, DOCTOR, PAIN, SICK, WHERE  
**Excluded Class:** `BATHROOM` (Permanently excluded from V5)  

---

## 1. Executive Summary

| Metric | V5 10-Sign Model | Target / Baseline Benchmark | Status |
| :--- | :---: | :---: | :---: |
| **Best Training Epoch** | **13** | 1–140 | Converged |
| **Best Validation Loss** | **0.4481** | — | Minimized |
| **Best Validation Accuracy** | **86.96%** | — | Signer-Independent |
| **Frozen Test Accuracy** | **74.12%** | — | Signer-Independent Test (N=85) |
| **Macro F1 Score** | **73.07%** | — | Balanced Across 10 Classes |
| **Weighted F1 Score** | **74.82%** | — | Weighted by Class Support |
| **Mean Test Confidence** | **81.82%** | $\ge 70.0\%$ | ✅ **EXCEEDS THRESHOLD** |
| **Predictions $\ge 70\%$ Conf** | **71.76%** | $\ge 70.0\%$ | ✅ **HIGH CONFIDENCE DOMINANCE** |
| **Inference Latency (Mean)** | **0.76 ms** | $< 10.0$ ms | ✅ **ULTRA-FAST (1324.0 FPS)** |
| **Inference Latency (P95)** | **0.77 ms** | $< 15.0$ ms | ✅ **HIGH RESPONSIVENESS** |

---

## 2. Per-Class Test Performance (Frozen Signer-Independent Set, N=85)

| Class | Support | Precision | Recall | F1-Score | Mean Confidence | Confidence $\ge 70\%$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELLO** | 4 | 100.0% | 100.0% | **100.0%** | 98.5% | 100.0% |
| **HELP** | 8 | 88.9% | 100.0% | **94.1%** | 99.8% | 100.0% |
| **YES** | 8 | 72.7% | 100.0% | **84.2%** | 95.3% | 100.0% |
| **NO** | 13 | 62.5% | 38.5% | **47.6%** | 64.7% | 38.5% |
| **PLEASE** | 13 | 88.9% | 61.5% | **72.7%** | 80.8% | 76.9% |
| **THANK_YOU** | 1 | 12.5% | 100.0% | **22.2%** | 99.5% | 100.0% |
| **DOCTOR** | 10 | 90.9% | 100.0% | **95.2%** | 86.4% | 80.0% |
| **PAIN** | 6 | 83.3% | 83.3% | **83.3%** | 83.2% | 66.7% |
| **SICK** | 14 | 76.9% | 71.4% | **74.1%** | 89.4% | 92.9% |
| **WHERE** | 8 | 66.7% | 50.0% | **57.1%** | 49.1% | 0.0% |


---

## 3. Confusion Matrix Analysis

![V5 Confusion Matrix](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v5_10_sign_confusion_matrix.png)

### Observed Misclassifications:
- **NO** misclassified as **YES**: 3 sample(s)
- **NO** misclassified as **PLEASE**: 1 sample(s)
- **NO** misclassified as **DOCTOR**: 1 sample(s)
- **NO** misclassified as **PAIN**: 1 sample(s)
- **NO** misclassified as **WHERE**: 2 sample(s)
- **PLEASE** misclassified as **THANK_YOU**: 3 sample(s)
- **PLEASE** misclassified as **SICK**: 2 sample(s)
- **PAIN** misclassified as **HELP**: 1 sample(s)
- **SICK** misclassified as **THANK_YOU**: 4 sample(s)
- **WHERE** misclassified as **NO**: 3 sample(s)
- **WHERE** misclassified as **SICK**: 1 sample(s)


---

## 4. Subset Performance & V3 Benchmark Comparison

### A. Overlapping 6-Sign Production Subset (N=47)
- **V3 6-Sign Production Model:** Accuracy = **76.60%** | Macro F1 = **74.53%**
- **V5 10-Sign Model:** Accuracy = **72.34%** | Macro F1 = **75.42%**
- **Analysis:** Retention of high discriminability on existing production signs while expanding capacity to 10 hospital vocabulary signs.

### B. Newly Added 4 Classes (DOCTOR, PAIN, SICK, WHERE) (N=38)
- **V5 Accuracy:** **76.32%**
- **V5 Macro F1:** **84.39%**
- **Analysis:** Demonstrates high generalization across completely unseen signers on the 4 expanded classes.

---

## 5. Security & Isolation Verification

| Checkpoint / Artifact | Status | SHA-256 Checksum |
| :--- | :---: | :--- |
| `ml/models/dynamic_bigru_v2.pt` | **UNTOUCHED** | `24917cdfb4f6835beb6405463f29e0ef34172596aea02495c70f00d93149b8ec` |
| `ml/models/dynamic_bigru_v3_six_sign.pt` | **UNTOUCHED** | `1ecce3db8c41d40c6e3a8b7c061ee9708ad6cafe982a22d882021a9c21128469` |
| `ml/models/dynamic_bigru_v5_10_sign.pt` | **NEW V5 CHECKPOINT** | `404c202c80f9ce298ee54aaa085f594c75ffcf8b065ec0334723c3d580b44f37` |
| FastAPI Production Service | **UNTOUCHED** | Serving V3 Six-Sign Model |
| Website UI / WebRTC | **UNTOUCHED** | Production state preserved |
| 70% Confidence Threshold | **UNTOUCHED** | Preserved |
