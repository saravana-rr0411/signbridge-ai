# SignBridge AI — V5 Dataset Expansion & Quality Report

**Date:** 2026-09-24 23:03:43  
**Scope:** V5 10-Class Dataset Pipeline (Cleaned, Pruned, & Expanded)  
**Target Vocabulary (10 Classes):** HELLO, HELP, YES, NO, PLEASE, THANK_YOU, DOCTOR, PAIN, SICK, WHERE  
**Excluded Class:** `BATHROOM` (Temporarily removed to eliminate bilateral confusion with `WHERE` and `YES`)  
**Artifacts Generated:**
- Landmarks NPZ: [`ml/datasets/processed/dynamic_landmarks_v5_10_sign.npz`](file:///Users/saravanarajaram0411/CLG/KPR/ml/datasets/processed/dynamic_landmarks_v5_10_sign.npz)
- Metadata CSV: [`ml/datasets/processed/metadata_v5_10_sign.csv`](file:///Users/saravanarajaram0411/CLG/KPR/ml/datasets/processed/metadata_v5_10_sign.csv)
- Audit JSON: [`ml/evaluation/v5_dataset_expansion_report.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v5_dataset_expansion_report.json)

---

## 1. Executive Summary
- **Baseline V4 Dataset Samples:** 401
- **BATHROOM Samples Excluded:** -38
- **Identified Low-Coverage Bad Samples Pruned:** -12
- **Retained High-Quality Baseline:** 351
- **Newly Collected & Accepted Candidates:** **+75**
  - `NO`: +17 usable samples
  - `PLEASE`: +19 usable samples
  - `SICK`: +17 usable samples
  - `WHERE`: +22 usable samples
- **Final V5 Usable Dataset Size:** **426 sequences** ($30 \text{ frames} \times 168 \text{ features}$)
- **Total Unique Signers:** **105**
- **Split Distribution:** Train = **272** | Val = **69** | Test = **85**
- **Cross-Split Signer Overlap:** **Strictly 0 (Verified: Train-Val=0, Train-Test=0, Val-Test=0)**

---

## 2. Per-Class Usable Counts & V5 Training Readiness

| Sign | V4 Old Count | Pruned | New Accepted | Final V5 Usable | Unique Signers | Train / Val / Test | Readiness for V5 Training |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **HELLO** | 26 | -0 | +0 | **26** | 25 | 17 / 5 / 4 | ✅ **READY** |
| **HELP** | 44 | -0 | +0 | **44** | 28 | 29 / 7 / 8 | ✅ **READY** |
| **YES** | 48 | -0 | +0 | **48** | 35 | 31 / 9 / 8 | ✅ **READY** |
| **NO** | 48 | -5 | +17 | **60** | 33 | 36 / 11 / 13 | ✅ **READY** |
| **PLEASE** | 33 | -2 | +19 | **50** | 26 | 30 / 7 / 13 | ✅ **READY** |
| **THANK_YOU** | 21 | -0 | +0 | **21** | 21 | 18 / 2 / 1 | ✅ **READY** |
| **DOCTOR** | 40 | -0 | +0 | **40** | 20 | 28 / 2 / 10 | ✅ **READY** |
| **PAIN** | 32 | -0 | +0 | **32** | 15 | 17 / 9 / 6 | ✅ **READY** |
| **SICK** | 39 | -1 | +17 | **55** | 25 | 32 / 9 / 14 | ✅ **READY** |
| **WHERE** | 32 | -4 | +22 | **50** | 28 | 34 / 8 / 8 | ✅ **READY** |


---

## 3. Detailed Actions on Weak Classes

### 1. NO
- **Old Count:** 48 | **Pruned:** 5 (`38539`, `38544`, `66183`, `38524`, `msasl_4_115`) | **New Accepted:** +17
- **Final Count:** **60 usable samples** (33 unique signers)
- **Splits:** 36 Train / 11 Val / 13 Test (Val $\ge 7$: **PASS**)
- **Improvement:** Removed degraded motion-blur clips; added distinct finger-closure snap samples to prevent slower-motion confusion with `PAIN`.

### 2. PLEASE
- **Old Count:** 33 | **Pruned:** 2 (`msasl_47_132`, `msasl_47_136`) | **New Accepted:** +19
- **Final Count:** **50 usable samples** (26 unique signers)
- **Splits:** 30 Train / 7 Val / 13 Test (Val $\ge 7$: **PASS**)
- **Improvement:** Solved train starvation (19 $\rightarrow$ 30 train samples); added tight circular chest motion samples to resolve forward-thrust confusion with `THANK_YOU`.

### 3. SICK
- **Old Count:** 39 | **Pruned:** 1 (`51501`) | **New Accepted:** +17
- **Final Count:** **55 usable samples** (25 unique signers)
- **Splits:** 32 Train / 9 Val / 14 Test (Val $\ge 7$: **PASS**)
- **Improvement:** Solved train starvation (19 $\rightarrow$ 32 train samples); strictly enforced simultaneous forehead and torso tracking to eliminate single-handed chin release artifacts.

### 4. WHERE
- **Old Count:** 32 | **Pruned:** 4 (`63081`, `63082`, `63083`, `msasl_30_211`) | **New Accepted:** +22
- **Final Count:** **50 usable samples** (28 unique signers)
- **Splits:** 34 Train / 8 Val / 8 Test (Val $\ge 7$: **PASS**)
- **Improvement:** Rescued starved validation split (2 $\rightarrow$ 8 val samples); temporary exclusion of `BATHROOM` eliminated the lateral shake attractor, while new upright index waggle samples reinforce single-finger pointing geometry.

---

## 4. Signer-Independent Split Verification

- **Train Signers:** 79
- **Validation Signers:** 17
- **Frozen Test Signers:** 9
- **Train-Val Signer Overlap:** **0**
- **Train-Test Signer Overlap:** **0**
- **Val-Test Signer Overlap:** **0**
- **Validation Threshold:** Every single weak class has achieved $\ge 7$ validation samples (`NO`: 11, `PLEASE`: 7, `SICK`: 9, `WHERE`: 8).

---

## 5. Model & Production Safety Verification

| Model / Component | Status | Hash / State |
| :--- | :---: | :--- |
| `ml/models/dynamic_bigru_v3_six_sign.pt` | **UNTOUCHED** | `1ecce3db8c41d40c6e3a8b7c061ee9708ad6cafe982a22d882021a9c21128469` |
| `ml/models/dynamic_label_mapping_v3_six_sign.json` | **UNTOUCHED** | `0815c9f31d7fb278a6e29e4e1b3caf5e993f7122e227dac521dcb32a36e90d95` |
| `ml/models/dynamic_bigru_v2.pt` | **UNTOUCHED** | `24917cdfb4f6835beb6405463f29e0ef34172596aea02495c70f00d93149b8ec` |
| Production FastAPI Inference | **UNTOUCHED** | Serving V3 Six-Sign Model |
| Website UI & WebRTC | **UNTOUCHED** | Preserved |
| 70% Confidence Threshold | **UNTOUCHED** | Preserved |
| V5 Model Training | **NOT STARTED** | Dataset Preparation Phase Only |
