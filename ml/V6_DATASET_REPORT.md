# SignBridge AI — V6 10-Sign Dataset Report

**Date:** 2026-09-24 23:35:48  
**Target Vocabulary (10 Signs):** HELLO, HELP, YES, NO, PLEASE, THANK_YOU, DOCTOR, PAIN, SICK, WHERE  
**Scope:** Controlled V6 experimental dataset combining the validated V3 six-sign dataset with 4 hospital expansion signs (`DOCTOR`, `PAIN`, `SICK`, `WHERE`).  
**Artifacts Generated:**
- Processed Landmarks NPZ: [`ml/datasets/processed/dynamic_landmarks_v6_10_sign.npz`](file:///Users/saravanarajaram0411/CLG/KPR/ml/datasets/processed/dynamic_landmarks_v6_10_sign.npz)
- Metadata CSV: [`ml/datasets/processed/metadata_v6_10_sign.csv`](file:///Users/saravanarajaram0411/CLG/KPR/ml/datasets/processed/metadata_v6_10_sign.csv)

---

## 1. Executive Summary

- **Total Sequences:** **390** ($30 \text{ frames} \times 168 \text{ features}$)
- **Split Distribution:**
  - **Train:** **251** samples (80 signers)
  - **Validation:** **65** samples (17 signers)
  - **Frozen Test:** **74** samples (8 signers)
- **Total Unique Signers:** **105**
- **Signer Overlap Check:** **Strictly 0 across Train, Val, and Test** (Train-Val=0, Train-Test=0, Val-Test=0)
- **BATHROOM Status:** **Permanently Excluded**
- **Baseline Preservation:** V3 six-sign dataset (220 samples, 39 test samples) was preserved byte-for-byte and merged with audited hospital additions.

---

## 2. Complete Pruning Audit Log ($N=7$ Samples Pruned)

To address the severe tracking collapses identified in the V5 forensic analysis (such as `SICK` losing its lower-torso anchor and collapsing into `THANK_YOU`), the following 7 defective samples were explicitly removed:

| # | Video ID | Class | Signer ID | Split | Source | Pruning Reason |
| -: | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | `msasl_51_55` | **DOCTOR** | `msasl_signer_72` | train | MS-ASL | Severe hand occlusion and tracking failure (hand presence 40.0% - 12/30 frames) |
| 2 | `msasl_76_69` | **PAIN** | `msasl_signer_388` | train | MS-ASL | Severe hand tracking occlusion (hand presence 16.7% - 5/30 frames) |
| 3 | `msasl_76_97` | **PAIN** | `msasl_signer_72` | train | MS-ASL | Severe motion blur and tracking failure (hand presence 20.0% - 6/30 frames) |
| 4 | `msasl_63_155` | **SICK** | `msasl_signer_218` | train | MS-ASL | Secondary lower-stomach anchor cropped below frame (dual-hand presence 3.3%) |
| 5 | `msasl_63_160` | **SICK** | `msasl_signer_124` | test | MS-ASL | Secondary lower-stomach anchor cropped below frame (dual-hand presence 3.3%) |
| 6 | `msasl_63_168` | **SICK** | `msasl_signer_172` | test | MS-ASL | Secondary lower-stomach anchor completely missing (dual-hand presence 0.0%) |
| 7 | `msasl_63_169` | **SICK** | `msasl_signer_172` | test | MS-ASL | Secondary lower-stomach anchor occluded/cropped (dual-hand presence 13.3%) |


---

## 3. Dataset Distribution & Composition by Class

| Sign Class | Category | Train | Val | Test | Total | Signers | Source Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **HELLO** | V3 Production | 17 | 5 | 4 | **26** | 25 | MS-ASL: 19, WLASL: 6, Public Educational ASL (Grab Official ASL Dictionary): 1 |
| **HELP** | V3 Production | 29 | 7 | 8 | **44** | 28 | MS-ASL: 29, WLASL: 14, Public Educational ASL (ASL Interactive): 1 |
| **YES** | V3 Production | 31 | 9 | 8 | **48** | 35 | MS-ASL: 29, WLASL: 17, Public Educational ASL (Signs ASL): 1, Public Educational ASL (I Like Signing Songs): 1 |
| **NO** | V3 Production | 28 | 10 | 10 | **48** | 36 | MS-ASL: 31, WLASL: 16, Public Educational ASL (Start ASL): 1 |
| **PLEASE** | V3 Production | 21 | 4 | 8 | **33** | 25 | MS-ASL: 24, WLASL: 8, Public Educational ASL (Kidcasts ASL): 1 |
| **THANK_YOU** | V3 Production | 18 | 2 | 1 | **21** | 21 | WLASL: 8, Public Educational ASL (Sign Tribe Academy): 1, Public Educational ASL (Learn How to Sign): 1, Public Educational ASL (Emma Kist ASL): 1, Public Educational ASL (DEAF TV): 1, Public Educational ASL (Signing With Omar): 1, Public Educational ASL (ASL Body Language): 1, Public Educational ASL (ASL Sign Language): 1, Public Educational ASL (ASL Korina): 1, Public Educational ASL (Jaiden’s Life Vlogs): 1, Public Educational ASL (The Goddard School ASL): 1, Public Educational ASL (J. J. Lightel ASL): 1, Public Educational ASL (Sign Language Lessons): 1, Public Educational ASL (Coleen Bleza ASL): 1 |
| **DOCTOR** | New 4 Sign | 27 | 2 | 10 | **39** | 19 | MS-ASL: 30, WLASL: 9 |
| **PAIN** | New 4 Sign | 15 | 9 | 6 | **30** | 15 | MS-ASL: 26, WLASL: 4 |
| **SICK** | New 4 Sign | 31 | 9 | 11 | **51** | 23 | MS-ASL: 42, WLASL: 9 |
| **WHERE** | New 4 Sign | 34 | 8 | 8 | **50** | 28 | MS-ASL: 45, WLASL: 5 |


---

## 4. Key Differences from V5

1. **Pruned Stomach-Cropped `SICK` Clips:**
   - In V5, single-hand `SICK` clips (`msasl_63_168`, `msasl_63_160`, `msasl_63_169`, `msasl_63_155`) collapsed into `THANK_YOU` with $>84\%$ confidence because their lower-torso anchor was off-camera. In V6, all `SICK` samples strictly exhibit dual-hand anchors ($\ge 20\%$ dual-hand frames).
2. **Preserved V3 Production Baseline:**
   - The exact 220 samples from V3 (`HELLO`, `HELP`, `YES`, `NO`, `PLEASE`, `THANK_YOU`) are preserved without disruption.
3. **Exact Apples-to-Apples Six-Sign Benchmark:**
   - The frozen test set contains the exact 39 test samples of V3, enabling a 100% direct comparative evaluation between V3 and V6 on the production vocabulary.

---

## 5. Security & Isolation Verification

| Checkpoint / Artifact | Status | SHA-256 Checksum |
| :--- | :---: | :--- |
| `ml/models/dynamic_bigru_v2.pt` | **UNTOUCHED** | `24917cdfb4f6835beb6405463f29e0ef34172596aea02495c70f00d93149b8ec` |
| `ml/models/dynamic_bigru_v3_six_sign.pt` | **UNTOUCHED** | `1ecce3db8c41d40c6e3a8b7c061ee9708ad6cafe982a22d882021a9c21128469` |
| `ml/models/dynamic_label_mapping_v3_six_sign.json` | **UNTOUCHED** | `0815c9f31d7fb278a6e29e4e1b3caf5e993f7122e227dac521dcb32a36e90d95` |
| `ml/models/dynamic_bigru_v5_10_sign.pt` | **UNTOUCHED** | `404c202c80f9ce298ee54aaa085f594c75ffcf8b065ec0334723c3d580b44f37` |
| Production FastAPI Inference | **UNTOUCHED** | Serving V3 Six-Sign Model |
| Website UI / WebRTC | **UNTOUCHED** | Production state preserved |
| 70% Confidence Threshold | **UNTOUCHED** | Preserved |
