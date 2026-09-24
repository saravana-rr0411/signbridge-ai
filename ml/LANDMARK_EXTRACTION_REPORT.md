# SignBridge AI — Landmark Extraction & Quality Control Report (Phase E)

**Project:** SignBridge AI — Accessibility-First Sign Language Communication Bridge  
**Problem Statement:** PS-09  
**Phase:** MediaPipe Landmark Extraction & Coordinate Normalization  
**Date:** September 2026  
**Status:** **READY_FOR_MODEL_TRAINING**  

---

## 1. Executive Summary

The MediaPipe landmark extraction pipeline has successfully processed the verified 18-class civic sign vocabulary.
Video clips were decoded, temporal sequences were normalized to 30 frames, and hand and pose landmarks were extracted with invariant geometric transformations.

- **Total Dynamic Sequences Extracted:** 156 sequences
- **Total Static 'letter_a' Handshapes Extracted:** 356 samples
- **Total Dataset Size:** 512 instances
- **Average Original Video Length:** 77.2 frames
- **Target Normalized Window:** 30 frames @ 150 features/frame = 4,500 dimensions per sequence
- **Distinct Signers Represented:** 44 signers
- **Missing Hand Frame Rate:** 36.76% (handled gracefully via binary presence masks)

---

## 2. Invariant Landmark Normalization Scheme

Each extracted frame contains 150 normalized features:
1. **Left Hand [0:64]:**
   - 21 3D landmarks centered at wrist $(p' = p - p_0)$ and scaled by hand span $\|p_9 - p_0\|$ (63 coordinates).
   - Index 63: Left Hand Presence Flag ($1.0$ if detected, $0.0$ if absent).
2. **Right Hand [64:128]:**
   - 21 3D landmarks centered at wrist and scaled by hand span (63 coordinates).
   - Index 127: Right Hand Presence Flag ($1.0$ if detected, $0.0$ if absent).
3. **Upper Body Pose [128:150]:**
   - 7 keypoints (nose, L/R shoulder, L/R elbow, L/R wrist) centered at mid-shoulder and scaled by shoulder width (21 coordinates).
   - Index 149: Pose Presence Flag ($1.0$ if detected, $0.0$ if absent).

---

## 3. Dataset Split Distribution (Signer-Aware)

| Split | Sample Count | Percentage | Role in Training Pipeline |
| :--- | :---: | :---: | :--- |
| **Train** | 105 | 67.3% | Parameter optimization |
| **Validation** | 25 | 16.0% | Hyperparameter tuning & checkpoint selection |
| **Test** | 26 | 16.7% | Final unbiased benchmark evaluation |

---

## 4. Class Distribution Table

| Class Label | Modality | Extracted Samples | Status |
| :--- | :---: | :---: | :--- |
| `appointment` | Dynamic | 9 | Train/Val/Test eligible |
| `bathroom` | Dynamic | 9 | Train/Val/Test eligible |
| `doctor` | Dynamic | 10 | Train/Val/Test eligible |
| `document` | Dynamic | 12 | Train/Val/Test eligible |
| `help` | Dynamic | 9 | Train/Val/Test eligible |
| `hospital` | Dynamic | 7 | Train/Val/Test eligible |
| `money` | Dynamic | 8 | Train/Val/Test eligible |
| `no` | Dynamic | 13 | Train/Val/Test eligible |
| `pay` | Dynamic | 6 | Train/Val/Test eligible |
| `please` | Dynamic | 7 | Train/Val/Test eligible |
| `problem` | Dynamic | 9 | Train/Val/Test eligible |
| `sick` | Dynamic | 9 | Train/Val/Test eligible |
| `thank_you` | Dynamic | 7 | Train/Val/Test eligible |
| `understand` | Dynamic | 8 | Train/Val/Test eligible |
| `wait` | Dynamic | 10 | Train/Val/Test eligible |
| `where` | Dynamic | 9 | Train/Val/Test eligible |
| `yes` | Dynamic | 14 | Train/Val/Test eligible |
| `letter_a` | Static | 356 | Train/Val eligible |

---

## 5. Artifacts Created

- **`ml/datasets/processed/dynamic_landmarks.npz`**: Compressed NumPy archive of shape `(156, 30, 150)`.
- **`ml/datasets/processed/static_landmarks.npz`**: Compressed NumPy archive of shape `(356, 64)`.
- **`ml/datasets/processed/metadata.csv`**: Tabular catalog linking video IDs, signer IDs, frame counts, and splits.
- **`ml/evaluation/landmark_visualizations/landmark_features_sample.png`**: Visual quality diagnostic plot.

---

## 6. Readiness for Model Training

The dataset is **READY FOR MODEL TRAINING**.
The feature matrices and validity masks are stored in compact, zero-overhead `.npz` format, fully isolating dynamic sequences from static handshapes and completely avoiding data leakage.
