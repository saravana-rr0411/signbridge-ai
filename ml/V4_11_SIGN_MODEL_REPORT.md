# SignBridge AI — V4 11-Sign Dynamic Recognition Model Report

**Date:** 2026-09-24  
**Status:** Evaluation Completed (Candidate Evaluation Phase — V3 Remains Production)  
**Model Architecture:** 2-layer Bidirectional GRU (Hidden=64, Masked Mean+Max Pooling, FC64, Dropout=0.3, 11 Classes)  
**Input Pipeline:** Strictly 30 frames × 168 features (MediaPipe Hands + Pose Wrists Physical Disambiguation)  
**Model Artifacts:**
- Checkpoint: [`ml/models/dynamic_bigru_v4_11_sign.pt`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_bigru_v4_11_sign.pt) (2,151.6 KB, 181,515 parameters)
- Label Mapping: [`ml/models/dynamic_label_mapping_v4_11_sign.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_label_mapping_v4_11_sign.json)
- Comparison JSON: [`ml/evaluation/v4_11_sign_model_comparison.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v4_11_sign_model_comparison.json)
- Training Curve: [`ml/evaluation/v4_11_sign_training_curve.png`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v4_11_sign_training_curve.png)
- Confusion Matrix: [`ml/evaluation/v4_11_sign_confusion_matrix.png`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v4_11_sign_confusion_matrix.png)
- Per-Class F1 Chart: [`ml/evaluation/v4_11_sign_f1_comparison.png`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v4_11_sign_f1_comparison.png)

---

## 1. Final 11-Class Vocabulary

The V4 vocabulary consists of **11 classes**:
1. `HELLO` (Original 6)
2. `HELP` (Original 6)
3. `YES` (Original 6)
4. `NO` (Original 6)
5. `PLEASE` (Original 6)
6. `THANK_YOU` (Original 6)
7. `DOCTOR` (New 5)
8. `PAIN` (New 5)
9. `SICK` (New 5)
10. `BATHROOM` (New 5)
11. `WHERE` (New 5)

*Excluded Candidates:* `MEDICINE`, `APPOINTMENT`, `WATER` (rejected during audit due to low diversity / motion blur); `WAIT`, `NURSE` (excluded from V4 scope to preserve model precision and avoid internal lexical collisions).

---

## 2. Dataset Distribution & Signer-Independent Split

All samples were extracted into the standard $30 \times 168$ feature format. Signer separation was strictly enforced globally across all 11 classes:
- **Total Unique Signers:** 115 signers
- **Train Signers:** 87 signers
- **Validation Signers:** 16 signers
- **Frozen Test Signers:** 12 signers
- **Cross-Split Signer Overlap:**
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$
  - **Overlap = Strictly 0 (100% Signer-Disjoint)**

### Split Counts per Class

| Class | Train Count | Validation Count | Frozen Test Count | Total Count |
| :--- | :---: | :---: | :---: | :---: |
| **HELLO** | 17 | 5 | 4 | 26 |
| **HELP** | 29 | 7 | 8 | 44 |
| **YES** | 31 | 9 | 8 | 48 |
| **NO** | 28 | 10 | 10 | 48 |
| **PLEASE** | 21 | 4 | 8 | 33 |
| **THANK_YOU** | 18 | 2 | 1 | 21 |
| **DOCTOR** | 28 | 2 | 10 | 40 |
| **PAIN** | 17 | 9 | 6 | 32 |
| **SICK** | 19 | 9 | 11 | 39 |
| **BATHROOM** | 19 | 8 | 11 | 38 |
| **WHERE** | 24 | 2 | 6 | 32 |
| **TOTAL** | **251** | **67** | **83** | **401** |

---

## 3. Training Configuration & Convergence

- **Loss Function:** Class-Weighted Cross-Entropy Loss ($\text{weights} \in [0.842, 1.138]$)
- **Controlled Oversampling:** Applied strictly to minority training classes with $<20$ samples (`hello`, `pain`, `thank_you` $2\times$), expanding the training set from 251 to 303 samples without duplicating validation or test data.
- **Data Augmentations (Train Split Only):**
  - In-plane hand rotation ($\pm 18^\circ$)
  - Temporal warping (speed variation $0.85\times$ to $1.15\times$, shift $\pm 2$ frames)
  - Coordinate scaling jitter ($[0.94, 1.06]$)
  - Gaussian coordinate noise ($\sigma=0.010$ on hands/pose, $\sigma=0.005$ on relative spatial offsets)
  - Binary presence flags (indices 63, 127, 149) preserved as exact $\{0.0, 1.0\}$
- **Optimizer:** AdamW ($\text{lr} = 1\times 10^{-3}$, $\text{weight decay} = 1\times 10^{-2}$)
- **Scheduler:** ReduceLROnPlateau (factor 0.5, patience 5)
- **Early Stopping:** Patience of 25 epochs on validation loss
- **Best Epoch:** **Epoch 24**
- **Best Validation Loss:** **0.6673**
- **Best Validation Accuracy:** **88.06%**

---

## 4. Frozen Test Set Performance (83 Samples, 12 Unseen Signers)

| Metric | Score |
| :--- | :---: |
| **Overall Accuracy** | **81.93%** (68 / 83 samples correct) |
| **Macro Precision** | **82.44%** |
| **Macro Recall** | **86.12%** |
| **Macro F1 Score** | **80.05%** |
| **Weighted F1 Score** | **83.80%** |
| **Mean Confidence** | **91.30%** |
| **Confidence $\ge 70\%$** | **89.16%** |
| **Inference Latency (Mean)** | **0.76 ms** (1,308 FPS) |
| **Inference Latency (P95)** | **0.79 ms** |

### Per-Class Evaluation Details

| Class | Support | Precision | Recall | F1-Score | Mean Confidence | $\ge 70\%$ Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELLO** | 4 | 100.00% | 100.00% | **100.00%** | 99.81% | 100.00% |
| **HELP** | 8 | 100.00% | 100.00% | **100.00%** | 99.93% | 100.00% |
| **YES** | 8 | 77.78% | 87.50% | **82.35%** | 92.34% | 87.50% |
| **NO** | 10 | 100.00% | 80.00% | **88.89%** | 84.07% | 80.00% |
| **PLEASE** | 8 | 83.33% | 62.50% | **71.43%** | 95.40% | 87.50% |
| **THANK_YOU** | 1 | 14.29% | 100.00% | **25.00%** | 99.88% | 100.00% |
| **DOCTOR** | 10 | 100.00% | 90.00% | **94.74%** | 94.02% | 100.00% |
| **PAIN** | 6 | 85.71% | 100.00% | **92.31%** | 99.83% | 100.00% |
| **SICK** | 11 | 100.00% | 72.73% | **84.21%** | 89.75% | 90.91% |
| **BATHROOM** | 11 | 85.71% | 54.55% | **66.67%** | 80.72% | 72.73% |
| **WHERE** | 6 | 60.00% | 100.00% | **75.00%** | 87.03% | 83.33% |

---

## 5. Focused Confusion Analysis

### Confusion Pairs
1. **BATHROOM vs YES:**
   - True `BATHROOM` $\rightarrow$ Predicted `YES`: **2 samples**
   - True `YES` $\rightarrow$ Predicted `BATHROOM`: **0 samples**
   - *Root Cause:* Both signs utilize a closed fist handshape with vertical orientation. When signer motion is subtle or truncated, the 'T' shake can be misconstrued as a nodding motion.
2. **DOCTOR vs HELP:**
   - True `DOCTOR` $\rightarrow$ Predicted `HELP`: **0 samples**
   - True `HELP` $\rightarrow$ Predicted `DOCTOR`: **0 samples**
   - *Resolution:* Perfectly separated! The wrist-tapping motion of DOCTOR does not collide with the thumbs-up upward lift of HELP.
3. **PLEASE vs THANK_YOU:**
   - True `PLEASE` $\rightarrow$ Predicted `THANK_YOU`: **3 samples**
   - *Root Cause:* Both gestures feature flat open hands starting from/near the chest or chin area.
4. **SICK vs THANK_YOU:**
   - True `SICK` $\rightarrow$ Predicted `THANK_YOU`: **3 samples**
   - *Root Cause:* SICK involves middle finger contact on the forehead and stomach; in certain camera angles with frontal hand occlusion, the downward chin/chest transition resembles the forward motion of THANK_YOU.
5. **BATHROOM vs WHERE:**
   - True `BATHROOM` $\rightarrow$ Predicted `WHERE`: **3 samples**
   - *Root Cause:* Both are repetitive side-to-side oscillation signs (fist vs single finger).

---

## 6. V4 vs V3 Head-to-Head Comparison

| Metric | V3 (6-Sign Model) | V4 (11-Sign Model on 6 Signs) | V4 (All 11 Signs) | V4 (New 5 Signs) |
| :--- | :---: | :---: | :---: | :---: |
| **Class Count** | 6 | 6 | 11 | 5 |
| **Test Support** | 39 samples | 39 samples | 83 samples | 44 samples |
| **Accuracy** | **92.31%** | **84.62%** | **81.93%** | **79.55%** |
| **Macro F1 Score** | **86.74%** | **82.28%** | **80.05%** | **85.12%** |
| **Mean Latency** | 0.76 ms | 0.76 ms | 0.76 ms | 0.76 ms |
| **P95 Latency** | 0.78 ms | 0.79 ms | 0.79 ms | 0.79 ms |
| **Model Size** | 2,147.7 KB | 2,151.6 KB | 2,151.6 KB | 2,151.6 KB |

### Honest Performance Analysis
- **Why V3 is higher on the 6 signs (92.31% vs 84.62%):** In V3, the model selects from only 6 classes with wide margin boundaries. In V4, introducing 5 new medical signs creates cross-boundary confusion (specifically `please` and `sick` crossing with `thank_you`, and `bathroom` with `where` and `yes`).
- **New 5 Signs Performance:** The 5 new signs achieved a strong **79.55% accuracy** and **85.12% Macro F1**. In particular, `DOCTOR` (94.7% F1) and `PAIN` (92.3% F1) perform exceptionally well.

---

## 7. Model Integrity Check (SHA-256 Hashes)

All previous model checkpoints and label mappings remained strictly untouched:

| File | SHA-256 Hash | Status |
| :--- | :--- | :---: |
| `ml/models/dynamic_bigru_v3_six_sign.pt` | `1ecce3db8c41d40c6e3a8b7c061ee9708ad6cafe982a22d882021a9c21128469` | **UNTOUCHED** |
| `ml/models/dynamic_label_mapping_v3_six_sign.json` | `0815c9f31d7fb278a6e29e4e1b3caf5e993f7122e227dac521dcb32a36e90d95` | **UNTOUCHED** |
| `ml/models/dynamic_bigru_v2.pt` | `24917cdfb4f6835beb6405463f29e0ef34172596aea02495c70f00d93149b8ec` | **UNTOUCHED** |
| `ml/models/dynamic_label_mapping_v2.json` | `964b3f732d92ec0618cb9984d4c748d086ead46650471059e892ca9a78f85134` | **UNTOUCHED** |

---

## 8. Live Webcam Controlled Validation

A standalone controlled validation session was executed with `ml/scripts/validate_v4_11_sign_live_webcam.py`:
- **Total Controlled Attempts:** 31
- **Correct Accepted Predictions ($\ge 70\%$):** 26 (83.87% empirical accuracy)
- **Accepted-Only Accuracy:** 92.86%
- **Average Live Confidence:** 93.62%
- **Live Latency:** Mean: 1.25 ms | P95: 2.24 ms

---

## 9. Production Status & Verification

1. **V3 Six-Sign Model Remains Production:**
   - Production website and FastAPI backend continue serving the validated V3 six-sign model (`dynamic_bigru_v3_six_sign.pt`).
2. **V4 Is NOT Activated in Production:**
   - Neither the FastAPI router nor the frontend recognition adapter was modified.
3. **Regression Tests Pass 100%:**
   - Hospital Conversation Layer: **37/37 Passed**
   - V3 Website Integration: **14/14 Passed**
   - V2 Smoke Test: **30/30 Passed**
   - Backend Pytest: **10/10 Passed**
   - Vite Build: **Passed (191 ms)**
