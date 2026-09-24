# SignBridge AI — Final Production Model Status (Phase 9 Integration)

**Document Type:** Final ML Production Deployment & System Status Specification  
**Status:** **V2 MODEL OFFICIALLY DEPLOYED TO PRODUCTION**  
**Active Production Model (V2):** [`ml/models/dynamic_bigru_v2.pt`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_bigru_v2.pt)  
**Previous Baseline Model (V1):** [`ml/models/dynamic_bigru_best.pt`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_bigru_best.pt) (Preserved as fallback backup)  
**Active Label Mapping (V2):** [`ml/models/dynamic_label_mapping_v2.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_label_mapping_v2.json)  
**Static Model:** [`ml/models/static_mlp_best.pt`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/static_mlp_best.pt) (Fingerspelling letter 'A')  
**Backend API Service:** FastAPI on port `8000` (Endpoints: `/health`, `/labels`, `/predict/sequence`, `/predict/static`)  

---

## 1. Production Model Specifications

| Attribute | Specification | Operational Status |
| :--- | :--- | :---: |
| **Active Production Checkpoint** | `ml/models/dynamic_bigru_v2.pt` | **LIVE & ACTIVE** |
| **Archival Fallback Checkpoint** | `ml/models/dynamic_bigru_best.pt` (V1) | **PRESERVED UNTOUCHED** |
| **Model Architecture** | 2-layer Bidirectional GRU (Hidden Dim: 64, Dropout: 0.3) | **UNCHANGED** |
| **Temporal Pooling** | Concatenated Masked Mean + Max Pooling (256 dimensions) | **UNCHANGED** |
| **Classification Head** | `Linear(256, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 17)` | **UNCHANGED** |
| **Input Sequence Format** | Exactly $30 \text{ frames} \times 150 \text{ features}$ | **PRESERVED** |
| **Active Vocabulary Size** | 18 classes (17 dynamic sequences + 1 static handshape) | **PRESERVED** |
| **Confidence Threshold** | **0.70** (Minimum accepted probability for automatic dispatch) | **ACTIVE** |
| **Temporal Stabilization** | 2 consecutive identical inference windows required before dispatch | **ACTIVE** |
| **Duplicate Debounce Window**| 3,500 ms suppression window for identical consecutive predictions | **ACTIVE** |

---

## 2. Quantitative Performance & Benchmark Summary

All metrics evaluated against the **official untouched, frozen WLASL test split** ($N=26$).

| Performance Dimension | Baseline V1 (`dynamic_bigru_best.pt`) | Deployed V2 (`dynamic_bigru_v2.pt`) | Delta (V2 vs V1) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Test Accuracy** | **76.92%** (20 / 26) | **80.77%** (21 / 26) | **+3.85%** | **Superior** |
| **Macro F1-Score** | 0.6745 | **0.7137** | **+0.0392** | **Superior** |
| **Weighted F1-Score** | 0.7538 | **0.7923** | **+0.0385** | **Superior** |
| **Macro Precision** | 0.7225 | 0.7525 | +0.0300 | **Superior** |
| **Macro Recall** | 0.7059 | 0.7353 | +0.0294 | **Superior** |
| **Validation Loss** | 0.4578 | **0.4062** | **-0.0516** | **Superior** |
| **Validation Accuracy** | 84.00% | **88.00%** | **+4.00%** | **Superior** |
| **Trainable Parameters** | 174,993 | 174,993 | 0 (Identical) | **Matched** |
| **Checkpoint File Size** | 2,124,197 bytes (2.12 MB) | 2,124,217 bytes (2.12 MB) | +20 B | **Matched** |
| **CPU Inference Latency** | 0.77 ms / sequence | 0.78 ms / sequence | +0.01 ms | **Matched ($<1\text{ms}$)** |
| **p95 CPU Latency** | 0.98 ms / sequence | 0.99 ms / sequence | +0.01 ms | **Matched ($<1\text{ms}$)** |

---

## 3. Retraining Enhancements Verified in V2

1. **In-Plane Hand Rotation Augmentation ($\pm 18^\circ$):**
   - Hand coordinate pairs $(x, y)$ rotated in-plane without corrupting landmark presence flags (indices 63, 127) or upper pose presence (index 149).
   - **Result:** Orientation vulnerability in `HELP` $\to$ `MONEY` under $\pm 15^\circ$ hand tilt dropped from **33.3% error rate in V1 (3/9 errors) down to 0.0% in V2 (0/9 errors)**, with every tilted sample recognized at $>96.3\%$ confidence.
2. **Temporal Speed & Shift Warping:**
   - Applied controlled velocity jitter ($\pm 15\%$) and temporal shifting ($\pm 2$ frames) resampled to 30 frames.
   - Preserves gesture recognition across varying signing speeds.
3. **Class Imbalance & Minority Class Handling:**
   - Class-aware oversampling of minority signs (`pay`, `doctor`, `help`, `money`, `hospital`, `please`, `thank_you`).
   - Smoothed inverse-frequency class-weighted `CrossEntropyLoss`.
4. **Gaussian Landmark Coordinate Jitter:**
   - Coordinate-only noise $\mathcal{N}(0, 0.010)$ applied during training to prevent overfitting to fixed camera angles.

---

## 4. Known Edge-Case Limitations & Diagnostic Findings

### `DOCTOR` $\longleftrightarrow$ `PAY` Kinematic Overlap
- **Observation:** In the official WLASL dataset, `doctor` has **0 test samples** and `pay` has **only 1 test sample** (and only 4 training samples).
- **Finding:** Signer 5 performed both signs with nearly identical initial arm trajectory and hand approach geometry toward the non-dominant wrist/palm.
- **Impact:**
  - One validation sample of `doctor` (signer 60) is predicted as `pay`.
  - The single test sample of `pay` (signer 5) is predicted as `doctor`.
- **Status:** V2 maintains parity with V1 on this pair without regression. Complete resolution of this edge case requires capturing additional training videos with high-framerate finger-flick landmark velocity tracking rather than further architectural changes.

---

## 5. System Health & Integration Verification Results

### 5.1 Final Backend Test Status
- **Test Command:** `./backend/venv/bin/pytest -q`
- **Result:** **10 PASSED, 0 FAILED** (Contract verification for `/health`, `/labels`, `/predict/sequence`, `/predict/static`, and confidence filtering).
- **`/health` Output:**
  ```json
  {
    "status": "ok",
    "dynamic_model": "loaded",
    "static_model": "loaded",
    "vocabulary_size": 18
  }
  ```
- **`/labels` Output:** 18 distinct civic sign classes verified.

### 5.2 Final Frontend Build Status
- **Build Command:** `npm run build`
- **Result:** **Vite v6.4.3 production bundle built successfully in 183ms**.
- **Output Bundle:** `dist/assets/index-30gOsLJh.js` (600.80 kB │ gzip: 162.33 kB).

### 5.3 Final End-to-End Smoke Test Status
- **Harness Command:** `node test_v2_e2e_smoke_test.mjs`
- **Result:** **30 PASSED, 0 FAILED (100% Pass Rate)**.
- **Verified Workflow Steps:**
  1. FastAPI service online on port 8000 with V2 Bi-GRU loaded.
  2. Deaf client connected to live recognition adapter.
  3. Feature extraction pipeline extracts 21 hand landmarks, correct handedness (`[0..63]` left, `[64..127]` right), upper pose (`[128..148]`), and pose presence flag (`[149] = 1.0`).
  4. Sequence downsampling generates strictly $30 \times 150$.
  5. Performed `HELP` $\to$ recognized with **99.6% confidence** by V2 Bi-GRU $\to$ passed temporal stabilization $\to$ dispatched to Admin.
  6. Admin console received recognized Deaf message.
  7. Admin sent preset response ("An officer will assist you immediately").
  8. Deaf interface received Admin response.
  9. Avatar sign animation played **exactly twice**.
  10. Green camera border/glow activated on message reception.
  11. Complete conversation saved in History records.
  12. Low-confidence test case ($< 0.70$) verified:
      - Prediction confidence $= 50.09\%$
      - Status correctly reported as **Uncertain**
      - **Zero messages dispatched to Admin**.

---

## 6. Deployment State

```
[LIVE WEBCAM]
     │
     ▼
[MediaPipe Hands + Pose]
     │
     ▼  (Physical Left -> [0..63], Physical Right -> [64..127], Upper Pose -> [128..149])
[150-dim Feature Vector]
     │
     ▼  (2.60s rolling temporal buffer)
[Linear Downsampling to 30 × 150]
     │
     ▼  (HTTP POST /predict/sequence)
[FastAPI ML Service (Port 8000)]
     │
     ▼
[Dynamic Bi-GRU V2 (dynamic_bigru_v2.pt)]
  • 2-layer Bi-GRU (H=64, 174k params)
  • Masked Mean + Max Pooling
  • Latency: 0.78 ms
     │
     ▼
[Confidence Gate (Threshold = 0.70)]
     ├── If < 0.70  ──> Display "Uncertain" (Suppressed from Admin)
     │
     └── If >= 0.70 ──> Temporal Stabilization (2 cycles) ──> Dispatched to Admin Console
```

**SignBridge AI V2 ML deployment is complete, verified, and active.**
