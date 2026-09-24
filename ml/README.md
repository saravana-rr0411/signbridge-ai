# SignBridge AI — Machine Learning Pipeline Architecture

This directory houses the real machine learning pipeline, dataset configurations, landmark extraction routines, neural network architectures, and evaluation benchmarks for the **SignBridge AI** platform (PS-09).

---

## 1. Datasets Used

| Dataset | Modality | Primary Role | License |
| :--- | :--- | :--- | :--- |
| **WLASL** (Word-Level American Sign Language) | Video clips (MP4/AVI) | Temporal sign recognition (17 dynamic classes) | C-UDA v1.0 |
| **ASL Alphabet** | 200×200 Static RGB images | Static handshape recognition (1 static class: `letter_a`) | CC BY-SA 4.0 |
| **MS-ASL** *(Supplementary)* | Video clips (YouTube) | Cross-dataset background/lighting validation | Microsoft Open Data |

> [!WARNING]
> **Modality Integrity:** Static images (ASL Alphabet) and video sequences (WLASL) are never mixed blindly. Static images contain no velocity or path trajectories and are routed solely through static landmark MLP models, whereas dynamic signs pass through temporal sequence models.

---

## 2. Verified 18-Class Civic Vocabulary

The MVP implements an 18-class high-utility vocabulary designed for public-service reception desks (hospitals, municipal desks, banks, and transit centers):

```json
[
  "help", "doctor", "hospital", "emergency", "appointment",
  "where", "bathroom", "yes", "no", "please", "thank_you",
  "wait", "understand", "problem", "money", "bank", "document",
  "letter_a"
]
```

Full details and linguistic mappings are defined in [`ml/config/vocabulary.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/config/vocabulary.json) and [`ml/config/dataset_manifest.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/config/dataset_manifest.json).

---

## 3. Landmark Extraction & Coordinate Normalization

To ensure invariant recognition regardless of camera resolution, distance from camera, or lighting shifts, raw video frames are converted into normalized geometric landmarks via Google MediaPipe.

### Landmark Spec
- **Hand Landmarks:** 21 3D points per hand $(x, y, z) \times 2 = 42$ coordinates.
- **Pose Landmarks:** Key upper-body anchor points (nose, shoulders, elbows, wrists) to anchor arm trajectories relative to the torso.
- **Temporal Window:** Fixed 30-frame sliding window ($T = 30$) sampled at 20–25 FPS (representing ~1.2 to 1.5 seconds of signing motion).

### Invariant Coordinate Normalization
For every detected hand:
1. **Translation Invariance:** Subtract the wrist landmark $(x_0, y_0, z_0)$ from all 21 keypoints:
   $$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0$$
2. **Scale Invariance:** Compute Euclidean distance between wrist (0) and middle finger MCP joint (9) as normalization scalar $s$:
   $$s = \|\mathbf{p}_9 - \mathbf{p}_0\|_2$$
   $$\hat{\mathbf{p}}_i = \frac{\mathbf{p}'_i}{s}$$
3. **Missing Hand Imputation:** When one hand is out of frame (one-handed signs), missing landmarks are zero-padded with a missingness mask flag rather than causing a runtime crash.

---

## 4. Model Architectures

### A. Dynamic Temporal Sequence Model (WLASL Classes)
- **Input:** Tensor of shape `(Batch, 30 frames, 126 landmark features)`.
- **Architecture Options:**
  - **Option 1 (Baseline):** 2-layer Bidirectional GRU (Hidden dim: 128, Dropout: 0.3) followed by a Linear classification head.
  - **Option 2 (Spatial-Temporal):** 1D Temporal Convolution (kernel size 3) + 1-layer LSTM (Hidden dim: 128) + Dense softmax.
- **Inference Latency:** $< 12\text{ ms}$ on standard Apple Silicon / modern CPU.

### B. Static Handshape Model (ASL Alphabet Classes)
- **Input:** Tensor of shape `(Batch, 42 normalized hand features)`.
- **Architecture:** 3-layer MLP (`42 -> 128 -> 64 -> N_classes`) with BatchNorm and ReLU activations.
- **Inference Latency:** $< 2\text{ ms}$.

---

## 5. Training, Validation & Evaluation Strategy

1. **Signer-Disjoint Splitting:**
   - Samples are grouped strictly by signer identity (`GroupKFold`, 70% train, 15% validation, 15% test).
   - Zero test-set leakage: the model is tested on signers it has never seen during training.
2. **Evaluation Metrics:**
   - Multi-class Accuracy, Top-3 Accuracy.
   - Per-class Precision, Recall, and F1-score.
   - Confusion Matrix saved to `ml/evaluation/confusion_matrix.png`.
   - Environmental validation across Condition A (Controlled/Studio) vs Condition B (Ambient/Wild).

---

## 6. Real-Time Inference & FastAPI Integration

Inference is hosted in a decoupled FastAPI backend (`backend/app/main.py`), preventing any interference with the React/Vite frontend.

### Endpoints
- `GET /health` — Check backend and model loading status.
- `GET /labels` — Retrieve active 18-class label list and metadata.
- `POST /predict` — Single-frame static prediction.
- `POST /predict/sequence` — 30-frame temporal landmark sequence prediction.

### Temporal Debouncing & Stability Filter
Real-time webcams produce frame-by-frame jitter. To avoid sending premature or duplicate messages:
1. **Sliding Buffer:** The frontend maintains a 30-frame FIFO landmark buffer.
2. **Confidence Threshold:** Predictions with softmax confidence $< 0.70$ are classified as `"unknown"`.
3. **Temporal Debounce:** A sign must be continuously predicted with $> 0.70$ confidence across at least 3 consecutive windows before firing an event to the dialogue state.

---

## 7. Safety, Quality & Linguistic Boundaries

1. **Limited Vocabulary:** This model recognizes **only** the 18 trained classes. It does **not** understand arbitrary conversational sign language.
2. **ASL vs. ISL:** The benchmark datasets (WLASL and ASL Alphabet) represent **American Sign Language (ASL)**, not Indian Sign Language (ISL) or British Sign Language (BSL).
3. **Non-Universal:** This system is an accessibility bridge for structured counter interactions and must not be portrayed as universal sign language translation.
4. **Human Interpreter Safeguard:** Legal, surgical, or high-liability public services must always provide certified human sign language interpreters.

---

## 8. Directory Organization

```
ml/
├── config/
│   ├── vocabulary.json          # 18-class MVP vocabulary & schema
│   └── dataset_manifest.json    # Dataset provenance, licenses, and splits
├── datasets/
│   ├── raw/                     # Raw video and image archives (gitignored)
│   └── processed/               # Extracted landmark numpy arrays (gitignored)
├── notebooks/                   # Prototyping and EDA notebooks
├── scripts/
│   ├── extract_landmarks.py     # MediaPipe extraction & normalization
│   ├── train.py                 # PyTorch training & checkpointing
│   └── evaluate.py              # Performance reporting & confusion matrix
├── models/                      # Saved .pt / ONNX model artifacts
├── checkpoints/                 # Training epoch checkpoints
├── evaluation/                  # Reports and confusion matrix plots
├── DATASET_REPORT.md            # Human-readable dataset & ethics report
└── README.md                    # This architecture specification
```
