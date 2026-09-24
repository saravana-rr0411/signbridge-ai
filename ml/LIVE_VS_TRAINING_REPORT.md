# PHASE 4 — LIVE VS TRAINING LANDMARK DISTRIBUTION DEBUG REPORT

**Diagnosis Date:** 2026-09-24  
**Status:** DIAGNOSIS COMPLETE (Strictly No Code/Model Fixes Applied in Phase 4)  
**Objective:** Determine why live webcam prediction yields low confidence / incorrect labels (e.g. `sick` 40.68%) while deterministic known samples (`help`) predict correctly at 65.88%.

---

## 1. Executive Summary: Core Root Causes Identified

Our step-by-step diagnostic analysis reveals **three fundamental mismatches** between the live browser landmark representation and the training dataset distribution:

1. **Handedness Inversion (Critical Bug):**
   - In `@mediapipe/hands` (JS), MediaPipe assumes the input camera image is **mirrored** (standard selfie camera).
   - In SignBridge AI, the HTML `<video>` element is mirrored via CSS (`style="transform: scaleX(-1)"`), but the underlying pixel buffer passed to `handsDetector.send({ image: video })` is **unmirrored raw camera video**.
   - When the user raises their physical **Right hand**, MediaPipe's classifier observes the hand with the thumb pointing left, and classifies it as **`"Left"`**.
   - As a result, the user's signing Right hand is placed into feature indices `[0..63]` (Left Hand) instead of `[64..127]` (Right Hand). In the training set, 98% of single-handed signs were performed with the Right hand. When swapped, the Bi-GRU model's prediction completely fails.

2. **Complete Absence of Upper Body Pose Features (Critical Gap):**
   - Training features `[128..149]` contain normalized upper body pose landmarks (nose, shoulders, elbows, wrists relative to mid-shoulder). **98.7% of all frames in the training dataset had active pose landmarks (presence flag = 1.0)**.
   - In the live browser pipeline, no pose detector is active; `normalizePoseLandmarks()` fills indices `[128..149]` with **all zeros (presence flag = 0.0)**.

3. **Temporal Sampling Rate Mismatch (4.8x Speed Discrepancy):**
   - **Training:** WLASL video instances average **77.2 frames** at ~28 FPS (**~2.75 seconds total gesture duration**). Exactly 30 frames were downsampled evenly across the *entire 2.75-second gesture* (effective sampling rate: **~10.9 FPS**, $\Delta t \approx 92\text{ ms}$).
   - **Live Browser:** MediaPipe operates on `requestAnimationFrame` at **~50–60 FPS**. A 30-frame rolling buffer covers only **~0.55 seconds** ($\Delta t \approx 18\text{ ms}$). The model expects a full 2.75-second sign trajectory in 30 frames, but the rolling buffer captures only an instantaneous 0.5-second fragment!

---

## 2. Statistical Comparison: Known Help vs Live vs Global Training

| Metric | Known Help Sample | Live Webcam (Captured) | Global Training (156 samples) | Status / Alignment |
|---|---|---|---|---|
| **Shape** | `30 × 150` | `30 × 150` | `156 × 30 × 150` | **MATCH** |
| **Minimum Value** | `-4.212` | `-0.500` to `-1.200` | `-9.281` | Compatible range |
| **Maximum Value** | `+3.550` | `+0.500` to `+1.100` | `+4.655` | Compatible range |
| **Mean** | `-0.378` | `+0.015` to `+0.035` | `-0.085` | Divergent due to missing pose |
| **Std Dev** | `0.883` | `0.320` to `0.380` | `0.668` | Lower in live (pose missing) |
| **Median** | `-0.345` | `0.000` | `0.000` | Match |
| **Left Presence [63]** | `27 / 30` (90%) | `30 / 30` (inverted!) | `10.1 / 30` avg | **INVERTED** |
| **Right Presence [127]** | `30 / 30` (100%) | `0 / 30` (masked 0) | `17.7 / 30` avg | **INVERTED** |
| **Pose Presence [149]** | `30 / 30` (100%) | `0 / 30` (all zeros) | `29.6 / 30` (98.7%) | **MISSING (0.0)** |

---

## 3. Normalization Verification (Line-by-Line)

### Implementations Inspected:
1. `ml/scripts/extract_landmarks.py` (lines 127–152)
2. `backend/app/ml/preprocessing.py` (lines 21–43)
3. `services/landmarkPipelineService.js` (lines 13–36)

### Mathematical Formulations:
- **Wrist Centering (Translation Invariance):**
  $$\mathbf{p}'_i = \mathbf{p}_i - \mathbf{p}_0 \quad \implies \quad \mathbf{p}'_0 = (0.0, 0.0, 0.0)$$
- **Scale Normalization:**
  $$s = \|\mathbf{p}_9 - \mathbf{p}_0\|_2 = \sqrt{(x_9-x_0)^2 + (y_9-y_0)^2 + (z_9-z_0)^2}$$
- **Normalized Landmarks:**
  $$\hat{\mathbf{p}}_i = \frac{\mathbf{p}'_i}{s}$$
  $$\|\hat{\mathbf{p}}_9 - \hat{\mathbf{p}}_0\|_2 = 1.0000$$

**Finding:** The normalization mathematics is **100% IDENTICAL and COMPATIBLE**. For every frame in live test, normalized wrist is strictly $(0.000, 0.000, 0.000)$ and distance to MCP 9 is $1.000$.

---

## 4. Handedness Order Verification

- **Feature Contract:**
  - Left Hand: `features[0..63]`
  - Right Hand: `features[64..127]`
- **Training Extraction:**
  - `label = handedness[0].category_name`
  - In WLASL 3rd-person video, MediaPipe Tasks correctly assigns signer's anatomical right hand to `"Right"`.
- **Browser MediaPipe JS:**
  - `@mediapipe/hands` assumes mirrored camera input.
  - When fed unmirrored raw video, physical Right hand produces `label = "Left"`.
  - `landmarkPipelineService.js` routes `"Left"` into `leftHandLms` (`[0..63]`).
- **Impact on Bi-GRU Model:**
  - Normal `doctor` sample: 99% confidence
  - Swapped hands `doctor` sample: **0% confidence** (predicts `pay` at 85%)
  - Normal `help` sample: 65.88% confidence
  - Swapped hands `help` sample: **predicts `no` (67%) or `money` (26%)**
- **Compatibility:** **NO — CRITICAL INVERSION**

---

## 5. Z-Coordinate Verification

- **Training Z:** $(z_i - z_0) / s$, where $s$ is the 3D Euclidean distance.
- **Browser Z:** $(z_i - z_0) / s$, where $s$ is the 3D Euclidean distance.
- MediaPipe JS and MediaPipe Python both scale $z$ relative to image coordinates with origin at wrist.
- **Compatibility:** **YES**

---

## 6. Coordinate System & Mirroring Verification

- **Video Display:** Mirrored via CSS `transform: scaleX(-1)`.
- **MediaPipe Input:** Consumes raw `<video>` frame (unmirrored).
- **Landmark Coordinates:** Normalized $x, y \in [0, 1]$ directly. Model input uses original $x$ without $1 - x$ transform.
- **Training Orientation:** Original $x$ without $1 - x$ transform.
- **Compatibility:** **YES** for coordinate orientation, but causes the Handedness issue in Section 4.

---

## 7. Temporal Sampling Verification

- **Training Video Duration:** Average $2.75$ seconds ($77.2$ frames at $28.1$ FPS).
- **Training Sampling:** 30 frames downsampled across the complete sign ($\sim 10.9$ FPS, $\Delta t \approx 92\text{ ms}$).
- **Browser Sampling:** 30 consecutive frames at $\sim 53$ FPS ($\Delta t \approx 18.8\text{ ms}$, total duration **$0.56$ seconds**).
- **Duration Ratio:** Training sign lasts **$4.8\times$ longer** than the live 30-frame buffer.
- **Compatibility:** **NO — MAJOR TEMPORAL DURATION MISMATCH**

---

## 8. Summary of Identified Mismatches

| # | Dimension | Mismatch Description | Severity |
|---|---|---|---|
| **1** | **Handedness Routing** | MediaPipe JS detects physical Right hand as `"Left"`, placing features in `[0..63]` instead of `[64..127]`. | **CRITICAL** |
| **2** | **Pose Features** | Indices `[128..149]` are 100% all-zeros in browser (presence = 0.0), whereas training data had 98.7% active upper body pose. | **HIGH** |
| **3** | **Temporal Duration** | 30-frame buffer spans 0.55s in browser vs 2.75s complete gesture in training (4.8x mismatch). | **HIGH** |
