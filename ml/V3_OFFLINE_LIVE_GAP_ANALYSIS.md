# SignBridge AI — V3 Offline-to-Live Feature Distribution Gap Analysis

**Document Type:** Root-Cause Architectural Diagnostic Report  
**Investigated Model:** Candidate Model V3 (`dynamic_bigru_v3.pt`, $30 \times 168$, 22 dynamic classes)  
**Status:** **V3 REJECTED FOR PRODUCTION ACTIVATION — V2 REMAINS ACTIVE**  
**Core Finding:** While V3 achieved **78.79%** accuracy on the frozen offline WLASL test set, live webcam testing failed on almost all signs (0/3 on `HELP`, `YES`, `NO`, `THANK YOU`, `PLEASE`, `GOOD`, `BAD`, with only `HELLO` scoring 3/3).  
**Primary Diagnosis:** Severe offline-to-live feature distribution shift caused by:
1. **Camera Mirroring & Handedness Inversion** in the live capture script.
2. **Cross-Detector Z-Coordinate Incompatibility** between MediaPipe HandLandmarker ($z=0$ at wrist) and PoseLandmarker ($z=0$ at hips), creating meaningless body-relative depth features.
3. **Webcam Framing & Perspective Skew** (close-up upward laptop angle vs wide standing studio cameras), transforming neutral chest gestures into negative Y-offsets that act as a mathematical attractor for `HELLO`.
4. **Temporal Sampling Discrepancy** (1.0s raw buffer vs 2.6s uniform temporal window vs full isolated video resampling).

---

## 1. Executive Summary & Live Validation Trace

In the real webcam validation experiment, sign attempts produced the following empirical predictions:

| Tested Sign | Real Physical Gesture | Live V3 Predictions (3 Attempts) | Result | Primary Model Bias |
| :--- | :--- | :--- | :---: | :--- |
| **HELP** | Closed fist resting on flat palm | `problem` (98.6%), `problem` (99.0%), `problem` (99.0%) | 0 / 3 | Symmetrical two-handed chest attractor |
| **YES** | Closed fist nodding up/down | `where` (34.4%), `hello` (96.0%), `hello` (94.1%) | 0 / 3 | High hand elevation / single-hand attractor |
| **NO** | Index & middle snapping to thumb | `hello` (97.4%), `hello` (96.4%), `hello` (89.0%) | 0 / 3 | Single-hand high elevation attractor |
| **THANK YOU** | Flat hand chin moving forward | `hello` (94.5%), `pay` (93.7%), `hello` (94.9%) | 0 / 3 | Chin-level height mapped to forehead locus |
| **PLEASE** | Palm circular rubbing on chest | `problem` (90.3%), `problem` (72.9%), `wait` (89.8%) | 0 / 3 | Chest height mapped to two-handed prior |
| **HELLO** | Flat hand salute from temple | `hello` (95.9%), `hello` (93.9%), `hello` (95.1%) | **3 / 3** | **Only sign matching negative-Y prior** |
| **GOOD** | Flat hand chin to forward | `problem` (92.0%), `problem` (88.3%), `problem` (84.4%) | 0 / 3 | Two-handed desk presence bias |
| **BAD** | Flat hand chin flipping down | `appointment` (81.0%), `problem` (90.1%), `appointment` (97.7%) | 0 / 3 | Symmetrical two-handed prior |

**Observations:**
- Almost every single-handed sign collapsed into **`HELLO`** ($>94\%$ confidence).
- Almost every chest-level or two-handed sign collapsed into **`PROBLEM`** ($>88\%$ confidence) or **`APPOINTMENT`**.
- This extreme bi-modal collapse indicates that the Bi-GRU model was receiving features that severely violated the training distribution, triggering only the dominant priors of the classifier.

---

## 2. Comprehensive 14-Point Architectural Audit

Below is the line-by-line comparison between:
- **Pipeline A (Offline Training):** [`ml/scripts/extract_landmarks.py`](file:///Users/saravanarajaram0411/CLG/KPR/ml/scripts/extract_landmarks.py)
- **Pipeline B (Browser Live):** [`services/landmarkPipelineService.js`](file:///Users/saravanarajaram0411/CLG/KPR/services/landmarkPipelineService.js)
- **Pipeline C (Live Script):** [`ml/scripts/validate_v3_live_webcam.py`](file:///Users/saravanarajaram0411/CLG/KPR/ml/scripts/validate_v3_live_webcam.py)

---

### Audit 1: Handedness Mapping & Camera Mirroring

#### Pipeline A (`extract_landmarks.py`):
```python
if hand_res.hand_landmarks and hand_res.handedness:
  for lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
    label = handedness[0].category_name
    if label == "Left" and l_hand_lms is None:
      l_hand_lms = lms
    elif label == "Right" and r_hand_lms is None:
      r_hand_lms = lms
```
- Processes unmirrored third-person video frames directly from WLASL `.mp4` files.
- In WLASL, signers face the camera. The dominant hand is on the viewer's left, which MediaPipe correctly detects as **`Right`**.
- **Result in Training Data:** $99.0\%$ of all dynamic sign samples have the right hand populated (`features[:, :, 64..127]`). Only $1.0\%$ have left-hand-only signing.

#### Pipeline B (`services/landmarkPipelineService.js`):
```javascript
// MediaPipe JS on unmirrored raw camera feed inverts handedness:
// raw "Left" -> physical Right hand
// raw "Right" -> physical Left hand
const rawLabel = handednessObj
  ? handednessObj.label
  : i === 0
    ? 'Left'
    : 'Right';
const physicalLabel = rawLabel === 'Left' ? 'Right' : 'Left';
if (physicalLabel === 'Left') leftHandLms = lms;
else if (physicalLabel === 'Right') rightHandLms = lms;
```
- Correctly inverts raw MediaPipe JS label because raw video passed from `getUserMedia()` is unmirrored.

#### Pipeline C (`validate_v3_live_webcam.py` — The Script Executed):
```python
frame = cv2.flip(frame, 1)  # Line 296: Flips image horizontally
...
if label == "Left" and l_hand is None:  # Line 227: Maps 'Left' to l_hand!
  l_hand = hand_res.hand_landmarks[idx]
```
- **The Critical Flaw:** The camera frame was horizontally flipped with `cv2.flip(frame, 1)` *before* being processed by MediaPipe.
- Empirical test: When an image is flipped, MediaPipe's internal classifier flips its prediction:
  ```
  Vid 38538.mp4: Unflipped=['Right'] vs Flipped=['Left']
  ```
- Because `validate_v3_live_webcam.py` mapped `label == "Left"` to `l_hand` without inverting the label, the user's **dominant physical Right hand was placed into `l_hand` (indices `0..63`)**, and `r_hand` (indices `64..127`) was set to **all zeros**!
- When tested with simulated left/right swapping in Python, test set accuracy collapses immediately to $<35\%$.

---

### Audit 2: Wrist-Centered Normalization (Indices 0..63 and 64..127)

#### Pipeline A vs Pipeline B:
- **Formula (Both):**
  $$x_{norm} = \frac{x - wrist.x}{scale}, \quad y_{norm} = \frac{y - wrist.y}{scale}, \quad z_{norm} = \frac{z - wrist.z}{scale}$$
  where $scale = \sqrt{(p9.x - wrist.x)^2 + (p9.y - wrist.y)^2 + (p9.z - wrist.z)^2}$.
- **Verification:**
  - Translation invariance (wrist at origin $0,0,0$): **MATCHED**
  - Middle finger MCP scale normalization: **MATCHED**
  - Epsilon clamp ($1\times 10^{-4}$): **MATCHED**
  - Presence flags ($feat[63] = 1.0, feat[127] = 1.0$): **MATCHED**

---

### Audit 3: Cross-Detector Z-Coordinate Incompatibility (Indices 152, 155, 158, 161, 164, 167)

This is the **most fundamental mathematical flaw** in the V3 18-feature body-relative design.

#### The Problem:
In `compute_body_relative_features`:
```python
l_wrist = np.array(
    [l_hand_lms[0].x, l_hand_lms[0].y, l_hand_lms[0].z], dtype=np.float32
)
rel_feat[0:3] = (l_wrist - shoulder_center) / shoulder_width  # Z is index 2
rel_feat[6:9] = (l_wrist - nose) / shoulder_width  # Z is index 8
rel_feat[12:15] = (l_wrist - chest_center) / shoulder_width  # Z is index 14
```

1. **`l_wrist[2]` (MediaPipe HandLandmarker):**
   - MediaPipe HandLandmarker processes a localized crop around the hand.
   - Landmark 0 IS the wrist.
   - Therefore, by definition of the coordinate system, **`l_wrist[2]` is ALWAYS $0.0000$**.
2. **`nose[2]` and `shoulder_center[2]` (MediaPipe PoseLandmarker):**
   - MediaPipe PoseLandmarker outputs 3D coordinates relative to the person's hips (in meters).
   - In training data: `nose.z` is typically $-1.0$ to $-1.2\text{m}$, `shoulders.z` is $-0.3$ to $-0.5\text{m}$.
3. **The Result:**
   $$dz_{nose} = \frac{l\_wrist[2] - nose[2]}{shoulder\_width} = \frac{0.0 - (-1.18)}{0.35} \approx +3.37$$
   $$dz_{shoulder} = \frac{l\_wrist[2] - shoulder\_center[2]}{shoulder\_width} = \frac{0.0 - (-0.52)}{0.35} \approx +1.48$$

**Mathematical Consequence:**
The Z-coordinates in features 152, 155, 158, 161, 164, 167 do **NOT** measure the distance from the hand to the body! Because `wrist.z` is identically $0.0$, the numerator is literally just $-nose.z$ or $-shoulder.z$.
- In WLASL videos, signers stand 2.5 meters away; $nose.z$ is around $-1.2$.
- In a desktop webcam, users sit 50 cm away; $nose.z$ changes drastically.
- When the user leans in or out, these 6 features swing between $+1.0$ and $+5.6$, completely disorienting the Bi-GRU.

---

### Audit 4: Shoulder-Center & Chest-Center Calculation (Indices 150..167)

#### Offline Formula (`extract_landmarks.py`):
```python
shoulder_center = (l_shoulder + r_shoulder) / 2.0
shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
down_vec = shoulder_center - nose
unit_down = down_vec / np.linalg.norm(down_vec)
chest_center = shoulder_center + unit_down * (0.5 * shoulder_width)
```

#### Browser Formula (`landmarkPipelineService.js`):
```javascript
const midX = (lShoulder.x + rShoulder.x) / 2.0;
const midY = (lShoulder.y + rShoulder.y) / 2.0;
const midZ = ((lShoulder.z || 0) + (rShoulder.z || 0)) / 2.0;
let downX = midX - nose.x;
let downY = midY - nose.y;
let downZ = midZ - (nose.z || 0);
let uDown = down / Math.hypot(downX, downY, downZ);
const chestX = midX + uDownX * (0.5 * shoulderWidth);
const chestY = midY + uDownY * (0.5 * shoulderWidth);
const chestZ = midZ + uDownZ * (0.5 * shoulderWidth);
```

- **Formula Alignment:** The vector math between JavaScript and Python is mathematically identical.
- **Physical Defect:** When a user is seated at a laptop, the camera angle is tilted upwards. The neck appears compressed from this low perspective, so $shoulder\_center.y - nose.y$ is smaller by $40\%-60\%$ compared to eye-level studio cameras.
- This places `chest_center` significantly higher relative to the torso than it was during training, causing chest gestures (`PLEASE`) to appear displaced into the abdomen region in feature space.

---

### Audit 5: Camera Perspective & Elevation Bias (The `HELLO` Attractor)

Why did almost every sign trigger `HELLO`?

In `dynamic_landmarks_v3.npz`, we inspect the vertical offset of the right wrist relative to the shoulders ($Y_{rel} = \frac{wrist.y - shoulder.y}{shoulder\_width}$):

| Class | Mean $Y_{rel}$ (Wrist to Shoulder) | Physical Meaning |
| :--- | :---: | :--- |
| **HELLO** | **$-0.2103$** | **Hand is ABOVE shoulder line (near forehead/temple)** |
| **FOOD** | $+0.0513$ | Hand is at chin level |
| **GOOD** | $+0.1840$ | Hand moves from chin to mid-chest |
| **PLEASE** | $+0.4383$ | Hand is on chest center |
| **HELP** | $+0.4275$ | Hand rests on palm at waist/chest |
| **PROBLEM** | $+0.3565$ | Hands are at lower chest |

- In a laptop camera looking up from the desk, the user's shoulders are at the bottom of the frame ($y \approx 0.85$).
- When the user signs *anything* with their hand raised in front of the screen, the hand's $y$ coordinate is $\approx 0.50$.
- Therefore, $y_{wrist} - y_{shoulder} \approx 0.50 - 0.85 = -0.35$ (NEGATIVE!).
- **Because `HELLO` is the ONLY sign in the entire 22-class vocabulary with a negative vertical shoulder offset**, the Bi-GRU's classifier assigns $>95\%$ probability to `HELLO` regardless of finger shapes!

---

### Audit 6: Symmetrical Desk Presence (The `PROBLEM` Attractor)

Why did `HELP`, `PLEASE`, `GOOD`, and `BAD` trigger `PROBLEM`?

- In ASL, `PROBLEM` is a two-handed sign performed with both hands at lower chest level.
- Symmetrical features in training data for `PROBLEM`:
  - `Left wrist to shoulder Y`: $+0.3570$
  - `Right wrist to shoulder Y`: $+0.3565$
  - Both Left (index 63) and Right (index 127) presence flags are $1.0$.
- In a live webcam setup:
  - If a user rests their non-dominant hand near the keyboard or desk while signing with their right hand, MediaPipe detects both hands.
  - The model observes two active hands in front of the lower torso.
  - Coupled with the handedness inversion (which placed the moving hand into the Left slot), the Bi-GRU matches the dual-hand chest pattern of `PROBLEM`.

---

### Audit 7: Temporal Window Discrepancy

| Parameter | Offline Training (`extract_landmarks.py`) | Browser Pipeline (`landmarkPipelineService.js`) | Live Script (`validate_v3_live_webcam.py`) |
| :--- | :---: | :---: | :---: |
| **Capture Mechanism** | Isolated video from start to end | 2.6-second rolling temporal buffer | 30-frame FIFO `deque(maxlen=30)` |
| **Time Span** | Entire sign duration ($100\%$ gesture) | Fixed 2.6 seconds | Exactly 1.0 second ($30 \text{ frames} / 30 \text{ FPS}$) |
| **Downsampling** | `np.linspace(0, total-1, 30)` | Uniform sample across 2.6s | No downsampling (instantaneous frames) |
| **Movement Captured**| Complete trajectory (start $\to$ apex $\to$ end) | Gesture with idle lead/tail padding | Only the last 1.0s or frozen static pose |

In `validate_v3_live_webcam.py`, pressing `[SPACE]` captured the instantaneous 30-frame buffer. If the user held their hand still while pressing space, all 30 frames were near-identical static frames. As tested, static frames cause the Bi-GRU to default to dominant training priors.

---

## 3. Summary of Code Audits across the 14 Points

| # | Checkpoint | Status | Impact / Finding |
| :-: | :--- | :---: | :--- |
| **1** | **Handedness Mapping** | **FAIL in Live Script** | `cv2.flip` inverted MediaPipe handedness; right hand stored in left slot (indices 0..63). |
| **2** | **Camera Mirroring** | **FAIL in Live Script** | Live script flipped frame before inference without reversing handedness logic. |
| **3** | **Wrist Normalization** | **PASS** | Exact translation & scale formulas matched across all pipelines. |
| **4** | **Scale Normalization** | **PASS** | Middle MCP distance normalization matched. |
| **5** | **Z Normalization** | **ARCHITECTURAL FLAW** | `HandLandmarker` wrist $z=0$ subtracted from `PoseLandmarker` hip-centered $z$. Uncorrelated with physical hand depth. |
| **6** | **Pose Normalization** | **PASS** | Mid-shoulder translation and shoulder-width scale matched. |
| **7** | **Shoulder-Center** | **PASS in code / SKEWED in live** | Formula matches, but camera tilt shifts shoulder vertical locus. |
| **8** | **Chest-Center** | **PASS in code / SKEWED in live** | Formula matches, but upwards camera angle shifts chest locus upwards. |
| **9** | **Wrist-to-Nose** | **AFFECTED by Z & Angle** | Z component corrupted; Y component distorted by camera tilt. |
| **10** | **Wrist-to-Chest** | **AFFECTED by Z & Angle** | Z component corrupted; Y component distorted by camera tilt. |
| **11** | **Coordinate Axes** | **PASS** | Normalized image coordinates $[0, 1]$ used consistently. |
| **12** | **Presence Flags** | **PASS** | Exact binary flags ($1.0 / 0.0$) at indices 63, 127, 149. |
| **13** | **Temporal Sampling** | **MISMATCH in Live Script** | 1.0s instantaneous deque used instead of 2.6s rolling window downsampling. |
| **14** | **30-Frame Sequence** | **PASS in shape / MISMATCH in dynamics** | Strictly $30 \times 168$, but temporal velocity and progression mismatched. |

---

## 4. Final Recommendation & Operational Status

### **V3 MUST REMAIN INACTIVE. DO NOT TRAIN V4 AT THIS STAGE.**

1. **Preserve Production V2:**
   - The production system ([`backend/app/main.py`](file:///Users/saravanarajaram0411/CLG/KPR/backend/app/main.py), WebRTC, and the Deaf/Admin UI) remains running on the **V2 model** (`dynamic_bigru_v2.pt`, $30 \times 150$, 17 classes).
   - V2 does NOT depend on the flawed cross-detector body-relative Z features ($150..167$) and has been verified to work on the live browser interface.
2. **Key Architectural Lesson:**
   - Combining depth coordinates across separate MediaPipe models (Hands vs Pose) without a shared 3D camera extrinsic calibration produces unnormalized noise.
   - Body-relative spatial features in 2D ($X, Y$ normalized by shoulder width) are geometrically sound, but 3D ($Z$) must be excluded unless derived from a unified holistic model or depth sensor.
   - Live camera applications must strictly preserve the 2.6s temporal downsampling buffer to match offline sequence velocity.
