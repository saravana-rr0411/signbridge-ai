# SignBridge AI — V5 Dataset Improvement & Audit Plan

**Date:** 2026-09-24  
**Audit Scope:** Re-audit & Expansion Plan for 4 Weak Signs (`NO`, `PLEASE`, `SICK`, `WHERE`)  
**Status:** Audit & Collection Planning Phase — **NO MODEL TRAINING & NO PRODUCTION ACTIVATION**  
**Policy Directives:**
- Production V3 Six-Sign Model remains active and untouched.
- V2 remains fallback.
- V4 remains experimental.
- WebRTC, FastAPI, UI, and the 70% threshold remain completely untouched.
- **`BATHROOM` is temporarily excluded from V5 scope** to decouple critical confusion pairs.

---

## 1. Executive Summary & Policy Directives

In the two independent live webcam validation sessions of the V4 11-sign model, 6 signs (`HELLO`, `HELP`, `YES`, `THANK_YOU`, `DOCTOR`, `PAIN`) demonstrated **100% empirical repeat accuracy** with $>99\%$ mean confidence.

However, four existing signs exhibited lower empirical accuracy and specific kinematic confusion:
- **`NO` (60.0% live accuracy):** Rapid snap causes motion blur / tracking loss; slower execution mimics `PAIN`.
- **`PLEASE` (60.0% live accuracy):** Flat hand circling chest overlaps with `THANK_YOU`'s forward extension.
- **`SICK` (60.0% live accuracy):** Forehead hand release descending past chin mimics `THANK_YOU`.
- **`WHERE` (80.0% live accuracy):** Horizontal waggle collides with `BATHROOM`'s lateral fist shake; starved validation split ($N=2$).
- **`BATHROOM` (20.0% live accuracy):** Severe bilateral confusion with `WHERE` and `YES`.

### Strategic Decision: Temporary Exclusion of BATHROOM from V5
1. **Kinematic Decoupling:** `BATHROOM`'s lateral oscillatory shaking directly confounded `WHERE`'s single-finger sway. Removing `BATHROOM` instantly removes the primary false high-confidence attractor for `WHERE`.
2. **Handshape Ambiguity:** At standard webcam resolutions, the 'T' fist (thumb between index and middle) is visually indistinguishable from an S-fist (`YES`) under motion blur.
3. **Target V5 Vocabulary (10 Classes):**
   - **Original 6:** `HELLO`, `HELP`, `YES`, `NO`, `PLEASE`, `THANK_YOU`
   - **Approved Medical 4:** `DOCTOR`, `PAIN`, `SICK`, `WHERE`

---

## 2. Re-Audit of Existing Dataset for the 4 Target Signs

Detailed inspection of [`ml/datasets/processed/dynamic_landmarks_v4_11_sign.npz`](file:///Users/saravanarajaram0411/CLG/KPR/ml/datasets/processed/dynamic_landmarks_v4_11_sign.npz) and [`metadata_v4_11_sign.csv`](file:///Users/saravanarajaram0411/CLG/KPR/ml/datasets/processed/metadata_v4_11_sign.csv):

| Metric | NO | PLEASE | SICK | WHERE |
| :--- | :---: | :---: | :---: | :---: |
| **Total Usable Samples** | **48** | **33** | **39** | **32** |
| **Unique Signers** | 36 signers | 25 signers | 26 signers | 29 signers |
| **Train Count** | 28 | 21 | 19 | 24 |
| **Validation Count** | 10 | 4 | 9 | **2** *(Starved)* |
| **Test Count** | 10 | 8 | 11 | 6 |
| **Source: MS-ASL** | 31 (64.6%) | 24 (72.7%) | 29 (74.4%) | 24 (75.0%) |
| **Source: WLASL** | 16 (33.3%) | 8 (24.2%) | 10 (25.6%) | 8 (25.0%) |
| **Source: Educational ASL** | 1 (2.1%) | 1 (3.0%) | 0 (0.0%) | 0 (0.0%) |
| **Mean Frame Count** | 91.5 frames | 97.8 frames | 96.8 frames | 102.8 frames |
| **Duration Range (sec)** | 0.73s – 10.9s | 0.97s – 8.3s | 0.87s – 12.5s | 1.2s – 6.3s |
| **Mean Hand Coverage %** | 77.1% | 81.9% | 80.5% | **74.8%** *(Lowest)* |
| **Low Hand Coverage (<50%)**| **5 samples** | **2 samples** | **1 sample** | **4 samples** |

### Specific Problematic Samples Identified for Pruning
These 12 samples suffer from severe hand tracking degradation ($<50\%$ frame coverage) or camera crop artifacts and must be pruned prior to V5 training:
1. **`NO` (5 samples):**
   - `38539` (WLASL, signer 47, train): 26.7% coverage — hand drops out of frame immediately after snap.
   - `38544` (WLASL, signer 41, train): 30.0% coverage — severe motion blur during snap.
   - `66183` (WLASL, signer 102, train): 36.7% coverage — low contrast / dark lighting.
   - `38524` (WLASL, signer 56, train): 30.0% coverage — hand occluded by signer body angle.
   - `msasl_4_115` (MS-ASL, signer 38, train): 30.0% coverage — extreme high camera tilt angle.
2. **`PLEASE` (2 samples):**
   - `msasl_47_132` (MS-ASL, signer 72, train): 43.3% coverage — hand tracking lost against patterned shirt.
   - `msasl_47_136` (MS-ASL, signer 215, train): 46.7% coverage — low resolution webcam blur.
3. **`SICK` (1 sample):**
   - `51501` (WLASL, signer 63, test): 46.7% coverage — forehead shadow causes landmark jitter.
4. **`WHERE` (4 samples):**
   - `63081` (WLASL, signer 42, train): 30.0% coverage — severe wrist blur during waggle.
   - `63082` (WLASL, signer 36, train): 46.7% coverage — lower frame edge crop.
   - `63083` (WLASL, signer 70, test): 43.3% coverage — single-finger tracking dropped to palm center.
   - `msasl_30_211` (MS-ASL, signer 72, train): 43.3% coverage — background hand confusion.

---

## 3. Root Cause Analysis of Confusion Pairs

### 1. `NO` vs `PAIN`
- **Observed:** In live validation Round 2, attempt 4 of `NO` produced a false high-confidence prediction of `PAIN` (99.4%).
- **Kinematic Mechanism:** `NO` is executed by snapping the extended index and middle fingers down onto the thumb. `PAIN` is executed by pointing both index fingers inward towards each other with twisting/converging motions. When a signer executes `NO` with a slower tempo or slight diagonal wrist orientation, the two extended fingers moving across the frame mimic the converging trajectory of `PAIN`.
- **Temporal Aliasing:** The snap phase of `NO` often lasts only 3–5 frames ($100$–$160$ ms). When uniformly subsampled to 30 frames, if the closure moment falls between sample points, the model only sees extended fingers moving across space.
- **Improvement Strategy:**
  - Prune the 5 low-coverage samples.
  - Ingest 17 new high-FPS samples showing distinct 2-finger-to-thumb pinches.
  - Implement temporal boundary centering and multi-tempo snap augmentation.

### 2. `PLEASE` vs `THANK_YOU`
- **Observed:** In Round 1, 1 attempt confused with `THANK_YOU`; in Round 2, 2 attempts confused with `THANK_YOU` (99.7% and 99.9% confidence).
- **Kinematic Mechanism:** Both gestures feature open flat palms positioned on the upper body (`PLEASE` rubs the center chest; `THANK_YOU` starts at the chin/lips and extends forward/downward). If a signer leans forward while performing `PLEASE`, or if the circular stroke is wide, the outward/downward recovery of the hand produces large positive $z$ and negative $y$ displacements that match `THANK_YOU`.
- **Under-Representation:** `PLEASE` only has 21 training samples. The 2-layer BiGRU lacks sufficient variance to decouple a closed circular chest manifold from a linear outward release.
- **Improvement Strategy:**
  - Ingest 19 new candidate samples with tight chest contact across diverse body types.
  - Add body-relative spatial anchoring penalties for open-palm gestures moving beyond the chest plane.

### 3. `SICK` vs `THANK_YOU`
- **Observed:** In Round 2, attempts 4 & 5 of `SICK` produced false high-confidence predictions of `THANK_YOU` (88.3% and 79.5%).
- **Kinematic Mechanism:** `SICK` requires contacting the forehead with the dominant middle finger and the stomach with the non-dominant middle finger. In several training clips, after contacting the forehead, the dominant hand drops downward past the chin/chest to rest. The model learned to latch onto this post-contact downward descent, confusing it with `THANK_YOU`'s release.
- **Missing Secondary Anchor:** In many single-camera crops, the non-dominant hand at the stomach was either occluded or cut off by the lower frame boundary.
- **Improvement Strategy:**
  - Ingest 17 new strictly two-handed clips where both forehead and stomach contacts are explicitly detected and tracked.
  - Truncate post-sign recovery frames to eliminate the downward hand drop artifact.

### 4. `WHERE` vs `BATHROOM`
- **Observed:** In Round 2, `BATHROOM` caused 2 false high-confidence `WHERE` predictions (86.5%, 88.7%), and `WHERE` had a low-confidence rejection (57.0%).
- **Kinematic Mechanism:** Both gestures utilize horizontal oscillatory motion. When MediaPipe finger detection certainty is low, the spatial feature pipeline tracks the hand centroid oscillating back and forth, rendering the index waggle and the 'T'-fist shake identical in feature space.
- **Improvement Strategy:**
  - **Temporarily exclude `BATHROOM` from V5.** This completely eliminates the bilateral attractor.
  - Collect 22 new samples for `WHERE` with clear upright index finger pointing and minimal palm-body occlusion.

---

## 4. Candidate Discovery Pool for the 4 Weak Signs

An extensive audit of available public repositories identified abundant candidate instances to reach and exceed the 40+ sample threshold:

### Identified Available Candidates

| Sign | Currently Usable | Available in WLASL (Uncollected) | Available in MS-ASL (Uncollected) | Educational ASL Candidates | Total Available Pool | Target V5 Usable |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NO** | 48 | 6 | 24 | 4 | **82** | **60** |
| **PLEASE** | 33 | 7 | 23 | 5 | **68** | **50** |
| **SICK** | 39 | 8 | 14 | 4 | **65** | **55** |
| **WHERE** | 32 | 7 | 27 | 4 | **70** | **50** |
| **TOTAL** | **152** | **28** | **88** | **17** | **285** | **215** |

### Quality & Diversity Criteria for V5 Ingestion:
1. **Resolution & Lighting:** Clear hand-to-background contrast with $>80\%$ MediaPipe hand detection coverage across all 30 frames.
2. **Signer Independence:** Ensure newly collected signers do not violate the 0 cross-split signer overlap constraint against existing test signers.
3. **Kinematic Verification:**
   - `NO`: Must include distinct finger closure (snap) without immediate drop out of frame.
   - `PLEASE`: Must feature clear circular chest trajectory without outward camera thrust.
   - `SICK`: Must have simultaneous visibility of both forehead and stomach keypoints.
   - `WHERE`: Must exhibit clear upright index finger orientation throughout the lateral waggle.

---

## 5. Current vs Target Counts & Readiness Assessment

| Sign | Current Usable | Pruning Count | Recommended to Collect | Target Usable Count | Target Train / Val / Test | Ready for V5 Training? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NO** | 48 | -5 | +17 | **60** | 42 / 10 / 8 | ❌ **NOT READY** *(Needs pruning + snap-centering)* |
| **PLEASE** | 33 | -2 | +19 | **50** | 35 / 7 / 8 | ❌ **NOT READY** *(Under 40 samples, train starved at 21)* |
| **SICK** | 39 | -1 | +17 | **55** | 37 / 9 / 9 | ❌ **NOT READY** *(Train starved at 19, needs 2-hand anchor)* |
| **WHERE** | 32 | -4 | +22 | **50** | 35 / 8 / 7 | ❌ **NOT READY** *(Under 40 samples, val starved at 2)* |
| **BATHROOM**| 38 | N/A | N/A | **EXCLUDED** | N/A | ⛔ **EXCLUDED FROM V5** *(Accuracy 20%, severe collisions)*|

### Summary Statement on Training Readiness
**None of the 4 weak signs are currently ready for V5 training.**  
While the remaining 6 classes (`HELLO`, `HELP`, `YES`, `THANK_YOU`, `DOCTOR`, `PAIN`) have proven 100% stable, training V5 immediately on the current raw dataset would perpetuate the exact same confusion pairs and validation starvation.

---

## 6. Exact Next Actions

1. **Step 1 — Prune Degraded Samples:**
   - Execute automated dataset filter script to drop the 12 identified low-coverage samples (<50% hand detection) across `NO`, `PLEASE`, `SICK`, and `WHERE`.
2. **Step 2 — Download & Ingest Candidate Pool (75 new clips):**
   - Ingest 17 new `NO` clips from MS-ASL / WLASL.
   - Ingest 19 new `PLEASE` clips with tight chest circles.
   - Ingest 17 new `SICK` clips with verified two-handed visibility.
   - Ingest 22 new `WHERE` clips with clear upright index waggle.
3. **Step 3 — Extract Landmarks & Validate Signer Disjointness:**
   - Extract $30 \times 168$ features using the existing MediaPipe + Pose wrist pipeline.
   - Run programmatic signer check to enforce strictly 0 cross-split signer overlap.
4. **Step 4 — V5 Model Preparation:**
   - Adjust classification head to 10 classes (`HELLO`, `HELP`, `YES`, `NO`, `PLEASE`, `THANK_YOU`, `DOCTOR`, `PAIN`, `SICK`, `WHERE`).
   - Keep V3 production completely untouched.
