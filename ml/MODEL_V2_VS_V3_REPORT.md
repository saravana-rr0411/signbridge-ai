# SignBridge AI — Model V2 vs V3 Comparative Evaluation Report

**Document Purpose:** Rigorous Offline Benchmark & Multi-Class Spatial Feature Analysis  
**Evaluation Scope:** Head-to-Head Comparison of Production Model (V2) vs Candidate Model (V3)  
**Production Model (V2):** [`ml/models/dynamic_bigru_v2.pt`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_bigru_v2.pt) (30 × 150 input, 17 classes)  
**Candidate Model (V3):** [`ml/models/dynamic_bigru_v3.pt`](file:///Users/saravanarajaram0411/CLG/KPR/ml/models/dynamic_bigru_v3.pt) (30 × 168 input, 22 classes)  
**Evaluation Dataset:** Frozen WLASL Test Split (N=26 for V2; N=33 for V3: 26 existing-class + 7 new-class instances)  
**Status:** **V3 INACTIVE** — Live system strictly preserves V2 in production  

---

## 1. Executive Summary & Key Evaluation Findings

The V3 model introduces two architectural advancements over the production V2 model:
1. **Body-Relative Spatial Coordinate Expansion (Option 3):** Appends 18 normalized spatial coordinates (wrist-to-shoulder-center, wrist-to-nose, wrist-to-chest-center) to resolve chest-anchored signs such as `PLEASE` and mouth-directed signs such as `FOOD`.
2. **Vocabulary Expansion (Option 2):** Adds 5 dynamic civic classes (`HELLO`, `GOOD`, `BAD`, `WATER`, `FOOD`), expanding the dynamic vocabulary from 17 to 22 classes.

### Primary Benchmark Results:
- **Existing 17-Class Performance:**
  - **V2:** **80.77%** test accuracy (21 / 26 correct), Macro F1 = **0.7137**, Weighted F1 = **0.7923**
  - **V3:** **76.92%** test accuracy (20 / 26 correct), Macro F1 = **0.6255**, Weighted F1 = **0.7410**
  - *Delta:* $-3.85\%$ accuracy (1 additional test mistake: `where` $\to$ `bathroom`).
- **New 5-Class Performance (V3):**
  - **V3:** **85.71%** test accuracy (6 / 7 correct), Macro F1 = **0.8000**, Weighted F1 = **0.8571**
  - High individual accuracy on `HELLO` (100%), `BAD` (100%), `WATER` (100%), `FOOD` (100%).
  - 1 mistake on `GOOD` (predicted as `THANK_YOU`, sharing the identical chin-contact open-palm hand movement).
- **Overall 22-Class Performance (V3):**
  - **V3:** **78.79%** overall test accuracy (26 / 33 correct), Macro F1 = **0.6652**, Weighted F1 = **0.7556**
- **Inference Latency (CPU):**
  - **V2:** Mean **0.70 ms**, P95 **0.71 ms**
  - **V3:** Mean **0.70 ms**, P95 **0.71 ms** (identical, $<1\text{ms}$ real-time constraint satisfied).
- **Model Parameters & Storage:**
  - **V2:** 174,993 parameters, 2.12 MB checkpoint
  - **V3:** 182,230 parameters (+7,237 params, $+4.14\%$), 2.21 MB checkpoint (+0.09 MB)

> [!IMPORTANT]
> **Activation Decision:** **V3 should REMAIN INACTIVE pending physical webcam validation.**
> Although V3 successfully learned 4 of the 5 new classes with high confidence, its test accuracy on the existing 17 classes dropped slightly (80.77% $\to$ 76.92%), `GOOD` is frequently confused with `THANK_YOU`, and live coordinate normalization stability in varying webcam framings must be verified before replacing the stable V2 production engine.

---

## 2. Quantitative Head-to-Head Comparison

| Benchmark Metric | V2 Production (`dynamic_bigru_v2.pt`) | V3 Candidate (`dynamic_bigru_v3.pt`) | Delta (V3 vs V2) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Input Shape** | 30 × 150 | 30 × 168 | +18 features | Body-relative features added |
| **Dynamic Vocabulary** | 17 classes | 22 classes | +5 classes | Vocabulary expanded |
| **Total Test Samples** | 26 | 33 | +7 samples | Full official test set |
| **Existing-17 Test Accuracy** | **80.77%** (21 / 26) | **76.92%** (20 / 26) | -3.85% (1 error) | V2 higher on existing 17 |
| **Existing-17 Macro F1** | **0.7137** | 0.6255 | -0.0882 | V2 higher |
| **Existing-17 Weighted F1** | **0.7923** | 0.7410 | -0.0513 | V2 higher |
| **New-5 Test Accuracy** | N/A | **85.71%** (6 / 7) | New Capability | **Strong new class performance** |
| **New-5 Macro F1** | N/A | **0.8000** | New Capability | **Strong new class performance** |
| **New-5 Weighted F1** | N/A | **0.8571** | New Capability | **Strong new class performance** |
| **Overall Test Accuracy** | 80.77% (17 classes) | **78.79%** (22 classes) | -1.98% | Maintained across 22 classes |
| **Overall Macro F1** | 0.7137 | 0.6652 | -0.0485 | 22-class macro average |
| **Overall Weighted F1** | 0.7923 | 0.7556 | -0.0367 | 22-class weighted average |
| **Trainable Parameters** | 174,993 | 182,230 | +7,237 (+4.14%) | Minimal overhead |
| **Checkpoint File Size** | 2.12 MB (2,124,217 B) | 2.21 MB (2,211,263 B) | +0.09 MB (+4.10%) | Minimal overhead |
| **Mean CPU Latency** | **0.70 ms** | **0.70 ms** | +0.00 ms | **Identical (<1 ms)** |
| **p95 CPU Latency** | **0.71 ms** | **0.71 ms** | +0.00 ms | **Identical (<1 ms)** |
| **p99 CPU Latency** | **0.73 ms** | **0.73 ms** | +0.00 ms | **Identical (<1 ms)** |

---

## 3. Per-Class Performance Breakdown

| Class | Type | V2 Test Support | V2 Precision | V2 Recall | V2 F1 | V3 Test Support | V3 Precision | V3 Recall | V3 F1 | Status / Observation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **help** | Existing | 2 | 1.000 | 1.000 | 1.000 | 2 | 1.000 | 1.000 | 1.000 | Matched |
| **doctor** | Existing | 0 | 0.000 | 0.000 | 0.000 | 0 | 0.000 | 0.000 | 0.000 | Matched |
| **hospital** | Existing | 2 | 1.000 | 1.000 | 1.000 | 2 | 1.000 | 1.000 | 1.000 | Matched |
| **sick** | Existing | 2 | 1.000 | 0.500 | 0.667 | 2 | 1.000 | 0.500 | 0.667 | Matched |
| **appointment** | Existing | 2 | 1.000 | 1.000 | 1.000 | 2 | 1.000 | 1.000 | 1.000 | Matched |
| **where** | Existing | 1 | 1.000 | 1.000 | 1.000 | 1 | 0.000 | 0.000 | 0.000 | V2 higher |
| **bathroom** | Existing | 2 | 1.000 | 0.500 | 0.667 | 2 | 0.500 | 0.500 | 0.500 | V2 higher |
| **yes** | Existing | 2 | 1.000 | 0.500 | 0.667 | 2 | 1.000 | 0.500 | 0.667 | Matched |
| **no** | Existing | 2 | 0.500 | 1.000 | 0.667 | 2 | 0.500 | 1.000 | 0.667 | Matched |
| **please** | Existing | 1 | 1.000 | 1.000 | 1.000 | 1 | 1.000 | 1.000 | 1.000 | Matched |
| **thank_you** | Existing | 1 | 1.000 | 1.000 | 1.000 | 1 | 0.500 | 1.000 | 0.667 | V2 higher |
| **wait** | Existing | 1 | 0.500 | 1.000 | 0.667 | 1 | 0.500 | 1.000 | 0.667 | Matched |
| **understand** | Existing | 2 | 1.000 | 1.000 | 1.000 | 2 | 1.000 | 1.000 | 1.000 | Matched |
| **problem** | Existing | 2 | 0.667 | 1.000 | 0.800 | 2 | 0.667 | 1.000 | 0.800 | Matched |
| **money** | Existing | 2 | 1.000 | 1.000 | 1.000 | 2 | 1.000 | 1.000 | 1.000 | Matched |
| **pay** | Existing | 1 | 0.000 | 0.000 | 0.000 | 1 | 0.000 | 0.000 | 0.000 | Matched |
| **document** | Existing | 1 | 0.000 | 0.000 | 0.000 | 1 | 0.000 | 0.000 | 0.000 | Matched |
| **hello** | New | 0 | - | - | - | 1 | 1.000 | 1.000 | 1.000 | New Class (85.7% sub-acc) |
| **good** | New | 0 | - | - | - | 1 | 0.000 | 0.000 | 0.000 | New Class (85.7% sub-acc) |
| **bad** | New | 0 | - | - | - | 2 | 1.000 | 1.000 | 1.000 | New Class (85.7% sub-acc) |
| **water** | New | 0 | - | - | - | 2 | 1.000 | 1.000 | 1.000 | New Class (85.7% sub-acc) |
| **food** | New | 0 | - | - | - | 1 | 1.000 | 1.000 | 1.000 | New Class (85.7% sub-acc) |

---

## 4. Inspection of Specifically Requested Signs

### 4.1 `PLEASE`
- **Biomechanical Context:** Circular rubbing motion of open palm flat against the center of the chest.
- **V2 Result:** 1 / 1 test correct (100.0%), 7 / 7 overall correct (100.0%), Mean Confidence = 99.8%.
- **V3 Result:** 1 / 1 test correct (100.0%), 7 / 7 overall correct (100.0%), Mean Confidence = 98.8%.
- **Assessment:** Both models classify `PLEASE` with 100% precision and recall across all dataset splits. In V3, chest-relative coordinates explicitly anchor the hand at the chest center (`chest_center`), preventing ambiguity with non-chest flat-hand signs.

### 4.2 `FOOD` (New Dynamic Class)
- **Biomechanical Context:** Squished O-handshape / bunched fingers repeatedly tapping the mouth/lips.
- **V2 Result:** Not present in V2 vocabulary.
- **V3 Result:** 1 / 1 test correct (100.0%), 7 / 8 overall correct (87.5%), Mean Confidence = 85.2%.
- **Single Training/Validation Error:** Sample 198 (validation split) predicted as `BAD` with very low confidence (24.4%).
- **Assessment:** Successfully integrated. Nose-relative coordinates ($z_{nose}, y_{nose}$) provide clear spatial guidance locating the sign at the mouth rather than the chest.

### 4.3 `HELLO` (New Dynamic Class)
- **Biomechanical Context:** Open flat hand saluting outwards from temple/forehead.
- **V2 Result:** Not present in V2 vocabulary.
- **V3 Result:** 1 / 1 test correct (100.0%), 6 / 6 overall correct (100.0%), Mean Confidence = 94.4%.
- **Assessment:** Flawless classification across all splits. High confidence and zero confusion with other waving or single-hand signs.

### 4.4 `GOOD` (New Dynamic Class)
- **Biomechanical Context:** Open flat hand fingers touching chin/lips, then moving forward/downward toward non-dominant palm or forward into open space.
- **V2 Result:** Not present in V2 vocabulary.
- **V3 Result:** 0 / 1 test correct (0.0%), 4 / 10 overall correct (40.0%), Mean Confidence = 71.5%.
- **Error Analysis:** 5 of the 6 misclassifications predicted `THANK_YOU` (confidences: 70.4%, 78.9%, 72.8%, 57.0%, 74.6%).
- **Linguistic/Phonological Root Cause:** In American Sign Language, `GOOD` and `THANK YOU` are near-identical minimal pairs. Both signs initiate with the fingertips touching the chin/lower lip and transition forward. When signers execute `GOOD` without the non-dominant receiving hand (common in casual signing and single-hand camera setups), the trajectory and handshape are virtually indistinguishable from `THANK_YOU`. This represents a known phonological overlap that requires either non-dominant hand presence or facial context to fully disambiguate.

### 4.5 `BAD` (New Dynamic Class)
- **Biomechanical Context:** Open flat hand touching chin, then flipping downward and away with palm facing down.
- **V2 Result:** Not present in V2 vocabulary.
- **V3 Result:** 2 / 2 test correct (100.0%), 10 / 10 overall correct (100.0%), Mean Confidence = 87.6%.
- **Assessment:** Flawless classification across all 10 instances (train, val, and test). The distinct downward flip distinguishes it cleanly from `GOOD` and `THANK_YOU`.

### 4.6 `WATER` (New Dynamic Class)
- **Biomechanical Context:** 'W' handshape (index, middle, ring fingers upright) tapping index finger against the chin twice.
- **V2 Result:** Not present in V2 vocabulary.
- **V3 Result:** 2 / 2 test correct (100.0%), 10 / 11 overall correct (90.9%), Mean Confidence = 74.5%.
- **Single Training Error:** Sample 189 (train split) predicted as `HELLO` (confidence 54.9%).
- **Assessment:** Robust test performance (100% on test split). The chin-relative features correctly capture the chin-tapping locus.

---

## 5. Inspection of Known Confusion Pairs

### 5.1 `DOCTOR` vs `PAY`
- **Biomechanical Overlap:** Both signs involve one hand interacting with the upturned palm of the other hand (fingers bent tapping wrist for `DOCTOR`, fingertips sliding across palm for `PAY`).
- **Head-to-Head Comparison:**
  - **`DOCTOR` $\to$ `PAY` Error Rate:**
    - V2: 1 / 10 samples (10.0% error rate, in validation split)
    - V3: 1 / 11 samples (9.1% error rate, in validation split)
  - **`PAY` $\to$ `DOCTOR` Error Rate:**
    - V2: 1 / 6 samples (16.7% error rate, test sample 142 misclassified as `doctor` with 99.2% confidence)
    - V3: 1 / 6 samples (16.7% error rate, test sample 144 misclassified as `doctor` with 91.5% confidence)
- **Assessment:** Both models exhibit identical behavior on this difficult fine-grained finger-to-palm contact pair. Body-relative features do not substantially alter wrist-to-palm contacts because the spatial location of the contact is virtually identical relative to the torso.

### 5.2 `HELP` vs `MONEY`
- **Biomechanical Overlap:** In `HELP`, a closed fist rests on the open palm. In `MONEY`, a flattened 'O' handshape repeatedly taps the open palm. Under orientation tilt, the thumb projection of `HELP` can visually resemble the flattened 'O' of `MONEY`.
- **Standard Evaluation:**
  - V2: 0 / 9 `HELP` $\to$ `MONEY` errors (0.0%), 0 / 8 `MONEY` $\to$ `HELP` errors (0.0%)
  - V3: 0 / 9 `HELP` $\to$ `MONEY` errors (0.0%), 0 / 8 `MONEY` $\to$ `HELP` errors (0.0%)
- **Rotational Stress Test ($+15^\circ$ In-Plane Tilt):**
  - **V2:** 0 / 9 `HELP` $\to$ `MONEY` errors (0.0%) — completely immune to rotational misclassification.
  - **V3:** 1 / 9 `HELP` $\to$ `MONEY` errors (11.1%) — 1 sample flipped to `money` under $+15^\circ$ tilt.
- **Assessment:** V2 remains slightly more robust against rotation-induced `HELP` $\to$ `MONEY` cross-confusion than V3.

---

## 6. Confusion Matrix & Distribution Plots

The following evaluation visualizations have been generated and archived in `ml/evaluation/`:
- **Confusion Matrix Comparison:** [`ml/evaluation/model_v2_v3_confusion_matrices.png`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/model_v2_v3_confusion_matrices.png)
- **Per-Class F1 Score Comparison:** [`ml/evaluation/model_v2_v3_per_class_f1.png`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/model_v2_v3_per_class_f1.png)

---

## 7. Operational Recommendation: V3 Activation Status

### **Recommendation: KEEP V3 INACTIVE IN PRODUCTION**

| Evaluation Criterion | Production Standard | V2 Status | V3 Status | Risk / Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Existing 17-Class Accuracy** | $\ge 80\%$ | **80.77%** | 76.92% | V2 is $+3.85\%$ higher; V3 lost 1 sample on `WHERE` |
| **New Class Integration** | $\ge 80\%$ | N/A | **85.71%** | V3 achieved strong performance on 4/5 new classes |
| **Phonological Independence** | Low confusion | Stable | `GOOD` $\to$ `THANK_YOU` (60% err) | Severe confusion between `GOOD` and `THANK_YOU` |
| **Rotational Robustness** | 0% HELP $\to$ MONEY | **0.0%** (0/9) | 11.1% (1/9) | V2 has stronger rotation immunity |
| **Inference Latency** | $<10\text{ms}$ | **0.70 ms** | **0.70 ms** | Both models satisfy real-time budget |
| **Live Pipeline Verification** | Verified live | **Working in live UI** | Unvalidated on live camera | V3 requires live empirical verification |

### Action Plan:
1. **Preserve V2 in Production:** FastAPI uvicorn daemon and live WebRTC camera feed remain connected to `dynamic_bigru_v2.pt` (150-feature input, 17 classes).
2. **Phase 10 Future Work:** Conduct physical live camera testing with signers performing `GOOD` vs `THANK_YOU` and body-relative coordinate calibration under varied camera distances before promoting V3 to production.
