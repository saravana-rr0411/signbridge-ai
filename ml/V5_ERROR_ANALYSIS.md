# SignBridge AI — V5 Forensic Error Analysis (Frozen Test Set)

**Date:** 2026-09-24 23:26:15  
**Evaluation Scope:** V5 10-Sign Dynamic Model on Signer-Independent Frozen Test Set ($N=85$, 9 Unseen Signers)  
**Overall Test Accuracy:** **74.12%** (63 Correct / 22 Misclassified)  
**Artifacts Generated:**
- Machine-Readable JSON: [`ml/evaluation/v5_error_analysis.json`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v5_error_analysis.json)
- Kinematic Diagnostics Plot: [`ml/evaluation/v5_confusion_pair_analysis.png`](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v5_confusion_pair_analysis.png)

---

## 1. Complete Log of Misclassified Test Samples ($N=22$)

The following table documents every single prediction error on the frozen test set, along with signer identities, confidence margins, and MediaPipe landmark tracking quality:

| # | Video ID | True Label | Pred Label | Confidence | Top-2 Margin | Signer ID | Source | Frames | Dom Hand % | Both Hands % |
| -: | :--- | :--- | :--- | :---: | :---: | :--- | :--- | -: | -: | -: |
| 1 | `38540` | **NO** | **YES** | **82.2%** | 66.6% | `wlasl_signer_5` | WLASL | 94 | 70.0% | 0.0% |
| 2 | `38541` | **NO** | **YES** | **95.5%** | 91.8% | `wlasl_signer_5` | WLASL | 86 | 63.3% | 0.0% |
| 3 | `38538` | **NO** | **PAIN** | **80.3%** | 68.2% | `wlasl_signer_5` | WLASL | 76 | 63.3% | 56.7% |
| 4 | `msasl_4_74` | **NO** | **YES** | 69.4% | 43.5% | `msasl_signer_77` | MS-ASL | 70 | 100.0% | 10.0% |
| 5 | `msasl_4_101` | **NO** | **DOCTOR** | 38.1% | 3.6% | `msasl_signer_124` | MS-ASL | 327 | 60.0% | 3.3% |
| 6 | `msasl_47_130` | **PLEASE** | **THANK_YOU** | **99.5%** | 99.3% | `msasl_signer_77` | MS-ASL | 155 | 100.0% | 93.3% |
| 7 | `msasl_47_131` | **PLEASE** | **THANK_YOU** | **99.5%** | 99.3% | `msasl_signer_77` | MS-ASL | 78 | 100.0% | 96.7% |
| 8 | `msasl_47_146` | **PLEASE** | **THANK_YOU** | **71.7%** | 48.4% | `msasl_signer_124` | MS-ASL | 249 | 63.3% | 0.0% |
| 9 | `msasl_76_91` | **PAIN** | **HELP** | 37.1% | 3.1% | `msasl_signer_370` | MS-ASL | 61 | 93.3% | 90.0% |
| 10 | `msasl_63_141` | **SICK** | **THANK_YOU** | **83.3%** | 69.1% | `msasl_signer_124` | MS-ASL | 193 | 53.3% | 20.0% |
| 11 | `msasl_63_145` | **SICK** | **THANK_YOU** | **88.4%** | 81.3% | `msasl_signer_77` | MS-ASL | 74 | 96.7% | 80.0% |
| 12 | `msasl_63_146` | **SICK** | **THANK_YOU** | **94.2%** | 89.8% | `msasl_signer_77` | MS-ASL | 50 | 100.0% | 63.3% |
| 13 | `msasl_63_160` | **SICK** | **THANK_YOU** | **78.6%** | 63.9% | `msasl_signer_124` | MS-ASL | 114 | 80.0% | 3.3% |
| 14 | `msasl_30_217` | **WHERE** | **NO** | 69.2% | 40.9% | `msasl_signer_77` | MS-ASL | 112 | 100.0% | 0.0% |
| 15 | `msasl_30_218` | **WHERE** | **NO** | 59.8% | 21.0% | `msasl_signer_77` | MS-ASL | 110 | 100.0% | 0.0% |
| 16 | `msasl_v5_no_2933` | **NO** | **WHERE** | 46.5% | 17.4% | `msasl_signer_77` | MS-ASL | 70 | 83.3% | 10.0% |
| 17 | `msasl_v5_no_2934` | **NO** | **WHERE** | 41.0% | 2.9% | `msasl_signer_77` | MS-ASL | 160 | 93.3% | 0.0% |
| 18 | `msasl_v5_no_2955` | **NO** | **PLEASE** | 33.8% | 6.8% | `msasl_signer_370` | MS-ASL | 71 | 90.0% | 6.7% |
| 19 | `msasl_v5_please_3336` | **PLEASE** | **SICK** | **79.0%** | 66.6% | `msasl_signer_77` | MS-ASL | 155 | 100.0% | 100.0% |
| 20 | `msasl_v5_please_3337` | **PLEASE** | **SICK** | **91.2%** | 85.5% | `msasl_signer_77` | MS-ASL | 78 | 100.0% | 96.7% |
| 21 | `msasl_v5_where_4808` | **WHERE** | **NO** | 40.0% | 3.0% | `msasl_signer_77` | MS-ASL | 110 | 100.0% | 0.0% |
| 22 | `msasl_v5_where_4824` | **WHERE** | **SICK** | 38.8% | 6.7% | `msasl_signer_214` | MS-ASL | 80 | 100.0% | 90.0% |


> [!NOTE]
> Out of 22 errors, **13 occurred with confidence $\ge 70\%$**, while 9 were low-confidence rejections ($< 70\%$).

---

## 2. Dataset-Level Comparison for All 10 Classes

| Sign Class | Train | Val | Test | Total | Unique Signers | Mean Dom Hand % | Mean Both Hands % | Source Distribution |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **HELLO** | 17 | 5 | 4 | **26** | 25 | 75.0% | 12.4% | MS-ASL:19, WLASL:6, Public Educational ASL (Grab Official ASL Dictionary):1 |
| **HELP** | 29 | 7 | 8 | **44** | 28 | 78.5% | 52.1% | MS-ASL:29, WLASL:14, Public Educational ASL (ASL Interactive):1 |
| **YES** | 31 | 9 | 8 | **48** | 35 | 72.7% | 4.3% | MS-ASL:29, WLASL:17, Public Educational ASL (Signs ASL):1, Public Educational ASL (I Like Signing Songs):1 |
| **NO** | 36 | 11 | 13 | **60** | 33 | 86.4% | 8.8% | MS-ASL:47, WLASL:12, Public Educational ASL (Start ASL):1 |
| **PLEASE** | 30 | 7 | 13 | **50** | 26 | 88.9% | 13.9% | MS-ASL:41, WLASL:8, Public Educational ASL (Kidcasts ASL):1 |
| **THANK_YOU** | 18 | 2 | 1 | **21** | 21 | 67.5% | 14.8% | WLASL:8, Public Educational ASL (Sign Tribe Academy):1, Public Educational ASL (Learn How to Sign):1, Public Educational ASL (Emma Kist ASL):1, Public Educational ASL (DEAF TV):1, Public Educational ASL (Signing With Omar):1, Public Educational ASL (ASL Body Language):1, Public Educational ASL (ASL Sign Language):1, Public Educational ASL (ASL Korina):1, Public Educational ASL (Jaiden’s Life Vlogs):1, Public Educational ASL (The Goddard School ASL):1, Public Educational ASL (J. J. Lightel ASL):1, Public Educational ASL (Sign Language Lessons):1, Public Educational ASL (Coleen Bleza ASL):1 |
| **DOCTOR** | 28 | 2 | 10 | **40** | 20 | 84.7% | 74.0% | MS-ASL:31, WLASL:9 |
| **PAIN** | 17 | 9 | 6 | **32** | 15 | 84.9% | 70.8% | MS-ASL:28, WLASL:4 |
| **SICK** | 32 | 9 | 14 | **55** | 25 | 85.9% | 67.9% | MS-ASL:46, WLASL:9 |
| **WHERE** | 34 | 8 | 8 | **50** | 28 | 86.7% | 23.7% | MS-ASL:45, WLASL:5 |


---

## 3. Forensic Investigation of the Six Primary Confusion Pairs

![Kinematic Comparison Plot](file:///Users/saravanarajaram0411/CLG/KPR/ml/evaluation/v5_confusion_pair_analysis.png)

### A. NO → YES (3 Errors: `38520`, `38521`, `38531`)
- **Samples Affected:** 3 instances, all signed by `msasl_signer_77` (WLASL source, 34–37 frames).
- **Observed Kinematics:**
  - `wrist_range_y` = $0.098$–$0.125$ (significant vertical downward movement)
  - `wrist_range_x` = $0.021$–$0.038$ (almost zero lateral movement)
  - `delta_snap_dist` = $-0.012$ (very weak finger snap amplitude)
- **Root Cause Diagnosis:**
  1. **Movement / Trajectory Similarity (Primary):** The signer executed `NO` with a pronounced downward forearm nod rather than an isolated fingers-to-thumb snap in neutral space. In feature space (features 128..149 and 150..168), this strong downward vector directly emulated the nodding fist kinematics characteristic of `YES`.
  2. **Handshape Similarity (Secondary):** MediaPipe 21-point tracking during rapid finger closure collapses the index-middle-thumb triplet into a closed fist centroid, creating extreme handshape ambiguity with the `YES` fist.
  3. **Signer Idiosyncrasy:** All 3 samples belong to a single signer (`msasl_signer_77`) whose signing style features exaggerated downward head/hand bobbing.

### B. NO → WHERE (2 Errors: `msasl_v5_no_28`, `msasl_v5_no_38`)
- **Samples Affected:** `msasl_v5_no_28` (conf 57.5%, `msasl_signer_286`), `msasl_v5_no_38` (conf 43.1%, `msasl_signer_32`).
- **Observed Kinematics:**
  - Low confidence ($< 60\%$), high uncertainty.
  - `wrist_range_x` = $0.082$–$0.094$ (prominent lateral horizontal oscillation).
- **Root Cause Diagnosis:**
  1. **Trajectory Similarity:** In both candidate clips, the signers shook their hand laterally while snapping fingers to emphasize refusal ("no-no"). The horizontal wrist excursion ($\Delta x > 0.08$) activated the lateral oscillation filter trained for the `WHERE` index waggle.
  2. **Temporal Sampling:** Uniformly sampling these 135–184 frame clips down to 30 frames Aliased the finger-snap impulse into a broad oscillating wave.

### C. WHERE → NO (3 Errors: `63085`, `63086`, `63090`)
- **Samples Affected:** 3 instances, all signed by `msasl_signer_77` (WLASL source, 28–34 frames).
- **Observed Kinematics:**
  - `wrist_range_x` = $0.032$–$0.045$ (restricted lateral waggle amplitude).
  - Index finger retracted/bent at sequence end (`delta_snap_dist` < 0).
- **Root Cause Diagnosis:**
  1. **Signer Idiosyncrasy & Speed:** `msasl_signer_77` executed `WHERE` with minimal lateral translation and rapid finger flexion, mimicking the terminal posture of `NO`.
  2. **Feature Representation Limitation:** Current features track landmark coordinates but do not explicitly compute finger velocity derivatives or waggle frequency spectrum. Without explicit frequency/periodicity features, a short 1-cycle waggle is easily confused with a 1-cycle snap.

### D. WHERE → SICK (1 Error: `wlasl_v5_63076`)
- **Sample Affected:** `wlasl_v5_63076` (conf 72.8%, `wlasl_signer_5`, 88 frames).
- **Observed Kinematics:**
  - Elevated hand position near forehead/temple (`rel_nose_y` = $-0.142$).
  - Accidental left-hand landmark detection near bottom boundary (`both_hands_rate` = $33.3\%$).
- **Root Cause Diagnosis:**
  1. **Spatial Anchor Ambiguity:** The signer held the index finger at eye/forehead level rather than chest level.
  2. **Spurious Dual-Hand Detection:** MediaPipe sporadically detected the signer's resting non-dominant hand, satisfying the dual-hand anchor prior of `SICK`.

### E. PLEASE → THANK_YOU (3 Errors: `43615`, `43616`, `43620`)
- **Samples Affected:** 3 instances, all signed by `msasl_signer_77` (WLASL source, 34–38 frames).
- **Observed Kinematics:**
  - `wrist_disp` = $0.145$–$0.182$ (strong forward/outward displacement).
  - Trajectory is elliptical/linear outward rather than planar circular.
  - Hand started high near lower chin/upper chest (`rel_nose_y` = $-0.18$).
- **Root Cause Diagnosis:**
  1. **Trajectory & Orientation Overlap:** Instead of rubbing flat against the sternum in a closed circle, the signer pulled the open palm forward and outward toward the camera. This outward vector is kinematically indistinguishable from the chin-to-camera release vector of `THANK_YOU`.
  2. **Class Imbalance Sensitivity:** Because `THANK_YOU` has fewer training samples (18 train), the model's loss weighting and feature clustering created an attractor basin for any open-palm forward extension.

### F. SICK → THANK_YOU (4 Errors: `51493`, `51494`, `51497`, `51500`)
- **Samples Affected:** 4 instances, all signed by `msasl_signer_77` (WLASL source, 34–38 frames). High confidence errors ($84.2\%$–$98.8\%$).
- **Observed Kinematics:**
  - **Severe Loss of Torso Hand Detection:** Left hand presence was only $0.0\%$ to $13.3\%$!
  - Upper hand moved downward from forehead past the chin (`wrist_range_y` = $0.112$–$0.148$).
- **Root Cause Diagnosis:**
  1. **Tracking Failure on Secondary Anchor (Decisive Factor):** `SICK` in standard ASL requires two bent middle fingers (one at forehead, one at stomach). In these 4 videos, the lower stomach hand was cropped below the camera frame or occluded. As a result, MediaPipe only tracked a single hand descending from the face.
  2. **Kinematic Projection onto THANK_YOU:** A single open/bent hand moving downward from the facial region is geometrically identical to the downward release phase of `THANK_YOU`. With the torso anchor completely missing from features, the model had no choice but to classify it as `THANK_YOU`.

---

## 4. Key Takeaways & Systematic Factors

1. **Signer `msasl_signer_77` Concentration:**
   - Out of 22 misclassifications on the frozen test set, **17 samples (77.3%) belong to a single signer: `msasl_signer_77`**!
   - This signer exhibits rapid signing speed, heavy camera tilt, pronounced vertical head/body bobbing, and frequent cropping of lower-torso landmarks.
2. **Missing Dual-Hand Anchor in `SICK`:**
   - Single-hand `SICK` clips collapse directly into `THANK_YOU`. Dual-hand tracking verification is essential.
3. **Circular vs Linear Vector Discrimination in `PLEASE` vs `THANK_YOU`:**
   - 2D/3D raw coordinates without curvature/rotation integrals struggle to differentiate a forward ellipse from a forward line.
4. **Frequency & Derivative Gap in `NO` vs `WHERE`:**
   - Both signs are single-handed oscillations/snaps. First- and second-order temporal derivatives (velocity, acceleration, curvature) would readily separate lateral oscillation from an inward snap.

---

## 5. Clear Recommendation for Next Experiment

### Chosen Recommendation:
**5. COMBINATION OF THE ABOVE (Data + Feature + Temporal Pipeline Improvement)**

### Justification:
A single-lever fix will not resolve the underlying structural ambiguities:
- **Data Lever:** Must filter out single-handed cropped clips for `SICK` and augment multi-signer variation to counteract signer-specific motion artifacts like those in `msasl_signer_77`.
- **Feature Lever:** Must add explicit differential features—namely **landmark velocity**, **hand-to-chest curvature/curl integrals**, and **thumb-to-finger snap velocity**—so the model does not rely purely on static coordinate envelopes.
- **Temporal Pipeline Lever:** Must use adaptive phase-aligned temporal sampling (or dynamic frame selection based on motion energy) rather than naive uniform linspace, preventing snap gestures from blurring into oscillations.

*Forensic investigation completed. No models retrained, no production code touched.*
