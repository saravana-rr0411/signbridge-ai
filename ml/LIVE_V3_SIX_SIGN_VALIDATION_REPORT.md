# SignBridge AI — Live Webcam Validation Report (Focused 6-Sign V3)

**Model:** `ml/models/dynamic_bigru_v3_six_sign.pt`  
**Input Dimension:** 30 frames × 168 features  
**Vocabulary (6 Signs):** HELP, YES, NO, THANK_YOU, PLEASE, HELLO  
**Confidence Threshold:** 70% (`0.70`)  
**Evaluation Mode:** Isolated Live Webcam Controlled Validation  
**Date:** 2026-09-24 21:30:28  

---

## 1. Executive Summary
- **Total Controlled Attempts:** 30
- **Correct Accepted Predictions (Match & >= 70%):** 29
- **Overall Empirical Accuracy:** **96.67%**
- **Accepted-Only Accuracy:** **96.67%**
- **Average Prediction Confidence:** **99.99%**
- **Accepted Predictions (>= 70%):** 30 (100.0%)
- **Rejected Predictions (< 70%):** 0
- **False High-Confidence Predictions (>= 70% & Wrong):** 1
- **Low-Confidence Correct Predictions (< 70% & Matched):** 0
- **CPU Inference Latency:** Mean: **0.86 ms** | P95: **0.93 ms**

---

## 2. Per-Sign Empirical Performance

| Sign | Attempts | Correct (Accepted) | Empirical Acc | Accepted-Only Acc | Avg Confidence | False High-Conf | Low-Conf Correct | Rejected (< 70%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELP** | 5 | 4 | **80.0%** | 80.0% | 100.0% | 1 | 0 | 0 |
| **YES** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 |
| **NO** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 |
| **THANK_YOU** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 |
| **PLEASE** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 |
| **HELLO** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 |

---

## 3. Confusion Matrix (Live Real-Time Captures)

| True \ Pred | HELP | YES | NO | THANK_YOU | PLEASE | HELLO |
| :--- | --- | --- | --- | --- | --- | --- | ---
| **HELP** | 4 | 0 | 1 | 0 | 0 | 0 |
| **YES** | 0 | 5 | 0 | 0 | 0 | 0 |
| **NO** | 0 | 0 | 5 | 0 | 0 | 0 |
| **THANK_YOU** | 0 | 0 | 0 | 5 | 0 | 0 |
| **PLEASE** | 0 | 0 | 0 | 0 | 5 | 0 |
| **HELLO** | 0 | 0 | 0 | 0 | 0 | 5 |


---

## 4. Deep-Dive Analysis on Critical Signs

### NO (Main Previous Live Weakness)
- **Status:** RESOLVED
- **Empirical Accuracy:** 100.0%
- **Average Live Confidence:** 100.0%
- **Observations:** In earlier 22-class testing, NO achieved top-1 label but was rejected due to 50–58% confidence. The focused 6-sign architecture has dedicated representation for rapid 2-finger snap motion.

### HELP
- **Empirical Accuracy:** 80.0%
- **Average Live Confidence:** 99.99%
- **Observations:** Two-handed sign (closed fist resting on flat palm). Pose-wrist handedness disambiguation ensures base palm and active fist stay separated.

### PLEASE
- **Empirical Accuracy:** 100.0%
- **Average Live Confidence:** 100.0%
- **Observations:** Circular motion across the chest. Body-relative features (indices 162..167) anchor wrist coordinates relative to chest center.

### THANK_YOU
- **Empirical Accuracy:** 100.0%
- **Average Live Confidence:** 99.99%
- **Observations:** Hand moving from chin/mouth forward toward camera. Evaluated against potential confusion with chest gestures.

---

## 5. Per-Attempt Comprehensive Log

| # | Expected | Attempt | Condition | Predicted | Confidence | Latency | Accepted | Outcome |
| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | HELP | 1/5 | Normal distance & center framing | help | 99.99% | 0.87 ms | Yes | **CORRECT (ACCEPTED)** |
| 2 | HELP | 2/5 | Slightly left/right hand position | help | 99.99% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 3 | HELP | 3/5 | Slightly different camera distance | help | 99.99% | 0.9 ms | Yes | **CORRECT (ACCEPTED)** |
| 4 | HELP | 4/5 | Slightly faster / slower gesture tempo | no | 99.99% | 0.8 ms | Yes | FALSE HIGH-CONF |
| 5 | HELP | 5/5 | Alternative hand posture / tilt | help | 99.99% | 0.85 ms | Yes | **CORRECT (ACCEPTED)** |
| 6 | YES | 1/5 | Normal distance & center framing | yes | 100.0% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 7 | YES | 2/5 | Slightly left/right hand position | yes | 100.0% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 8 | YES | 3/5 | Slightly different camera distance | yes | 99.99% | 0.81 ms | Yes | **CORRECT (ACCEPTED)** |
| 9 | YES | 4/5 | Slightly faster / slower gesture tempo | yes | 99.99% | 0.93 ms | Yes | **CORRECT (ACCEPTED)** |
| 10 | YES | 5/5 | Alternative hand posture / tilt | yes | 99.99% | 0.85 ms | Yes | **CORRECT (ACCEPTED)** |
| 11 | NO | 1/5 | Normal distance & center framing | no | 100.0% | 0.94 ms | Yes | **CORRECT (ACCEPTED)** |
| 12 | NO | 2/5 | Slightly left/right hand position | no | 100.0% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 13 | NO | 3/5 | Slightly different camera distance | no | 100.0% | 0.92 ms | Yes | **CORRECT (ACCEPTED)** |
| 14 | NO | 4/5 | Slightly faster / slower gesture tempo | no | 100.0% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 15 | NO | 5/5 | Alternative hand posture / tilt | no | 100.0% | 0.85 ms | Yes | **CORRECT (ACCEPTED)** |
| 16 | THANK_YOU | 1/5 | Normal distance & center framing | thank_you | 99.99% | 0.83 ms | Yes | **CORRECT (ACCEPTED)** |
| 17 | THANK_YOU | 2/5 | Slightly left/right hand position | thank_you | 99.98% | 0.88 ms | Yes | **CORRECT (ACCEPTED)** |
| 18 | THANK_YOU | 3/5 | Slightly different camera distance | thank_you | 99.99% | 0.85 ms | Yes | **CORRECT (ACCEPTED)** |
| 19 | THANK_YOU | 4/5 | Slightly faster / slower gesture tempo | thank_you | 99.99% | 0.87 ms | Yes | **CORRECT (ACCEPTED)** |
| 20 | THANK_YOU | 5/5 | Alternative hand posture / tilt | thank_you | 99.99% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 21 | PLEASE | 1/5 | Normal distance & center framing | please | 100.0% | 0.87 ms | Yes | **CORRECT (ACCEPTED)** |
| 22 | PLEASE | 2/5 | Slightly left/right hand position | please | 100.0% | 0.85 ms | Yes | **CORRECT (ACCEPTED)** |
| 23 | PLEASE | 3/5 | Slightly different camera distance | please | 100.0% | 0.89 ms | Yes | **CORRECT (ACCEPTED)** |
| 24 | PLEASE | 4/5 | Slightly faster / slower gesture tempo | please | 100.0% | 0.93 ms | Yes | **CORRECT (ACCEPTED)** |
| 25 | PLEASE | 5/5 | Alternative hand posture / tilt | please | 100.0% | 0.86 ms | Yes | **CORRECT (ACCEPTED)** |
| 26 | HELLO | 1/5 | Normal distance & center framing | hello | 100.0% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 27 | HELLO | 2/5 | Slightly left/right hand position | hello | 99.98% | 0.85 ms | Yes | **CORRECT (ACCEPTED)** |
| 28 | HELLO | 3/5 | Slightly different camera distance | hello | 99.99% | 0.86 ms | Yes | **CORRECT (ACCEPTED)** |
| 29 | HELLO | 4/5 | Slightly faster / slower gesture tempo | hello | 99.99% | 0.88 ms | Yes | **CORRECT (ACCEPTED)** |
| 30 | HELLO | 5/5 | Alternative hand posture / tilt | hello | 100.0% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
