# SignBridge AI — V3 Focused Live Webcam Validation Report

**Model Tested:** Candidate Model V3 (`dynamic_bigru_v3.pt`)  
**Input Dimension:** 30 frames × 168 features  
**Dynamic Classes:** 22 classes  
**Evaluation Mode:** Focused Live Webcam Continuous Recognition Session  
**Date:** 2026-09-24 20:37:21  

---

## 1. Summary Metrics
- **Total Controlled Attempts:** 14
- **Correct Recognitions (Accepted & Matching):** 6
- **Overall Empirical Accuracy:** **42.86%**
- **Average Prediction Confidence:** 73.73%
- **Accepted Predictions (>= 70%):** 7
- **False High-Confidence Predictions (>= 70% & Wrong):** 1
- **Rejected Low-Confidence Predictions (< 70%):** 7
- **Inference Latency:** Mean: **0.85 ms** | P95: **0.88 ms**

---

## 2. Per-Sign Accuracy Breakdown
| Sign | Target Attempts | Correct | Accuracy | Avg Confidence | False High-Conf | Rejected Low-Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NO** | 5 | 0 | **0.0%** | 54.8% | 0 | 5 |
| **YES** | 3 | 3 | **100.0%** | 85.9% | 0 | 0 |
| **HELP** | 3 | 1 | **33.3%** | 81.0% | 1 | 1 |
| **PLEASE** | 3 | 2 | **66.7%** | 85.9% | 0 | 1 |

---

## 3. Per-Attempt Detailed Log
| # | Expected Sign | Attempt | Condition | Predicted | Confidence | Latency | Accepted | Result |
| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 1 | NO | 1/5 | Normal distance & center framing | no | 49.9% | 0.85 ms | No | REJECTED (UNCERTAIN) |
| 2 | NO | 2/5 | Slightly left/right hand position | no | 52.61% | 0.86 ms | No | REJECTED (UNCERTAIN) |
| 3 | NO | 3/5 | Slightly different camera distance | no | 58.94% | 0.83 ms | No | REJECTED (UNCERTAIN) |
| 4 | NO | 4/5 | Slightly faster / slower gesture tempo | no | 54.02% | 0.85 ms | No | REJECTED (UNCERTAIN) |
| 5 | NO | 5/5 | Alternative hand posture / tilt | no | 58.32% | 0.85 ms | No | REJECTED (UNCERTAIN) |
| 6 | YES | 1/3 | Normal distance & center framing | yes | 90.25% | 0.86 ms | Yes | CORRECT |
| 7 | YES | 2/3 | Slightly left/right hand position | yes | 82.86% | 0.78 ms | Yes | CORRECT |
| 8 | YES | 3/3 | Slightly different camera distance | yes | 84.61% | 0.86 ms | Yes | CORRECT |
| 9 | HELP | 1/3 | Normal distance & center framing | yes | 83.97% | 0.82 ms | Yes | FALSE HIGH-CONF |
| 10 | HELP | 2/3 | Slightly left/right hand position | help | 92.27% | 0.86 ms | Yes | CORRECT |
| 11 | HELP | 3/3 | Slightly different camera distance | help | 66.73% | 0.86 ms | No | REJECTED (UNCERTAIN) |
| 12 | PLEASE | 1/3 | Normal distance & center framing | please | 95.3% | 0.92 ms | Yes | CORRECT |
| 13 | PLEASE | 2/3 | Slightly left/right hand position | please | 66.08% | 0.84 ms | No | REJECTED (UNCERTAIN) |
| 14 | PLEASE | 3/3 | Slightly different camera distance | please | 96.42% | 0.86 ms | Yes | CORRECT |
