# SignBridge AI — Live Validation Report (V4 11-Sign Model)

**Model:** `ml/models/dynamic_bigru_v4_11_sign.pt`  
**Input Dimension:** 30 frames × 168 features  
**Vocabulary (11 Signs):** HELLO, HELP, YES, NO, PLEASE, THANK_YOU, DOCTOR, PAIN, SICK, BATHROOM, WHERE  
**Confidence Threshold:** 70% (`0.70`)  
**Evaluation Mode:** Isolated Live Controlled Validation Session  
**Date:** 2026-09-24 22:35:23  

---

## 1. Executive Summary
- **Total Controlled Attempts:** 31
- **Correct Accepted Predictions (Match & >= 70%):** 26
- **Overall Empirical Accuracy:** **83.87%**
- **Accepted-Only Accuracy:** **92.86%**
- **Average Prediction Confidence:** **93.62%**
- **Accepted Predictions (>= 70%):** 28 (90.3%)
- **Rejected Predictions (< 70%):** 3
- **False High-Confidence Predictions (>= 70% & Wrong):** 2
- **Low-Confidence Correct Predictions (< 70% & Matched):** 2
- **CPU Inference Latency:** Mean: **1.25 ms** | P95: **2.24 ms**

---

## 2. Per-Sign Empirical Performance

| Sign | Attempts | Correct (Accepted) | Empirical Acc | Accepted-Only Acc | Avg Confidence | False High-Conf | Low-Conf Correct | Rejected (< 70%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELLO** | 3 | 3 | **100.0%** | 100.0% | 99.8% | 0 | 0 | 0 |
| **HELP** | 3 | 3 | **100.0%** | 100.0% | 99.9% | 0 | 0 | 0 |
| **YES** | 3 | 3 | **100.0%** | 100.0% | 99.4% | 0 | 0 | 0 |
| **NO** | 3 | 2 | **66.7%** | 100.0% | 81.2% | 0 | 1 | 1 |
| **PLEASE** | 3 | 2 | **66.7%** | 66.7% | 99.8% | 1 | 0 | 0 |
| **THANK_YOU** | 1 | 1 | **100.0%** | 100.0% | 99.9% | 0 | 0 | 0 |
| **DOCTOR** | 3 | 3 | **100.0%** | 100.0% | 99.8% | 0 | 0 | 0 |
| **PAIN** | 3 | 3 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 |
| **SICK** | 3 | 3 | **100.0%** | 100.0% | 98.0% | 0 | 0 | 0 |
| **BATHROOM** | 3 | 1 | **33.3%** | 50.0% | 77.2% | 1 | 0 | 1 |
| **WHERE** | 3 | 2 | **66.7%** | 100.0% | 79.0% | 0 | 1 | 1 |

---

## 3. Confusion Matrix (Controlled Real-Time Captures)

| True \ Pred | HELLO | HELP | YES | NO | PLEASE | THANK_YOU | DOCTOR | PAIN | SICK | BATHROOM | WHERE |
| :--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
| **HELLO** | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **HELP** | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **YES** | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **NO** | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **PLEASE** | 0 | 0 | 0 | 0 | 2 | 1 | 0 | 0 | 0 | 0 | 0 |
| **THANK_YOU** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| **DOCTOR** | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 0 |
| **PAIN** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| **SICK** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| **BATHROOM** | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| **WHERE** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 |


---

## 4. Per-Attempt Log (All 11 Signs)

| # | Expected | Attempt | Condition | Predicted | Confidence | Latency | Accepted | Outcome |
| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | HELLO | 1/3 | Normal distance & center framing | hello | 99.77% | 1.37 ms | Yes | **CORRECT (ACCEPTED)** |
| 2 | HELLO | 2/3 | Slightly left/right hand position | hello | 99.69% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 3 | HELLO | 3/3 | Slightly faster / slower gesture tempo | hello | 99.9% | 1.08 ms | Yes | **CORRECT (ACCEPTED)** |
| 4 | HELP | 1/3 | Normal distance & center framing | help | 99.94% | 1.11 ms | Yes | **CORRECT (ACCEPTED)** |
| 5 | HELP | 2/3 | Slightly left/right hand position | help | 99.98% | 0.82 ms | Yes | **CORRECT (ACCEPTED)** |
| 6 | HELP | 3/3 | Slightly faster / slower gesture tempo | help | 99.85% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 7 | YES | 1/3 | Normal distance & center framing | yes | 99.45% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 8 | YES | 2/3 | Slightly left/right hand position | yes | 99.26% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 9 | YES | 3/3 | Slightly faster / slower gesture tempo | yes | 99.51% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 10 | NO | 1/3 | Normal distance & center framing | no | 88.45% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 11 | NO | 2/3 | Slightly left/right hand position | no | 57.66% | 0.78 ms | No | REJECTED (LOW CONF MATCH) |
| 12 | NO | 3/3 | Slightly faster / slower gesture tempo | no | 97.61% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 13 | PLEASE | 1/3 | Normal distance & center framing | please | 99.96% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 14 | PLEASE | 2/3 | Slightly left/right hand position | please | 99.76% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 15 | PLEASE | 3/3 | Slightly faster / slower gesture tempo | thank_you | 99.7% | 0.8 ms | Yes | FALSE HIGH-CONF |
| 16 | THANK_YOU | 1/1 | Normal distance & center framing | thank_you | 99.88% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 17 | DOCTOR | 1/3 | Normal distance & center framing | doctor | 99.72% | 3.66 ms | Yes | **CORRECT (ACCEPTED)** |
| 18 | DOCTOR | 2/3 | Slightly left/right hand position | doctor | 99.86% | 1.43 ms | Yes | **CORRECT (ACCEPTED)** |
| 19 | DOCTOR | 3/3 | Slightly faster / slower gesture tempo | doctor | 99.73% | 1.34 ms | Yes | **CORRECT (ACCEPTED)** |
| 20 | PAIN | 1/3 | Normal distance & center framing | pain | 99.97% | 1.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 21 | PAIN | 2/3 | Slightly left/right hand position | pain | 99.98% | 1.19 ms | Yes | **CORRECT (ACCEPTED)** |
| 22 | PAIN | 3/3 | Slightly faster / slower gesture tempo | pain | 99.98% | 2.04 ms | Yes | **CORRECT (ACCEPTED)** |
| 23 | SICK | 1/3 | Normal distance & center framing | sick | 95.13% | 1.11 ms | Yes | **CORRECT (ACCEPTED)** |
| 24 | SICK | 2/3 | Slightly left/right hand position | sick | 99.19% | 1.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 25 | SICK | 3/3 | Slightly faster / slower gesture tempo | sick | 99.79% | 2.43 ms | Yes | **CORRECT (ACCEPTED)** |
| 26 | BATHROOM | 1/3 | Normal distance & center framing | bathroom | 96.82% | 1.55 ms | Yes | **CORRECT (ACCEPTED)** |
| 27 | BATHROOM | 2/3 | Slightly left/right hand position | where | 86.5% | 1.92 ms | Yes | FALSE HIGH-CONF |
| 28 | BATHROOM | 3/3 | Slightly faster / slower gesture tempo | yes | 48.17% | 0.9 ms | No | REJECTED (UNCERTAIN) |
| 29 | WHERE | 1/3 | Normal distance & center framing | where | 85.93% | 0.87 ms | Yes | **CORRECT (ACCEPTED)** |
| 30 | WHERE | 2/3 | Slightly left/right hand position | where | 57.01% | 1.82 ms | No | REJECTED (LOW CONF MATCH) |
| 31 | WHERE | 3/3 | Slightly faster / slower gesture tempo | where | 94.05% | 0.99 ms | Yes | **CORRECT (ACCEPTED)** |
