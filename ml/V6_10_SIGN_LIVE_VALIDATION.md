# SignBridge AI — Live Webcam Validation Report (V6 10-Sign Model)

**Model:** `ml/models/dynamic_bigru_v6_10_sign.pt`  
**Input Dimension:** 30 frames × 168 features  
**Vocabulary (10 Signs):** HELLO, HELP, YES, NO, PLEASE, THANK_YOU, DOCTOR, PAIN, SICK, WHERE  
**Confidence Threshold:** 70% (`0.70`)  
**Date:** 2026-09-24 23:39:38  

---

## 1. Executive Summary
- **Total Controlled Attempts:** 50 (5 attempts per sign across 10 classes)
- **Correct Accepted Predictions (Match & >= 70%):** 45
- **Overall Empirical Accuracy:** **90.00%**
- **Accepted-Only Accuracy:** **91.84%**
- **Acceptance Rate:** **98.00%** (49/50)
- **Average Prediction Confidence:** **98.14%**
- **Accepted Predictions (>= 70%):** 49
- **Rejected Predictions (< 70%):** 1
- **False High-Confidence Predictions (>= 70% & Wrong):** 4
- **Low-Confidence Correct Predictions (< 70% & Matched):** 1
- **CPU Inference Latency:** Mean: **0.80 ms** | P95: **0.84 ms**

---

## 2. Per-Sign Empirical Performance

| Sign | Attempts | Correct (Accepted) | Empirical Acc | Accepted-Only Acc | Avg Confidence | False High-Conf | Low-Conf Correct | Rejected (< 70%) | Stability & Repeat Consistency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **HELLO** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 | Highly Stable (100% Consistent) |
| **HELP** | 5 | 5 | **100.0%** | 100.0% | 99.9% | 0 | 0 | 0 | Highly Stable (100% Consistent) |
| **YES** | 5 | 5 | **100.0%** | 100.0% | 99.8% | 0 | 0 | 0 | Highly Stable (100% Consistent) |
| **NO** | 5 | 4 | **80.0%** | 80.0% | 95.4% | 1 | 0 | 0 | Variations: {'NO': 4, 'YES': 1} |
| **PLEASE** | 5 | 3 | **60.0%** | 60.0% | 100.0% | 2 | 0 | 0 | Variations: {'PLEASE': 3, 'THANK_YOU': 2} |
| **THANK_YOU** | 5 | 5 | **100.0%** | 100.0% | 100.0% | 0 | 0 | 0 | Highly Stable (100% Consistent) |
| **DOCTOR** | 5 | 5 | **100.0%** | 100.0% | 99.3% | 0 | 0 | 0 | Highly Stable (100% Consistent) |
| **PAIN** | 5 | 4 | **80.0%** | 100.0% | 93.4% | 0 | 1 | 1 | Variations: {'PAIN': 5} |
| **SICK** | 5 | 4 | **80.0%** | 80.0% | 99.7% | 1 | 0 | 0 | Variations: {'SICK': 4, 'THANK_YOU': 1} |
| **WHERE** | 5 | 5 | **100.0%** | 100.0% | 94.0% | 0 | 0 | 0 | Highly Stable (100% Consistent) |

---

## 3. Confusion Pairs & Stability Analysis

### Observed Confusion Pairs
- **PLEASE -> THANK_YOU:** 2 attempt(s)
- **NO -> YES:** 1 attempt(s)
- **SICK -> THANK_YOU:** 1 attempt(s)


### Prediction Repeat Stability Highlights
- **100% Repeat-Stable Signs:** Signs where 5/5 attempts consistently predicted the exact correct sign with >95% confidence.
- **Signs with Boundary Sensitivity:** Any sign where temporal tempo or hand position changes created ambiguity or triggered the 70% rejection safety gate.

---

## 4. Confusion Matrix (50 Controlled Attempts)

| True \ Pred | HELLO | HELP | YES | NO | PLEASE | THANK_YOU | DOCTOR | PAIN | SICK | WHERE |
| :--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
| **HELLO** | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **HELP** | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **YES** | 0 | 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **NO** | 0 | 0 | 1 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| **PLEASE** | 0 | 0 | 0 | 0 | 3 | 2 | 0 | 0 | 0 | 0 |
| **THANK_YOU** | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| **DOCTOR** | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 |
| **PAIN** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 |
| **SICK** | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 4 | 0 |
| **WHERE** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |


---

## 5. Per-Attempt Comprehensive Log (5 Attempts x 10 Signs)

| # | Expected | Attempt | Condition | Predicted | Confidence | Latency | Accepted | Outcome |
| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | HELLO | 1/5 | Normal distance & center framing | hello | 99.9% | 1.44 ms | Yes | **CORRECT (ACCEPTED)** |
| 2 | HELLO | 2/5 | Slightly left/right hand position | hello | 99.95% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 3 | HELLO | 3/5 | Slightly faster gesture tempo | hello | 99.98% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 4 | HELLO | 4/5 | Slightly slower gesture tempo | hello | 99.98% | 0.82 ms | Yes | **CORRECT (ACCEPTED)** |
| 5 | HELLO | 5/5 | Slightly tilted / alternative hand posture | hello | 99.92% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 6 | HELP | 1/5 | Normal distance & center framing | help | 99.92% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 7 | HELP | 2/5 | Slightly left/right hand position | help | 99.91% | 0.82 ms | Yes | **CORRECT (ACCEPTED)** |
| 8 | HELP | 3/5 | Slightly faster gesture tempo | help | 99.91% | 0.81 ms | Yes | **CORRECT (ACCEPTED)** |
| 9 | HELP | 4/5 | Slightly slower gesture tempo | help | 99.93% | 0.76 ms | Yes | **CORRECT (ACCEPTED)** |
| 10 | HELP | 5/5 | Slightly tilted / alternative hand posture | help | 99.94% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 11 | YES | 1/5 | Normal distance & center framing | yes | 99.77% | 0.84 ms | Yes | **CORRECT (ACCEPTED)** |
| 12 | YES | 2/5 | Slightly left/right hand position | yes | 99.72% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 13 | YES | 3/5 | Slightly faster gesture tempo | yes | 99.77% | 0.82 ms | Yes | **CORRECT (ACCEPTED)** |
| 14 | YES | 4/5 | Slightly slower gesture tempo | yes | 99.82% | 0.81 ms | Yes | **CORRECT (ACCEPTED)** |
| 15 | YES | 5/5 | Slightly tilted / alternative hand posture | yes | 99.79% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 16 | NO | 1/5 | Normal distance & center framing | no | 99.55% | 0.8 ms | Yes | **CORRECT (ACCEPTED)** |
| 17 | NO | 2/5 | Slightly left/right hand position | no | 98.4% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 18 | NO | 3/5 | Slightly faster gesture tempo | no | 99.31% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 19 | NO | 4/5 | Slightly slower gesture tempo | yes | 80.5% | 0.81 ms | Yes | FALSE HIGH-CONF |
| 20 | NO | 5/5 | Slightly tilted / alternative hand posture | no | 99.39% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 21 | PLEASE | 1/5 | Normal distance & center framing | please | 99.99% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 22 | PLEASE | 2/5 | Slightly left/right hand position | please | 99.94% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 23 | PLEASE | 3/5 | Slightly faster gesture tempo | thank_you | 99.98% | 0.79 ms | Yes | FALSE HIGH-CONF |
| 24 | PLEASE | 4/5 | Slightly slower gesture tempo | thank_you | 99.98% | 0.78 ms | Yes | FALSE HIGH-CONF |
| 25 | PLEASE | 5/5 | Slightly tilted / alternative hand posture | please | 99.99% | 0.79 ms | Yes | **CORRECT (ACCEPTED)** |
| 26 | THANK_YOU | 1/5 | Normal distance & center framing | thank_you | 99.98% | 0.82 ms | Yes | **CORRECT (ACCEPTED)** |
| 27 | THANK_YOU | 2/5 | Slightly left/right hand position | thank_you | 99.98% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 28 | THANK_YOU | 3/5 | Slightly faster gesture tempo | thank_you | 99.98% | 0.76 ms | Yes | **CORRECT (ACCEPTED)** |
| 29 | THANK_YOU | 4/5 | Slightly slower gesture tempo | thank_you | 99.98% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 30 | THANK_YOU | 5/5 | Slightly tilted / alternative hand posture | thank_you | 99.98% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 31 | DOCTOR | 1/5 | Normal distance & center framing | doctor | 99.9% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 32 | DOCTOR | 2/5 | Slightly left/right hand position | doctor | 99.96% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 33 | DOCTOR | 3/5 | Slightly faster gesture tempo | doctor | 99.68% | 0.74 ms | Yes | **CORRECT (ACCEPTED)** |
| 34 | DOCTOR | 4/5 | Slightly slower gesture tempo | doctor | 96.99% | 0.74 ms | Yes | **CORRECT (ACCEPTED)** |
| 35 | DOCTOR | 5/5 | Slightly tilted / alternative hand posture | doctor | 99.81% | 0.76 ms | Yes | **CORRECT (ACCEPTED)** |
| 36 | PAIN | 1/5 | Normal distance & center framing | pain | 99.66% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 37 | PAIN | 2/5 | Slightly left/right hand position | pain | 99.95% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 38 | PAIN | 3/5 | Slightly faster gesture tempo | pain | 99.94% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 39 | PAIN | 4/5 | Slightly slower gesture tempo | pain | 99.99% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 40 | PAIN | 5/5 | Slightly tilted / alternative hand posture | pain | 67.64% | 0.74 ms | No | REJECTED (LOW CONF MATCH) |
| 41 | SICK | 1/5 | Normal distance & center framing | sick | 99.49% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 42 | SICK | 2/5 | Slightly left/right hand position | sick | 99.92% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 43 | SICK | 3/5 | Slightly faster gesture tempo | thank_you | 99.67% | 0.75 ms | Yes | FALSE HIGH-CONF |
| 44 | SICK | 4/5 | Slightly slower gesture tempo | sick | 99.73% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 45 | SICK | 5/5 | Slightly tilted / alternative hand posture | sick | 99.52% | 0.75 ms | Yes | **CORRECT (ACCEPTED)** |
| 46 | WHERE | 1/5 | Normal distance & center framing | where | 99.29% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 47 | WHERE | 2/5 | Slightly left/right hand position | where | 99.66% | 0.86 ms | Yes | **CORRECT (ACCEPTED)** |
| 48 | WHERE | 3/5 | Slightly faster gesture tempo | where | 99.93% | 0.77 ms | Yes | **CORRECT (ACCEPTED)** |
| 49 | WHERE | 4/5 | Slightly slower gesture tempo | where | 99.82% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
| 50 | WHERE | 5/5 | Slightly tilted / alternative hand posture | where | 71.26% | 0.78 ms | Yes | **CORRECT (ACCEPTED)** |
