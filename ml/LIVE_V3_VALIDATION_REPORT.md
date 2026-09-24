# SignBridge AI — Live Webcam V3 Validation Report (Pipeline Fixed)

**Model Tested:** Candidate Model V3 (`dynamic_bigru_v3.pt`)  
**Input Dimension:** 30 frames × 168 features  
**Dynamic Classes:** 22 classes  
**Evaluation Mode:** Live Physical Webcam Continuous Recognition Protocol  
**Date:** 2026-09-24 20:27:11  

## Summary Results
- **Total Controlled Attempts:** 21
- **Correct Recognitions:** 14
- **Live Empirical Accuracy:** **66.67%**
- **Average Prediction Confidence:** 82.59%
- **Accepted Predictions (>= 70%):** 16
- **Uncertain / Rejected (< 70%):** 5

## Per-Attempt Log
| # | Expected Sign | Attempt | Condition | Predicted | Confidence | Accepted | Result |
| :-: | :--- | :-: | :--- | :--- | :-: | :-: | :-: |
| 1 | HELP | 1/3 | Normal distance & center framing | help | 95.13% | Yes | CORRECT |
| 2 | HELP | 2/3 | Slightly left/right hand position | help | 89.9% | Yes | CORRECT |
| 3 | HELP | 3/3 | Slightly different camera distance | help | 95.71% | Yes | CORRECT |
| 4 | YES | 1/3 | Normal distance & center framing | yes | 94.73% | Yes | CORRECT |
| 5 | YES | 2/3 | Slightly left/right hand position | yes | 96.19% | Yes | CORRECT |
| 6 | YES | 3/3 | Slightly different camera distance | yes | 56.86% | No | INCORRECT |
| 7 | NO | 1/3 | Normal distance & center framing | no | 41.28% | No | INCORRECT |
| 8 | NO | 2/3 | Slightly left/right hand position | no | 44.19% | No | INCORRECT |
| 9 | NO | 3/3 | Slightly different camera distance | no | 41.2% | No | INCORRECT |
| 10 | THANK YOU | 1/3 | Normal distance & center framing | thank_you | 83.27% | Yes | CORRECT |
| 11 | THANK YOU | 2/3 | Slightly left/right hand position | thank_you | 87.04% | Yes | CORRECT |
| 12 | THANK YOU | 3/3 | Slightly different camera distance | thank_you | 83.38% | Yes | CORRECT |
| 13 | PLEASE | 1/3 | Normal distance & center framing | please | 98.04% | Yes | CORRECT |
| 14 | PLEASE | 2/3 | Slightly left/right hand position | please | 97.79% | Yes | CORRECT |
| 15 | PLEASE | 3/3 | Slightly different camera distance | please | 97.73% | Yes | CORRECT |
| 16 | HELLO | 1/3 | Normal distance & center framing | hello | 96.51% | Yes | CORRECT |
| 17 | HELLO | 2/3 | Slightly left/right hand position | hello | 94.82% | Yes | CORRECT |
| 18 | HELLO | 3/3 | Slightly different camera distance | hello | 96.66% | Yes | CORRECT |
| 19 | GOOD | 1/3 | Normal distance & center framing | good | 55.24% | No | INCORRECT |
| 20 | GOOD | 2/3 | Slightly left/right hand position | wait | 93.75% | Yes | INCORRECT |
| 21 | GOOD | 3/3 | Slightly different camera distance | wait | 94.97% | Yes | INCORRECT |
