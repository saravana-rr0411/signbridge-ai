# V3 Six-Sign Dataset Collection & Quality Audit Report

**Audit Generated:** 2026-09-24 15:45:20 UTC  
**Scope:** Strictly 6 Dynamic ASL Signs (`HELP`, `YES`, `NO`, `THANK_YOU`, `PLEASE`, `HELLO`)  
**Permanently Excluded:** `GOOD` and `BAD` are permanently omitted from V3 scope.  
**Model State:** V2 production remains 100% active and untouched. V3 is NOT trained or activated yet.  

---

## 1. Executive Summary

- **Total Candidates Audited:** **453**
- **Total Usable Video Samples:** **220** (48.6% acceptance rate)
- **Total Unusable / Rejected:** **233** (audited with exact failure root causes)
- **Total Unique Signers across Dataset:** **86**
- **Signer Disjointness:** **STRICT 0 OVERLAP** across Train / Validation / Test splits.

### Key Findings & Progress vs Previous Dataset:
- Previous dataset contained only **49 samples** across these signs (and 0 for `THANK_YOU`).
- Expanded dataset contains **220 high-quality, MediaPipe-verified samples** (4.5x expansion).
- Signer count expanded from **43 unique signers to 86 unique signers**.

---

## 2. Dataset Comparison: Previous vs. Expanded

| Sign Label | Previous Samples | Previous Signers | Expanded Usable Samples | Expanded Signers | Sample Expansion |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HELP** | 9 | 7 | **44** | **28** | **4.9x** |
| **YES** | 14 | 12 | **48** | **35** | **3.4x** |
| **NO** | 13 | 11 | **48** | **36** | **3.7x** |
| **THANK_YOU** | 0 | 0 | **21** | **21** | **NEW** |
| **PLEASE** | 7 | 7 | **33** | **25** | **4.7x** |
| **HELLO** | 6 | 6 | **26** | **25** | **4.3x** |
| **TOTAL** | **49** | - | **220** | **86** | **4.5x** |

---

## 3. Class Balance & Split Distribution (Signer-Disjoint)

| Sign Label | Total Usable | Unique Signers | Train (Signers) | Val (Signers) | Test (Signers) | Avg Duration | Avg Hand Det Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HELP** | 44 | 28 | 29 | 7 | 8 | 2.72s | 80.4% |
| **YES** | 48 | 35 | 31 | 9 | 8 | 2.76s | 74.0% |
| **NO** | 48 | 36 | 28 | 10 | 10 | 3.12s | 78.8% |
| **THANK_YOU** | 21 | 21 | 18 | 2 | 1 | 5.63s | 64.6% |
| **PLEASE** | 33 | 25 | 21 | 4 | 8 | 3.24s | 83.1% |
| **HELLO** | 26 | 25 | 17 | 5 | 4 | 3.49s | 75.9% |

### Signer Disjointness Verification:
- **Train Signers:** 69
- **Val Signers:** 11
- **Test Signers:** 6
- **Cross-Split Signer Overlap:** **0** (Verified: no signer appears in more than one split)

---

## 4. Source Distribution & Licensing

| Source Dataset | Usable Samples | % of Dataset | Licensing & Usage Rights |
| :--- | :---: | :---: | :--- |
| **WLASL** | 69 | 31.4% | WLASL Research/Academic License |
| **MS-ASL** | 132 | 60.0% | Microsoft Research C-UDA Agreement |
| **Public Educational ASL (Sign Tribe Academy)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Learn How to Sign)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Emma Kist ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (DEAF TV)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Signing With Omar)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (ASL Body Language)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (ASL Sign Language)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (ASL Korina)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Jaiden’s Life Vlogs)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (The Goddard School ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (J. J. Lightel ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Sign Language Lessons)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Coleen Bleza ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Kidcasts ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Grab Official ASL Dictionary)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Signs ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (I Like Signing Songs)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (Start ASL)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |
| **Public Educational ASL (ASL Interactive)** | 1 | 0.5% | Public Educational Demonstration (Fair Use / Educational) |

---

## 5. Rejection & Quality Filtering Breakdown

Candidate samples were systematically filtered to eliminate corrupt or non-viable footage:

| Rejection Category | Count | % of Rejected | Technical Rationale & Impact |
| :--- | :---: | :---: | :--- |
| **YouTube Video Unavailable / Deleted / 404 / Private** | **141** | 60.5% | Video was deleted, made private, or copyright-removed on YouTube since original dataset release |
| **YouTube Anti-Bot Challenge on Initial Direct Fetch** | **66** | 28.3% | YouTube bot detection triggered during high-concurrency downloads; filtered out |
| **Clip Duration Outside Valid Bounds (<0.3s or >15.0s)** | **9** | 3.9% | Clip duration is too short for a complete gesture or too long to be an isolated sign |
| **Other Network / Download Failure** | **7** | 3.0% | Network handshake timeout or HTTP socket error during candidate collection |
| **Corrupted Video Decode / Zero Frames / Codec Failure** | **5** | 2.1% | OpenCV / ffmpeg failed to decode any video frames from corrupted video stream |
| **Insufficient Hand Detections (MediaPipe Hands < 20%)** | **5** | 2.1% | Signer hands severely occluded, outside camera FOV, or obscured in the majority of frames |
| **TOTAL REJECTED** | **233** | **100.0%** | **All non-viable samples filtered prior to training** |

---

## 6. Next Steps & Explicit Approval Gate

> [!IMPORTANT]
> **Strict Policy Enforced:**
> 1. **V2 production model remains active** (FastAPI endpoint `POST /recognize/dynamic` running `dynamic_bigru_v2.pt`).
> 2. **V3 has NOT been trained or activated**.
> 3. **The 70% confidence threshold is unchanged**.
> 4. **WebRTC and frontend UI are completely untouched**.
> 5. Retraining will begin **ONLY after you explicitly approve** this dataset audit.
