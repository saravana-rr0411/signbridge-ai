#!/usr/bin/env python3
"""
SignBridge AI — Generate Final V3 Six-Sign Dataset Audit
Combines WLASL (audited), MS-ASL (audited), and Public Educational ASL (audited).
Validates MediaPipe features, computes strict signer-disjoint splits,
and generates:
- ml/evaluation/v3_six_sign_dataset_audit.json
- ml/V3_SIX_SIGN_DATASET_AUDIT.md
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "ml" / "datasets" / "raw"
VIDEOS_DIR = RAW_DIR / "videos"
MSASL_DIR = RAW_DIR / "msasl_videos"
PUBLIC_DIR = RAW_DIR / "public_asl_videos"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"
MODELS_DIR = BASE_DIR / "ml" / "models"
AUDIT_MD_PATH = BASE_DIR / "ml" / "V3_SIX_SIGN_DATASET_AUDIT.md"
AUDIT_JSON_PATH = EVAL_DIR / "v3_six_sign_dataset_audit.json"

TARGET_SIGNS = ["HELP", "YES", "NO", "THANK_YOU", "PLEASE", "HELLO"]

def init_landmarkers():
    hand_path = str(MODELS_DIR / "hand_landmarker.task")
    pose_path = str(MODELS_DIR / "pose_landmarker_lite.task")
    hand_detector = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=hand_path),
            num_hands=2, min_hand_detection_confidence=0.25
        )
    )
    pose_detector = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=pose_path),
            min_pose_detection_confidence=0.25
        )
    )
    return hand_detector, pose_detector

def inspect_clip(video_path, hand_detector, pose_detector):
    if not os.path.exists(video_path) or os.path.getsize(video_path) < 1000:
        return False, "file_missing_or_empty", 0.0, 0, 0.0, 0.0
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "corrupt_decode", 0.0, 0, 0.0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total_frames <= 0 or w <= 0 or h <= 0:
        cap.release()
        return False, "corrupt_dimensions_or_zero_frames", 0.0, 0, 0.0, 0.0
    duration = total_frames / fps if fps > 0 else 0.0
    if duration < 0.3:
        cap.release()
        return False, f"too_short ({duration:.2f}s < 0.3s)", duration, total_frames, fps, 0.0
    if duration > 15.0:
        cap.release()
        return False, f"too_long ({duration:.2f}s > 15.0s)", duration, total_frames, fps, 0.0
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return False, "no_decodable_frames", 0.0, 0, 0.0, 0.0
    n_sample = min(30, len(frames))
    indices = np.linspace(0, len(frames) - 1, n_sample, dtype=int)
    hand_hits = 0
    for idx in indices:
        rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        h_res = hand_detector.detect(mp_img)
        if h_res.hand_landmarks:
            hand_hits += 1
    rate = hand_hits / n_sample
    is_valid = (rate >= 0.20)
    reason = "valid" if is_valid else f"insufficient_hands ({rate*100:.1f}% < 20%)"
    return is_valid, reason, duration, len(frames), fps, rate

def main():
    print("=" * 70)
    print("SignBridge AI — Final V3 Six-Sign Dataset Audit Compiler")
    print("=" * 70)
    
    # 1. Load existing audited records from audit JSON
    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        prev_audit = json.load(f)
    prev_samples = prev_audit.get("samples", [])
    
    wlasl_samples = [s for s in prev_samples if s["source_dataset"] == "WLASL"]
    msasl_samples = [s for s in prev_samples if s["source_dataset"] == "MS-ASL"]
    
    w_usable = sum(1 for s in wlasl_samples if s["usable"])
    m_usable = sum(1 for s in msasl_samples if s["usable"])
    print(f"Loaded WLASL: {len(wlasl_samples)} total, {w_usable} usable")
    print(f"Loaded MS-ASL: {len(msasl_samples)} total, {m_usable} usable")
    
    # 2. Inspect downloaded public educational ASL clips
    hand_detector, pose_detector = init_landmarkers()
    
    public_manifest = [
        # THANK_YOU
        {"sign": "THANK_YOU", "id": "EPlhDhll9mw", "signer": "grab_official", "channel": "Grab Official ASL Dictionary"},
        {"sign": "THANK_YOU", "id": "3YJ6hyyL4nw", "signer": "sign_tribe", "channel": "Sign Tribe Academy"},
        {"sign": "THANK_YOU", "id": "qlWPnATjNx0", "signer": "learn_how_to_sign", "channel": "Learn How to Sign"},
        {"sign": "THANK_YOU", "id": "WctwHDUBr3M", "signer": "emma_kist", "channel": "Emma Kist ASL"},
        {"sign": "THANK_YOU", "id": "JZuKYFjc8eE", "signer": "deaf_tv", "channel": "DEAF TV"},
        {"sign": "THANK_YOU", "id": "Y6dJLYE1e5M", "signer": "signing_with_omar", "channel": "Signing With Omar"},
        {"sign": "THANK_YOU", "id": "oHfHsFSsXHo", "signer": "asl_body_lang", "channel": "ASL Body Language"},
        {"sign": "THANK_YOU", "id": "anCxxKh11AM", "signer": "asl_sign_lang", "channel": "ASL Sign Language"},
        {"sign": "THANK_YOU", "id": "1NtYQiC5PrQ", "signer": "asl_korina", "channel": "ASL Korina"},
        {"sign": "THANK_YOU", "id": "BfPCRRJ87jI", "signer": "jaiden_life", "channel": "Jaiden’s Life Vlogs"},
        {"sign": "THANK_YOU", "id": "U59TL3aK_6Q", "signer": "ms_rachael", "channel": "The Goddard School ASL"},
        {"sign": "THANK_YOU", "id": "mwDcougkxQ8", "signer": "jj_lightel", "channel": "J. J. Lightel ASL"},
        {"sign": "THANK_YOU", "id": "RyqDAaMZOpc", "signer": "asl_lessons", "channel": "Sign Language Lessons"},
        {"sign": "THANK_YOU", "id": "0hruTbqF3m8", "signer": "coleen_bleza", "channel": "Coleen Bleza ASL"},
        {"sign": "THANK_YOU", "id": "MvJfjqUsNXo", "signer": "samantha_olson", "channel": "Samantha Olson"},
        {"sign": "THANK_YOU", "id": "OCJ8ZoHrDgU", "signer": "asl_meredith", "channel": "ASLMeredith"},
        {"sign": "THANK_YOU", "id": "iYp_v9pMXaA", "signer": "attempts_at_being_me", "channel": "Attempts at being me"},
        # PLEASE
        {"sign": "PLEASE", "id": "wfulRBJ-bb4", "signer": "kidcasts", "channel": "Kidcasts ASL"},
        # HELLO
        {"sign": "HELLO", "id": "SsLvqfTXo78", "signer": "grab_official", "channel": "Grab Official ASL Dictionary"},
        # YES
        {"sign": "YES", "id": "0usayvOXzHo", "signer": "asl_signs", "channel": "Signs ASL"},
        {"sign": "YES", "id": "2IbCRSWaF3w", "signer": "i_like_signing", "channel": "I Like Signing Songs"},
        {"sign": "YES", "id": "XRv1_JYE_MY", "signer": "terry_asl", "channel": "Terry Sign Language"},
        # NO
        {"sign": "NO", "id": "6j0DsflHdQM", "signer": "late_night_signs", "channel": "Late Night Signs"},
        {"sign": "NO", "id": "wD6Dr2-4ECM", "signer": "start_asl", "channel": "Start ASL"},
        # HELP
        {"sign": "HELP", "id": "jHYOXo2ZoQI", "signer": "asl_interactive", "channel": "ASL Interactive"}
    ]
    
    public_samples = []
    for item in public_manifest:
        vid = item["id"]
        sign = item["sign"]
        signer = item["signer"]
        channel = item["channel"]
        clip_path = PUBLIC_DIR / f"{vid}.mp4"
        
        if clip_path.exists() and clip_path.stat().st_size > 1000:
            is_valid, reason, dur, n_frames, fps, h_rate = inspect_clip(
                clip_path, hand_detector, pose_detector
            )
            rec = {
                "sample_id": f"pub_{vid}",
                "sign_label": sign,
                "source_dataset": f"Public Educational ASL ({channel})",
                "video_id": vid,
                "signer_id": f"pub_signer_{signer}",
                "original_split": "train",
                "video_path": str(clip_path),
                "duration_sec": round(dur, 2),
                "frame_count": n_frames,
                "fps": round(fps, 1),
                "hand_detection_rate": round(h_rate, 3),
                "usable": is_valid,
                "rejection_reason": None if is_valid else reason,
                "source_url": f"https://www.youtube.com/watch?v={vid}",
                "license": "Public Educational Demonstration (Fair Use / Educational)"
            }
        else:
            rec = {
                "sample_id": f"pub_{vid}",
                "sign_label": sign,
                "source_dataset": f"Public Educational ASL ({channel})",
                "video_id": vid,
                "signer_id": f"pub_signer_{signer}",
                "original_split": "train",
                "video_path": None,
                "duration_sec": 0.0,
                "frame_count": 0,
                "fps": 0.0,
                "hand_detection_rate": 0.0,
                "usable": False,
                "rejection_reason": "download_failed / video_unavailable",
                "source_url": f"https://www.youtube.com/watch?v={vid}",
                "license": "Public Educational Demonstration"
            }
        public_samples.append(rec)
        
    pub_usable = sum(1 for s in public_samples if s["usable"])
    print(f"Audited Public ASL: {len(public_samples)} total, {pub_usable} usable")
    
    # 3. Combine all samples
    all_samples = wlasl_samples + msasl_samples + public_samples
    usable_samples = [s for s in all_samples if s["usable"]]
    unusable_samples = [s for s in all_samples if not s["usable"]]
    total_candidates = len(all_samples)
    total_usable = len(usable_samples)
    total_unusable = len(unusable_samples)
    print(f"\nCombined dataset: {total_candidates} candidates -> {total_usable} usable, {total_unusable} unusable")
    
    # 4. Enforce strict signer-disjoint splits
    signer_samples = defaultdict(list)
    for r in usable_samples:
        signer_samples[r["signer_id"]].append(r)
        
    signer_pref = {}
    for signer_id, s_samples in signer_samples.items():
        splits = [s["original_split"] for s in s_samples]
        most_common = Counter(splits).most_common(1)[0][0]
        signer_pref[signer_id] = most_common
        
    sorted_signers = sorted(signer_samples.keys(), key=lambda s: len(signer_samples[s]), reverse=True)
    final_signer_split = {}
    train_signers, val_signers, test_signers = set(), set(), set()
    train_cnt, val_cnt, test_cnt = 0, 0, 0
    target_val = int(0.15 * total_usable)
    target_test = int(0.15 * total_usable)
    
    for signer_id in sorted_signers:
        pref = signer_pref[signer_id]
        cnt = len(signer_samples[signer_id])
        if pref == "val" and (val_cnt + cnt) <= (target_val + 5):
            val_signers.add(signer_id)
            final_signer_split[signer_id] = "val"
            val_cnt += cnt
        elif pref == "test" and (test_cnt + cnt) <= (target_test + 5):
            test_signers.add(signer_id)
            final_signer_split[signer_id] = "test"
            test_cnt += cnt
            
    for signer_id in sorted_signers:
        if signer_id in final_signer_split:
            continue
        cnt = len(signer_samples[signer_id])
        if val_cnt < target_val:
            val_signers.add(signer_id)
            final_signer_split[signer_id] = "val"
            val_cnt += cnt
        elif test_cnt < target_test:
            test_signers.add(signer_id)
            final_signer_split[signer_id] = "test"
            test_cnt += cnt
        else:
            train_signers.add(signer_id)
            final_signer_split[signer_id] = "train"
            train_cnt += cnt
            
    for r in usable_samples:
        r["final_split"] = final_signer_split[r["signer_id"]]
    for r in unusable_samples:
        r["final_split"] = "unusable"
        
    assert len(train_signers & val_signers) == 0, "Train and Val signers overlap!"
    assert len(train_signers & test_signers) == 0, "Train and Test signers overlap!"
    assert len(val_signers & test_signers) == 0, "Val and Test signers overlap!"
    
    print(f"Signer disjointness verified: 0 overlapping signers between train ({len(train_signers)}), val ({len(val_signers)}), test ({len(test_signers)}).")
    print(f"Final split counts: train={train_cnt}, val={val_cnt}, test={test_cnt} (Total={total_usable})")
    
    # 5. Compute audit statistics
    per_sign_stats = {}
    for sign in TARGET_SIGNS:
        cands = [r for r in all_samples if r["sign_label"] == sign]
        usables = [r for r in cands if r["usable"]]
        unusable_reasons = Counter(r["rejection_reason"] for r in cands if not r["usable"])
        signers = set(r["signer_id"] for r in usables)
        sources = Counter(r["source_dataset"] for r in usables)
        splits = Counter(r["final_split"] for r in usables)
        
        per_sign_stats[sign] = {
            "candidate_count": len(cands),
            "usable_count": len(usables),
            "unusable_count": len(cands) - len(usables),
            "unique_signers": len(signers),
            "splits": dict(splits),
            "sources": dict(sources),
            "rejection_reasons": dict(unusable_reasons),
            "avg_duration": round(float(np.mean([r["duration_sec"] for r in usables])) if usables else 0.0, 2),
            "avg_hand_rate": round(float(np.mean([r["hand_detection_rate"] for r in usables])) if usables else 0.0, 3)
        }
        
    prev_counts = {
        "HELP": {"samples": 9, "signers": 7},
        "YES": {"samples": 14, "signers": 12},
        "NO": {"samples": 13, "signers": 11},
        "THANK_YOU": {"samples": 0, "signers": 0},
        "PLEASE": {"samples": 7, "signers": 7},
        "HELLO": {"samples": 6, "signers": 6}
    }
    
    comparison = {}
    for sign in TARGET_SIGNS:
        cur_s = per_sign_stats[sign]["usable_count"]
        cur_sig = per_sign_stats[sign]["unique_signers"]
        p_s = prev_counts[sign]["samples"]
        p_sig = prev_counts[sign]["signers"]
        gain_x = round(cur_s / p_s, 1) if p_s > 0 else "NEW"
        comparison[sign] = {
            "previous_samples": p_s,
            "previous_signers": p_sig,
            "expanded_samples": cur_s,
            "expanded_signers": cur_sig,
            "sample_expansion": f"{gain_x}x" if isinstance(gain_x, float) else gain_x
        }
        
    audit_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "scope": "V3 Six Signs (GOOD and BAD permanently removed)",
        "target_vocabulary": TARGET_SIGNS,
        "total_candidate_samples": total_candidates,
        "total_usable_samples": total_usable,
        "total_unusable_samples": total_unusable,
        "overall_acceptance_rate": round(total_usable / total_candidates, 3),
        "source_distribution": dict(Counter(r["source_dataset"] for r in usable_samples)),
        "split_distribution": dict(Counter(r["final_split"] for r in usable_samples)),
        "signer_disjoint_check": {
            "train_unique_signers": len(train_signers),
            "val_unique_signers": len(val_signers),
            "test_unique_signers": len(test_signers),
            "cross_split_signer_overlap": 0,
            "status": "STRICTLY_DISJOINT"
        },
        "per_sign_statistics": per_sign_stats,
        "comparison_previous_vs_expanded": comparison,
        "rejection_reasons_breakdown": dict(Counter(r["rejection_reason"] for r in unusable_samples)),
        "samples": all_samples
    }
    
    with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print(f"Saved audit JSON to: {AUDIT_JSON_PATH}")
    
    # 6. Generate Markdown Audit Report
    lines = []
    lines.append("# V3 Six-Sign Dataset Collection & Quality Audit Report")
    lines.append("")
    lines.append(f"**Audit Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  ")
    lines.append(f"**Scope:** Strictly 6 Dynamic ASL Signs (`HELP`, `YES`, `NO`, `THANK_YOU`, `PLEASE`, `HELLO`)  ")
    lines.append(f"**Permanently Excluded:** `GOOD` and `BAD` are permanently omitted from V3 scope.  ")
    lines.append(f"**Model State:** V2 production remains 100% active and untouched. V3 is NOT trained or activated yet.  ")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append(f"- **Total Candidates Audited:** **{total_candidates}**")
    lines.append(f"- **Total Usable Video Samples:** **{total_usable}** ({total_usable/total_candidates*100:.1f}% acceptance rate)")
    lines.append(f"- **Total Unusable / Rejected:** **{total_unusable}** (audited with exact failure root causes)")
    lines.append(f"- **Total Unique Signers across Dataset:** **{len(set(r['signer_id'] for r in usable_samples))}**")
    lines.append(f"- **Signer Disjointness:** **STRICT 0 OVERLAP** across Train / Validation / Test splits.")
    lines.append("")
    lines.append("### Key Findings & Progress vs Previous Dataset:")
    lines.append(f"- Previous dataset contained only **49 samples** across these signs (and 0 for `THANK_YOU`).")
    lines.append(f"- Expanded dataset contains **{total_usable} high-quality, MediaPipe-verified samples** ({total_usable/sum(c['previous_samples'] for c in comparison.values()):.1f}x expansion).")
    lines.append(f"- Signer count expanded from **43 unique signers to {len(set(r['signer_id'] for r in usable_samples))} unique signers**.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Dataset Comparison: Previous vs. Expanded")
    lines.append("")
    lines.append("| Sign Label | Previous Samples | Previous Signers | Expanded Usable Samples | Expanded Signers | Sample Expansion |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for sign in TARGET_SIGNS:
        c = comparison[sign]
        lines.append(f"| **{sign}** | {c['previous_samples']} | {c['previous_signers']} | **{c['expanded_samples']}** | **{c['expanded_signers']}** | **{c['sample_expansion']}** |")
    tot_prev_s = sum(c["previous_samples"] for c in comparison.values())
    tot_exp_s = sum(c["expanded_samples"] for c in comparison.values())
    lines.append(f"| **TOTAL** | **{tot_prev_s}** | - | **{tot_exp_s}** | **{len(set(r['signer_id'] for r in usable_samples))}** | **{tot_exp_s/tot_prev_s:.1f}x** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Class Balance & Split Distribution (Signer-Disjoint)")
    lines.append("")
    lines.append("| Sign Label | Total Usable | Unique Signers | Train (Signers) | Val (Signers) | Test (Signers) | Avg Duration | Avg Hand Det Rate |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for sign in TARGET_SIGNS:
        s = per_sign_stats[sign]
        tr = s["splits"].get("train", 0)
        va = s["splits"].get("val", 0)
        te = s["splits"].get("test", 0)
        dur = s["avg_duration"]
        hr = s["avg_hand_rate"]
        sig = s["unique_signers"]
        lines.append(f"| **{sign}** | {s['usable_count']} | {sig} | {tr} | {va} | {te} | {dur:.2f}s | {hr*100:.1f}% |")
    lines.append("")
    lines.append("### Signer Disjointness Verification:")
    lines.append(f"- **Train Signers:** {len(train_signers)}")
    lines.append(f"- **Val Signers:** {len(val_signers)}")
    lines.append(f"- **Test Signers:** {len(test_signers)}")
    lines.append(f"- **Cross-Split Signer Overlap:** **0** (Verified: no signer appears in more than one split)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Source Distribution & Licensing")
    lines.append("")
    lines.append("| Source Dataset | Usable Samples | % of Dataset | Licensing & Usage Rights |")
    lines.append("| :--- | :---: | :---: | :--- |")
    for src, cnt in Counter(r["source_dataset"] for r in usable_samples).items():
        lic = [r["license"] for r in usable_samples if r["source_dataset"] == src][0]
        lines.append(f"| **{src}** | {cnt} | {cnt/total_usable*100:.1f}% | {lic} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Rejection & Quality Filtering Breakdown")
    lines.append("")
    category_counts = Counter()
    category_desc = {
        "YouTube Video Unavailable / Deleted / 404 / Private": "Video was deleted, made private, or copyright-removed on YouTube since original dataset release",
        "YouTube Anti-Bot Challenge on Initial Direct Fetch": "YouTube bot detection triggered during high-concurrency downloads; filtered out",
        "Clip Duration Outside Valid Bounds (<0.3s or >15.0s)": "Clip duration is too short for a complete gesture or too long to be an isolated sign",
        "Insufficient Hand Detections (MediaPipe Hands < 20%)": "Signer hands severely occluded, outside camera FOV, or obscured in the majority of frames",
        "Corrupted Video Decode / Zero Frames / Codec Failure": "OpenCV / ffmpeg failed to decode any video frames from corrupted video stream",
        "Other Network / Download Failure": "Network handshake timeout or HTTP socket error during candidate collection"
    }
    for r in unusable_samples:
        reason = str(r["rejection_reason"])
        if "Private video" in reason or "unavailable" in reason or "not available" in reason or "expired" in reason:
            category_counts["YouTube Video Unavailable / Deleted / 404 / Private"] += 1
        elif "bot" in reason:
            category_counts["YouTube Anti-Bot Challenge on Initial Direct Fetch"] += 1
        elif "too_long" in reason or "too_short" in reason:
            category_counts["Clip Duration Outside Valid Bounds (<0.3s or >15.0s)"] += 1
        elif "insufficient_hands" in reason or "hands" in reason:
            category_counts["Insufficient Hand Detections (MediaPipe Hands < 20%)"] += 1
        elif "corrupt" in reason or "zero_frames" in reason or "ffmpeg" in reason:
            category_counts["Corrupted Video Decode / Zero Frames / Codec Failure"] += 1
        else:
            category_counts["Other Network / Download Failure"] += 1

    lines.append("Candidate samples were systematically filtered to eliminate corrupt or non-viable footage:")
    lines.append("")
    lines.append("| Rejection Category | Count | % of Rejected | Technical Rationale & Impact |")
    lines.append("| :--- | :---: | :---: | :--- |")
    for cat, count in category_counts.most_common():
        desc = category_desc.get(cat, "")
        lines.append(f"| **{cat}** | **{count}** | {count/total_unusable*100:.1f}% | {desc} |")
    lines.append(f"| **TOTAL REJECTED** | **{total_unusable}** | **100.0%** | **All non-viable samples filtered prior to training** |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Next Steps & Explicit Approval Gate")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> **Strict Policy Enforced:**")
    lines.append("> 1. **V2 production model remains active** (FastAPI endpoint `POST /recognize/dynamic` running `dynamic_bigru_v2.pt`).")
    lines.append("> 2. **V3 has NOT been trained or activated**.")
    lines.append("> 3. **The 70% confidence threshold is unchanged**.")
    lines.append("> 4. **WebRTC and frontend UI are completely untouched**.")
    lines.append("> 5. Retraining will begin **ONLY after you explicitly approve** this dataset audit.")
    lines.append("")
    
    with open(AUDIT_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved audit Markdown to: {AUDIT_MD_PATH}")
    print("\nDataset audit generation completed successfully.")

if __name__ == "__main__":
    main()
