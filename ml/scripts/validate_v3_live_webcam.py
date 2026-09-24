"""
SignBridge AI - Real Webcam Live Validation Tool for Model V3
Phase: STEP 4 - Live Camera Validation Only (Feature Pipeline Fixed)

Fixes applied:
1. Unmirrored MediaPipe Inference: Frame passed to MediaPipe is unmirrored (raw), matching WLASL training convention.
2. Physical Handedness Verification: Uses Pose landmarks 15 (Left wrist) and 16 (Right wrist) to verify physical Left vs Right hand.
   - Physical Left hand -> features 0..63
   - Physical Right hand -> features 64..127
   - Left body-relative -> features 150..152, 156..158, 162..164
   - Right body-relative -> features 153..155, 159..161, 165..167
3. Mirror Preview for Display: cv2.flip(frame, 1) is used ONLY for the GUI preview window so the user sees a natural reflection.
4. Temporal Sampling Engine: Buffers a rolling 2.6-second window (~78 frames) and uniformly downsamples to strictly 30x168 frames.
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"
REPORT_MD_PATH = BASE_DIR / "ml" / "V3_FOCUSED_LIVE_VALIDATION_REPORT.md"
REPORT_JSON_PATH = EVAL_DIR / "v3_focused_live_validation_report.json"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.70
CAPTURE_WINDOW_SEC = 2.6

# Focused 24-attempt session: NO x5, GOOD x5, BAD x5, YES x3, HELP x3, PLEASE x3
TARGET_SIGNS = [
    {"name": "NO", "expected": "no", "gesture": "Index & middle fingers snap to thumb", "attempts": 5},
    {"name": "GOOD", "expected": "good", "gesture": "Flat hand chin to forward / thumbs up", "attempts": 5},
    {"name": "BAD", "expected": "bad", "gesture": "Flat hand chin flipping down / thumbs down", "attempts": 5},
    {"name": "YES", "expected": "yes", "gesture": "Fist nodding up/down", "attempts": 3},
    {"name": "HELP", "expected": "help", "gesture": "Closed fist on open palm", "attempts": 3},
    {"name": "PLEASE", "expected": "please", "gesture": "Open palm circular rubbing on chest", "attempts": 3},
]

ATTEMPT_CONDITIONS = [
    "Normal distance & center framing",
    "Slightly left/right hand position",
    "Slightly different camera distance",
    "Slightly faster / slower gesture tempo",
    "Alternative hand posture / tilt"
]


class DynamicSignBiGRU(nn.Module):
    def __init__(self, input_dim=168, hidden_dim=64, num_layers=2, num_classes=22, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.pool_dim = hidden_dim * 4
        self.fc1 = nn.Linear(self.pool_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, mask):
        gru_out, _ = self.gru(x)
        mask_expanded = mask.unsqueeze(-1)
        masked_out = gru_out * mask_expanded
        sum_out = torch.sum(masked_out, dim=1)
        lengths = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        mean_pool = sum_out / lengths
        masked_for_max = masked_out + (1.0 - mask_expanded) * -1e9
        max_pool, _ = torch.max(masked_for_max, dim=1)
        combined = torch.cat([mean_pool, max_pool], dim=-1)
        h = self.dropout(self.relu(self.fc1(combined)))
        logits = self.fc2(h)
        return logits


def init_landmarkers():
    hand_model_path = str(MODELS_DIR / "hand_landmarker.task")
    pose_model_path = str(MODELS_DIR / "pose_landmarker_lite.task")

    hand_base = python.BaseOptions(model_asset_path=hand_model_path)
    hand_opts = vision.HandLandmarkerOptions(
        base_options=hand_base,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    pose_base = python.BaseOptions(model_asset_path=pose_model_path)
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=pose_base,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)
    return hand_detector, pose_detector


def normalize_hand(landmarks):
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(64, dtype=np.float32)
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[0].copy()
    coords_centered = coords - wrist
    scale = np.linalg.norm(coords[9] - coords[0])
    if scale < 1e-4:
        scale = 1.0
    coords_norm = coords_centered / scale
    feat = np.zeros(64, dtype=np.float32)
    feat[:63] = coords_norm.flatten()
    feat[63] = 1.0
    return feat


def normalize_pose(landmarks):
    if landmarks is None or len(landmarks) < 17:
        return np.zeros(22, dtype=np.float32)
    keypoints = [0, 11, 12, 13, 14, 15, 16]
    raw_points = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in keypoints], dtype=np.float32)
    l_shoulder = raw_points[1] # pt 11
    r_shoulder = raw_points[2] # pt 12
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    scale = np.linalg.norm(r_shoulder - l_shoulder)
    if scale < 1e-4:
        scale = 1.0
    points_norm = (raw_points - mid_shoulder) / scale
    feat = np.zeros(22, dtype=np.float32)
    feat[:21] = points_norm.flatten()
    feat[21] = 1.0
    return feat


def compute_body_relative_features(l_hand_lms, r_hand_lms, pose_lms):
    rel_feat = np.zeros(18, dtype=np.float32)
    if pose_lms is None or len(pose_lms) < 17:
        return rel_feat

    l_shoulder = np.array([pose_lms[11].x, pose_lms[11].y, pose_lms[11].z], dtype=np.float32)
    r_shoulder = np.array([pose_lms[12].x, pose_lms[12].y, pose_lms[12].z], dtype=np.float32)
    nose = np.array([pose_lms[0].x, pose_lms[0].y, pose_lms[0].z], dtype=np.float32)

    shoulder_center = (l_shoulder + r_shoulder) / 2.0
    shoulder_width = np.linalg.norm(r_shoulder - l_shoulder)
    if shoulder_width < 1e-4:
        shoulder_width = 1.0

    down_vec = shoulder_center - nose
    down_dist = np.linalg.norm(down_vec)
    unit_down = down_vec / down_dist if down_dist > 1e-4 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    chest_center = shoulder_center + unit_down * (0.5 * shoulder_width)

    has_left = (l_hand_lms is not None and len(l_hand_lms) >= 21)
    has_right = (r_hand_lms is not None and len(r_hand_lms) >= 21)

    if has_left:
        l_wrist = np.array([l_hand_lms[0].x, l_hand_lms[0].y, l_hand_lms[0].z], dtype=np.float32)
        rel_feat[0:3] = (l_wrist - shoulder_center) / shoulder_width   # 150..152
        rel_feat[6:9] = (l_wrist - nose) / shoulder_width              # 156..158
        rel_feat[12:15] = (l_wrist - chest_center) / shoulder_width   # 162..164

    if has_right:
        r_wrist = np.array([r_hand_lms[0].x, r_hand_lms[0].y, r_hand_lms[0].z], dtype=np.float32)
        rel_feat[3:6] = (r_wrist - shoulder_center) / shoulder_width   # 153..155
        rel_feat[9:12] = (r_wrist - nose) / shoulder_width             # 159..161
        rel_feat[15:18] = (r_wrist - chest_center) / shoulder_width   # 165..167

    return rel_feat


def extract_168_vector(frame_bgr_unflipped, hand_detector, pose_detector):
    """
    Extracts strictly 168 features from RAW UNMIRRORED frame.
    Ensures physical Right hand -> [64..127], physical Left hand -> [0..63].
    """
    frame_rgb = cv2.cvtColor(frame_bgr_unflipped, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    hand_res = hand_detector.detect(mp_img)
    pose_res = pose_detector.detect(mp_img)

    pose_lms = pose_res.pose_landmarks[0] if (pose_res.pose_landmarks and len(pose_res.pose_landmarks) > 0) else None

    l_hand = None
    r_hand = None

    if hand_res.hand_landmarks:
        has_pose_wrists = (pose_lms is not None and len(pose_lms) >= 17)

        for idx, lms in enumerate(hand_res.hand_landmarks):
            hw = lms[0] # Hand wrist
            assigned_side = None

            # Geometric Physical Verification using Pose Wrists (15=Left, 16=Right)
            if has_pose_wrists:
                pw_l = pose_lms[15] # Physical Left wrist
                pw_r = pose_lms[16] # Physical Right wrist
                d_l = np.hypot(hw.x - pw_l.x, hw.y - pw_l.y)
                d_r = np.hypot(hw.x - pw_r.x, hw.y - pw_r.y)
                if abs(d_l - d_r) > 0.05:
                    assigned_side = "Left" if d_l < d_r else "Right"

            if assigned_side is None:
                # Fallback to unmirrored MediaPipe label (on unmirrored feed, MediaPipe Right = physical Right)
                label = hand_res.handedness[idx][0].category_name
                assigned_side = label # Direct unmirrored label

            if assigned_side == "Left" and l_hand is None:
                l_hand = lms
            elif assigned_side == "Right" and r_hand is None:
                r_hand = lms
            elif r_hand is None:
                r_hand = lms
            elif l_hand is None:
                l_hand = lms

    f_left = normalize_hand(l_hand)
    f_right = normalize_hand(r_hand)
    f_pose = normalize_pose(pose_lms)
    f_rel = compute_body_relative_features(l_hand, r_hand, pose_lms)

    vec = np.concatenate([f_left, f_right, f_pose, f_rel], axis=0).astype(np.float32)
    has_activity = bool(f_left[63] > 0.5 or f_right[63] > 0.5)

    return vec, has_activity, l_hand, r_hand, pose_lms


def main():
    print("=" * 75)
    print("SignBridge AI — Live Webcam Model V3 Validation Tool (Pipeline Fixed)")
    print("=" * 75)

    # 1. Load V3 Model
    v3_path = MODELS_DIR / "dynamic_bigru_v3.pt"
    map_path = MODELS_DIR / "dynamic_label_mapping_v3.json"

    if not v3_path.exists():
        print(f"Error: V3 Checkpoint missing: {v3_path}")
        return
    if not map_path.exists():
        print(f"Error: V3 Label mapping missing: {map_path}")
        return

    with open(map_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    chk = torch.load(v3_path, map_location="cpu")
    model = DynamicSignBiGRU(input_dim=168, hidden_dim=64, num_layers=2, num_classes=len(label_map), dropout=0.0)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    print(f"V3 Model Loaded ({len(label_map)} classes, 168 input dim).")

    # 2. Check Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n" + "!" * 75)
        print("CAMERA ACCESS FAILED / TERMINAL NOT AUTHORIZED")
        print("Please run this command directly in your native Terminal app:")
        print("    ./backend/venv/bin/python ml/scripts/validate_v3_live_webcam.py")
        print("!" * 75)
        return

    hand_detector, pose_detector = init_landmarkers()
    print("MediaPipe detectors ready. Opening camera window...")

    # Rolling 2.6s temporal buffer (~78 frames at 30 FPS)
    raw_temporal_buffer = deque(maxlen=90)
    results_log = []

    target_idx = 0
    attempt_idx = 0
    total_targets = len(TARGET_SIGNS)

    while target_idx < total_targets:
        cur_sign = TARGET_SIGNS[target_idx]
        max_attempts = cur_sign.get("attempts", 3)
        cur_cond = ATTEMPT_CONDITIONS[attempt_idx % len(ATTEMPT_CONDITIONS)]

        ret, frame_raw = cap.read()
        if not ret:
            break

        now = time.time()
        # CRITICAL FIX 1: Extract landmarks from RAW UNFLIPPED frame
        vec, has_act, l_hand, r_hand, pose_lms = extract_168_vector(frame_raw, hand_detector, pose_detector)
        raw_temporal_buffer.append((now, vec))

        # Evict frames older than 2.6 seconds
        cutoff = now - CAPTURE_WINDOW_SEC
        while len(raw_temporal_buffer) > 0 and raw_temporal_buffer[0][0] < cutoff:
            raw_temporal_buffer.popleft()

        # CRITICAL FIX 2: Mirror frame ONLY for user GUI display
        preview_frame = cv2.flip(frame_raw, 1)
        h, w, _ = preview_frame.shape

        # Uniform 30-frame temporal downsampling across the rolling 2.6s buffer
        raw_count = len(raw_temporal_buffer)
        pred_label = "Buffering gesture window..."
        confidence = 0.0
        accepted = False
        sampled_seq = None
        current_latency_ms = 0.0

        if raw_count >= 30 and has_act:
            # Sample exactly 30 frames uniformly across full captured buffer
            indices = np.linspace(0, raw_count - 1, 30, dtype=int)
            sampled_seq = np.array([raw_temporal_buffer[idx][1] for idx in indices], dtype=np.float32)

            seq_t = torch.tensor(sampled_seq).unsqueeze(0)
            mask_t = torch.ones(1, 30, dtype=torch.float32)

            t_start = time.perf_counter()
            with torch.inference_mode():
                logits = model(seq_t, mask_t)
                probs = torch.softmax(logits, dim=-1).squeeze(0)
                pred_idx = torch.argmax(probs).item()
                confidence = float(probs[pred_idx].item())
                pred_label = label_map[pred_idx]
                accepted = bool(confidence >= CONFIDENCE_THRESHOLD)
            current_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Draw UI overlay on preview frame
        cv2.rectangle(preview_frame, (0, 0), (w, 140), (20, 20, 20), -1)
        cv2.putText(preview_frame, "SIGNBRIDGE AI — V3 FOCUSED VALIDATION (Pipeline Fixed)",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
        cv2.putText(preview_frame, f"Target [{target_idx+1}/{total_targets}]: {cur_sign['name']} | Attempt: {attempt_idx+1}/{max_attempts}",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(preview_frame, f"Guidance: {cur_sign['gesture']} ({cur_cond})",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)

        status_color = (0, 255, 0) if accepted else (0, 165, 255)
        status_txt = f"Live Pred: {pred_label.upper()} ({confidence*100:.1f}%) [{'ACCEPTED' if accepted else 'UNCERTAIN'}] [{current_latency_ms:.1f}ms]"
        cv2.putText(preview_frame, status_txt, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

        # Buffer progress indicator
        buf_pct = min(100, int((raw_count / 75.0) * 100))
        cv2.putText(preview_frame, f"Window: {buf_pct}% ({raw_count} frames)", (w - 220, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.putText(preview_frame, "[SPACE] Record Attempt | [R] Retry | [S] Skip | [Q] Quit",
                    (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow("SignBridge AI — V3 Live Camera Validation", preview_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if sampled_seq is not None:
                is_correct = bool(accepted and pred_label.lower() == cur_sign["expected"].lower())
                record = {
                    "attempt_id": len(results_log) + 1,
                    "target_sign": cur_sign["name"],
                    "expected_label": cur_sign["expected"],
                    "condition": cur_cond,
                    "attempt_number": attempt_idx + 1,
                    "max_attempts": max_attempts,
                    "predicted_sign": pred_label,
                    "confidence": round(confidence * 100, 2),
                    "accepted": accepted,
                    "is_correct": is_correct,
                    "latency_ms": round(current_latency_ms, 2)
                }
                results_log.append(record)
                status_str = "ACCEPTED & CORRECT" if is_correct else ("ACCEPTED (WRONG)" if accepted else "REJECTED (UNCERTAIN)")
                print(f"Recorded [{cur_sign['name']} - Attempt {attempt_idx+1}/{max_attempts}]: Pred={pred_label} ({confidence*100:.1f}%) Latency={current_latency_ms:.1f}ms [{status_str}]")

                attempt_idx += 1
                if attempt_idx >= max_attempts:
                    attempt_idx = 0
                    target_idx += 1
                raw_temporal_buffer.clear()
            else:
                print("Buffer not ready yet — perform the sign in front of the camera and press SPACE when motion is detected.")

        elif key == ord('r'):
            print("Retrying attempt...")
            raw_temporal_buffer.clear()

        elif key == ord('s'):
            print(f"Skipping {cur_sign['name']} attempt {attempt_idx+1}/{max_attempts}")
            attempt_idx += 1
            if attempt_idx >= max_attempts:
                attempt_idx = 0
                target_idx += 1
            raw_temporal_buffer.clear()

        elif key == ord('q'):
            print("Quit requested by user.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(results_log) > 0:
        save_reports(results_log)


def save_reports(records):
    total = len(records)
    correct = sum(1 for r in records if r["is_correct"])
    accepted = sum(1 for r in records if r["accepted"])
    rejected_low_conf = sum(1 for r in records if not r["accepted"])
    false_high_conf = sum(1 for r in records if r["accepted"] and not r["is_correct"])
    
    avg_conf = float(np.mean([r["confidence"] for r in records])) if total > 0 else 0.0
    latencies = [r["latency_ms"] for r in records]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0
    overall_acc = (correct / total * 100.0) if total > 0 else 0.0

    # Per-sign statistics
    per_sign_stats = {}
    unique_signs = []
    for r in records:
        if r["target_sign"] not in unique_signs:
            unique_signs.append(r["target_sign"])

    for s_name in unique_signs:
        s_recs = [r for r in records if r["target_sign"] == s_name]
        s_total = len(s_recs)
        s_correct = sum(1 for r in s_recs if r["is_correct"])
        s_accepted = sum(1 for r in s_recs if r["accepted"])
        s_false_high = sum(1 for r in s_recs if r["accepted"] and not r["is_correct"])
        s_rejected = sum(1 for r in s_recs if not r["accepted"])
        s_avg_conf = float(np.mean([r["confidence"] for r in s_recs])) if s_total > 0 else 0.0
        s_acc = (s_correct / s_total * 100.0) if s_total > 0 else 0.0
        
        per_sign_stats[s_name] = {
            "total_attempts": s_total,
            "correct": s_correct,
            "accuracy_pct": round(s_acc, 2),
            "average_confidence_pct": round(s_avg_conf, 2),
            "accepted_count": s_accepted,
            "false_high_confidence": s_false_high,
            "rejected_low_confidence": s_rejected
        }

    report_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "dynamic_bigru_v3.pt",
        "input_dimension": 168,
        "classes_count": 22,
        "total_attempts": total,
        "correct_count": correct,
        "overall_accuracy_pct": round(overall_acc, 2),
        "average_confidence_pct": round(avg_conf, 2),
        "accepted_count": accepted,
        "false_high_confidence_count": false_high_conf,
        "rejected_low_confidence_count": rejected_low_conf,
        "latency_stats": {
            "mean_ms": round(mean_latency, 2),
            "p95_ms": round(p95_latency, 2)
        },
        "per_sign_statistics": per_sign_stats,
        "attempts": records
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Saved: {REPORT_JSON_PATH}")

    md = f"""# SignBridge AI — V3 Focused Live Webcam Validation Report

**Model Tested:** Candidate Model V3 (`dynamic_bigru_v3.pt`)  
**Input Dimension:** 30 frames × 168 features  
**Dynamic Classes:** 22 classes  
**Evaluation Mode:** Focused Live Webcam Continuous Recognition Session  
**Date:** {report_dict['timestamp']}  

---

## 1. Summary Metrics
- **Total Controlled Attempts:** {total}
- **Correct Recognitions (Accepted & Matching):** {correct}
- **Overall Empirical Accuracy:** **{overall_acc:.2f}%**
- **Average Prediction Confidence:** {avg_conf:.2f}%
- **Accepted Predictions (>= 70%):** {accepted}
- **False High-Confidence Predictions (>= 70% & Wrong):** {false_high_conf}
- **Rejected Low-Confidence Predictions (< 70%):** {rejected_low_conf}
- **Inference Latency:** Mean: **{mean_latency:.2f} ms** | P95: **{p95_latency:.2f} ms**

---

## 2. Per-Sign Accuracy Breakdown
| Sign | Target Attempts | Correct | Accuracy | Avg Confidence | False High-Conf | Rejected Low-Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s_name, stats in per_sign_stats.items():
        md += f"| **{s_name}** | {stats['total_attempts']} | {stats['correct']} | **{stats['accuracy_pct']:.1f}%** | {stats['average_confidence_pct']:.1f}% | {stats['false_high_confidence']} | {stats['rejected_low_confidence']} |\n"

    md += f"""
---

## 3. Per-Attempt Detailed Log
| # | Expected Sign | Attempt | Condition | Predicted | Confidence | Latency | Accepted | Result |
| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :---: |
"""
    for r in records:
        if r["is_correct"]:
            res = "CORRECT"
        elif r["accepted"]:
            res = "FALSE HIGH-CONF"
        else:
            res = "REJECTED (UNCERTAIN)"
        md += f"| {r['attempt_id']} | {r['target_sign']} | {r['attempt_number']}/{r['max_attempts']} | {r['condition']} | {r['predicted_sign']} | {r['confidence']}% | {r['latency_ms']} ms | {'Yes' if r['accepted'] else 'No'} | {res} |\n"

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Saved: {REPORT_MD_PATH}")


if __name__ == "__main__":
    main()
