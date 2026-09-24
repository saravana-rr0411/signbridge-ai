"""
SignBridge AI - Isolated Live Webcam Validation for Focused 6-Sign V3 Model
Model Checkpoint: ml/models/dynamic_bigru_v3_six_sign.pt
Label Mapping:    ml/models/dynamic_label_mapping_v3_six_sign.json
Input Dimension:  30 frames x 168 features
Vocabulary:       HELP, YES, NO, THANK_YOU, PLEASE, HELLO (Strictly 6 Signs)

Pipeline Features:
  1. Raw unmirrored frame passed to MediaPipe (preserves training geometry).
  2. Physical Handedness Verification using Pose Wrists (15=Left, 16=Right).
  3. Strictly 168 features per frame:
     - 0..63: Physical Left Hand
     - 64..127: Physical Right Hand
     - 128..149: Upper Body Pose
     - 150..167: Body-Relative Spatial Coordinates (wrists to mid-shoulder, nose, chest-center)
  4. Rolling temporal buffer uniformly sampled to strictly 30 frames.
  5. Direct PyTorch CPU inference (completely isolated from FastAPI production).
  6. 70% confidence threshold gating.
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
from sklearn.metrics import confusion_matrix

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"
REPORT_MD_PATH = BASE_DIR / "ml" / "LIVE_V3_SIX_SIGN_VALIDATION_REPORT.md"
REPORT_JSON_PATH = EVAL_DIR / "live_v3_six_sign_validation_report.json"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.70
CAPTURE_WINDOW_SEC = 2.6

# Strictly the 6 focused target signs (5 attempts minimum each = 30 total)
TARGET_SIGNS = [
    {
        "name": "HELP",
        "expected": "help",
        "gesture": "Closed fist (thumbs up) resting on flat open palm, moving upward together",
        "attempts": 5
    },
    {
        "name": "YES",
        "expected": "yes",
        "gesture": "S-fist nodding up and down at wrist",
        "attempts": 5
    },
    {
        "name": "NO",
        "expected": "no",
        "gesture": "Index and middle fingers snapping down onto thumb",
        "attempts": 5
    },
    {
        "name": "THANK_YOU",
        "expected": "thank_you",
        "gesture": "Flat hand with fingers touching chin/lips, then moving forward/downward toward person",
        "attempts": 5
    },
    {
        "name": "PLEASE",
        "expected": "please",
        "gesture": "Flat open palm placed on center chest, moving in gentle clockwise circles",
        "attempts": 5
    },
    {
        "name": "HELLO",
        "expected": "hello",
        "gesture": "Open flat hand near temple/forehead, waving or moving outward in salute",
        "attempts": 5
    },
]

ATTEMPT_CONDITIONS = [
    "Normal distance & center framing",
    "Slightly left/right hand position",
    "Slightly different camera distance",
    "Slightly faster / slower gesture tempo",
    "Alternative hand posture / tilt"
]

class DynamicSignBiGRUV3(nn.Module):
    def __init__(self, input_dim=168, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
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
    l_shoulder = raw_points[1]  # pt 11
    r_shoulder = raw_points[2]  # pt 12
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
            hw = lms[0]
            assigned_side = None

            # Geometric Physical Verification using Pose Wrists (15=Left, 16=Right)
            if has_pose_wrists:
                pw_l = pose_lms[15]  # Physical Left wrist
                pw_r = pose_lms[16]  # Physical Right wrist
                d_l = np.hypot(hw.x - pw_l.x, hw.y - pw_l.y)
                d_r = np.hypot(hw.x - pw_r.x, hw.y - pw_r.y)
                if abs(d_l - d_r) > 0.05:
                    assigned_side = "Left" if d_l < d_r else "Right"

            if assigned_side is None:
                # Direct unmirrored MediaPipe label
                label = hand_res.handedness[idx][0].category_name
                assigned_side = label

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
    print("SignBridge AI — Isolated Live Webcam Validation (Focused 6-Sign V3)")
    print("=" * 75)

    # 1. Load V3 Checkpoint and Mapping
    v3_path = MODELS_DIR / "dynamic_bigru_v3_six_sign.pt"
    map_path = MODELS_DIR / "dynamic_label_mapping_v3_six_sign.json"

    if not v3_path.exists():
        print(f"Error: Model checkpoint missing at {v3_path}")
        return
    if not map_path.exists():
        print(f"Error: Label mapping missing at {map_path}")
        return

    with open(map_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
        label_map = {int(k): v for k, v in raw_map.items()}

    print(f"Loaded 6-Sign Mapping: {label_map}")
    num_classes = len(label_map)

    chk = torch.load(v3_path, map_location="cpu", weights_only=False)
    model = DynamicSignBiGRUV3(input_dim=168, hidden_dim=64, num_layers=2, num_classes=num_classes, dropout=0.0)
    model.load_state_dict(chk["model_state_dict"])
    model.eval()

    print(f"Model V3 loaded successfully ({num_classes} classes, input 30x168).")

    # 2. Camera Access
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n" + "!" * 75)
        print("CAMERA ACCESS FAILED / PERMISSION REQUIRED")
        print("Please run this command directly in your macOS Terminal:")
        print("    ./backend/venv/bin/python ml/scripts/validate_v3_six_sign_live_webcam.py")
        print("!" * 75)
        return

    hand_detector, pose_detector = init_landmarkers()
    print("MediaPipe detectors ready. Opening camera window...")

    raw_temporal_buffer = deque(maxlen=90)
    results_log = []

    target_idx = 0
    attempt_idx = 0
    total_targets = len(TARGET_SIGNS)

    fps_tracker = deque(maxlen=30)
    last_time = time.time()

    while target_idx < total_targets:
        cur_sign = TARGET_SIGNS[target_idx]
        max_attempts = cur_sign.get("attempts", 5)
        cur_cond = ATTEMPT_CONDITIONS[attempt_idx % len(ATTEMPT_CONDITIONS)]

        ret, frame_raw = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_tracker.append(1.0 / dt)
        cur_fps = float(np.mean(fps_tracker)) if len(fps_tracker) > 0 else 30.0

        # Feature pipeline on unmirrored frame
        vec, has_act, l_hand, r_hand, pose_lms = extract_168_vector(frame_raw, hand_detector, pose_detector)
        raw_temporal_buffer.append((now, vec))

        # Evict frames older than 2.6s
        cutoff = now - CAPTURE_WINDOW_SEC
        while len(raw_temporal_buffer) > 0 and raw_temporal_buffer[0][0] < cutoff:
            raw_temporal_buffer.popleft()

        # Mirror frame ONLY for display preview
        preview_frame = cv2.flip(frame_raw, 1)
        h, w, _ = preview_frame.shape

        raw_count = len(raw_temporal_buffer)
        pred_label = "Buffering gesture window..."
        confidence = 0.0
        accepted = False
        sampled_seq = None
        current_latency_ms = 0.0

        if raw_count >= 30 and has_act:
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

        # UI Overlay
        cv2.rectangle(preview_frame, (0, 0), (w, 140), (15, 15, 20), -1)
        cv2.putText(preview_frame, f"SIGNBRIDGE AI — FOCUSED 6-SIGN V3 VALIDATION | {cur_fps:.1f} FPS",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2)
        cv2.putText(preview_frame, f"Target [{target_idx+1}/{total_targets}]: {cur_sign['name']} | Attempt: {attempt_idx+1}/{max_attempts}",
                    (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(preview_frame, f"Guidance: {cur_sign['gesture']}",
                    (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1)

        status_color = (0, 255, 0) if accepted else (0, 140, 255)
        status_tag = "ACCEPTED" if accepted else "UNCERTAIN (< 70%)"
        status_txt = f"Live: {pred_label.upper()} ({confidence*100:.1f}%) [{status_tag}] [{current_latency_ms:.1f}ms]"
        cv2.putText(preview_frame, status_txt, (15, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2)

        # Buffer progress
        buf_pct = min(100, int((raw_count / 75.0) * 100))
        cv2.putText(preview_frame, f"Window: {buf_pct}% ({raw_count} frames)", (w - 230, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # Control Bar
        cv2.rectangle(preview_frame, (0, h - 35), (w, h), (10, 10, 15), -1)
        cv2.putText(preview_frame, "[SPACE] Capture Prediction | [R] Retry | [S] Skip | [Q] Quit & Save",
                    (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("SignBridge AI — V3 Focused 6-Sign Validation", preview_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if sampled_seq is not None:
                # IMPORTANT RULE: Do NOT count low-confidence prediction as correct even if top-1 label matches
                is_match = bool(pred_label.lower() == cur_sign["expected"].lower())
                is_correct = bool(accepted and is_match)
                is_low_conf_match = bool(is_match and not accepted)
                is_false_high_conf = bool(accepted and not is_match)

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
                    "is_low_conf_match": is_low_conf_match,
                    "is_false_high_conf": is_false_high_conf,
                    "latency_ms": round(current_latency_ms, 2)
                }
                results_log.append(record)

                if is_correct:
                    status_str = "ACCEPTED & CORRECT"
                elif is_false_high_conf:
                    status_str = "FALSE HIGH-CONFIDENCE"
                elif is_low_conf_match:
                    status_str = f"REJECTED (< 70% conf: {confidence*100:.1f}%) BUT LABEL MATCHED"
                else:
                    status_str = f"REJECTED (< 70% conf: {confidence*100:.1f}%)"

                print(f"Recorded [{cur_sign['name']} - Attempt {attempt_idx+1}/{max_attempts}]: Pred={pred_label} ({confidence*100:.1f}%) Latency={current_latency_ms:.1f}ms [{status_str}]")

                attempt_idx += 1
                if attempt_idx >= max_attempts:
                    attempt_idx = 0
                    target_idx += 1
                raw_temporal_buffer.clear()
            else:
                print("Buffer not ready yet — perform the gesture in camera view and press SPACE when active.")

        elif key == ord('r'):
            print("Retrying current attempt buffer...")
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
        save_reports(results_log, label_map)
    else:
        print("No attempts were recorded.")

def save_reports(records, label_map):
    total = len(records)
    correct_accepted = sum(1 for r in records if r["is_correct"])
    accepted_total = sum(1 for r in records if r["accepted"])
    rejected_total = sum(1 for r in records if not r["accepted"])
    false_high_conf = sum(1 for r in records if r["is_false_high_conf"])
    low_conf_correct = sum(1 for r in records if r["is_low_conf_match"])

    overall_acc = (correct_accepted / total * 100.0) if total > 0 else 0.0
    accepted_acc = (correct_accepted / accepted_total * 100.0) if accepted_total > 0 else 0.0
    avg_conf = float(np.mean([r["confidence"] for r in records])) if total > 0 else 0.0

    latencies = [r["latency_ms"] for r in records if r["latency_ms"] > 0]
    mean_latency = float(np.mean(latencies)) if latencies else 0.0
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    # Ordered sign list
    ordered_signs = ["HELP", "YES", "NO", "THANK_YOU", "PLEASE", "HELLO"]
    labels_lower = [s.lower() for s in ordered_signs]

    # Per-sign breakdown
    per_sign_stats = {}
    for s_name in ordered_signs:
        s_recs = [r for r in records if r["target_sign"] == s_name]
        s_tot = len(s_recs)
        s_corr = sum(1 for r in s_recs if r["is_correct"])
        s_acc = sum(1 for r in s_recs if r["accepted"])
        s_f_high = sum(1 for r in s_recs if r["is_false_high_conf"])
        s_rej = sum(1 for r in s_recs if not r["accepted"])
        s_low_corr = sum(1 for r in s_recs if r["is_low_conf_match"])
        s_avg_conf = float(np.mean([r["confidence"] for r in s_recs])) if s_tot > 0 else 0.0
        s_emp_acc = (s_corr / s_tot * 100.0) if s_tot > 0 else 0.0
        s_acc_acc = (s_corr / s_acc * 100.0) if s_acc > 0 else 0.0

        per_sign_stats[s_name] = {
            "total_attempts": s_tot,
            "correct_accepted": s_corr,
            "accepted_count": s_acc,
            "rejected_count": s_rej,
            "false_high_confidence": s_f_high,
            "low_confidence_correct": s_low_corr,
            "average_confidence_pct": round(s_avg_conf, 2),
            "empirical_accuracy_pct": round(s_emp_acc, 2),
            "accepted_only_accuracy_pct": round(s_acc_acc, 2)
        }

    # Confusion matrix
    y_true = [r["expected_label"].lower() for r in records]
    y_pred = [r["predicted_sign"].lower() for r in records]
    cm = confusion_matrix(y_true, y_pred, labels=labels_lower).tolist()

    report_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": "ml/models/dynamic_bigru_v3_six_sign.pt",
        "vocabulary": ordered_signs,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "input_shape": "30 x 168",
        "summary": {
            "total_attempts": total,
            "correct_accepted_predictions": correct_accepted,
            "incorrect_accepted_predictions": false_high_conf,
            "rejected_predictions": rejected_total,
            "low_confidence_correct_predictions": low_conf_correct,
            "overall_empirical_accuracy_pct": round(overall_acc, 2),
            "accepted_only_accuracy_pct": round(accepted_acc, 2),
            "average_confidence_pct": round(avg_conf, 2),
            "latency": {
                "mean_ms": round(mean_latency, 2),
                "p95_ms": round(p95_latency, 2)
            }
        },
        "per_sign_results": per_sign_stats,
        "confusion_matrix": {
            "labels": ordered_signs,
            "matrix": cm
        },
        "attempts": records
    }

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[OK] Validation JSON saved to: {REPORT_JSON_PATH}")

    # Generate Markdown Report
    cm_header = " | ".join(ordered_signs)
    cm_sep = " | ".join(["---"] * (len(ordered_signs) + 1))
    cm_rows = ""
    for idx, true_s in enumerate(ordered_signs):
        row_vals = " | ".join(str(cm[idx][j]) for j in range(len(ordered_signs)))
        cm_rows += f"| **{true_s}** | {row_vals} |\n"

    md = f"""# SignBridge AI — Live Webcam Validation Report (Focused 6-Sign V3)

**Model:** `ml/models/dynamic_bigru_v3_six_sign.pt`  
**Input Dimension:** 30 frames × 168 features  
**Vocabulary (6 Signs):** HELP, YES, NO, THANK_YOU, PLEASE, HELLO  
**Confidence Threshold:** 70% (`0.70`)  
**Evaluation Mode:** Isolated Live Webcam Controlled Validation  
**Date:** {report_dict['timestamp']}  

---

## 1. Executive Summary
- **Total Controlled Attempts:** {total}
- **Correct Accepted Predictions (Match & >= 70%):** {correct_accepted}
- **Overall Empirical Accuracy:** **{overall_acc:.2f}%**
- **Accepted-Only Accuracy:** **{accepted_acc:.2f}%**
- **Average Prediction Confidence:** **{avg_conf:.2f}%**
- **Accepted Predictions (>= 70%):** {accepted_total} ({accepted_total/total*100:.1f}%)
- **Rejected Predictions (< 70%):** {rejected_total}
- **False High-Confidence Predictions (>= 70% & Wrong):** {false_high_conf}
- **Low-Confidence Correct Predictions (< 70% & Matched):** {low_conf_correct}
- **CPU Inference Latency:** Mean: **{mean_latency:.2f} ms** | P95: **{p95_latency:.2f} ms**

---

## 2. Per-Sign Empirical Performance

| Sign | Attempts | Correct (Accepted) | Empirical Acc | Accepted-Only Acc | Avg Confidence | False High-Conf | Low-Conf Correct | Rejected (< 70%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s_name in ordered_signs:
        st = per_sign_stats[s_name]
        md += f"| **{s_name}** | {st['total_attempts']} | {st['correct_accepted']} | **{st['empirical_accuracy_pct']:.1f}%** | {st['accepted_only_accuracy_pct']:.1f}% | {st['average_confidence_pct']:.1f}% | {st['false_high_confidence']} | {st['low_confidence_correct']} | {st['rejected_count']} |\n"

    md += f"""
---

## 3. Confusion Matrix (Live Real-Time Captures)

| True \\ Pred | {cm_header} |
| :--- | {cm_sep}
{cm_rows}

---

## 4. Deep-Dive Analysis on Critical Signs

### NO (Main Previous Live Weakness)
- **Status:** {'RESOLVED' if per_sign_stats['NO']['empirical_accuracy_pct'] >= 70 else 'INVESTIGATE'}
- **Empirical Accuracy:** {per_sign_stats['NO']['empirical_accuracy_pct']}%
- **Average Live Confidence:** {per_sign_stats['NO']['average_confidence_pct']}%
- **Observations:** In earlier 22-class testing, NO achieved top-1 label but was rejected due to 50–58% confidence. The focused 6-sign architecture has dedicated representation for rapid 2-finger snap motion.

### HELP
- **Empirical Accuracy:** {per_sign_stats['HELP']['empirical_accuracy_pct']}%
- **Average Live Confidence:** {per_sign_stats['HELP']['average_confidence_pct']}%
- **Observations:** Two-handed sign (closed fist resting on flat palm). Pose-wrist handedness disambiguation ensures base palm and active fist stay separated.

### PLEASE
- **Empirical Accuracy:** {per_sign_stats['PLEASE']['empirical_accuracy_pct']}%
- **Average Live Confidence:** {per_sign_stats['PLEASE']['average_confidence_pct']}%
- **Observations:** Circular motion across the chest. Body-relative features (indices 162..167) anchor wrist coordinates relative to chest center.

### THANK_YOU
- **Empirical Accuracy:** {per_sign_stats['THANK_YOU']['empirical_accuracy_pct']}%
- **Average Live Confidence:** {per_sign_stats['THANK_YOU']['average_confidence_pct']}%
- **Observations:** Hand moving from chin/mouth forward toward camera. Evaluated against potential confusion with chest gestures.

---

## 5. Per-Attempt Comprehensive Log

| # | Expected | Attempt | Condition | Predicted | Confidence | Latency | Accepted | Outcome |
| :-: | :--- | :---: | :--- | :--- | :---: | :---: | :---: | :--- |
"""
    for r in records:
        if r["is_correct"]:
            res = "**CORRECT (ACCEPTED)**"
        elif r["is_false_high_conf"]:
            res = "FALSE HIGH-CONF"
        elif r["is_low_conf_match"]:
            res = "REJECTED (LOW CONF MATCH)"
        else:
            res = "REJECTED (UNCERTAIN)"
        md += f"| {r['attempt_id']} | {r['target_sign']} | {r['attempt_number']}/{r['max_attempts']} | {r['condition']} | {r['predicted_sign']} | {r['confidence']}% | {r['latency_ms']} ms | {'Yes' if r['accepted'] else 'No'} | {res} |\n"

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[OK] Validation Markdown report saved to: {REPORT_MD_PATH}")

if __name__ == "__main__":
    main()
