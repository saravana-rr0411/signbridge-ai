"""
SignBridge AI - Step 1 & 2: Landmark Extraction & Temporal Pipeline for Focused V3 Six-Sign Dataset
Input: ml/evaluation/v3_six_sign_dataset_audit.json (220 usable samples across 6 signs)
Output:
  - ml/datasets/processed/dynamic_landmarks_v3_six_sign.npz
  - ml/datasets/processed/metadata_v3_six_sign.csv

Pipeline:
  - MediaPipe Hands (num_hands=2)
  - MediaPipe Pose (upper body anchors)
  - Physical handedness verification with pose wrists
  - Exactly 168 features per frame:
      [0..63]   : Left Hand (63 coords + presence)
      [64..127] : Right Hand (63 coords + presence)
      [128..149]: Upper Body Pose (21 coords + presence)
      [150..167]: Body-Relative Spatial Features (18 coords)
  - Exactly 30 frames per video sequence
  - Sequence shape: (30, 168)
  - Preserves strict signer-disjoint splits (train / val / test)
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "ml" / "models"
PROCESSED_DIR = BASE_DIR / "ml" / "datasets" / "processed"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"
AUDIT_JSON_PATH = EVAL_DIR / "v3_six_sign_dataset_audit.json"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 30
VOCABULARY = ["help", "yes", "no", "thank_you", "please", "hello"]
LABEL_TO_ID = {lbl: idx for idx, lbl in enumerate(VOCABULARY)}

def init_landmarkers():
    hand_model_path = str(MODELS_DIR / "hand_landmarker.task")
    pose_model_path = str(MODELS_DIR / "pose_landmarker_lite.task")

    hand_opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=hand_model_path),
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    pose_opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=pose_model_path),
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)
    pose_detector = vision.PoseLandmarker.create_from_options(pose_opts)
    return hand_detector, pose_detector

def normalize_hand_landmarks(landmarks):
    """
    Normalizes 21 3D hand landmarks:
    - Translation invariant: wrist (landmark 0) subtracted.
    - Scale invariant: divided by Euclidean distance from wrist (0) to middle finger MCP (9).
    Returns 63-dim flat array + 1.0 presence flag = 64 dimensions.
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(64, dtype=np.float32)

    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    wrist = coords[0].copy()
    coords_centered = coords - wrist

    scale = np.linalg.norm(coords[9] - coords[0])
    if scale < 1e-4:
        scale = 1.0

    coords_norm = coords_centered / scale
    features = np.zeros(64, dtype=np.float32)
    features[:63] = coords_norm.flatten()
    features[63] = 1.0
    return features

def normalize_pose_landmarks(landmarks):
    """
    Normalizes upper body anchor landmarks:
    Points: 0 (nose), 11 (L shoulder), 12 (R shoulder), 13 (L elbow), 14 (R elbow), 15 (L wrist), 16 (R wrist).
    - Centered relative to mid-shoulder: (p11 + p12) / 2
    - Scale normalized by shoulder distance ||p12 - p11||
    Returns 21-dim flat array + 1.0 presence flag = 22 dimensions.
    """
    if landmarks is None or len(landmarks) < 17:
        return np.zeros(22, dtype=np.float32)

    keypoints = [0, 11, 12, 13, 14, 15, 16]
    raw_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in keypoints], dtype=np.float32)

    l_shoulder = raw_points[1]  # pt 11
    r_shoulder = raw_points[2]  # pt 12
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0

    scale = np.linalg.norm(r_shoulder - l_shoulder)
    if scale < 1e-4:
        scale = 1.0

    points_norm = (raw_points - mid_shoulder) / scale
    features = np.zeros(22, dtype=np.float32)
    features[:21] = points_norm.flatten()
    features[21] = 1.0
    return features

def compute_body_relative_features(l_hand_lms, r_hand_lms, pose_lms):
    """
    Computes 18 body-relative spatial features normalized by shoulder width:
    A) Left wrist relative to shoulder-center: 3 values (indices 150..152)
    B) Right wrist relative to shoulder-center: 3 values (indices 153..155)
    C) Left wrist relative to nose: 3 values (indices 156..158)
    D) Right wrist relative to nose: 3 values (indices 159..161)
    E) Left wrist relative to chest-center: 3 values (indices 162..164)
    F) Right wrist relative to chest-center: 3 values (indices 165..167)
    Total: 18 features.
    """
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

def extract_frame_landmarks(frame_bgr, hand_detector, pose_detector):
    """
    Extracts strictly 168 features from unmirrored frame with physical handedness disambiguation.
    Layout:
      0..63   : Left Hand (64)
      64..127 : Right Hand (64)
      128..149: Pose (22)
      150..167: Body-Relative (18)
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
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
            if has_pose_wrists:
                pw_l = pose_lms[15]  # Physical Left wrist
                pw_r = pose_lms[16]  # Physical Right wrist
                d_l = np.hypot(hw.x - pw_l.x, hw.y - pw_l.y)
                d_r = np.hypot(hw.x - pw_r.x, hw.y - pw_r.y)
                if abs(d_l - d_r) > 0.05:
                    assigned_side = "Left" if d_l < d_r else "Right"
            if assigned_side is None:
                label = hand_res.handedness[idx][0].category_name
                assigned_side = label  # Unmirrored feed label

            if assigned_side == "Left" and l_hand is None:
                l_hand = lms
            elif assigned_side == "Right" and r_hand is None:
                r_hand = lms
            elif r_hand is None:
                r_hand = lms
            elif l_hand is None:
                l_hand = lms

    f_left = normalize_hand_landmarks(l_hand)
    f_right = normalize_hand_landmarks(r_hand)
    f_pose = normalize_pose_landmarks(pose_lms)
    f_rel = compute_body_relative_features(l_hand, r_hand, pose_lms)

    vec = np.concatenate([f_left, f_right, f_pose, f_rel], axis=0).astype(np.float32)
    has_activity = bool(f_left[63] > 0.5 or f_right[63] > 0.5)

    return vec, has_activity

def process_video(video_path, hand_detector, pose_detector, target_frames=30):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, None, 0, 0

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return None, None, 0, 0

    # Temporal sampling
    if total_frames >= target_frames:
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        valid_mask = np.ones(target_frames, dtype=np.float32)
    else:
        sampled_frames = frames
        valid_mask = np.zeros(target_frames, dtype=np.float32)
        valid_mask[:total_frames] = 1.0

    seq = np.zeros((target_frames, 168), dtype=np.float32)
    hand_hits = 0

    for i, frame in enumerate(sampled_frames):
        vec, active = extract_frame_landmarks(frame, hand_detector, pose_detector)
        seq[i] = vec
        if active:
            hand_hits += 1

    # Quality and integrity checks
    if np.isnan(seq).any() or np.isinf(seq).any():
        return None, None, total_frames, 0

    return seq, valid_mask, total_frames, hand_hits

def main():
    print("=" * 75)
    print("SignBridge AI — Step 1 & 2: Landmark Extraction (6-Sign Focused V3)")
    print("=" * 75)

    if not AUDIT_JSON_PATH.exists():
        print(f"Error: Audit manifest missing at {AUDIT_JSON_PATH}")
        sys.exit(1)

    with open(AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        audit_data = json.load(f)

    usable_samples = [s for s in audit_data.get("samples", []) if s.get("usable")]
    print(f"Loaded {len(usable_samples)} usable samples from audit manifest.")

    hand_detector, pose_detector = init_landmarkers()
    print("MediaPipe HandLandmarker and PoseLandmarker initialized.")

    all_features = []
    all_masks = []
    all_labels = []
    all_class_ids = []
    all_video_ids = []
    all_signer_ids = []
    all_splits = []
    all_frame_counts = []
    metadata_rows = []

    succeeded = 0
    failed = 0
    t0 = time.time()

    for idx, s in enumerate(usable_samples, 1):
        vid_path = s["video_path"]
        sign_label = s["sign_label"].lower()  # normalize to 'help', 'yes', etc.
        class_id = LABEL_TO_ID[sign_label]
        signer_id = s["signer_id"]
        split = s["final_split"]
        sample_id = s["sample_id"]
        src = s["source_dataset"]

        seq, mask, total_f, hand_hits = process_video(vid_path, hand_detector, pose_detector, TARGET_FRAMES)

        if seq is None or hand_hits == 0:
            print(f"[{idx:3d}/{len(usable_samples)}] FAILED: {sample_id} ({sign_label}) - extraction failed or 0 hands")
            failed += 1
            continue

        all_features.append(seq)
        all_masks.append(mask)
        all_labels.append(sign_label)
        all_class_ids.append(class_id)
        all_video_ids.append(sample_id)
        all_signer_ids.append(signer_id)
        all_splits.append(split)
        all_frame_counts.append(total_f)

        metadata_rows.append({
            "sample_id": sample_id,
            "sign_label": sign_label,
            "class_id": class_id,
            "source_dataset": src,
            "video_id": s["video_id"],
            "signer_id": signer_id,
            "split": split,
            "original_frames": total_f,
            "target_frames": TARGET_FRAMES,
            "hand_detections": hand_hits,
            "hand_detection_rate": round(hand_hits / TARGET_FRAMES, 3),
            "video_path": vid_path
        })

        succeeded += 1
        if idx % 20 == 0 or idx == len(usable_samples):
            elapsed = time.time() - t0
            fps_overall = (succeeded * TARGET_FRAMES) / elapsed
            print(f"[{idx:3d}/{len(usable_samples)}] Processed {succeeded} OK, {failed} Fail | {elapsed:.1f}s ({fps_overall:.1f} fps)")

    # Build numpy arrays
    features_arr = np.array(all_features, dtype=np.float32)
    masks_arr = np.array(all_masks, dtype=np.float32)
    labels_arr = np.array(all_labels)
    class_ids_arr = np.array(all_class_ids, dtype=np.int32)
    video_ids_arr = np.array(all_video_ids)
    signer_ids_arr = np.array(all_signer_ids)
    splits_arr = np.array(all_splits)
    fcounts_arr = np.array(all_frame_counts, dtype=np.int32)

    # Hard Assertions
    assert features_arr.ndim == 3, f"Shape must be (N, 30, 168), got {features_arr.shape}"
    assert features_arr.shape[1] == 30, f"Frames must be 30, got {features_arr.shape[1]}"
    assert features_arr.shape[2] == 168, f"Features must be 168, got {features_arr.shape[2]}"
    assert not np.isnan(features_arr).any(), "Found NaN in extracted features!"
    assert not np.isinf(features_arr).any(), "Found Inf in extracted features!"
    assert len(features_arr) == succeeded, "Count mismatch"

    out_npz = PROCESSED_DIR / "dynamic_landmarks_v3_six_sign.npz"
    np.savez_compressed(
        out_npz,
        features=features_arr,
        masks=masks_arr,
        labels=labels_arr,
        class_ids=class_ids_arr,
        video_ids=video_ids_arr,
        signer_ids=signer_ids_arr,
        splits=splits_arr,
        frame_counts=fcounts_arr
    )
    print(f"\nSaved dynamic landmarks to: {out_npz} (Shape: {features_arr.shape})")

    out_csv = PROCESSED_DIR / "metadata_v3_six_sign.csv"
    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(out_csv, index=False)
    print(f"Saved metadata CSV to: {out_csv} ({len(df_meta)} rows)")

    # Print summary of splits and signer overlap
    train_idx = (splits_arr == "train")
    val_idx = (splits_arr == "val")
    test_idx = (splits_arr == "test")

    print("\n--- Final Dataset Summary ---")
    print(f"Total samples: {len(features_arr)}")
    print(f"Train samples: {np.sum(train_idx)} | Signers: {len(set(signer_ids_arr[train_idx]))}")
    print(f"Val samples:   {np.sum(val_idx)} | Signers: {len(set(signer_ids_arr[val_idx]))}")
    print(f"Test samples:  {np.sum(test_idx)} | Signers: {len(set(signer_ids_arr[test_idx]))}")

    train_signers = set(signer_ids_arr[train_idx])
    val_signers = set(signer_ids_arr[val_idx])
    test_signers = set(signer_ids_arr[test_idx])

    ov_tv = len(train_signers & val_signers)
    ov_tt = len(train_signers & test_signers)
    ov_vt = len(val_signers & test_signers)

    print(f"Train-Val Signer Overlap: {ov_tv}")
    print(f"Train-Test Signer Overlap: {ov_tt}")
    print(f"Val-Test Signer Overlap: {ov_vt}")
    assert ov_tv == 0 and ov_tt == 0 and ov_vt == 0, "ERROR: Cross-split signer leakage detected!"

    print("\nPer-sign breakdown:")
    for lbl in VOCABULARY:
        c_tr = np.sum((labels_arr == lbl) & train_idx)
        c_val = np.sum((labels_arr == lbl) & val_idx)
        c_te = np.sum((labels_arr == lbl) & test_idx)
        print(f"  {lbl.upper():10s}: Train={c_tr:2d}, Val={c_val:2d}, Test={c_te:2d} | Total={c_tr+c_val+c_te:2d}")

if __name__ == "__main__":
    main()
