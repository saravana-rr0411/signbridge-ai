#!/usr/bin/env python3
"""
SignBridge AI - Landmark Extraction & Dataset Assembly for V4 11-Sign Model
Combines:
  1. Audited V3 6-Sign dataset (220 samples): HELP, YES, NO, THANK_YOU, PLEASE, HELLO
  2. Audited Approved 5-Sign candidate dataset (181 samples): DOCTOR, PAIN, SICK, BATHROOM, WHERE
Total: 401 samples across 11 classes.

Pipeline:
  - Exact 30 frames x 168 features pipeline
  - Physical handedness disambiguation with Pose wrists
  - Strictly 0 cross-split signer overlap (Signer-Independent Split)
Outputs:
  - ml/datasets/processed/dynamic_landmarks_v4_11_sign.npz
  - ml/datasets/processed/metadata_v4_11_sign.csv
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

V3_NPZ_PATH = PROCESSED_DIR / "dynamic_landmarks_v3_six_sign.npz"
V3_CSV_PATH = PROCESSED_DIR / "metadata_v3_six_sign.csv"
V4_AUDIT_JSON_PATH = EVAL_DIR / "v4_candidate_dataset_audit.json"

OUT_NPZ_PATH = PROCESSED_DIR / "dynamic_landmarks_v4_11_sign.npz"
OUT_CSV_PATH = PROCESSED_DIR / "metadata_v4_11_sign.csv"

TARGET_FRAMES = 30
VOCABULARY_11 = [
    "hello",
    "help",
    "yes",
    "no",
    "please",
    "thank_you",
    "doctor",
    "pain",
    "sick",
    "bathroom",
    "where"
]
LABEL_TO_ID = {lbl: idx for idx, lbl in enumerate(VOCABULARY_11)}

APPROVED_NEW_SIGNS = {"DOCTOR", "PAIN", "SICK", "BATHROOM", "WHERE"}

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
        rel_feat[0:3] = (l_wrist - shoulder_center) / shoulder_width
        rel_feat[6:9] = (l_wrist - nose) / shoulder_width
        rel_feat[12:15] = (l_wrist - chest_center) / shoulder_width

    if has_right:
        r_wrist = np.array([r_hand_lms[0].x, r_hand_lms[0].y, r_hand_lms[0].z], dtype=np.float32)
        rel_feat[3:6] = (r_wrist - shoulder_center) / shoulder_width
        rel_feat[9:12] = (r_wrist - nose) / shoulder_width
        rel_feat[15:18] = (r_wrist - chest_center) / shoulder_width

    return rel_feat

def extract_frame_landmarks(frame_bgr, hand_detector, pose_detector):
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
                pw_l = pose_lms[15]
                pw_r = pose_lms[16]
                d_l = np.hypot(hw.x - pw_l.x, hw.y - pw_l.y)
                d_r = np.hypot(hw.x - pw_r.x, hw.y - pw_r.y)
                if abs(d_l - d_r) > 0.05:
                    assigned_side = "Left" if d_l < d_r else "Right"
            if assigned_side is None:
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
        feat_vec, has_hand = extract_frame_landmarks(frame, hand_detector, pose_detector)
        seq[i] = feat_vec
        if has_hand:
            hand_hits += 1

    return seq, valid_mask, total_frames, hand_hits

def extract_and_combine():
    print("=" * 70)
    print("EXTRACTING & COMBINING V4 11-SIGN DATASET")
    print("=" * 70)

    # 1. Load V3 Six-Sign Dataset
    print(f"\n1. Loading existing V3 Six-Sign dataset from {V3_NPZ_PATH.name}...")
    v3_data = np.load(V3_NPZ_PATH)
    v3_df = pd.read_csv(V3_CSV_PATH)
    print(f"   V3 samples loaded: {len(v3_df)} sequences of shape {v3_data['features'].shape[1:]}")

    # Re-map V3 class IDs to the 11-class vocabulary mapping
    v3_features = v3_data["features"]
    v3_masks = v3_data["masks"]
    v3_labels = [str(l).lower() for l in v3_df["sign_label"]]
    v3_class_ids = np.array([LABEL_TO_ID[l] for l in v3_labels], dtype=np.int64)
    v3_video_ids = list(v3_df["video_id"])
    v3_signer_ids = list(v3_df["signer_id"])
    v3_splits = list(v3_df["split"])
    v3_frame_counts = list(v3_data["frame_counts"])
    v3_source_datasets = list(v3_df.get("source_dataset", ["WLASL"] * len(v3_df)))

    # 2. Load V4 Candidate Audit
    print(f"\n2. Loading V4 candidate audit from {V4_AUDIT_JSON_PATH.name}...")
    with open(V4_AUDIT_JSON_PATH, "r", encoding="utf-8") as f:
        audit = json.load(f)

    all_samples = audit["samples"]
    approved_usable = [
        s for s in all_samples
        if s.get("usable") and s.get("sign_label") in APPROVED_NEW_SIGNS
    ]
    print(f"   Approved usable candidate samples to extract: {len(approved_usable)}")

    # Global Signer Partition:
    # 1. Any signer already present in V3 MUST retain their exact V3 split.
    # 2. Any new signer is partitioned deterministically via MD5 hash to guarantee 0 cross-split overlap globally.
    v3_signer_to_split = dict(zip(v3_df["signer_id"], v3_df["split"]))
    import hashlib

    # Check for landmark cache
    cache_seqs_path = PROCESSED_DIR / "candidate_landmarks_cache.npz"
    cached_tensors = {}
    if cache_seqs_path.exists():
        try:
            cached_data = np.load(cache_seqs_path)
            for k in cached_data.files:
                cached_tensors[k] = (cached_data[k], cached_data[k + "_mask"], int(cached_data[k + "_fc"]))
            print(f"   Loaded {len(cached_tensors)} cached candidate landmark sequences.")
        except Exception:
            cached_tensors = {}

    new_features = []
    new_masks = []
    new_labels = []
    new_class_ids = []
    new_video_ids = []
    new_signer_ids = []
    new_splits = []
    new_frame_counts = []
    new_source_datasets = []

    hand_detector, pose_detector = init_landmarkers()
    success_cnt = 0
    t0 = time.time()

    for idx, sample in enumerate(approved_usable):
        v_path = sample["video_path"]
        sign_label = sample["sign_label"].lower()
        vid_id = sample["video_id"]
        signer_id = sample["signer_id"]
        src = sample["source_dataset"]

        # Global signer partition
        if signer_id in v3_signer_to_split:
            split = v3_signer_to_split[signer_id]
        else:
            h = int(hashlib.md5(signer_id.encode()).hexdigest(), 16) % 100
            if h < 70:
                split = "train"
            elif h < 85:
                split = "val"
            else:
                split = "test"

        # Check cache
        if vid_id in cached_tensors:
            seq, mask, total_frames = cached_tensors[vid_id]
        else:
            seq, mask, total_frames, hand_hits = process_video(
                v_path, hand_detector, pose_detector, target_frames=TARGET_FRAMES
            )
            if seq is not None:
                cached_tensors[vid_id] = (seq, mask, total_frames)

        if seq is None:
            print(f"   Warning: failed to read {v_path}, skipping.")
            continue

        new_features.append(seq)
        new_masks.append(mask)
        new_labels.append(sign_label)
        new_class_ids.append(LABEL_TO_ID[sign_label])
        new_video_ids.append(vid_id)
        new_signer_ids.append(signer_id)
        new_splits.append(split)
        new_frame_counts.append(total_frames)
        new_source_datasets.append(src)

        success_cnt += 1
        if success_cnt % 25 == 0 or success_cnt == len(approved_usable):
            print(f"   Processed {success_cnt}/{len(approved_usable)} videos in {time.time()-t0:.1f}s...")

    print(f"   Extracted/loaded {len(new_features)} new candidate sequences.")

    # Save cache
    cache_dict = {}
    for vid, (seq, mask, fc) in cached_tensors.items():
        cache_dict[vid] = seq
        cache_dict[vid + "_mask"] = mask
        cache_dict[vid + "_fc"] = np.array(fc, dtype=np.int64)
    np.savez_compressed(cache_seqs_path, **cache_dict)

    # 4. Combine Datasets
    print("\n4. Combining original 6-sign data with new 5-sign data...")
    comb_features = np.concatenate([v3_features, np.array(new_features, dtype=np.float32)], axis=0)
    comb_masks = np.concatenate([v3_masks, np.array(new_masks, dtype=np.float32)], axis=0)
    comb_labels = np.array(v3_labels + new_labels, dtype=object)
    comb_class_ids = np.concatenate([v3_class_ids, np.array(new_class_ids, dtype=np.int64)], axis=0)
    comb_video_ids = np.array(v3_video_ids + new_video_ids, dtype=object)
    comb_signer_ids = np.array(v3_signer_ids + new_signer_ids, dtype=object)
    comb_splits = np.array(v3_splits + new_splits, dtype=object)
    comb_frame_counts = np.array(v3_frame_counts + new_frame_counts, dtype=np.int64)
    comb_sources = np.array(v3_source_datasets + new_source_datasets, dtype=object)

    print(f"   Combined total sequences: {len(comb_features)} of shape {comb_features.shape}")

    # 5. Signer Disjoint Validation
    print("\n5. Validating Signer Disjoint Splits (0 Overlap Check)...")
    train_mask = (comb_splits == "train")
    val_mask = (comb_splits == "val")
    test_mask = (comb_splits == "test")

    train_signers = set(comb_signer_ids[train_mask])
    val_signers = set(comb_signer_ids[val_mask])
    test_signers = set(comb_signer_ids[test_mask])

    tv_overlap = train_signers.intersection(val_signers)
    tt_overlap = train_signers.intersection(test_signers)
    vt_overlap = val_signers.intersection(test_signers)

    print(f"   Unique signers: Train={len(train_signers)}, Val={len(val_signers)}, Test={len(test_signers)}")
    print(f"   Train-Val signer overlap: {len(tv_overlap)}")
    print(f"   Train-Test signer overlap: {len(tt_overlap)}")
    print(f"   Val-Test signer overlap: {len(vt_overlap)}")

    assert len(tv_overlap) == 0, f"ERROR: Train-Val signer overlap detected: {tv_overlap}"
    assert len(tt_overlap) == 0, f"ERROR: Train-Test signer overlap detected: {tt_overlap}"
    assert len(vt_overlap) == 0, f"ERROR: Val-Test signer overlap detected: {vt_overlap}"
    print("   [VERIFIED] Strictly ZERO signer overlap across Train, Val, and Test!")

    # 6. Save Combined NPZ and Metadata CSV
    print(f"\n6. Saving artifacts to {OUT_NPZ_PATH.name} and {OUT_CSV_PATH.name}...")
    np.savez_compressed(
        OUT_NPZ_PATH,
        features=comb_features,
        masks=comb_masks,
        labels=comb_labels,
        class_ids=comb_class_ids,
        video_ids=comb_video_ids,
        signer_ids=comb_signer_ids,
        splits=comb_splits,
        frame_counts=comb_frame_counts
    )

    comb_df = pd.DataFrame({
        "video_id": comb_video_ids,
        "sign_label": comb_labels,
        "class_id": comb_class_ids,
        "signer_id": comb_signer_ids,
        "split": comb_splits,
        "frame_count": comb_frame_counts,
        "source_dataset": comb_sources
    })
    comb_df.to_csv(OUT_CSV_PATH, index=False)

    print(f"   Successfully saved {OUT_NPZ_PATH} ({OUT_NPZ_PATH.stat().st_size / 1e6:.2f} MB)")
    print(f"   Successfully saved {OUT_CSV_PATH}")

    # Summary table
    print("\n--- V4 11-Sign Dataset Split Distribution ---")
    ct = pd.crosstab(comb_df["sign_label"], comb_df["split"], margins=True)
    print(ct)

if __name__ == "__main__":
    extract_and_combine()
