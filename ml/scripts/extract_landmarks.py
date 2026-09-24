"""
SignBridge AI - Landmark Extraction Pipeline
Reproducible extraction and normalization of 3D MediaPipe hand and pose landmarks
from WLASL dynamic videos and ASL Alphabet static images.
"""

import os
import sys
import json
import ssl
import time
import urllib.request
import subprocess
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup SSL context for downloading
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Referer': 'https://www.google.com'
}

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "ml" / "config"
RAW_DIR = BASE_DIR / "ml" / "datasets" / "raw"
PROCESSED_DIR = BASE_DIR / "ml" / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
VIDEOS_DIR = RAW_DIR / "videos"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"
VIZ_DIR = EVAL_DIR / "landmark_visualizations"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FRAMES = 30
HAND_LANDMARKS_COUNT = 21
POSE_KEYPOINTS = [0, 11, 12, 13, 14, 15, 16] # nose, L/R shoulder, L/R elbow, L/R wrist

def load_vocabulary():
    vocab_path = CONFIG_DIR / "vocabulary.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_wlasl_metadata():
    wlasl_path = RAW_DIR / "WLASL_v0.3.json"
    with open(wlasl_path, "r", encoding="utf-8") as f:
        return json.load(f)

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

def download_video(inst):
    vid_id = inst["video_id"]
    target_path = VIDEOS_DIR / f"{vid_id}.mp4"
    if target_path.exists() and target_path.stat().st_size > 1000:
        return target_path
    return None

def normalize_hand_landmarks(landmarks):
    """
    Normalizes 21 3D hand landmarks:
    - Translation invariant: wrist (landmark 0) subtracted.
    - Scale invariant: divided by Euclidean distance from wrist to middle finger MCP (landmark 9).
    Returns 63-dim flat array + 1.0 presence flag = 64 dimensions.
    """
    if landmarks is None or len(landmarks) < 21:
        return np.zeros(64, dtype=np.float32)
        
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32) # (21, 3)
    wrist = coords[0].copy()
    
    # Translation invariance
    coords_centered = coords - wrist
    
    # Scale invariance: distance from wrist (0) to middle finger base (9)
    scale = np.linalg.norm(coords[9] - coords[0])
    if scale < 1e-4:
        scale = 1.0
        
    coords_norm = coords_centered / scale
    features = np.zeros(64, dtype=np.float32)
    features[:63] = coords_norm.flatten()
    features[63] = 1.0 # presence flag
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
        
    raw_points = np.array([[landmarks[idx].x, landmarks[idx].y, landmarks[idx].z] for idx in POSE_KEYPOINTS], dtype=np.float32)
    
    # Mid-shoulder anchor (landmarks[1] = L shoulder, landmarks[2] = R shoulder in raw_points)
    l_shoulder = raw_points[1]
    r_shoulder = raw_points[2]
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0
    
    scale = np.linalg.norm(r_shoulder - l_shoulder)
    if scale < 1e-4:
        scale = 1.0
        
    points_norm = (raw_points - mid_shoulder) / scale
    features = np.zeros(22, dtype=np.float32)
    features[:21] = points_norm.flatten()
    features[21] = 1.0 # presence flag
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

    If left hand or pose is missing, left hand relative features are 0.0.
    If right hand or pose is missing, right hand relative features are 0.0.
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
    if down_dist > 1e-4:
        unit_down = down_vec / down_dist
    else:
        unit_down = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    chest_center = shoulder_center + unit_down * (0.5 * shoulder_width)

    has_left = (l_hand_lms is not None and len(l_hand_lms) >= 21)
    has_right = (r_hand_lms is not None and len(r_hand_lms) >= 21)

    if has_left:
        l_wrist = np.array([l_hand_lms[0].x, l_hand_lms[0].y, l_hand_lms[0].z], dtype=np.float32)
        rel_feat[0:3] = (l_wrist - shoulder_center) / shoulder_width   # A: 150..152
        rel_feat[6:9] = (l_wrist - nose) / shoulder_width              # C: 156..158
        rel_feat[12:15] = (l_wrist - chest_center) / shoulder_width   # E: 162..164

    if has_right:
        r_wrist = np.array([r_hand_lms[0].x, r_hand_lms[0].y, r_hand_lms[0].z], dtype=np.float32)
        rel_feat[3:6] = (r_wrist - shoulder_center) / shoulder_width   # B: 153..155
        rel_feat[9:12] = (r_wrist - nose) / shoulder_width             # D: 159..161
        rel_feat[15:18] = (r_wrist - chest_center) / shoulder_width   # F: 165..167

    return rel_feat

def extract_frame_landmarks(frame_bgr, hand_detector, pose_detector):
    """
    Extracts features for a single BGR image.
    Feature layout:
    - [0:64]    Left Hand (63 coords + presence)
    - [64:128]  Right Hand (63 coords + presence)
    - [128:150] Upper Body Pose (21 coords + presence)
    - [150:168] Body-Relative Spatial Features (18 coords)
    Total: 168 features.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    hand_res = hand_detector.detect(mp_image)
    pose_res = pose_detector.detect(mp_image)
    
    l_hand_lms = None
    r_hand_lms = None
    
    if hand_res.hand_landmarks and hand_res.handedness:
        for lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
            label = handedness[0].category_name
            # In selfie/mirror mode or standard camera, store Left vs Right consistently
            if label == 'Left' and l_hand_lms is None:
                l_hand_lms = lms
            elif label == 'Right' and r_hand_lms is None:
                r_hand_lms = lms
            elif l_hand_lms is None:
                l_hand_lms = lms
            elif r_hand_lms is None:
                r_hand_lms = lms
                
    l_features = normalize_hand_landmarks(l_hand_lms)
    r_features = normalize_hand_landmarks(r_hand_lms)
    
    pose_lms = pose_res.pose_landmarks[0] if pose_res.pose_landmarks else None
    pose_features = normalize_pose_landmarks(pose_lms)
    rel_features = compute_body_relative_features(l_hand_lms, r_hand_lms, pose_lms)
    
    frame_feature = np.concatenate([l_features, r_features, pose_features, rel_features])
    return frame_feature, l_features[63] > 0.5, r_features[63] > 0.5

def process_video_to_sequence(video_path, frame_start, frame_end, hand_detector, pose_detector):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0
        
    frames = []
    f_idx = 1
    start = max(1, frame_start) if frame_start is not None and frame_start > 0 else 1
    end = frame_end if frame_end is not None and frame_end > 0 else int(1e9)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx >= start and f_idx <= end:
            frames.append(frame)
        if f_idx > end:
            break
        f_idx += 1
        
    cap.release()
    
    total_usable = len(frames)
    if total_usable == 0:
        return None, 0, 0, 0
        
    # Uniform downsample or direct selection to 30 frames
    if total_usable >= TARGET_FRAMES:
        indices = np.linspace(0, total_usable - 1, TARGET_FRAMES, dtype=int)
        sampled_frames = [frames[i] for i in indices]
        valid_mask = np.ones(TARGET_FRAMES, dtype=np.float32)
    else:
        sampled_frames = frames
        valid_mask = np.zeros(TARGET_FRAMES, dtype=np.float32)
        valid_mask[:total_usable] = 1.0
        
    sequence_features = np.zeros((TARGET_FRAMES, 168), dtype=np.float32)
    left_hand_hits = 0
    right_hand_hits = 0
    
    for i, frame in enumerate(sampled_frames):
        feat, has_l, has_r = extract_frame_landmarks(frame, hand_detector, pose_detector)
        sequence_features[i] = feat
        if has_l:
            left_hand_hits += 1
        if has_r:
            right_hand_hits += 1
            
    return sequence_features, valid_mask, total_usable, (left_hand_hits + right_hand_hits)

def run_extraction():
    print("=" * 60)
    print("SignBridge AI — Phase E: Landmark Extraction Pipeline")
    print("=" * 60)
    
    vocab = load_vocabulary()
    wlasl = load_wlasl_metadata()
    g_map = {e["gloss"].lower(): e for e in wlasl}
    
    hand_detector, pose_detector = init_landmarkers()
    print("MediaPipe HandLandmarker and PoseLandmarker loaded.")
    
    dynamic_classes = [c for c in vocab["classes"] if c["dataset"] == "WLASL"]
    print(f"Processing {len(dynamic_classes)} dynamic WLASL classes...")
    
    all_features = []
    all_masks = []
    all_labels = []
    all_class_ids = []
    all_video_ids = []
    all_signer_ids = []
    all_splits = []
    all_frame_counts = []
    metadata_rows = []
    
    total_videos_attempted = 0
    total_videos_succeeded = 0
    total_videos_failed = 0
    
    class_success_counts = Counter()
    split_counts = Counter()
    signer_counts = Counter()
    
    for c in dynamic_classes:
        sc = c["source_class"]
        label = c["label"]
        cid = c["id"]
        entry = g_map[sc]
        instances = entry["instances"]
        
        print(f"\n--- Class [{cid}] '{label}' (WLASL gloss: '{sc}') | Total instances: {len(instances)} ---")
        
        for inst in instances:
            total_videos_attempted += 1
            vid_id = inst["video_id"]
            signer_id = inst.get("signer_id", -1)
            split = inst.get("split", "train")
            f_start = inst.get("frame_start", 1)
            f_end = inst.get("frame_end", -1)
            
            # Assign test sample for food if needed
            if label == "food" and vid_id == "22746":
                split = "test"

            # Download or use cached video
            vid_path = download_video(inst)
            if vid_path is None:
                total_videos_failed += 1
                continue
                
            try:
                seq, mask, f_count, hand_hits = process_video_to_sequence(
                    vid_path, f_start, f_end, hand_detector, pose_detector
                )
                if seq is None or hand_hits == 0:
                    total_videos_failed += 1
                    continue
                    
                all_features.append(seq)
                all_masks.append(mask)
                all_labels.append(label)
                all_class_ids.append(cid)
                all_video_ids.append(vid_id)
                all_signer_ids.append(signer_id)
                all_splits.append(split)
                all_frame_counts.append(f_count)
                
                total_videos_succeeded += 1
                class_success_counts[label] += 1
                split_counts[split] += 1
                signer_counts[signer_id] += 1
                
                metadata_rows.append({
                    "video_id": vid_id,
                    "class_id": cid,
                    "label": label,
                    "modality": "dynamic",
                    "signer_id": signer_id,
                    "split": split,
                    "original_frames": f_count,
                    "target_frames": TARGET_FRAMES,
                    "hand_detections": hand_hits,
                    "source": inst.get("source", "wlasl")
                })
                
                print(f"  + Video {vid_id:6} | Signer: {signer_id:2} | Split: {split:5} | Frames: {f_count:3} -> {TARGET_FRAMES} | Hands detected: {hand_hits:2}")
            except Exception as e:
                total_videos_failed += 1
                print(f"  x Error processing video {vid_id}: {e}")
                
    # Save Dynamic Landmarks V3 (168-dim)
    dynamic_features_arr = np.array(all_features, dtype=np.float32)
    dynamic_masks_arr = np.array(all_masks, dtype=np.float32)
    dynamic_labels_arr = np.array(all_labels)
    dynamic_cids_arr = np.array(all_class_ids, dtype=np.int32)
    dynamic_vids_arr = np.array(all_video_ids)
    dynamic_signers_arr = np.array(all_signer_ids, dtype=np.int32)
    dynamic_splits_arr = np.array(all_splits)
    dynamic_fcounts_arr = np.array(all_frame_counts, dtype=np.int32)
    
    out_dynamic_v3 = PROCESSED_DIR / "dynamic_landmarks_v3.npz"
    np.savez_compressed(
        out_dynamic_v3,
        features=dynamic_features_arr,
        masks=dynamic_masks_arr,
        labels=dynamic_labels_arr,
        class_ids=dynamic_cids_arr,
        video_ids=dynamic_vids_arr,
        signer_ids=dynamic_signers_arr,
        splits=dynamic_splits_arr,
        frame_counts=dynamic_fcounts_arr
    )
    print(f"\nSaved dynamic landmarks V3: {out_dynamic_v3} (Shape: {dynamic_features_arr.shape})")
    
    df_meta_v3 = pd.DataFrame(metadata_rows)
    df_meta_v3.to_csv(PROCESSED_DIR / "metadata_v3.csv", index=False)
    
    # Process Static Sign: letter_a from ASL Alphabet
    print("\n" + "=" * 60)
    print("Processing Static Sign 'letter_a' from ASL Alphabet Parquet...")
    print("=" * 60)
    
    asl_parquet_path = RAW_DIR / "asl_alphabet_v03.parquet"
    static_features = []
    static_labels = []
    static_cids = []
    
    out_static = PROCESSED_DIR / "static_landmarks.npz"
    if out_static.exists():
        print(f"Static landmarks already exist at {out_static}. Skipping re-extraction.")
    elif asl_parquet_path.exists():
        table = pq.read_table(asl_parquet_path)
        labels = table["label"].to_pylist()
        images_data = table["image"].to_pylist()
        a_count = 0
        for i, (lbl, img_dict) in enumerate(zip(labels, images_data)):
            if lbl == 0: # class A
                img_bytes = img_dict["bytes"]
                img_np = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
                
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                    res = hand_detector.detect(mp_img)
                    if res.hand_landmarks:
                        lms = res.hand_landmarks[0]
                        feat = normalize_hand_landmarks(lms)
                        static_features.append(feat)
                        static_labels.append("letter_a")
                        static_cids.append(17)
                        a_count += 1
                        
                        metadata_rows.append({
                            "video_id": f"asl_a_{a_count}",
                            "class_id": 17,
                            "label": "letter_a",
                            "modality": "static",
                            "signer_id": 999,
                            "split": "train",
                            "original_frames": 1,
                            "target_frames": 1,
                            "hand_detections": 1,
                            "source": "asl_alphabet"
                        })
                        
        static_features_arr = np.array(static_features, dtype=np.float32)
        out_static = PROCESSED_DIR / "static_landmarks.npz"
        np.savez_compressed(
            out_static,
            features=static_features_arr,
            labels=np.array(static_labels),
            class_ids=np.array(static_cids, dtype=np.int32)
        )
        print(f"Saved static landmarks: {out_static} (Shape: {static_features_arr.shape})")
    
    # Save Metadata CSV
    df_meta = pd.DataFrame(metadata_rows)
    meta_csv_path = PROCESSED_DIR / "metadata.csv"
    df_meta.to_csv(meta_csv_path, index=False)
    print(f"Saved metadata catalog: {meta_csv_path} ({len(df_meta)} records)")
    
    # Generate Visual Quality Check
    generate_visual_samples(dynamic_features_arr, dynamic_labels_arr)
    
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY:")
    print(f"Total videos attempted: {total_videos_attempted}")
    print(f"Total videos successfully extracted: {total_videos_succeeded}")
    print(f"Total videos failed/unreachable: {total_videos_failed}")
    print(f"Split distribution: {dict(split_counts)}")
    print(f"Total signers represented: {len(signer_counts)}")
    print(f"Static 'letter_a' samples extracted: {len(static_features)}")
    print("=" * 60)

def generate_visual_samples(features, labels):
    import matplotlib.pyplot as plt
    
    # Find sample indices for 1-hand, 2-hand
    one_hand_idx = None
    two_hand_idx = None
    
    for i, (feat, lbl) in enumerate(zip(features, labels)):
        # feat is (30, 150)
        # L hand present is feat[:, 63], R hand present is feat[:, 127]
        l_pres = np.mean(feat[:, 63])
        r_pres = np.mean(feat[:, 127])
        
        if one_hand_idx is None and (l_pres > 0.5 and r_pres < 0.2):
            one_hand_idx = i
        if two_hand_idx is None and (l_pres > 0.5 and r_pres > 0.5):
            two_hand_idx = i
        if one_hand_idx is not None and two_hand_idx is not None:
            break
            
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. One hand sign trajectory
    if one_hand_idx is not None:
        ax = axes[0, 0]
        f = features[one_hand_idx]
        # Plot wrist, thumb tip (4), index tip (8) over 30 frames
        # index tip x is index 8*3=24, y is 25
        index_x = f[:, 24]
        index_y = f[:, 25]
        ax.plot(index_x, -index_y, marker='o', color='#3b82f6', label='Index Tip (L)')
        ax.set_title(f"One-Hand Sign Trajectory: '{labels[one_hand_idx]}'")
        ax.set_xlabel("Normalized X (Wrist Relative)")
        ax.set_ylabel("Normalized -Y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # 2. Two hand sign trajectory
    if two_hand_idx is not None:
        ax = axes[0, 1]
        f = features[two_hand_idx]
        l_index_x = f[:, 24]
        l_index_y = f[:, 25]
        r_index_x = f[:, 64 + 24]
        r_index_y = f[:, 64 + 25]
        ax.plot(l_index_x, -l_index_y, marker='o', color='#10b981', label='Left Hand Index')
        ax.plot(r_index_x, -r_index_y, marker='s', color='#f59e0b', label='Right Hand Index')
        ax.set_title(f"Two-Hand Sign Trajectory: '{labels[two_hand_idx]}'")
        ax.set_xlabel("Normalized X (Wrist Relative)")
        ax.set_ylabel("Normalized -Y")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    # 3. Hand presence over time
    ax = axes[1, 0]
    sample_i = two_hand_idx if two_hand_idx is not None else 0
    f = features[sample_i]
    ax.plot(f[:, 63], label='Left Hand Present', color='#10b981', lw=2)
    ax.plot(f[:, 127], label='Right Hand Present', color='#f59e0b', lw=2)
    ax.plot(f[:, 149], label='Pose Present', color='#6366f1', lw=2, linestyle='--')
    ax.set_title("Temporal Presence Flags (30 Frames)")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Presence Flag (0.0 / 1.0)")
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Normalized Landmark Coordinate Scatter (Frame 15)
    ax = axes[1, 1]
    mid_f = features[sample_i, 15]
    # 21 hand landmarks for left hand: mid_f[0:63]
    l_pts = mid_f[:63].reshape(21, 3)
    ax.scatter(l_pts[:, 0], -l_pts[:, 1], c='#10b981', s=60, label='Left Hand Joints')
    if mid_f[127] > 0.5:
        r_pts = mid_f[64:127].reshape(21, 3)
        ax.scatter(r_pts[:, 0], -r_pts[:, 1], c='#f59e0b', s=60, label='Right Hand Joints')
    ax.set_title("Scale-Normalized Hand Skeleton (Frame 15)")
    ax.set_xlabel("Wrist-Relative X")
    ax.set_ylabel("Wrist-Relative -Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    viz_path = VIZ_DIR / "landmark_features_sample.png"
    plt.savefig(viz_path, dpi=200)
    plt.close()
    print(f"Generated landmark quality visualization: {viz_path}")

if __name__ == "__main__":
    run_extraction()
