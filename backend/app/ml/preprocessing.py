"""
SignBridge AI - Landmark Preprocessing & Validation
Reuses the exact normalization mathematics from ml/scripts/extract_landmarks.py:
1. Hand Landmarks:
   - Translation centered to wrist root (landmark 0)
   - Scale normalized by distance from wrist (0) to middle finger base (9)
   - 63 coordinates + 1.0 presence flag = 64 features per hand
2. Pose Landmarks:
   - Centered to mid-shoulder: (L_shoulder + R_shoulder) / 2
   - Scale normalized by shoulder width: ||R_shoulder - L_shoulder||
   - 21 coordinates + 1.0 presence flag = 22 features
3. Frame Total: Left Hand (64) + Right Hand (64) + Pose (22) = 150 features.
"""

import numpy as np
from typing import List, Tuple, Union, Any

POSE_KEYPOINTS = [0, 11, 12, 13, 14, 15, 16]  # nose, L/R shoulder, L/R elbow, L/R wrist


def normalize_hand_landmarks_array(coords: np.ndarray) -> np.ndarray:
    """
    coords: (21, 3) numpy array of 3D hand coordinates.
    Returns: 64-element array (63 normalized coords + presence flag).
    """
    if coords is None or len(coords) < 21:
        return np.zeros(64, dtype=np.float32)

    coords = np.array(coords, dtype=np.float32)
    wrist = coords[0].copy()
    coords_centered = coords - wrist

    # Scale: Euclidean distance from wrist (0) to middle finger MCP (9)
    scale = float(np.linalg.norm(coords[9] - coords[0]))
    if scale < 1e-4:
        scale = 1.0

    coords_norm = coords_centered / scale
    features = np.zeros(64, dtype=np.float32)
    features[:63] = coords_norm.flatten()
    features[63] = 1.0
    return features


def normalize_pose_landmarks_array(coords_all: np.ndarray) -> np.ndarray:
    """
    coords_all: list or array of all 33 pose landmarks, or 7 POSE_KEYPOINTS.
    Returns: 22-element array (21 normalized coords + presence flag).
    """
    if coords_all is None:
        return np.zeros(22, dtype=np.float32)

    coords = np.array(coords_all, dtype=np.float32)
    if len(coords) == 33:
        raw_points = coords[POSE_KEYPOINTS]
    elif len(coords) == 7:
        raw_points = coords
    else:
        return np.zeros(22, dtype=np.float32)

    l_shoulder = raw_points[1]
    r_shoulder = raw_points[2]
    mid_shoulder = (l_shoulder + r_shoulder) / 2.0

    scale = float(np.linalg.norm(r_shoulder - l_shoulder))
    if scale < 1e-4:
        scale = 1.0

    points_norm = (raw_points - mid_shoulder) / scale
    features = np.zeros(22, dtype=np.float32)
    features[:21] = points_norm.flatten()
    features[21] = 1.0
    return features


def validate_and_format_sequence(frames: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates input frames and formats into:
    - features: (1, 30, 150) np.float32
    - mask: (1, 30) np.float32
    """
    if not isinstance(frames, list):
        raise ValueError(f"Input frames must be a list, got {type(frames)}")

    # Handle flat 4500 list
    if len(frames) == 4500 and not isinstance(frames[0], list):
        arr = np.array(frames, dtype=np.float32).reshape(1, 30, 150)
        mask = np.ones((1, 30), dtype=np.float32)
        return arr, mask

    # Handle 30 x 150 list
    if len(frames) != 30:
        raise ValueError(f"Sequence must contain exactly 30 frames, got {len(frames)}")

    formatted_frames = []
    for idx, f in enumerate(frames):
        if not isinstance(f, (list, np.ndarray)):
            raise ValueError(f"Frame {idx} must be a list of 150 features")
        if len(f) != 150:
            raise ValueError(f"Frame {idx} must have 150 features, got {len(f)}")
        formatted_frames.append(f)

    arr = np.array(formatted_frames, dtype=np.float32).reshape(1, 30, 150)
    # Check if any frames are dummy/all-zeros (can be masked if desired; default all active)
    mask = np.ones((1, 30), dtype=np.float32)
    for t in range(30):
        if np.all(arr[0, t] == 0.0):
            mask[0, t] = 0.0

    # Ensure at least 1 frame is valid to avoid div by zero in mean pooling
    if np.sum(mask) == 0.0:
        mask[0, :] = 1.0

    return arr, mask


def validate_and_format_sequence_v3(frames: List[Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates input frames for V3 six-sign model and formats into:
    - features: (1, 30, 168) np.float32
    - mask: (1, 30) np.float32
    """
    if not isinstance(frames, list):
        raise ValueError(f"Input frames must be a list, got {type(frames)}")

    # Handle flat 5040 list
    if len(frames) == 5040 and not isinstance(frames[0], list):
        arr = np.array(frames, dtype=np.float32).reshape(1, 30, 168)
        mask = np.ones((1, 30), dtype=np.float32)
        return arr, mask

    # Handle 30 x 168 list
    if len(frames) != 30:
        raise ValueError(f"Sequence must contain exactly 30 frames, got {len(frames)}")

    formatted_frames = []
    for idx, f in enumerate(frames):
        if not isinstance(f, (list, np.ndarray)):
            raise ValueError(f"Frame {idx} must be a list of 168 features")
        if len(f) != 168:
            raise ValueError(f"Frame {idx} must have 168 features, got {len(f)}")
        formatted_frames.append(f)

    arr = np.array(formatted_frames, dtype=np.float32).reshape(1, 30, 168)
    mask = np.ones((1, 30), dtype=np.float32)
    for t in range(30):
        if np.all(arr[0, t] == 0.0):
            mask[0, t] = 0.0

    if np.sum(mask) == 0.0:
        mask[0, :] = 1.0

    return arr, mask


def validate_and_format_static(features: List[float]) -> np.ndarray:
    """
    Validates single-frame static handshape features and formats into:
    - (1, 64) np.float32
    """
    if not isinstance(features, (list, np.ndarray)) or len(features) != 64:
        raise ValueError(f"Static features must have exactly 64 values, got {len(features) if isinstance(features, (list, np.ndarray)) else type(features)}")
    return np.array(features, dtype=np.float32).reshape(1, 64)
