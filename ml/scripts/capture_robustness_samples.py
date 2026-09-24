"""
SignBridge AI - Real Camera Robustness Sample Capture Tool
Allows recording live test samples from webcam across PS-09 environmental conditions:
1. Normal Indoor Lighting (normal/)
2. Low / Dim Indoor Lighting (low_light/)
3. Bright / Backlit Indoor Lighting (bright_light/)
4. Cluttered / Different Background (different_background/)
5. Varied Hand Position / Distance (different_position/)

Saves raw MP4 clips under: ml/datasets/robustness_raw/<condition>/
Does NOT modify the original dataset or retrain existing models.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
import cv2

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = BASE_DIR / "ml" / "config"
ROBUSTNESS_DIR = BASE_DIR / "ml" / "datasets" / "robustness_raw"

VALID_CONDITIONS = [
    "normal",
    "low_light",
    "bright_light",
    "different_background",
    "different_position"
]

def load_vocabulary():
    vocab_path = CONFIG_DIR / "vocabulary.json"
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return [c["label"] for c in data.get("classes", [])]

def parse_args():
    vocab = load_vocabulary()
    parser = argparse.ArgumentParser(description="SignBridge AI Real-Camera Robustness Capture Tool")
    parser.add_argument("--class-name", type=str, choices=vocab, default=vocab[0],
                        help=f"Target sign to record (choices: {', '.join(vocab)})")
    parser.add_argument("--condition", type=str, choices=VALID_CONDITIONS, default="normal",
                        help=f"Environmental condition to record (choices: {', '.join(VALID_CONDITIONS)})")
    parser.add_argument("--duration", type=float, default=2.5,
                        help="Duration of the recorded sign in seconds (default: 2.5s)")
    parser.add_argument("--fps", type=int, default=25,
                        help="Video recording frame rate (default: 25 FPS)")
    parser.add_argument("--camera-idx", type=int, default=0,
                        help="OpenCV webcam index (default: 0)")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Record immediately without waiting for keypress")
    return parser.parse_args()

def capture():
    args = parse_args()
    condition_dir = ROBUSTNESS_DIR / args.condition
    condition_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SignBridge AI — Real Camera Robustness Capture Utility")
    print("=" * 65)
    print(f"Target Sign:     '{args.class_name}'")
    print(f"Condition:       '{args.condition}'")
    print(f"Duration:        {args.duration}s @ {args.fps} FPS")
    print(f"Output Folder:   {condition_dir}")
    print("=" * 65)

    cap = cv2.VideoCapture(args.camera_idx)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {args.camera_idx}.")
        print("Note: In headless or remote server environments, camera access requires a local GUI session.")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_filename = f"{args.class_name}_{args.condition}_{timestamp_str}.mp4"
    out_filepath = condition_dir / out_filename

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(str(out_filepath), fourcc, args.fps, (width, height))

    print("\nLive camera ready. Press SPACE in preview window to start 3-second countdown (or Q to quit).")

    # Countdown loop
    countdown_secs = 3
    t_start_countdown = None
    recording = False
    t_start_recording = None
    frames_recorded = 0
    total_frames_target = int(args.duration * args.fps)

    if args.non_interactive:
        t_start_countdown = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        now = time.time()
        display_frame = frame.copy()

        # UI Overlay
        cv2.putText(display_frame, f"Sign: {args.class_name.upper()} | Condition: {args.condition}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if not recording and t_start_countdown is None:
            cv2.putText(display_frame, "Press SPACE to Begin Recording",
                        (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        elif not recording and t_start_countdown is not None:
            elapsed_cd = now - t_start_countdown
            remain = int(countdown_secs - elapsed_cd) + 1
            if remain > 0:
                cv2.putText(display_frame, f"Get Ready... {remain}",
                            (width // 2 - 120, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
            else:
                recording = True
                t_start_recording = time.time()
        elif recording:
            elapsed_rec = now - t_start_recording
            out_writer.write(frame)
            frames_recorded += 1

            # Red recording indicator
            cv2.circle(display_frame, (35, 75), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, f"RECORDING ({frames_recorded}/{total_frames_target})",
                        (60, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            if frames_recorded >= total_frames_target or elapsed_rec >= args.duration:
                print(f"\nRecording complete! Captured {frames_recorded} frames.")
                break

        try:
            cv2.imshow("SignBridge AI — Robustness Capture", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                t_start_countdown = time.time()
            elif key == ord('q'):
                print("Capture aborted by user.")
                break
        except Exception:
            # Headless environment fallback
            if recording and frames_recorded >= total_frames_target:
                break

    cap.release()
    out_writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    if frames_recorded > 0:
        manifest_path = ROBUSTNESS_DIR / "manifest.json"
        manifest_data = []
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
            except Exception:
                manifest_data = []

        manifest_data.append({
            "filename": out_filename,
            "filepath": str(out_filepath),
            "label": args.class_name,
            "condition": args.condition,
            "frames": frames_recorded,
            "fps": args.fps,
            "duration_sec": args.duration,
            "timestamp": timestamp_str
        })

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        print(f"Saved clip: {out_filepath}")
        print(f"Updated catalog: {manifest_path}")
        return True
    return False

if __name__ == "__main__":
    capture()
