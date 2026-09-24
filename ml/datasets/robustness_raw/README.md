# SignBridge AI — Real-Camera Robustness Raw Dataset

This directory stores raw MP4 test recordings captured with `ml/scripts/capture_robustness_samples.py`.
Samples in this directory are reserved for empirical camera robustness stress testing and are **NEVER** automatically merged into the training set.

## Structure:
- `normal/`: Baseline indoor lighting (approx. 300–500 lux), neutral positioning.
- `low_light/`: Dim/low-light environments (< 100 lux).
- `bright_light/`: Harsh indoor illumination or direct backlighting (> 800 lux).
- `different_background/`: Cluttered domestic, outdoor, or complex moving backgrounds.
- `different_position/`: Non-central framing, lateral shifts, or varying camera distances.

## Catalog
The recording script automatically logs metadata to `ml/datasets/robustness_raw/manifest.json`.
