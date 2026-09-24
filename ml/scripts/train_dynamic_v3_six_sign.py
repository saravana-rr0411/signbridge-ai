"""
SignBridge AI - Step 4: Train Focused 6-Sign V3 Dynamic Recognition Model
Classes: HELP, YES, NO, THANK_YOU, PLEASE, HELLO (strictly 6 classes)
Dataset: ml/datasets/processed/dynamic_landmarks_v3_six_sign.npz

Architecture:
  - 2-layer Bidirectional GRU (hidden_size=64, input_dim=168)
  - Masked temporal mean + max pooling (dim 256)
  - FC64 -> ReLU -> Dropout(0.3) -> Linear(64, 6)
  - Class-weighted CrossEntropyLoss + minority oversampling
  - Spatial and temporal augmentations
Checkpoints:
  - Checkpoint: ml/models/dynamic_bigru_v3_six_sign.pt
  - Mapping: ml/models/dynamic_label_mapping_v3_six_sign.json
"""

import os
import sys
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "ml" / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

class DynamicLandmarkDataset(Dataset):
    def __init__(self, features, masks, class_ids, is_train=False, oversample=False, label_names=None):
        self.is_train = is_train
        self.label_names = label_names

        if is_train and oversample and label_names is not None:
            # Rebalance minority classes (thank_you: 18 -> 36, hello: 17 -> 34, please: 21 -> 31)
            rep_multipliers = {
                "thank_you": 2,
                "hello": 2,
                "please": 1,
                "help": 1,
                "yes": 1,
                "no": 1,
            }
            indices = []
            for idx in range(len(features)):
                lbl = label_names[idx]
                r = rep_multipliers.get(lbl, 1)
                for _ in range(r):
                    indices.append(idx)
            self.features = torch.tensor(features[indices], dtype=torch.float32)
            self.masks = torch.tensor(masks[indices], dtype=torch.float32)
            self.class_ids = torch.tensor(class_ids[indices], dtype=torch.long)
        else:
            self.features = torch.tensor(features, dtype=torch.float32)
            self.masks = torch.tensor(masks, dtype=torch.float32)
            self.class_ids = torch.tensor(class_ids, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def _apply_rotation(self, feat, max_deg=18.0):
        # Rotate Left hand (0..62) and Right hand (64..126) in-plane (x, y)
        deg_l = np.random.uniform(-max_deg, max_deg)
        deg_r = np.random.uniform(-max_deg, max_deg)

        rad_l = np.deg2rad(deg_l)
        cos_l, sin_l = np.cos(rad_l), np.sin(rad_l)
        rad_r = np.deg2rad(deg_r)
        cos_r, sin_r = np.cos(rad_r), np.sin(rad_r)

        # Left hand coords
        if feat[:, 63].sum() > 0:
            for k in range(21):
                x = feat[:, 3 * k].clone()
                y = feat[:, 3 * k + 1].clone()
                feat[:, 3 * k] = x * cos_l - y * sin_l
                feat[:, 3 * k + 1] = x * sin_l + y * cos_l

        # Right hand coords
        if feat[:, 127].sum() > 0:
            for k in range(21):
                x = feat[:, 64 + 3 * k].clone()
                y = feat[:, 64 + 3 * k + 1].clone()
                feat[:, 64 + 3 * k] = x * cos_r - y * sin_r
                feat[:, 64 + 3 * k + 1] = x * sin_r + y * cos_r

        return feat

    def _apply_temporal_warp(self, feat, mask, speed_range=(0.85, 1.15), shift_range=(-2.0, 2.0)):
        T = feat.shape[0]  # 30
        valid_len = int(mask.sum().item())
        if valid_len < 10:
            return feat, mask

        speed = np.random.uniform(*speed_range)
        shift = np.random.uniform(*shift_range)

        t_target = np.linspace(0, 1, T)
        t_src = np.clip(t_target * speed + (shift / T), 0.0, 1.0) * (T - 1)

        t_floor = np.floor(t_src).astype(int)
        t_ceil = np.clip(t_floor + 1, 0, T - 1)
        alpha = torch.tensor(t_src - t_floor, dtype=torch.float32).unsqueeze(-1)

        f_floor = feat[t_floor]
        f_ceil = feat[t_ceil]
        warped_feat = (1.0 - alpha) * f_floor + alpha * f_ceil

        # Preserve binary presence flags (63, 127, 149)
        warped_feat[:, 63] = (warped_feat[:, 63] > 0.5).float()
        warped_feat[:, 127] = (warped_feat[:, 127] > 0.5).float()
        warped_feat[:, 149] = (warped_feat[:, 149] > 0.5).float()

        m_floor = mask[t_floor]
        m_ceil = mask[t_ceil]
        warped_mask = (1.0 - alpha.squeeze(-1)) * m_floor + alpha.squeeze(-1) * m_ceil
        warped_mask = (warped_mask > 0.5).float()

        return warped_feat, warped_mask

    def __getitem__(self, idx):
        feat = self.features[idx].clone()
        mask = self.masks[idx].clone()
        target = self.class_ids[idx]

        if self.is_train:
            # 1. Temporal warping
            if np.random.rand() > 0.2:
                feat, mask = self._apply_temporal_warp(feat, mask, speed_range=(0.85, 1.15), shift_range=(-2.0, 2.0))

            # 2. In-plane rotation
            if np.random.rand() > 0.15:
                feat = self._apply_rotation(feat, max_deg=18.0)

            # 3. Scale jitter [0.94, 1.06] on hand coordinates
            scale = np.random.uniform(0.94, 1.06)

            # 4. Coordinate Gaussian noise
            noise_l = torch.randn_like(feat[:, 0:63]) * 0.010
            noise_r = torch.randn_like(feat[:, 64:127]) * 0.010
            noise_p = torch.randn_like(feat[:, 128:149]) * 0.010
            noise_rel = torch.randn_like(feat[:, 150:168]) * 0.005

            feat[:, 0:63] = feat[:, 0:63] * scale + noise_l
            feat[:, 64:127] = feat[:, 64:127] * scale + noise_r
            feat[:, 128:149] = feat[:, 128:149] * scale + noise_p
            feat[:, 150:168] = feat[:, 150:168] + noise_rel

            # Ensure presence flags remain exact binary
            feat[:, 63] = (feat[:, 63] > 0.5).float()
            feat[:, 127] = (feat[:, 127] > 0.5).float()
            feat[:, 149] = (feat[:, 149] > 0.5).float()

        return feat, mask, target

class DynamicSignBiGRU(nn.Module):
    def __init__(self, input_dim=168, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.pool_dim = hidden_dim * 4  # mean (2*H) + max (2*H) = 256
        self.fc1 = nn.Linear(self.pool_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, mask):
        gru_out, _ = self.gru(x)  # (B, T, 2*H)

        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        masked_out = gru_out * mask_expanded

        # Mean pooling over valid frames
        sum_out = torch.sum(masked_out, dim=1)  # (B, 2*H)
        lengths = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        mean_pool = sum_out / lengths

        # Max pooling over valid frames
        masked_for_max = masked_out + (1.0 - mask_expanded) * -1e9
        max_pool, _ = torch.max(masked_for_max, dim=1)  # (B, 2*H)

        combined = torch.cat([mean_pool, max_pool], dim=-1)  # (B, 4*H)
        h = self.dropout(self.relu(self.fc1(combined)))
        logits = self.fc2(h)
        return logits

def train():
    print("=" * 75)
    print("SignBridge AI — Step 4: Focused 6-Sign V3 Model Training")
    print("=" * 75)

    dataset_path = PROCESSED_DIR / "dynamic_landmarks_v3_six_sign.npz"
    if not dataset_path.exists():
        print(f"Error: Dataset file missing: {dataset_path}")
        sys.exit(1)

    data = np.load(dataset_path)
    features = data["features"]      # (N, 30, 168)
    masks = data["masks"]            # (N, 30)
    labels = data["labels"]          # (N,)
    class_ids = data["class_ids"]    # (N,)
    splits = data["splits"]          # (N,)
    signers = data["signer_ids"]      # (N,)

    # Vocabulary mapping
    vocab = ["help", "yes", "no", "thank_you", "please", "hello"]
    label_mapping = {i: vocab[i] for i in range(len(vocab))}
    num_classes = len(vocab)
    print(f"Target Vocabulary ({num_classes} classes): {label_mapping}")

    # Save Label Mapping JSON
    map_path = MODELS_DIR / "dynamic_label_mapping_v3_six_sign.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved label mapping to: {map_path}")

    # Splits
    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    x_train, m_train, y_train, l_train = features[train_mask], masks[train_mask], class_ids[train_mask], labels[train_mask]
    x_val, m_val, y_val, l_val = features[val_mask], masks[val_mask], class_ids[val_mask], labels[val_mask]
    x_test, m_test, y_test, l_test = features[test_mask], masks[test_mask], class_ids[test_mask], labels[test_mask]

    # Verify Signer Disjoint Splits Programmatically
    train_signers = set(signers[train_mask])
    val_signers = set(signers[val_mask])
    test_signers = set(signers[test_mask])

    overlap_tv = len(train_signers & val_signers)
    overlap_tt = len(train_signers & test_signers)
    overlap_vt = len(val_signers & test_signers)

    print("\n--- Programmatic Split & Signer Integrity Check ---")
    print(f"Train samples:      {len(x_train):3d} | Unique signers: {len(train_signers):2d}")
    print(f"Validation samples: {len(x_val):3d} | Unique signers: {len(val_signers):2d}")
    print(f"Test samples:       {len(x_test):3d} | Unique signers: {len(test_signers):2d}")
    print(f"Cross-split signer overlap: Train-Val={overlap_tv}, Train-Test={overlap_tt}, Val-Test={overlap_vt}")
    assert overlap_tv == 0 and overlap_tt == 0 and overlap_vt == 0, "FATAL: Signer overlap detected!"
    print("Signer-disjoint integrity: 100% VERIFIED (0 overlap across all splits)")

    # Class-weighted CrossEntropyLoss
    train_counts = np.bincount(y_train, minlength=num_classes)
    total_train = len(y_train)
    weights = np.sqrt(total_train / (num_classes * np.maximum(train_counts, 1).astype(np.float32)))
    weights = weights / np.mean(weights)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("\nClass weights applied to loss:")
    for cid in range(num_classes):
        print(f"  [{cid}] {label_mapping[cid]:10s}: count={train_counts[cid]:2d}, weight={weights[cid]:.3f}")

    # DataLoaders
    batch_size = 16
    train_ds = DynamicLandmarkDataset(x_train, m_train, y_train, is_train=True, oversample=True, label_names=l_train)
    val_ds = DynamicLandmarkDataset(x_val, m_val, y_val, is_train=False)
    test_ds = DynamicLandmarkDataset(x_test, m_test, y_test, is_train=False)

    print(f"Oversampled training dataset size: {len(train_ds)} (Original: {len(x_train)})")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cpu")
    model = DynamicSignBiGRU(input_dim=168, hidden_dim=64, num_layers=2, num_classes=num_classes, dropout=0.3).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 120
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 25
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    checkpoint_path = MODELS_DIR / "dynamic_bigru_v3_six_sign.pt"

    print("\nStarting training loop...")
    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_b, m_b, y_b in train_loader:
            x_b, m_b, y_b = x_b.to(device), m_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b, m_b)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_b).sum().item()
            train_total += len(y_b)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for x_b, m_b, y_b in val_loader:
                x_b, m_b, y_b = x_b.to(device), m_b.to(device), y_b.to(device)
                logits = model(x_b, m_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
                all_val_preds.extend(preds.cpu().numpy().tolist())
                all_val_targets.extend(y_b.cpu().numpy().tolist())

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(epoch_val_loss)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(epoch_train_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["val_acc"].append(round(epoch_val_acc, 4))
        history["lr"].append(current_lr)

        # Checkpoint saving on best validation loss
        saved_indicator = " "
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save({
                "version": "v3_six_sign",
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "label_mapping": label_mapping,
                "architecture": "2-layer Bidirectional GRU (Hidden=64, Masked Mean+Max Pooling, FC64, 6 Classes)",
                "input_dim": 168,
                "hidden_dim": 64,
                "num_classes": num_classes,
                "vocabulary": vocab,
                "total_train_samples": len(x_train),
                "total_val_samples": len(x_val),
                "total_test_samples": len(x_test),
                "augmentations": [
                    "in_plane_rotation_18deg",
                    "temporal_warp_speed_shift",
                    "minority_oversample",
                    "coordinate_jitter",
                    "body_relative_spatial_18d"
                ]
            }, checkpoint_path)
            saved_indicator = "*"
        else:
            patience_counter += 1

        if epoch % 5 == 0 or saved_indicator == "*":
            print(f"Epoch [{epoch:3d}/{num_epochs:3d}] | Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.3f} | Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:.3f} | LR: {current_lr:.6f} {saved_indicator}")

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch} (no val_loss improvement for {patience} epochs).")
            break

    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.4f}, Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Plot training curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("V3 Six-Sign Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc")
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("V3 Six-Sign Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = EVAL_DIR / "v3_six_sign_training_curve.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training curve plot to: {plot_path}")

if __name__ == "__main__":
    train()
