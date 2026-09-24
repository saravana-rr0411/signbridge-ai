"""
SignBridge AI - Dynamic Sign Recognition Training Pipeline
Trains a lightweight 2-layer Bidirectional GRU model over 30-frame MediaPipe landmark sequences.
Uses official WLASL train/val/test splits, class-weighted CrossEntropyLoss, and training-only data augmentation.
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set random seeds for exact reproducibility
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
CONFIG_DIR = BASE_DIR / "ml" / "config"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

class LandmarkSequenceDataset(Dataset):
    def __init__(self, features, masks, class_ids, is_train=False):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.float32)
        self.class_ids = torch.tensor(class_ids, dtype=torch.long)
        self.is_train = is_train

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx].clone()
        mask = self.masks[idx]
        target = self.class_ids[idx]

        if self.is_train:
            # Data Augmentation (Train set ONLY)
            # 1. Scale jitter: s in [0.96, 1.04]
            scale = np.random.uniform(0.96, 1.04)
            # 2. Gaussian coordinate jitter
            noise = torch.randn_like(feat) * 0.012

            # Left hand coords: 0..62
            feat[:, 0:63] = feat[:, 0:63] * scale + noise[:, 0:63]
            # Right hand coords: 64..126
            feat[:, 64:127] = feat[:, 64:127] * scale + noise[:, 64:127]
            # Pose coords: 128..148
            feat[:, 128:149] = feat[:, 128:149] * scale + noise[:, 128:149]

        return feat, mask, target

class DynamicSignBiGRU(nn.Module):
    def __init__(self, input_dim=150, hidden_dim=64, num_layers=2, num_classes=17, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Bidirectional produces 2 * hidden_dim
        # We concatenate temporal average pooling and temporal max pooling: 2 * (hidden_dim * 2) = 4 * hidden_dim
        self.pool_dim = hidden_dim * 4
        self.fc1 = nn.Linear(self.pool_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x, mask):
        # x: (B, T, D), mask: (B, T)
        gru_out, _ = self.gru(x) # (B, T, 2*H)

        # Masked pooling over temporal dimension
        mask_expanded = mask.unsqueeze(-1) # (B, T, 1)
        masked_out = gru_out * mask_expanded

        # Mean pooling over valid frames
        sum_out = torch.sum(masked_out, dim=1) # (B, 2*H)
        lengths = torch.clamp(torch.sum(mask_expanded, dim=1), min=1.0)
        mean_pool = sum_out / lengths

        # Max pooling
        # Set masked frames to large negative value before max
        masked_for_max = masked_out + (1.0 - mask_expanded) * -1e9
        max_pool, _ = torch.max(masked_for_max, dim=1) # (B, 2*H)

        combined = torch.cat([mean_pool, max_pool], dim=-1) # (B, 4*H)
        h = self.dropout(self.relu(self.fc1(combined)))
        logits = self.fc2(h)
        return logits

def train():
    print("=" * 60)
    print("SignBridge AI — Phase F: Dynamic Bi-GRU Model Training")
    print("=" * 60)

    # Load preprocessed dynamic landmarks
    dyn_path = PROCESSED_DIR / "dynamic_landmarks.npz"
    if not dyn_path.exists():
        raise FileNotFoundError(f"Processed dynamic dataset not found at {dyn_path}")

    data = np.load(dyn_path)
    features = data["features"]      # (N, 30, 150)
    masks = data["masks"]            # (N, 30)
    labels = data["labels"]          # (N,)
    class_ids = data["class_ids"]    # (N,)
    splits = data["splits"]          # (N,)

    # Create Label Mapping
    unique_pairs = sorted(list(set(zip(class_ids.tolist(), labels.tolist()))), key=lambda x: x[0])
    label_mapping = {int(cid): str(lbl) for cid, lbl in unique_pairs}
    num_classes = len(label_mapping)
    print(f"Total dynamic classes: {num_classes}")

    # Save Label Mapping JSON
    map_path = MODELS_DIR / "dynamic_label_mapping.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=2)
    print(f"Saved dynamic label mapping to {map_path}")

    # Partition by official splits
    train_mask = (splits == "train")
    val_mask = (splits == "val")
    test_mask = (splits == "test")

    x_train, m_train, y_train = features[train_mask], masks[train_mask], class_ids[train_mask]
    x_val, m_val, y_val = features[val_mask], masks[val_mask], class_ids[val_mask]
    x_test, m_test, y_test = features[test_mask], masks[test_mask], class_ids[test_mask]

    print(f"Split sizes: Train={len(x_train)} | Validation={len(x_val)} | Test={len(x_test)}")

    # Compute class weights for imbalanced classes
    train_counts = np.bincount(y_train, minlength=num_classes)
    total_train = len(y_train)
    class_weights = total_train / (num_classes * np.maximum(train_counts, 1).astype(np.float32))
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # DataLoaders
    batch_size = 16
    train_ds = LandmarkSequenceDataset(x_train, m_train, y_train, is_train=True)
    val_ds = LandmarkSequenceDataset(x_val, m_val, y_val, is_train=False)
    test_ds = LandmarkSequenceDataset(x_test, m_test, y_test, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    device = torch.device("cpu") # Optimized for MacBook CPU real-time inference
    model = DynamicSignBiGRU(input_dim=150, hidden_dim=64, num_layers=2, num_classes=num_classes, dropout=0.3).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Dynamic Bi-GRU Trainable Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 80
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_checkpoint_path = MODELS_DIR / "dynamic_bigru_best.pt"

    for epoch in range(1, num_epochs + 1):
        # Training loop
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

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x_b, m_b, y_b in val_loader:
                x_b, m_b, y_b = x_b.to(device), m_b.to(device), y_b.to(device)
                logits = model(x_b, m_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

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

        # Early stopping and checkpoint saving
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
                "label_mapping": label_mapping,
                "architecture": "2-layer Bi-directional GRU",
                "input_dim": 150,
                "hidden_dim": 64,
                "num_classes": num_classes
            }, best_checkpoint_path)
            saved_indicator = "*"
        else:
            patience_counter += 1
            saved_indicator = ""

        if epoch % 5 == 0 or saved_indicator == "*":
            print(f"Epoch {epoch:2d}/{num_epochs:2d} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.3f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.3f} (LR: {current_lr:.1e}) {saved_indicator}")

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
            break

    print("\nTraining completed.")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Accuracy: {best_val_acc:.3f}")
    print(f"Saved best checkpoint: {best_checkpoint_path}")

    # Save training history JSON
    history_json_path = EVAL_DIR / "dynamic_training_history.json"
    with open(history_json_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved training history to {history_json_path}")

    # Plot training curves
    plot_path = EVAL_DIR / "dynamic_training_curve.png"
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#3b82f6", lw=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#ef4444", lw=2)
    plt.title("CrossEntropy Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["train_acc"], label="Train Acc", color="#3b82f6", lw=2)
    plt.plot(history["epoch"], history["val_acc"], label="Val Acc", color="#10b981", lw=2)
    plt.title("Classification Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved training curve to {plot_path}")

if __name__ == "__main__":
    train()
