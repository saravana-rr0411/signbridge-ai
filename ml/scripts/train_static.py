"""
SignBridge AI - Static Handshape MLP Training Pipeline
Trains a lightweight 3-layer MLP classifier for 'letter_a' handshape verification.
Input: 64-dim normalized landmark feature vector (21 3D hand coordinates + presence flag).
"""

import os
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reproducible random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "ml" / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

class StaticHandshapeDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class StaticHandshapeMLP(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, num_classes=2, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_static():
    print("=" * 60)
    print("SignBridge AI — Phase F: Static 'letter_a' MLP Training")
    print("=" * 60)

    static_path = PROCESSED_DIR / "static_landmarks.npz"
    if not static_path.exists():
        raise FileNotFoundError(f"Processed static dataset not found at {static_path}")

    data = np.load(static_path)
    pos_features = data["features"] # (N, 64)
    num_pos = len(pos_features)
    print(f"Loaded {num_pos} verified 'letter_a' landmark vectors.")

    # Create balanced negative samples (non-A / open-hand / flat hand poses)
    # to allow the model to discriminate Letter A from casual hand movements
    num_neg = num_pos
    neg_features = []
    for feat in pos_features:
        neg = feat.copy()
        # Randomly extend fingers (simulating open palm or non-fist gestures)
        finger_indices = [4, 8, 12, 16, 20] # tips of thumb, index, middle, ring, pinky
        for tip in finger_indices:
            neg[tip * 3 : (tip + 1) * 3] += np.random.uniform(0.3, 0.8, size=3)
        # Add random joint perturbation
        neg[:63] += np.random.normal(0, 0.05, size=63)
        neg_features.append(neg)

    neg_features = np.array(neg_features, dtype=np.float32)

    all_features = np.concatenate([pos_features, neg_features], axis=0)
    all_labels = np.concatenate([np.ones(num_pos, dtype=np.int64), np.zeros(num_neg, dtype=np.int64)], axis=0)

    # Shuffle
    perm = np.random.permutation(len(all_features))
    all_features = all_features[perm]
    all_labels = all_labels[perm]

    # Train / Val Split (80 / 20)
    split_idx = int(0.8 * len(all_features))
    x_train, y_train = all_features[:split_idx], all_labels[:split_idx]
    x_val, y_val = all_features[split_idx:], all_labels[split_idx:]

    print(f"Static Training Split: {len(x_train)} | Validation Split: {len(x_val)}")

    train_ds = StaticHandshapeDataset(x_train, y_train)
    val_ds = StaticHandshapeDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = StaticHandshapeMLP(input_dim=64, hidden_dim=32, num_classes=2, dropout=0.2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    num_epochs = 30
    best_acc = 0.0
    best_path = MODELS_DIR / "static_mlp_best.pt"

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_b).sum().item()
            train_total += len(y_b)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                logits = model(x_b)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_acc = val_correct / val_total
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": best_acc,
                "architecture": "3-layer MLP",
                "input_dim": 64,
                "num_classes": 2,
                "label_mapping": {"0": "other", "1": "letter_a"}
            }, best_path)

        if epoch % 5 == 0:
            print(f"Static Epoch {epoch:2d}/{num_epochs:2d} | Train Acc: {train_correct/train_total:.3f} | Val Acc: {val_acc:.3f}")

    print(f"\nStatic MLP training complete. Best Validation Accuracy: {best_acc:.3f}")
    print(f"Saved static checkpoint: {best_path}")

    # Save Label Mapping JSON
    static_label_mapping = {
        "0": "other_handshape",
        "1": "letter_a"
    }
    mapping_path = MODELS_DIR / "static_label_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(static_label_mapping, f, indent=2)
    print(f"Saved static label mapping: {mapping_path}")

if __name__ == "__main__":
    train_static()
