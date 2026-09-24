"""
SignBridge AI - Model Evaluation & Benchmark Script
Evaluates the trained dynamic Bi-GRU model on the untouched official test set.
Computes multi-class accuracy, precision, recall, macro/weighted F1-score,
confusion matrix, and measured inference latency per sequence.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "ml" / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

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

def evaluate():
    print("=" * 60)
    print("SignBridge AI — Phase F: Independent Test Set Evaluation")
    print("=" * 60)

    checkpoint_path = MODELS_DIR / "dynamic_bigru_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_mapping = checkpoint["label_mapping"]
    num_classes = checkpoint["num_classes"]

    model = DynamicSignBiGRU(
        input_dim=checkpoint.get("input_dim", 150),
        hidden_dim=checkpoint.get("hidden_dim", 64),
        num_layers=2,
        num_classes=num_classes,
        dropout=0.0 # eval mode
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load processed dynamic dataset
    data = np.load(PROCESSED_DIR / "dynamic_landmarks.npz")
    features = data["features"]
    masks = data["masks"]
    class_ids = data["class_ids"]
    splits = data["splits"]

    # Filter strictly for test split
    test_mask = (splits == "test")
    x_test = features[test_mask]
    m_test = masks[test_mask]
    y_test = class_ids[test_mask]

    num_test = len(y_test)
    print(f"Total Untouched Test Samples: {num_test}")

    # Latency benchmarking (CPU)
    # Warm-up
    dummy_x = torch.tensor(x_test[:1], dtype=torch.float32)
    dummy_m = torch.tensor(m_test[:1], dtype=torch.float32)
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_x, dummy_m)

    latencies = []
    y_pred_list = []
    y_true_list = []

    with torch.no_grad():
        for i in range(num_test):
            x_sample = torch.tensor(x_test[i:i+1], dtype=torch.float32)
            m_sample = torch.tensor(m_test[i:i+1], dtype=torch.float32)

            t0 = time.perf_counter()
            logits = model(x_sample, m_sample)
            t1 = time.perf_counter()

            latencies.append((t1 - t0) * 1000.0) # in ms
            pred_class = int(torch.argmax(logits, dim=1).item())
            y_pred_list.append(pred_class)
            y_true_list.append(int(y_test[i]))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    # Calculate metrics
    accuracy = float(np.mean(y_true == y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    mean_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    print(f"\n--- TEST BENCHMARK RESULTS ---")
    print(f"Accuracy:        {accuracy * 100:.2f}%")
    print(f"Macro F1-Score:  {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")
    print(f"Mean Latency:    {mean_latency:.2f} ms / sequence (p95: {p95_latency:.2f} ms)")

    # Per-class metrics
    class_names = [label_mapping[str(i)] if str(i) in label_mapping else label_mapping.get(i, f"class_{i}") for i in range(num_classes)]
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class_results = {}
    for i, name in enumerate(class_names):
        per_class_results[name] = {
            "precision": round(float(p_per[i]), 3),
            "recall": round(float(r_per[i]), 3),
            "f1_score": round(float(f1_per[i]), 3),
            "test_support": int(sup_per[i])
        }

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    # Save Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Dynamic Sign Confusion Matrix (Test Set N={num_test})")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # Add text labels inside cells
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if val > 0:
                plt.text(j, i, format(val, 'd'),
                         ha="center", va="center",
                         color="white" if val > thresh else "black")

    plt.ylabel('True Class Label')
    plt.xlabel('Predicted Class Label')
    plt.tight_layout()
    cm_path = EVAL_DIR / "dynamic_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot: {cm_path}")

    # Compile Comprehensive Report JSON
    report_data = {
        "project": "SignBridge AI",
        "evaluation_phase": "Phase F: Dynamic Bi-GRU Model Benchmark",
        "checkpoint_evaluated": str(checkpoint_path),
        "total_test_samples": num_test,
        "overall_accuracy": round(accuracy, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "inference_latency_ms": {
            "mean": round(mean_latency, 2),
            "p95": round(p95_latency, 2),
            "min": round(float(np.min(latencies)), 2),
            "max": round(float(np.max(latencies)), 2)
        },
        "per_class_performance": per_class_results,
        "sample_size_statistical_limitation": (
            "Because the official WLASL test split contains 26 samples across 17 classes (~1-3 test samples per class), "
            "individual class F1 scores exhibit discrete step variations. Confidence intervals must be interpreted "
            "alongside validation loss and cross-signer generalization."
        ),
        "confusion_matrix": cm.tolist()
    }

    report_path = EVAL_DIR / "dynamic_test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved test evaluation report to {report_path}")

if __name__ == "__main__":
    evaluate()
