"""
SignBridge AI - Step 5 & 6: Offline Evaluation and Comparison for Focused 6-Sign V3 Model
Evaluates:
  1. Focused 6-Sign V3 Model (dynamic_bigru_v3_six_sign.pt, 168 features) on Frozen Test Split (39 samples, 6 signers)
  2. Comparable evaluation against previous V2 Model (dynamic_bigru_v2.pt, 150 features) on overlapping signs and test population
  3. Metrics:
     - Accuracy, Macro Precision, Recall, F1, Weighted F1
     - Per-class metrics for NO, HELP, YES, PLEASE, THANK_YOU, HELLO
     - Confusion matrix & confidence distribution
     - Latency (mean, median, p95), parameter count, model size
Outputs:
  - ml/evaluation/v3_six_sign_model_comparison.json
  - ml/evaluation/v3_six_sign_confusion_matrix.png
  - ml/evaluation/v3_six_sign_f1_comparison.png
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "ml" / "datasets" / "processed"
MODELS_DIR = BASE_DIR / "ml" / "models"
EVAL_DIR = BASE_DIR / "ml" / "evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Model Definitions
# -------------------------------------------------------------
class DynamicSignBiGRUV3(nn.Module):
    def __init__(self, input_dim=168, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3):
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

class DynamicSignBiGRUV2(nn.Module):
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

def measure_latency(model, input_tensor, mask_tensor, runs=200):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(50):
            _ = model(input_tensor, mask_tensor)
        latencies = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(input_tensor, mask_tensor)
            latencies.append((time.perf_counter() - t0) * 1000.0)
    latencies = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(latencies)), 2),
        "median_ms": round(float(np.median(latencies)), 2),
        "p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "min_ms": round(float(np.min(latencies)), 2),
        "max_ms": round(float(np.max(latencies)), 2),
        "throughput_fps": round(1000.0 / float(np.mean(latencies)), 1)
    }

def main():
    print("=" * 75)
    print("SignBridge AI — Step 5 & 6: Offline Evaluation & Comparison")
    print("=" * 75)

    dataset_path = PROCESSED_DIR / "dynamic_landmarks_v3_six_sign.npz"
    if not dataset_path.exists():
        print(f"Error: Dataset missing at {dataset_path}")
        sys.exit(1)

    data = np.load(dataset_path)
    features = data["features"]
    masks = data["masks"]
    labels = data["labels"]
    class_ids = data["class_ids"]
    splits = data["splits"]
    signers = data["signer_ids"]

    test_mask = (splits == "test")
    x_test = features[test_mask]
    m_test = masks[test_mask]
    y_test = class_ids[test_mask]
    l_test = labels[test_mask]
    s_test = signers[test_mask]

    n_test = len(x_test)
    unique_test_signers = len(set(s_test))
    print(f"Frozen Test Set: {n_test} samples from {unique_test_signers} signers")

    # -------------------------------------------------------------
    # 1. Evaluate Focused 6-Sign V3 Model
    # -------------------------------------------------------------
    v3_ckpt_path = MODELS_DIR / "dynamic_bigru_v3_six_sign.pt"
    v3_map_path = MODELS_DIR / "dynamic_label_mapping_v3_six_sign.json"

    if not v3_ckpt_path.exists() or not v3_map_path.exists():
        print(f"Error: V3 checkpoint or mapping missing")
        sys.exit(1)

    with open(v3_map_path, "r", encoding="utf-8") as f:
        v3_label_mapping = json.load(f)
    v3_classes = [v3_label_mapping[str(i)] for i in range(len(v3_label_mapping))]

    v3_ckpt = torch.load(v3_ckpt_path, map_location="cpu", weights_only=False)
    v3_model = DynamicSignBiGRUV3(input_dim=168, hidden_dim=64, num_layers=2, num_classes=len(v3_classes), dropout=0.3)
    v3_model.load_state_dict(v3_ckpt["model_state_dict"])
    v3_model.eval()

    v3_params = sum(p.numel() for p in v3_model.parameters())
    v3_size_kb = round(os.path.getsize(v3_ckpt_path) / 1024, 1)

    # Inference on test set
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    m_test_t = torch.tensor(m_test, dtype=torch.float32)

    with torch.no_grad():
        v3_logits = v3_model(x_test_t, m_test_t)
        v3_probs = torch.softmax(v3_logits, dim=-1).numpy()
        v3_preds = np.argmax(v3_probs, axis=-1)
        v3_confs = np.max(v3_probs, axis=-1)

    v3_acc = accuracy_score(y_test, v3_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, v3_preds, average="macro", zero_division=0)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, v3_preds, average="weighted", zero_division=0)

    # Per-class metrics
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_test, v3_preds, labels=range(len(v3_classes)), zero_division=0
    )

    per_class_v3 = {}
    for idx, cname in enumerate(v3_classes):
        per_class_v3[cname.upper()] = {
            "precision": round(float(per_class_p[idx]), 4),
            "recall": round(float(per_class_r[idx]), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "support": int(per_class_supp[idx])
        }

    # Confusion matrix
    cm_v3 = confusion_matrix(y_test, v3_preds, labels=range(len(v3_classes)))

    # Confidence distribution
    correct_mask = (v3_preds == y_test)
    conf_correct = v3_confs[correct_mask] if np.any(correct_mask) else np.array([0.0])
    conf_incorrect = v3_confs[~correct_mask] if np.any(~correct_mask) else np.array([0.0])
    accepted_mask = (v3_confs >= 0.70)

    v3_conf_stats = {
        "mean_overall": round(float(np.mean(v3_confs)), 4),
        "mean_correct": round(float(np.mean(conf_correct)), 4),
        "mean_incorrect": round(float(np.mean(conf_incorrect)), 4),
        "min_conf": round(float(np.min(v3_confs)), 4),
        "max_conf": round(float(np.max(v3_confs)), 4),
        "samples_above_70_thresh": int(np.sum(accepted_mask)),
        "accepted_percentage": round(float(np.mean(accepted_mask) * 100), 1),
        "accuracy_above_70_thresh": round(float(accuracy_score(y_test[accepted_mask], v3_preds[accepted_mask])) if np.any(accepted_mask) else 0.0, 4)
    }

    # Latency test (single sample batch size 1)
    single_x = x_test_t[0:1]
    single_m = m_test_t[0:1]
    v3_latency = measure_latency(v3_model, single_x, single_m)

    print("\n--- Focused 6-Sign V3 Test Results ---")
    print(f"Accuracy:        {v3_acc*100:.2f}%")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")
    print(f"Mean Confidence: {v3_conf_stats['mean_overall']*100:.1f}%")
    print(f"CPU Latency:     {v3_latency['mean_ms']} ms (p95: {v3_latency['p95_ms']} ms, {v3_latency['throughput_fps']} FPS)")

    # -------------------------------------------------------------
    # 2. Evaluate Previous V2 Model
    # -------------------------------------------------------------
    v2_ckpt_path = MODELS_DIR / "dynamic_bigru_v2.pt"
    v2_map_path = MODELS_DIR / "dynamic_label_mapping_v2.json"

    v2_evaluated = False
    v2_overlap_results = {}
    v2_all_results = {}
    v2_params = 0
    v2_size_kb = 0.0
    v2_latency = {}

    if v2_ckpt_path.exists() and v2_map_path.exists():
        with open(v2_map_path, "r", encoding="utf-8") as f:
            v2_label_mapping = json.load(f)
        v2_classes = [v2_label_mapping[str(i)] for i in range(len(v2_label_mapping))]

        v2_ckpt = torch.load(v2_ckpt_path, map_location="cpu", weights_only=False)
        v2_model = DynamicSignBiGRUV2(input_dim=150, hidden_dim=64, num_layers=2, num_classes=len(v2_classes), dropout=0.3)
        v2_model.load_state_dict(v2_ckpt["model_state_dict"])
        v2_model.eval()

        v2_params = sum(p.numel() for p in v2_model.parameters())
        v2_size_kb = round(os.path.getsize(v2_ckpt_path) / 1024, 1)

        # Slice 150 features from 168
        x_test_150 = torch.tensor(x_test[:, :, :150], dtype=torch.float32)
        v2_latency = measure_latency(v2_model, x_test_150[0:1], single_m)

        with torch.no_grad():
            v2_logits = v2_model(x_test_150, m_test_t)
            v2_probs = torch.softmax(v2_logits, dim=-1).numpy()
            v2_preds = np.argmax(v2_probs, axis=-1)
            v2_pred_labels = [v2_label_mapping[str(p)] for p in v2_preds]
            v2_confs = np.max(v2_probs, axis=-1)

        # Overlapping signs evaluation (HELP, YES, NO, PLEASE, THANK_YOU)
        # Note: HELLO was NOT in V2 vocabulary
        overlap_indices = [i for i, lbl in enumerate(l_test) if lbl in v2_classes]
        overlap_y_true = [l_test[i] for i in overlap_indices]
        overlap_y_pred = [v2_pred_labels[i] for i in overlap_indices]

        v2_overlap_acc = accuracy_score(overlap_y_true, overlap_y_pred)
        p_ov, r_ov, f1_ov, _ = precision_recall_fscore_support(overlap_y_true, overlap_y_pred, average="macro", zero_division=0)

        # V3 on the same overlapping subset
        v3_pred_labels = [v3_classes[p] for p in v3_preds]
        v3_overlap_y_pred = [v3_pred_labels[i] for i in overlap_indices]
        v3_overlap_acc = accuracy_score(overlap_y_true, v3_overlap_y_pred)
        v3_p_ov, v3_r_ov, v3_f1_ov, _ = precision_recall_fscore_support(overlap_y_true, v3_overlap_y_pred, average="macro", zero_division=0)

        v2_overlap_results = {
            "evaluated_samples": len(overlap_indices),
            "signs_evaluated": sorted(list(set(overlap_y_true))),
            "v2_accuracy": round(float(v2_overlap_acc), 4),
            "v2_macro_f1": round(float(f1_ov), 4),
            "v3_accuracy_on_same_subset": round(float(v3_overlap_acc), 4),
            "v3_macro_f1_on_same_subset": round(float(v3_f1_ov), 4),
            "delta_accuracy": round(float(v3_overlap_acc - v2_overlap_acc), 4),
            "delta_macro_f1": round(float(v3_f1_ov - f1_ov), 4)
        }

        # Per-class comparisons on overlapping signs
        per_sign_comparison = {}
        for sgn in ["help", "yes", "no", "please", "thank_you"]:
            sgn_indices = [i for i, lbl in enumerate(l_test) if lbl == sgn]
            n_sgn = len(sgn_indices)
            if n_sgn == 0:
                continue
            v2_corr = sum(1 for i in sgn_indices if v2_pred_labels[i] == sgn)
            v3_corr = sum(1 for i in sgn_indices if v3_pred_labels[i] == sgn)
            v2_sgn_confs = [float(v2_confs[i]) for i in sgn_indices]
            v3_sgn_confs = [float(v3_confs[i]) for i in sgn_indices]
            per_sign_comparison[sgn.upper()] = {
                "support": n_sgn,
                "v2_correct": v2_corr,
                "v2_recall": round(v2_corr / n_sgn, 4),
                "v2_mean_conf": round(float(np.mean(v2_sgn_confs)), 4),
                "v3_correct": v3_corr,
                "v3_recall": round(v3_corr / n_sgn, 4),
                "v3_mean_conf": round(float(np.mean(v3_sgn_confs)), 4),
                "recall_improvement": round((v3_corr - v2_corr) / n_sgn, 4)
            }

        # Check HELLO (OOD for V2)
        hello_indices = [i for i, lbl in enumerate(l_test) if lbl == "hello"]
        hello_v3_corr = sum(1 for i in hello_indices if v3_pred_labels[i] == "hello")
        hello_v2_preds = [v2_pred_labels[i] for i in hello_indices]
        per_sign_comparison["HELLO"] = {
            "support": len(hello_indices),
            "v2_correct": 0,
            "v2_recall": 0.0,
            "v2_note": "OOD (not in V2 17-class vocabulary; predicted: " + ", ".join(hello_v2_preds) + ")",
            "v3_correct": hello_v3_corr,
            "v3_recall": round(hello_v3_corr / len(hello_indices), 4),
            "v3_mean_conf": round(float(np.mean([v3_confs[i] for i in hello_indices])), 4),
            "recall_improvement": round(hello_v3_corr / len(hello_indices), 4)
        }

        v2_evaluated = True

    # -------------------------------------------------------------
    # 3. Generate Visualizations
    # -------------------------------------------------------------
    # Confusion Matrix Plot
    plt.figure(figsize=(7, 6))
    plt.imshow(cm_v3, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Focused 6-Sign V3 Confusion Matrix (Frozen Test Set)")
    plt.colorbar()
    tick_marks = np.arange(len(v3_classes))
    plt.xticks(tick_marks, [c.upper() for c in v3_classes], rotation=45)
    plt.yticks(tick_marks, [c.upper() for c in v3_classes])

    thresh = cm_v3.max() / 2.0
    for i in range(cm_v3.shape[0]):
        for j in range(cm_v3.shape[1]):
            plt.text(j, i, format(cm_v3[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm_v3[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = EVAL_DIR / "v3_six_sign_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}")

    # Per-Class F1 / Recall Plot
    plt.figure(figsize=(8, 4))
    classes_upper = [c.upper() for c in v3_classes]
    f1_vals = [per_class_v3[c]["f1_score"] for c in classes_upper]
    rec_vals = [per_class_v3[c]["recall"] for c in classes_upper]

    x = np.arange(len(classes_upper))
    width = 0.35

    plt.bar(x - width/2, rec_vals, width, label='Recall', color='#2563eb')
    plt.bar(x + width/2, f1_vals, width, label='F1 Score', color='#10b981')
    plt.axhline(y=0.70, color='red', linestyle='--', alpha=0.6, label='70% Threshold')
    plt.ylabel('Score')
    plt.title('V3 Six-Sign Per-Class Performance (Frozen Test Set)')
    plt.xticks(x, classes_upper)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    f1_plot_path = EVAL_DIR / "v3_six_sign_f1_comparison.png"
    plt.savefig(f1_plot_path, dpi=150)
    plt.close()
    print(f"Saved per-class F1 plot to: {f1_plot_path}")

    # -------------------------------------------------------------
    # 4. Save Comprehensive Comparison JSON
    # -------------------------------------------------------------
    comparison_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "v3_six_sign",
        "scope": "Focused 6-Sign Recognition (HELP, YES, NO, THANK_YOU, PLEASE, HELLO)",
        "production_status": {
            "v2_production_active": True,
            "v3_activated": False,
            "comment": "V2 production untouched and active; V3 evaluated strictly offline"
        },
        "dataset_summary": {
            "total_samples": len(features),
            "train_samples": int(np.sum(splits == "train")),
            "val_samples": int(np.sum(splits == "val")),
            "test_samples": int(np.sum(test_mask)),
            "test_signers": unique_test_signers,
            "signer_overlap_train_val": 0,
            "signer_overlap_train_test": 0,
            "signer_overlap_val_test": 0
        },
        "v3_six_sign_metrics": {
            "accuracy": round(float(v3_acc), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "parameters": v3_params,
            "model_size_kb": v3_size_kb,
            "latency": v3_latency,
            "confidence_distribution": v3_conf_stats,
            "per_class": per_class_v3,
            "confusion_matrix": cm_v3.tolist()
        },
        "v2_comparison": {
            "v2_production_baseline_17_classes": {
                "accuracy": 0.7647,
                "macro_f1": 0.7420,
                "vocabulary_size": 17,
                "features": 150,
                "note": "Reported on original frozen 17-class WLASL test split (17 samples)"
            },
            "v2_on_overlapping_test_samples": v2_overlap_results if v2_evaluated else None,
            "per_sign_direct_comparison": per_sign_comparison if v2_evaluated else None,
            "v2_model_specs": {
                "parameters": v2_params,
                "model_size_kb": v2_size_kb,
                "latency": v2_latency
            }
        }
    }

    out_comparison_json = EVAL_DIR / "v3_six_sign_model_comparison.json"
    with open(out_comparison_json, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\nSaved comparison JSON to: {out_comparison_json}")

if __name__ == "__main__":
    main()
