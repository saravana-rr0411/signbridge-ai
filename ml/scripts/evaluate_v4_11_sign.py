"""
SignBridge AI - Step 5 & 6: Offline Evaluation and Comparison for V4 11-Sign Model
Evaluates:
  1. V4 11-Sign Model (dynamic_bigru_v4_11_sign.pt, 168 features) on Frozen Test Split (83 samples, 12 signers)
  2. Comparable evaluation against V3 6-sign Model (dynamic_bigru_v3_six_sign.pt) on the overlapping 6-sign subset
  3. Metrics:
     - Accuracy, Macro Precision, Recall, F1, Weighted F1
     - Per-class metrics for all 11 classes (support counts, precision, recall, F1)
     - Full 11x11 Confusion matrix & detailed confusion pair analysis
     - Mean confidence and % above 70% confidence threshold
     - Latency (mean, median, p95)
Outputs:
  - ml/evaluation/v4_11_sign_model_comparison.json
  - ml/evaluation/v4_11_sign_confusion_matrix.png
  - ml/evaluation/v4_11_sign_f1_comparison.png
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
class DynamicSignBiGRU(nn.Module):
    def __init__(self, input_dim=168, hidden_dim=64, num_layers=2, num_classes=11, dropout=0.3):
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
    print("=" * 80)
    print("SignBridge AI — V4 11-Sign Model Offline Evaluation & V3 Comparison")
    print("=" * 80)

    # 1. Load V4 dataset
    dataset_path = PROCESSED_DIR / "dynamic_landmarks_v4_11_sign.npz"
    if not dataset_path.exists():
        print(f"Error: Dataset missing at {dataset_path}")
        sys.exit(1)

    data = np.load(dataset_path, allow_pickle=True)
    features = data["features"]
    masks = data["masks"]
    labels = data["labels"]
    class_ids = data["class_ids"]
    splits = data["splits"]
    signers = data["signer_ids"]

    # Vocabulary
    VOCABULARY_11 = [
        "hello", "help", "yes", "no", "please", "thank_you",
        "doctor", "pain", "sick", "bathroom", "where"
    ]
    ORIGINAL_6 = ["hello", "help", "yes", "no", "please", "thank_you"]
    NEW_5 = ["doctor", "pain", "sick", "bathroom", "where"]

    test_mask = (splits == "test")
    x_test = features[test_mask]
    m_test = masks[test_mask]
    y_test = class_ids[test_mask]
    l_test = labels[test_mask]
    s_test = signers[test_mask]

    n_test = len(x_test)
    unique_test_signers = len(set(s_test))
    print(f"\nFrozen Test Set: {n_test} total samples from {unique_test_signers} unique signers")

    # Verify 0 cross-split signer overlap
    train_signers = set(signers[splits == "train"])
    val_signers = set(signers[splits == "val"])
    test_signers = set(s_test)
    assert len(train_signers & test_signers) == 0, "FATAL: Train-Test signer overlap detected!"
    assert len(val_signers & test_signers) == 0, "FATAL: Val-Test signer overlap detected!"
    print(f"Cross-split signer overlap: Strictly 0 (Verified: Train-Test=0, Val-Test=0)")

    # 2. Load V4 Model
    v4_ckpt_path = MODELS_DIR / "dynamic_bigru_v4_11_sign.pt"
    if not v4_ckpt_path.exists():
        print(f"Error: V4 checkpoint missing at {v4_ckpt_path}")
        sys.exit(1)

    ckpt_v4 = torch.load(v4_ckpt_path, map_location="cpu", weights_only=False)
    model_v4 = DynamicSignBiGRU(input_dim=168, hidden_dim=64, num_layers=2, num_classes=11, dropout=0.3)
    model_v4.load_state_dict(ckpt_v4["model_state_dict"])
    model_v4.eval()

    v4_params = sum(p.numel() for p in model_v4.parameters() if p.requires_grad)
    v4_size_kb = round(os.path.getsize(v4_ckpt_path) / 1024.0, 1)
    print(f"\nV4 Model Loaded: {v4_params:,} parameters | Checkpoint Size: {v4_size_kb} KB | Best Epoch: {ckpt_v4.get('epoch')}")

    # Latency evaluation
    dummy_x = torch.tensor(x_test[:1], dtype=torch.float32)
    dummy_m = torch.tensor(m_test[:1], dtype=torch.float32)
    v4_latency = measure_latency(model_v4, dummy_x, dummy_m, runs=200)
    print(f"V4 Inference Latency: Mean={v4_latency['mean_ms']}ms, Median={v4_latency['median_ms']}ms, P95={v4_latency['p95_ms']}ms ({v4_latency['throughput_fps']} FPS)")

    # 3. Evaluate V4 on Full Frozen Test Set
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    m_test_t = torch.tensor(m_test, dtype=torch.float32)

    with torch.no_grad():
        logits_v4 = model_v4(x_test_t, m_test_t)
        probs_v4 = torch.softmax(logits_v4, dim=-1).cpu().numpy()
        preds_v4 = np.argmax(probs_v4, axis=-1)
        confs_v4 = np.max(probs_v4, axis=-1)

    v4_acc = float(accuracy_score(y_test, preds_v4))
    prec, rec, f1, support = precision_recall_fscore_support(y_test, preds_v4, labels=list(range(11)), zero_division=0)
    macro_prec = float(np.mean(prec))
    macro_rec = float(np.mean(rec))
    macro_f1 = float(np.mean(f1))
    weighted_prec, weighted_rec, weighted_f1, _ = precision_recall_fscore_support(y_test, preds_v4, average='weighted', zero_division=0)

    above_70 = float(np.mean(confs_v4 >= 0.70) * 100.0)
    mean_conf = float(np.mean(confs_v4) * 100.0)

    print("\n" + "=" * 80)
    print("V4 11-SIGN FROZEN TEST RESULTS (83 SAMPLES)")
    print("=" * 80)
    print(f"Overall Accuracy:       {v4_acc * 100:.2f}%")
    print(f"Macro Precision:        {macro_prec * 100:.2f}%")
    print(f"Macro Recall:           {macro_rec * 100:.2f}%")
    print(f"Macro F1 Score:         {macro_f1 * 100:.2f}%")
    print(f"Weighted F1 Score:      {weighted_f1 * 100:.2f}%")
    print(f"Mean Confidence:        {mean_conf:.2f}%")
    print(f"Confidence >= 70%:      {above_70:.2f}%")

    print("\n--- Per-Class Performance (All 11 Classes) ---")
    print(f"{'Class':<12} | {'Support':<8} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Mean Conf':<10} | {'>=70% Conf':<10}")
    print("-" * 80)
    per_class_v4 = {}
    for cid in range(11):
        cname = VOCABULARY_11[cid]
        c_mask = (y_test == cid)
        c_mean_conf = float(np.mean(confs_v4[c_mask])) * 100.0 if np.sum(c_mask) > 0 else 0.0
        c_above_70 = float(np.mean(confs_v4[c_mask] >= 0.70)) * 100.0 if np.sum(c_mask) > 0 else 0.0
        per_class_v4[cname] = {
            "class_id": cid,
            "support": int(support[cid]),
            "precision": round(float(prec[cid]), 4),
            "recall": round(float(rec[cid]), 4),
            "f1": round(float(f1[cid]), 4),
            "mean_confidence": round(c_mean_conf, 2),
            "pct_above_70": round(c_above_70, 2)
        }
        print(f"{cname:<12} | {support[cid]:<8} | {prec[cid]*100:>8.2f}% | {rec[cid]*100:>8.2f}% | {f1[cid]*100:>8.2f}% | {c_mean_conf:>8.2f}% | {c_above_70:>8.2f}%")

    # Confusion Matrix
    cm_v4 = confusion_matrix(y_test, preds_v4, labels=list(range(11)))

    # 4. Detailed Confusion Analysis
    print("\n--- Focused Pair Confusion Analysis ---")
    pairs_to_check = [
        ("bathroom", "yes"),
        ("doctor", "help"),
        ("doctor", "all_others"),
        ("pain", "all_others"),
        ("sick", "all_others"),
        ("where", "no"),
        ("where", "yes"),
        ("where", "all_others")
    ]
    
    confusion_report = []
    # BATHROOM vs YES
    c_bath_yes = cm_v4[VOCABULARY_11.index("bathroom")][VOCABULARY_11.index("yes")]
    c_yes_bath = cm_v4[VOCABULARY_11.index("yes")][VOCABULARY_11.index("bathroom")]
    print(f"  • BATHROOM vs YES: BATHROOM predicted as YES: {c_bath_yes} | YES predicted as BATHROOM: {c_yes_bath}")
    confusion_report.append({"pair": "BATHROOM vs YES", "bath_as_yes": int(c_bath_yes), "yes_as_bath": int(c_yes_bath)})

    # DOCTOR vs HELP
    c_doc_help = cm_v4[VOCABULARY_11.index("doctor")][VOCABULARY_11.index("help")]
    c_help_doc = cm_v4[VOCABULARY_11.index("help")][VOCABULARY_11.index("doctor")]
    print(f"  • DOCTOR vs HELP: DOCTOR predicted as HELP: {c_doc_help} | HELP predicted as DOCTOR: {c_help_doc}")
    confusion_report.append({"pair": "DOCTOR vs HELP", "doc_as_help": int(c_doc_help), "help_as_doc": int(c_help_doc)})

    # Other confusions
    actual_confusions = []
    for i in range(11):
        for j in range(11):
            if i != j and cm_v4[i][j] > 0:
                actual_confusions.append({
                    "true_class": VOCABULARY_11[i],
                    "pred_class": VOCABULARY_11[j],
                    "count": int(cm_v4[i][j])
                })
                print(f"  • Actual confusion: True '{VOCABULARY_11[i]}' -> Pred '{VOCABULARY_11[j]}' ({cm_v4[i][j]} sample(s))")

    # 5. Evaluate Performance on Subsets:
    # 5a. Original Six Signs subset
    orig6_indices = [VOCABULARY_11.index(s) for s in ORIGINAL_6]
    orig6_test_mask = np.isin(y_test, orig6_indices)
    orig6_y_test = y_test[orig6_test_mask]
    orig6_preds_v4 = preds_v4[orig6_test_mask]
    orig6_acc_v4 = float(accuracy_score(orig6_y_test, orig6_preds_v4))
    _, _, orig6_f1s_v4, _ = precision_recall_fscore_support(orig6_y_test, orig6_preds_v4, labels=orig6_indices, zero_division=0)
    orig6_macro_f1_v4 = float(np.mean(orig6_f1s_v4))

    # 5b. New Five Signs subset
    new5_indices = [VOCABULARY_11.index(s) for s in NEW_5]
    new5_test_mask = np.isin(y_test, new5_indices)
    new5_y_test = y_test[new5_test_mask]
    new5_preds_v4 = preds_v4[new5_test_mask]
    new5_acc_v4 = float(accuracy_score(new5_y_test, new5_preds_v4))
    _, _, new5_f1s_v4, _ = precision_recall_fscore_support(new5_y_test, new5_preds_v4, labels=new5_indices, zero_division=0)
    new5_macro_f1_v4 = float(np.mean(new5_f1s_v4))

    print("\n--- Subset Breakdowns (V4 11-Sign Model) ---")
    print(f"Original 6 Signs (N={len(orig6_y_test)}): Accuracy = {orig6_acc_v4*100:.2f}%, Macro F1 = {orig6_macro_f1_v4*100:.2f}%")
    print(f"New 5 Signs      (N={len(new5_y_test)}): Accuracy = {new5_acc_v4*100:.2f}%, Macro F1 = {new5_macro_f1_v4*100:.2f}%")

    # 6. Evaluate V3 Six-Sign Model on its test subset
    v3_ckpt_path = MODELS_DIR / "dynamic_bigru_v3_six_sign.pt"
    v3_map_path = MODELS_DIR / "dynamic_label_mapping_v3_six_sign.json"
    
    ckpt_v3 = torch.load(v3_ckpt_path, map_location="cpu", weights_only=False)
    with open(v3_map_path, "r", encoding="utf-8") as f:
        v3_label_mapping = json.load(f)

    # V3 architecture: num_classes=6
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

    model_v3 = DynamicSignBiGRUV3(input_dim=168, hidden_dim=64, num_layers=2, num_classes=6, dropout=0.3)
    model_v3.load_state_dict(ckpt_v3["model_state_dict"])
    model_v3.eval()

    v3_params = sum(p.numel() for p in model_v3.parameters() if p.requires_grad)
    v3_size_kb = round(os.path.getsize(v3_ckpt_path) / 1024.0, 1)
    v3_latency = measure_latency(model_v3, dummy_x, dummy_m, runs=200)

    # Evaluate V3 on the 6-sign test subset
    v3_label_to_id = {v: int(k) for k, v in v3_label_mapping.items()}
    # Filter 6-sign samples and map to V3 class IDs
    x_test_v3 = x_test[orig6_test_mask]
    m_test_v3 = m_test[orig6_test_mask]
    l_test_v3 = l_test[orig6_test_mask]
    y_test_v3 = np.array([v3_label_to_id[lbl] for lbl in l_test_v3])

    with torch.no_grad():
        logits_v3 = model_v3(torch.tensor(x_test_v3, dtype=torch.float32), torch.tensor(m_test_v3, dtype=torch.float32))
        probs_v3 = torch.softmax(logits_v3, dim=-1).cpu().numpy()
        preds_v3 = np.argmax(probs_v3, axis=-1)
        confs_v3 = np.max(probs_v3, axis=-1)

    v3_acc = float(accuracy_score(y_test_v3, preds_v3))
    v3_prec, v3_rec, v3_f1, v3_sup = precision_recall_fscore_support(y_test_v3, preds_v3, labels=list(range(6)), zero_division=0)
    v3_macro_f1 = float(np.mean(v3_f1))
    v3_weighted_f1 = float(precision_recall_fscore_support(y_test_v3, preds_v3, average='weighted', zero_division=0)[2])
    v3_mean_conf = float(np.mean(confs_v3) * 100.0)
    v3_above_70 = float(np.mean(confs_v3 >= 0.70) * 100.0)

    print("\n" + "=" * 80)
    print("V3 SIX-SIGN PRODUCTION MODEL ON SAME 6-SIGN SUBSET (39 SAMPLES)")
    print("=" * 80)
    print(f"V3 Accuracy (6 signs):  {v3_acc * 100:.2f}%")
    print(f"V3 Macro F1:            {v3_macro_f1 * 100:.2f}%")
    print(f"V3 Weighted F1:         {v3_weighted_f1 * 100:.2f}%")
    print(f"V3 Mean Confidence:     {v3_mean_conf:.2f}%")
    print(f"V3 >= 70% Confidence:   {v3_above_70:.2f}%")

    print("\n--- Head-to-Head Comparison: V3 vs V4 on Original 6 Signs ---")
    print(f"{'Metric':<25} | {'V3 (6-Sign Model)':<20} | {'V4 (11-Sign Model on 6 Signs)':<30}")
    print("-" * 80)
    print(f"{'Accuracy':<25} | {v3_acc*100:>18.2f}% | {orig6_acc_v4*100:>28.2f}%")
    print(f"{'Macro F1':<25} | {v3_macro_f1*100:>18.2f}% | {orig6_macro_f1_v4*100:>28.2f}%")
    print(f"{'Latency (Mean)':<25} | {v3_latency['mean_ms']:>18.2f}ms | {v4_latency['mean_ms']:>28.2f}ms")
    print(f"{'Latency (P95)':<25} | {v3_latency['p95_ms']:>18.2f}ms | {v4_latency['p95_ms']:>28.2f}ms")
    print(f"{'Model Parameters':<25} | {v3_params:>18,d} | {v4_params:>28,d}")

    # Plot Confusion Matrix
    plt.figure(figsize=(9, 8))
    plt.imshow(cm_v4, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("V4 11-Sign Confusion Matrix (Frozen Signer-Independent Test)")
    plt.colorbar()
    tick_marks = np.arange(len(VOCABULARY_11))
    plt.xticks(tick_marks, [s.upper() for s in VOCABULARY_11], rotation=45, ha='right')
    plt.yticks(tick_marks, [s.upper() for s in VOCABULARY_11])

    thresh = cm_v4.max() / 2.
    for i in range(cm_v4.shape[0]):
        for j in range(cm_v4.shape[1]):
            val = cm_v4[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, format(val, 'd'),
                     ha="center", va="center",
                     color=color, fontsize=10, fontweight='bold' if val > 0 else 'normal')

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = EVAL_DIR / "v4_11_sign_confusion_matrix.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"\nSaved confusion matrix plot to: {cm_path}")

    # Plot F1 Comparison Bar Chart
    plt.figure(figsize=(12, 5))
    x_pos = np.arange(len(VOCABULARY_11))
    f1_vals = [per_class_v4[s]["f1"] * 100 for s in VOCABULARY_11]
    colors = ['#1f77b4' if s in ORIGINAL_6 else '#ff7f0e' for s in VOCABULARY_11]

    bars = plt.bar(x_pos, f1_vals, color=colors, alpha=0.85, width=0.55, edgecolor='black')
    plt.xticks(x_pos, [s.upper() for s in VOCABULARY_11], rotation=30, ha='right')
    plt.ylabel('F1 Score (%)')
    plt.title('V4 11-Sign Per-Class F1 Scores (Blue = Original 6, Orange = New 5)')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 2, f'{h:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    f1_path = EVAL_DIR / "v4_11_sign_f1_comparison.png"
    plt.savefig(f1_path, dpi=180)
    plt.close()
    print(f"Saved F1 score bar plot to: {f1_path}")

    # Build Comparison JSON
    comparison_data = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset": {
            "total_test_samples": n_test,
            "unique_test_signers": unique_test_signers,
            "cross_split_signer_overlap": 0,
            "class_distribution_test": {s: int(np.sum(l_test == s)) for s in VOCABULARY_11}
        },
        "v4_11_sign_model": {
            "checkpoint": str(v4_ckpt_path.relative_to(BASE_DIR)),
            "parameters": v4_params,
            "model_size_kb": v4_size_kb,
            "latency": v4_latency,
            "overall_accuracy": round(v4_acc * 100, 2),
            "macro_precision": round(macro_prec * 100, 2),
            "macro_recall": round(macro_rec * 100, 2),
            "macro_f1": round(macro_f1 * 100, 2),
            "weighted_f1": round(weighted_f1 * 100, 2),
            "mean_confidence": round(mean_conf, 2),
            "percentage_above_70_conf": round(above_70, 2),
            "per_class_metrics": per_class_v4,
            "original_6_signs_performance": {
                "support": len(orig6_y_test),
                "accuracy": round(orig6_acc_v4 * 100, 2),
                "macro_f1": round(orig6_macro_f1_v4 * 100, 2)
            },
            "new_5_signs_performance": {
                "support": len(new5_y_test),
                "accuracy": round(new5_acc_v4 * 100, 2),
                "macro_f1": round(new5_macro_f1_v4 * 100, 2)
            },
            "confusion_pairs": confusion_report,
            "actual_confusions": actual_confusions,
            "confusion_matrix": cm_v4.tolist()
        },
        "v3_six_sign_model": {
            "checkpoint": str(v3_ckpt_path.relative_to(BASE_DIR)),
            "parameters": v3_params,
            "model_size_kb": v3_size_kb,
            "latency": v3_latency,
            "subset_support": len(y_test_v3),
            "accuracy": round(v3_acc * 100, 2),
            "macro_f1": round(v3_macro_f1 * 100, 2),
            "weighted_f1": round(v3_weighted_f1 * 100, 2),
            "mean_confidence": round(v3_mean_conf, 2),
            "percentage_above_70_conf": round(v3_above_70, 2)
        }
    }

    comp_path = EVAL_DIR / "v4_11_sign_model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"Saved full comparison data to: {comp_path}")

if __name__ == "__main__":
    main()
