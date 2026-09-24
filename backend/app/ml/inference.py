"""
SignBridge AI - PyTorch Inference Engine & Model Manager
Loads trained Bi-GRU (dynamic) and MLP (static) checkpoints once on application startup.
Runs deterministic CPU inference using torch.inference_mode().
Provides configurable confidence thresholding and temporal stabilization.
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from .preprocessing import (
    validate_and_format_sequence,
    validate_and_format_sequence_v3,
    validate_and_format_static,
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
ML_DIR = BASE_DIR / "ml"
MODELS_DIR = ML_DIR / "models"
CONFIG_DIR = ML_DIR / "config"

DEFAULT_CONFIDENCE_THRESHOLD = float(os.getenv("SIGNBRIDGE_CONFIDENCE_THRESHOLD", "0.70"))
DEFAULT_STABILIZATION_WINDOW = int(os.getenv("SIGNBRIDGE_STABILIZATION_WINDOW", "3"))
DEFAULT_COOLDOWN_SECONDS = float(os.getenv("SIGNBRIDGE_COOLDOWN_SECONDS", "3.0"))


class DynamicSignBiGRU(nn.Module):
    """
    2-layer Bidirectional GRU with masked dual temporal pooling.
    Matches ml/models/dynamic_bigru_best.pt architecture.
    """
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
        self.pool_dim = hidden_dim * 4  # Bi-directional (hidden*2) * 2 pools (mean + max) = 256
        self.fc1 = nn.Linear(self.pool_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
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


class StaticHandshapeMLP(nn.Module):
    """
    3-layer MLP classifier for static handshapes.
    Matches ml/models/static_mlp_best.pt architecture.
    """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionStabilizer:
    """
    Temporal stabilizer preventing flicker and duplicate emission:
    1. Requires N consecutive consistent predictions.
    2. Implements a cooldown window to prevent continuous duplicate messages.
    """
    def __init__(self, window_size: int = DEFAULT_STABILIZATION_WINDOW, cooldown_sec: float = DEFAULT_COOLDOWN_SECONDS):
        self.window_size = window_size
        self.cooldown_sec = cooldown_sec
        self.history = deque(maxlen=window_size)
        self.last_emitted_sign: Optional[str] = None
        self.last_emitted_time: float = 0.0

    def push_and_check(self, label: Optional[str], confidence: float, accepted: bool) -> Tuple[bool, Optional[str]]:
        now = time.time()
        if not accepted or not label:
            self.history.clear()
            return False, None

        self.history.append(label)

        # Check if window is full and all entries match
        if len(self.history) == self.window_size and all(s == label for s in self.history):
            # Check cooldown against last emitted sign
            if label == self.last_emitted_sign and (now - self.last_emitted_time) < self.cooldown_sec:
                # Suppressed by cooldown
                return False, label

            self.last_emitted_sign = label
            self.last_emitted_time = now
            return True, label

        return False, label

    def reset(self):
        self.history.clear()
        self.last_emitted_sign = None
        self.last_emitted_time = 0.0


class ModelManager:
    """
    Singleton Manager for SignBridge AI PyTorch Models.
    Loaded ONCE at startup on CPU.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.dynamic_model: Optional[DynamicSignBiGRU] = None
        self.static_model: Optional[StaticHandshapeMLP] = None
        
        self.dynamic_label_map: Dict[int, str] = {}
        self.static_label_map: Dict[int, str] = {}
        self.vocabulary: List[str] = []
        self.vocabulary_details: List[Dict[str, Any]] = []

        # Isolated V3 Six-Sign Model components
        self.dynamic_model_v3_six_sign: Optional[DynamicSignBiGRU] = None
        self.dynamic_label_map_v3_six_sign: Dict[int, str] = {}
        self.vocabulary_v3_six_sign: List[str] = ["help", "yes", "no", "thank_you", "please", "hello"]

        self.confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD
        self.stabilizer = PredictionStabilizer()
        self.is_loaded = False

    def load_models(self):
        """Loads checkpoints and label mappings into memory once."""
        if self.is_loaded:
            return

        print("[ModelManager] Loading SignBridge AI ML models on CPU...")

        # 1. Load Vocabulary Config
        vocab_path = CONFIG_DIR / "vocabulary.json"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                self.vocabulary_details = vocab_data.get("classes", [])
                self.vocabulary = [c["label"] for c in self.vocabulary_details]
        else:
            # Fallback exact 18-class list
            self.vocabulary = [
                "help", "doctor", "hospital", "sick", "appointment", "where",
                "bathroom", "yes", "no", "please", "thank_you", "wait",
                "understand", "problem", "money", "pay", "document", "letter_a"
            ]

        # 2. Load Dynamic Bi-GRU Checkpoint (V2 Production Model)
        dynamic_ckpt_path = MODELS_DIR / "dynamic_bigru_v2.pt"
        if not dynamic_ckpt_path.exists():
            # Graceful fallback to V1 if V2 is not found
            dynamic_ckpt_path = MODELS_DIR / "dynamic_bigru_best.pt"
        if not dynamic_ckpt_path.exists():
            raise FileNotFoundError(f"Dynamic model checkpoint missing at {dynamic_ckpt_path}")

        print(f"[ModelManager] Loading production dynamic model from: {dynamic_ckpt_path.name}")
        dynamic_ckpt = torch.load(dynamic_ckpt_path, map_location=self.device)
        num_dynamic_classes = dynamic_ckpt.get("num_classes", 17)
        input_dim = dynamic_ckpt.get("input_dim", 150)
        hidden_dim = dynamic_ckpt.get("hidden_dim", 64)

        self.dynamic_model = DynamicSignBiGRU(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=2,
            num_classes=num_dynamic_classes,
            dropout=0.0
        )
        self.dynamic_model.load_state_dict(dynamic_ckpt["model_state_dict"])
        self.dynamic_model.to(self.device)
        self.dynamic_model.eval()

        # Load dynamic label mapping V2 if available, fallback to checkpoint
        map_v2_path = MODELS_DIR / "dynamic_label_mapping_v2.json"
        if map_v2_path.exists():
            with open(map_v2_path, "r", encoding="utf-8") as f:
                raw_dyn_map = json.load(f)
        else:
            raw_dyn_map = dynamic_ckpt.get("label_mapping", {})
        self.dynamic_label_map = {int(k): str(v) for k, v in raw_dyn_map.items()}

        # 3. Load Static MLP Checkpoint
        static_ckpt_path = MODELS_DIR / "static_mlp_best.pt"
        if not static_ckpt_path.exists():
            raise FileNotFoundError(f"Static model checkpoint missing at {static_ckpt_path}")

        static_ckpt = torch.load(static_ckpt_path, map_location=self.device)
        static_num_classes = static_ckpt.get("num_classes", 2)
        static_input_dim = static_ckpt.get("input_dim", 64)

        self.static_model = StaticHandshapeMLP(
            input_dim=static_input_dim,
            hidden_dim=32,
            num_classes=static_num_classes,
            dropout=0.0
        )
        self.static_model.load_state_dict(static_ckpt["model_state_dict"])
        self.static_model.to(self.device)
        self.static_model.eval()

        raw_static_map = static_ckpt.get("label_mapping", {})
        self.static_label_map = {int(k): str(v) for k, v in raw_static_map.items()}

        # 4. Load Isolated V3 Six-Sign Bi-GRU Checkpoint (Candidate Model, 168 dims, 6 classes)
        v3_ckpt_path = MODELS_DIR / "dynamic_bigru_v3_six_sign.pt"
        v3_map_path = MODELS_DIR / "dynamic_label_mapping_v3_six_sign.json"
        if v3_ckpt_path.exists() and v3_map_path.exists():
            print(f"[ModelManager] Loading isolated V3 six-sign model from: {v3_ckpt_path.name}")
            v3_ckpt = torch.load(v3_ckpt_path, map_location=self.device, weights_only=False)
            with open(v3_map_path, "r", encoding="utf-8") as f:
                raw_v3_map = json.load(f)
            self.dynamic_label_map_v3_six_sign = {int(k): str(v) for k, v in raw_v3_map.items()}
            self.vocabulary_v3_six_sign = [self.dynamic_label_map_v3_six_sign[i] for i in range(len(self.dynamic_label_map_v3_six_sign))]

            self.dynamic_model_v3_six_sign = DynamicSignBiGRU(
                input_dim=168,
                hidden_dim=64,
                num_layers=2,
                num_classes=len(self.dynamic_label_map_v3_six_sign),
                dropout=0.0
            )
            self.dynamic_model_v3_six_sign.load_state_dict(v3_ckpt["model_state_dict"])
            self.dynamic_model_v3_six_sign.to(self.device)
            self.dynamic_model_v3_six_sign.eval()
            print(f"[ModelManager] V3 Six-Sign model ready: {len(self.vocabulary_v3_six_sign)} classes ({', '.join(self.vocabulary_v3_six_sign)}).")

        self.is_loaded = True
        print(f"[ModelManager] Successfully initialized: {len(self.dynamic_label_map)} dynamic classes, {len(self.static_label_map)} static classes. Active vocabulary: {len(self.vocabulary)} items.")

    def predict_sequence(self, raw_frames: List[Any], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs 30-frame sequence inference.
        Returns: label, class_id, confidence, accepted, top_k, latency_ms.
        """
        if not self.is_loaded or self.dynamic_model is None:
            raise RuntimeError("Dynamic Bi-GRU model is not loaded.")

        eff_threshold = threshold if threshold is not None else self.confidence_threshold

        t0 = time.perf_counter()
        arr, mask = validate_and_format_sequence(raw_frames)

        x_t = torch.from_numpy(arr).to(self.device)
        m_t = torch.from_numpy(mask).to(self.device)

        with torch.inference_mode():
            logits = self.dynamic_model(x_t, m_t)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        probs_np = probs.cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_indices = np.argsort(probs_np)[::-1]
        top_cid = int(top_indices[0])
        top_conf = float(probs_np[top_cid])

        top_k = []
        for idx in top_indices[:5]:
            cid = int(idx)
            lbl = self.dynamic_label_map.get(cid, f"class_{cid}")
            top_k.append({
                "label": lbl,
                "confidence": round(float(probs_np[cid]), 4)
            })

        predicted_label = self.dynamic_label_map.get(top_cid, f"class_{top_cid}")
        accepted = bool(top_conf >= eff_threshold)
        prob_sum = float(np.sum(probs_np))

        return {
            "label": predicted_label if accepted else None,
            "class_id": top_cid if accepted else None,
            "raw_prediction": predicted_label,
            "raw_class_id": top_cid,
            "confidence": round(top_conf, 4),
            "accepted": accepted,
            "top_k": top_k,
            "inference_latency_ms": round(latency_ms, 2),
            "tensor_shapes": {
                "frontend": [30, 150],
                "backend_received": [int(arr.shape[1]), int(arr.shape[2])],
                "pytorch_tensor": list(x_t.shape),
                "model_output": list(logits.shape)
            },
            "probability_sum": round(prob_sum, 4)
        }

    def predict_sequence_v3_six_sign(self, raw_frames: List[Any], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs 30-frame sequence inference for the focused 6-sign V3 model (30x168).
        Returns: label, class_id, confidence, accepted, top_k, latency_ms.
        """
        if not self.is_loaded or self.dynamic_model_v3_six_sign is None:
            raise RuntimeError("V3 six-sign Bi-GRU model is not loaded.")

        eff_threshold = threshold if threshold is not None else self.confidence_threshold

        t0 = time.perf_counter()
        arr, mask = validate_and_format_sequence_v3(raw_frames)

        x_t = torch.from_numpy(arr).to(self.device)
        m_t = torch.from_numpy(mask).to(self.device)

        with torch.inference_mode():
            logits = self.dynamic_model_v3_six_sign(x_t, m_t)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        probs_np = probs.cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        top_indices = np.argsort(probs_np)[::-1]
        top_cid = int(top_indices[0])
        top_conf = float(probs_np[top_cid])

        top_k = []
        for idx in top_indices:
            cid = int(idx)
            lbl = self.dynamic_label_map_v3_six_sign.get(cid, f"class_{cid}")
            top_k.append({
                "label": lbl,
                "confidence": round(float(probs_np[cid]), 4)
            })

        predicted_label = self.dynamic_label_map_v3_six_sign.get(top_cid, f"class_{top_cid}")
        accepted = bool(top_conf >= eff_threshold)
        prob_sum = float(np.sum(probs_np))

        return {
            "label": predicted_label if accepted else None,
            "class_id": top_cid if accepted else None,
            "raw_prediction": predicted_label,
            "raw_class_id": top_cid,
            "confidence": round(top_conf, 4),
            "accepted": accepted,
            "top_k": top_k,
            "inference_latency_ms": round(latency_ms, 2),
            "tensor_shapes": {
                "frontend": [30, 168],
                "backend_received": [int(arr.shape[1]), int(arr.shape[2])],
                "pytorch_tensor": list(x_t.shape),
                "model_output": list(logits.shape)
            },
            "probability_sum": round(prob_sum, 4)
        }

    def predict_static(self, raw_features: List[float], threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Runs single-frame static handshape classification (specifically letter_a).
        """
        if not self.is_loaded or self.static_model is None:
            raise RuntimeError("Static MLP model is not loaded.")

        eff_threshold = threshold if threshold is not None else self.confidence_threshold

        t0 = time.perf_counter()
        arr = validate_and_format_static(raw_features)
        x_t = torch.from_numpy(arr).to(self.device)

        with torch.inference_mode():
            logits = self.static_model(x_t)
            probs = torch.softmax(logits, dim=1).squeeze(0)

        probs_np = probs.cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Class 1 is 'letter_a', class 0 is 'other'
        a_conf = float(probs_np[1])
        accepted = bool(a_conf >= eff_threshold)

        return {
            "label": "letter_a" if accepted else None,
            "class_id": 17 if accepted else None,
            "confidence": round(a_conf, 4),
            "accepted": accepted,
            "inference_latency_ms": round(latency_ms, 2)
        }


# Global Singleton Instance
model_manager = ModelManager()
