"""
SignBridge AI - Machine Learning Package
Provides ModelManager, schemas, and preprocessing pipelines.
"""

from .schemas import (
    HealthResponse,
    LabelsResponse,
    SequencePredictionRequest,
    SequencePredictionResponse,
    StaticPredictionRequest,
    StaticPredictionResponse,
    PredictionTopK,
)
from .preprocessing import (
    validate_and_format_sequence,
    validate_and_format_static,
    normalize_hand_landmarks_array,
    normalize_pose_landmarks_array,
)
from .inference import (
    model_manager,
    ModelManager,
    DynamicSignBiGRU,
    StaticHandshapeMLP,
    PredictionStabilizer,
)

__all__ = [
    "HealthResponse",
    "LabelsResponse",
    "SequencePredictionRequest",
    "SequencePredictionResponse",
    "StaticPredictionRequest",
    "StaticPredictionResponse",
    "PredictionTopK",
    "validate_and_format_sequence",
    "validate_and_format_static",
    "normalize_hand_landmarks_array",
    "normalize_pose_landmarks_array",
    "model_manager",
    "ModelManager",
    "DynamicSignBiGRU",
    "StaticHandshapeMLP",
    "PredictionStabilizer",
]
