"""
SignBridge AI - Pydantic Request & Response Schemas
Defines API data contracts for /health, /labels, /predict/sequence, and /predict/static.
"""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str = "ok"
    dynamic_model: str = "loaded"
    static_model: str = "loaded"
    vocabulary_size: int = 18


class LabelsResponse(BaseModel):
    vocabulary_size: int = 18
    classes: List[str]
    details: Optional[List[Dict[str, Any]]] = None


class PredictionTopK(BaseModel):
    label: str
    confidence: float


class SequencePredictionRequest(BaseModel):
    """
    Input sequence of 30 frames, each with 150 landmark features.
    Accepts:
    - 30x150 nested list (30 frames, each 150 floats)
    - Flat list of 4500 floats (automatically reshaped to 30x150)
    """
    frames: List[Any] = Field(..., description="30x150 landmark frames or flat 4500 list")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for confidence threshold")

    @field_validator("frames")
    @classmethod
    def validate_frames_structure(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("frames must be a non-empty list")
        
        # Check flat 4500
        if len(v) == 4500 and all(isinstance(x, (int, float)) for x in v[:10]):
            # Valid flat shape
            return v
        
        # Check nested 30 frames
        if len(v) != 30:
            raise ValueError(f"Expected 30 frames, got {len(v)} frames")
        
        for idx, frame in enumerate(v):
            if not isinstance(frame, list):
                raise ValueError(f"Frame {idx} must be a list of 150 float features")
            if len(frame) != 150:
                raise ValueError(f"Frame {idx} must have exactly 150 features, got {len(frame)}")
        return v


class SequencePredictionResponse(BaseModel):
    label: Optional[str] = None
    class_id: Optional[int] = None
    confidence: float
    accepted: bool
    top_k: List[PredictionTopK] = []
    inference_latency_ms: Optional[float] = None
    tensor_shapes: Optional[Dict[str, Any]] = None
    probability_sum: Optional[float] = None
    raw_prediction: Optional[str] = None
    raw_class_id: Optional[int] = None


class SequencePredictionRequestV3(BaseModel):
    """
    Input sequence of 30 frames, each with 168 landmark features for the focused 6-sign model.
    Accepts:
    - 30x168 nested list (30 frames, each 168 floats)
    - Flat list of 5040 floats (automatically reshaped to 30x168)
    """
    frames: List[Any] = Field(..., description="30x168 landmark frames or flat 5040 list")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for confidence threshold")

    @field_validator("frames")
    @classmethod
    def validate_frames_structure(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("frames must be a non-empty list")

        # Check flat 5040
        if len(v) == 5040 and all(isinstance(x, (int, float)) for x in v[:10]):
            return v

        # Check nested 30 frames
        if len(v) != 30:
            raise ValueError(f"Expected 30 frames, got {len(v)} frames")

        for idx, frame in enumerate(v):
            if not isinstance(frame, list):
                raise ValueError(f"Frame {idx} must be a list of 168 float features")
            if len(frame) != 168:
                raise ValueError(f"Frame {idx} must have exactly 168 features, got {len(frame)}")
        return v


class LabelsResponseV3(BaseModel):
    vocabulary_size: int = 6
    classes: List[str]
    model_version: str = "v3_six_sign"


class SequencePredictionRequestV6(BaseModel):
    """
    Input sequence of 30 frames, each with 168 landmark features for the experimental 10-sign V6 model.
    Accepts:
    - 30x168 nested list (30 frames, each 168 floats)
    - Flat list of 5040 floats (automatically reshaped to 30x168)
    """
    frames: List[Any] = Field(..., description="30x168 landmark frames or flat 5040 list")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for confidence threshold")

    @field_validator("frames")
    @classmethod
    def validate_frames_structure(cls, v):
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError("frames must be a non-empty list")

        # Check flat 5040
        if len(v) == 5040 and all(isinstance(x, (int, float)) for x in v[:10]):
            return v

        # Check nested 30 frames
        if len(v) != 30:
            raise ValueError(f"Expected 30 frames, got {len(v)} frames")

        for idx, frame in enumerate(v):
            if not isinstance(frame, list):
                raise ValueError(f"Frame {idx} must be a list of 168 float features")
            if len(frame) != 168:
                raise ValueError(f"Frame {idx} must have exactly 168 features, got {len(frame)}")
        return v


class LabelsResponseV6(BaseModel):
    vocabulary_size: int = 10
    classes: List[str]
    model_version: str = "v6_10_sign"



class StaticPredictionRequest(BaseModel):
    """
    Single-frame 64-dimensional feature vector for static handshape classification.
    """
    features: List[float] = Field(..., description="64-element normalized hand landmark feature vector")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        if not isinstance(v, list) or len(v) != 64:
            raise ValueError(f"Static features must have exactly 64 elements, got {len(v) if isinstance(v, list) else type(v)}")
        return v


class StaticPredictionResponse(BaseModel):
    label: Optional[str] = None
    class_id: Optional[int] = None
    confidence: float
    accepted: bool
    inference_latency_ms: Optional[float] = None
