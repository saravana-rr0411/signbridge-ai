"""
SignBridge AI - FastAPI ML Backend Service
Provides real-time CPU inference for the 18-class civic sign vocabulary.
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.app.ml.schemas import (
    HealthResponse,
    LabelsResponse,
    LabelsResponseV3,
    LabelsResponseV6,
    SequencePredictionRequest,
    SequencePredictionRequestV3,
    SequencePredictionRequestV6,
    SequencePredictionResponse,
    StaticPredictionRequest,
    StaticPredictionResponse,
)
from backend.app.ml.inference import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads PyTorch models once at application startup."""
    try:
        model_manager.load_models()
        print("[SignBridge FastAPI] All models initialized successfully.")
    except Exception as e:
        print(f"[SignBridge FastAPI] Warning during model loading: {e}", file=sys.stderr)
    yield
    print("[SignBridge FastAPI] Shutting down service.")


app = FastAPI(
    title="SignBridge AI Real-Time Recognition Service",
    description="Accessibility-First Sign Language Communication Bridge — PS-09 ML Backend",
    version="1.1.0",
    lifespan=lifespan
)

# CORS Configuration: allow Vite frontend origins
allowed_origins_env = os.getenv("CORS_ORIGINS")
if allowed_origins_env:
    allowed_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
else:
    allowed_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns system status, model loading state, and vocabulary size.
    """
    dyn_loaded = "loaded" if (model_manager.is_loaded and model_manager.dynamic_model is not None) else "unloaded"
    stat_loaded = "loaded" if (model_manager.is_loaded and model_manager.static_model is not None) else "unloaded"
    
    return HealthResponse(
        status="ok",
        dynamic_model=dyn_loaded,
        static_model=stat_loaded,
        vocabulary_size=len(model_manager.vocabulary) if model_manager.vocabulary else 18
    )


@app.get("/labels", response_model=LabelsResponse)
async def get_labels():
    """
    Returns the exact frozen 18-class civic sign vocabulary.
    """
    if not model_manager.is_loaded:
        model_manager.load_models()
        
    return LabelsResponse(
        vocabulary_size=len(model_manager.vocabulary),
        classes=model_manager.vocabulary,
        details=model_manager.vocabulary_details
    )


@app.get("/labels/v3-six-sign", response_model=LabelsResponseV3)
async def get_labels_v3_six_sign():
    """
    Returns the focused 6-sign vocabulary for candidate model V3.
    """
    if not model_manager.is_loaded:
        model_manager.load_models()

    return LabelsResponseV3(
        vocabulary_size=len(model_manager.vocabulary_v3_six_sign),
        classes=model_manager.vocabulary_v3_six_sign,
        model_version="v3_six_sign"
    )


@app.get("/labels/v6-10-sign", response_model=LabelsResponseV6)
async def get_labels_v6_10_sign():
    """
    Returns the 10-sign vocabulary for the experimental model V6.
    """
    if not model_manager.is_loaded:
        model_manager.load_models()

    return LabelsResponseV6(
        vocabulary_size=len(model_manager.vocabulary_v6_10_sign),
        classes=model_manager.vocabulary_v6_10_sign,
        model_version="v6_10_sign"
    )


@app.post("/predict/sequence", response_model=SequencePredictionResponse)
async def predict_sequence(req: SequencePredictionRequest):
    """
    Consumes a 30-frame x 150-feature landmark sequence and returns the predicted sign.
    Applies configurable confidence thresholding (default: 0.70).
    """
    if not model_manager.is_loaded:
        model_manager.load_models()

    try:
        result = model_manager.predict_sequence(
            raw_frames=req.frames,
            threshold=req.confidence_threshold
        )
        return SequencePredictionResponse(**result)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid landmark tensor structure: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )


@app.post("/predict/sequence/v3-six-sign", response_model=SequencePredictionResponse)
async def predict_sequence_v3_six_sign(req: SequencePredictionRequestV3):
    """
    Consumes a 30-frame x 168-feature landmark sequence and returns prediction from the focused 6-sign V3 model.
    Applies configurable confidence thresholding (default: 0.70).
    """
    if not model_manager.is_loaded:
        model_manager.load_models()

    try:
        result = model_manager.predict_sequence_v3_six_sign(
            raw_frames=req.frames,
            threshold=req.confidence_threshold
        )
        return SequencePredictionResponse(**result)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid landmark tensor structure for V3: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"V3 inference error: {str(exc)}"
        )


@app.post("/predict/sequence/v6-10-sign", response_model=SequencePredictionResponse)
async def predict_sequence_v6_10_sign(req: SequencePredictionRequestV6):
    """
    Consumes a 30-frame x 168-feature landmark sequence and returns prediction from the experimental 10-sign V6 model.
    Applies configurable confidence thresholding (default: 0.70).
    """
    if not model_manager.is_loaded:
        model_manager.load_models()

    try:
        result = model_manager.predict_sequence_v6_10_sign(
            raw_frames=req.frames,
            threshold=req.confidence_threshold
        )
        return SequencePredictionResponse(**result)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid landmark tensor structure for V6: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"V6 inference error: {str(exc)}"
        )


@app.post("/predict/static", response_model=StaticPredictionResponse)
async def predict_static(req: StaticPredictionRequest):
    """
    Consumes a single-frame 64-feature hand landmark vector to classify static 'letter_a'.
    """
    if not model_manager.is_loaded:
        model_manager.load_models()

    try:
        result = model_manager.predict_static(
            raw_features=req.features,
            threshold=req.confidence_threshold
        )
        return StaticPredictionResponse(**result)
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid static feature dimension: {str(val_err)}"
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Static inference error: {str(exc)}"
        )
