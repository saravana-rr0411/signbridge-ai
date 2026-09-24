"""
SignBridge AI - FastAPI ML Backend Service
Provides real-time CPU inference for the 18-class civic sign vocabulary.
"""

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
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

# CORS Configuration: allow production Vercel frontend and local development origins
DEFAULT_ALLOWED_ORIGINS = [
    "https://signbridge-ai-kappa.vercel.app",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

allowed_origins_env = os.getenv("CORS_ORIGINS", "")
extra_origins = [orig.strip() for orig in allowed_origins_env.split(",") if orig.strip()]
allowed_origins = list(dict.fromkeys(DEFAULT_ALLOWED_ORIGINS + extra_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
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


# ============================================================================
# REAL-TIME WEBSOCKET RELAY: Cross-Device Signaling & Communication Bus
# ============================================================================

class ConnectionManager:
    """
    Manages active WebSockets segregated by room_id and client_role ('deaf' or 'admin').
    Ensures exactly one active connection per role per room, gracefully replacing duplicates
    and cleanly relaying messages to the opposite peer.
    """
    def __init__(self):
        self.rooms: dict[str, dict[str, WebSocket]] = {}

    async def connect(self, room_id: str, client_role: str, websocket: WebSocket):
        await websocket.accept()
        role = client_role.lower().strip()
        room = self.rooms.setdefault(room_id, {})

        # Handle duplicate same-role connection gracefully without breaking the room
        existing_ws = room.get(role)
        if existing_ws is not None and existing_ws != websocket:
            try:
                await existing_ws.close(code=1000, reason="Replaced by new connection")
            except Exception:
                pass

        room[role] = websocket

        # Notify the opposite peer that this role has connected
        opposite_role = "admin" if role == "deaf" else "deaf"
        opposite_ws = room.get(opposite_role)
        if opposite_ws is not None:
            try:
                await opposite_ws.send_json({
                    "roomId": room_id,
                    "senderRole": "system",
                    "targetRole": opposite_role,
                    "type": "PEER_CONNECTED",
                    "payload": {"connectedRole": role},
                    "timestamp": int(time.time() * 1000)
                })
            except Exception:
                pass

    def disconnect(self, room_id: str, client_role: str, websocket: WebSocket):
        role = client_role.lower().strip()
        if room_id in self.rooms:
            if self.rooms[room_id].get(role) == websocket:
                del self.rooms[room_id][role]
            if not self.rooms[room_id]:
                del self.rooms[room_id]

    async def notify_disconnect(self, room_id: str, client_role: str):
        role = client_role.lower().strip()
        opposite_role = "admin" if role == "deaf" else "deaf"
        if room_id in self.rooms:
            opposite_ws = self.rooms[room_id].get(opposite_role)
            if opposite_ws is not None:
                try:
                    await opposite_ws.send_json({
                        "roomId": room_id,
                        "senderRole": "system",
                        "targetRole": opposite_role,
                        "type": "PEER_DISCONNECTED",
                        "payload": {"disconnectedRole": role},
                        "timestamp": int(time.time() * 1000)
                    })
                except Exception:
                    pass

    async def relay_message(self, room_id: str, sender_role: str, data: dict):
        role = sender_role.lower().strip()
        opposite_role = "admin" if role == "deaf" else "deaf"
        room = self.rooms.get(room_id)
        if not room:
            return False

        target_ws = room.get(opposite_role)
        if target_ws is not None:
            try:
                await target_ws.send_json(data)
                return True
            except Exception as e:
                print(f"[SignBridge WS] Error relaying message to {opposite_role}: {e}", file=sys.stderr)
                return False
        return False


manager = ConnectionManager()


@app.websocket("/ws/relay/{room_id}/{client_role}")
async def websocket_relay_endpoint(websocket: WebSocket, room_id: str, client_role: str):
    """
    Bi-directional cross-device WebSocket relay for SignBridge AI.
    Relays contextual sign messages, Admin responses, and WebRTC SDP/ICE signaling
    between exactly one Deaf client and one Admin client in the room.
    """
    role = client_role.lower().strip()
    if role not in ("deaf", "admin"):
        await websocket.close(code=4003, reason="Invalid role. Must be 'deaf' or 'admin'.")
        return

    await manager.connect(room_id, role, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                continue

            envelope = {
                "roomId": room_id,
                "senderRole": role,
                "targetRole": "admin" if role == "deaf" else "deaf",
                "type": data.get("type", "UNKNOWN"),
                "payload": data.get("payload", {}),
                "timestamp": data.get("timestamp") or int(time.time() * 1000)
            }
            await manager.relay_message(room_id, role, envelope)
    except WebSocketDisconnect:
        manager.disconnect(room_id, role, websocket)
        await manager.notify_disconnect(room_id, role)
    except Exception:
        manager.disconnect(room_id, role, websocket)
        await manager.notify_disconnect(room_id, role)

