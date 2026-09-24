# SignBridge AI — FastAPI ML Backend Service

High-performance real-time sign language recognition inference service for SignBridge AI (PS-09). Powered by FastAPI and PyTorch on CPU.

---

## 1. Overview

The backend ingests 30-frame normalized MediaPipe landmark sequences from the client browser and performs sub-millisecond inference using the trained 2-layer Bidirectional GRU model (`ml/models/dynamic_bigru_best.pt`) and static handshape MLP (`ml/models/static_mlp_best.pt`).

- **Architecture:** 2-layer Bi-GRU with dual masked temporal pooling (mean + max).
- **Latency:** **0.90 ms** mean model inference latency on CPU (< 1.1 ms p95).
- **Frozen Vocabulary:** 18 civic/emergency signs (17 dynamic + 1 static).
- **Input Contract:** 30 frames $\times$ 150 geometric landmark coordinates (no raw video uploads).

---

## 2. Environment Setup

### 2.1 Virtual Environment
```bash
# Create Python 3.10+ virtual environment
python3 -m venv backend/venv

# Activate environment
source backend/venv/bin/activate

# Install required dependencies
pip install -r backend/requirements.txt
```

### 2.2 Model Checkpoint Prerequisites
Ensure the trained PyTorch checkpoints are present:
- `ml/models/dynamic_bigru_best.pt` (~680 KB, 174,993 parameters)
- `ml/models/dynamic_label_mapping.json` (17 dynamic classes)
- `ml/models/static_mlp_best.pt` (~15 KB, 2,690 parameters)
- `ml/models/static_label_mapping.json` (2 classes: other vs. letter_a)
- `ml/config/vocabulary.json` (18 verified civic classes)

---

## 3. Backend Startup Command

### 3.1 Local Development
Run the FastAPI Uvicorn ASGI server locally:
```bash
./backend/venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Or run directly:
```bash
python backend/app/main.py
```

### 3.2 Production Deployment (Render)
Render dynamically assigns a port via the `$PORT` environment variable.
- **Render Build Command:**
  ```bash
  pip install -r backend/requirements.txt
  ```
- **Render Start Command:**
  ```bash
  uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT
  ```

---

## 4. API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System health check, model load status, vocabulary size |
| `GET` | `/labels` | Exact 18-class frozen civic vocabulary list and metadata |
| `POST` | `/predict/sequence` | Dynamic 30-frame landmark sequence classification |
| `POST` | `/predict/static` | Static 64-feature handshape classification (`letter_a`) |
| `GET` | `/docs` | Interactive Swagger UI API documentation |

---

## 5. Model Loading

Models are loaded **ONCE** during application startup using FastAPI's modern `lifespan` context manager in [backend/app/main.py](file:///Users/saravanarajaram0411/CLG/KPR/backend/app/main.py). 

- Model checkpoints are kept in memory and shared across requests.
- All inference runs inside `torch.inference_mode()` with gradients disabled.
- Execution is pinned to CPU by default for broad server compatibility.

---

## 6. Confidence Thresholding

The default confidence threshold is **0.70** (70%).
- If the top prediction has confidence $\ge 0.70$:
  `{"label": "help", "class_id": 0, "confidence": 0.91, "accepted": true, ...}`
- If the top prediction has confidence $< 0.70$:
  `{"label": null, "class_id": null, "confidence": 0.52, "accepted": false, ...}`

Clients may optionally override the threshold per-request:
```json
{
  "frames": [...],
  "confidence_threshold": 0.60
}
```
Global default can also be set via the environment variable:
```bash
export SIGNBRIDGE_CONFIDENCE_THRESHOLD=0.70
```

---

## 7. Temporal Stabilization

To prevent spurious flicker and duplicate message spam:
1. **Consecutive Consistency Rule:** The backend and client adapters track recent prediction windows, requiring multiple consecutive matching windows before accepting a sign.
2. **Debounce / Cooldown:** Once a sign is confirmed and dispatched, an automatic cooldown window (3.5 seconds) prevents the same physical sign from generating rapid duplicate chat entries.

---

## 8. Frontend Environment Variable

The React/Vite frontend reads the API base URL from `.env`:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```
In JavaScript:
```javascript
import { API_ENDPOINTS } from './services/apiConfig.js';
// Uses ${VITE_API_BASE_URL}/predict/sequence
```

---

## 9. Running Automated Tests

Run the comprehensive pytest contract test suite:
```bash
./backend/venv/bin/pytest backend/tests/test_ml_api.py -v
```

All 10 contract tests validate:
- Health and label endpoints
- Valid and invalid tensor dimensions (30x150, flat 4500)
- Static handshape feature dimensions (64)
- Confidence threshold filtering
- Malformed request handling (HTTP 422)

---

## 10. Troubleshooting

| Issue | Cause | Resolution |
| :--- | :--- | :--- |
| `Address already in use` | Another process is on port 8000 | Kill existing process: `lsof -ti :8000 \| xargs kill -9` |
| `Checkpoint missing` | Model weights not at expected path | Ensure `ml/models/dynamic_bigru_best.pt` exists |
| `AI recognition unavailable` in UI | FastAPI server not started or wrong port | Start uvicorn on port 8000 and check `GET /health` |
| `CORS Error` in Browser Console | Origin mismatch | Set `CORS_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"` |
| `HTTP 422 Unprocessable Entity` | Incorrect landmark shape | Ensure input is exactly 30 frames $\times$ 150 features |
