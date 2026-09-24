# SignBridge AI — FastAPI Inference API Contract

**Service:** SignBridge AI ML Backend  
**Specification Version:** 1.1.0  
**Base URL (Development):** `http://127.0.0.1:8000`  
**Authentication:** None (Local Civic Terminal Deployment)  
**Content-Type:** `application/json`

---

## 1. System Health

### `GET /health`
Returns system operational status, model loading state, and active vocabulary size.

#### Request
```http
GET /health HTTP/1.1
Host: 127.0.0.1:8000
Accept: application/json
```

#### Response (200 OK)
```json
{
  "status": "ok",
  "dynamic_model": "loaded",
  "static_model": "loaded",
  "vocabulary_size": 18
}
```

---

## 2. Frozen Sign Vocabulary

### `GET /labels`
Returns the exact frozen 18-class civic sign vocabulary.

#### Request
```http
GET /labels HTTP/1.1
Host: 127.0.0.1:8000
Accept: application/json
```

#### Response (200 OK)
```json
{
  "vocabulary_size": 18,
  "classes": [
    "help",
    "doctor",
    "hospital",
    "sick",
    "appointment",
    "where",
    "bathroom",
    "yes",
    "no",
    "please",
    "thank_you",
    "wait",
    "understand",
    "problem",
    "money",
    "pay",
    "document",
    "letter_a"
  ]
}
```

---

## 3. Dynamic Sequence Prediction

### `POST /predict/sequence`
Consumes a 30-frame temporal landmark sequence and executes Bi-GRU classification across the 17 dynamic civic signs.

#### Request
```http
POST /predict/sequence HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
Accept: application/json

{
  "frames": [
    [... 150 float values ...],
    ... (30 frames total) ...
  ],
  "confidence_threshold": 0.70
}
```

- `frames`: Array of 30 frame feature vectors (each length 150) or flat array of 4500 float values.
  - Feature 0–63: Left Hand (63 normalized 3D coordinates + presence flag)
  - Feature 64–127: Right Hand (63 normalized 3D coordinates + presence flag)
  - Feature 128–149: Upper Body Pose (21 normalized 3D coordinates + presence flag)
- `confidence_threshold` *(optional)*: Float between 0.0 and 1.0 (default: 0.70).

#### Response: Confirmed Prediction (200 OK)
```json
{
  "label": "help",
  "class_id": 0,
  "confidence": 0.9124,
  "accepted": true,
  "top_k": [
    {
      "label": "help",
      "confidence": 0.9124
    },
    {
      "label": "please",
      "confidence": 0.0412
    },
    {
      "label": "wait",
      "confidence": 0.0215
    },
    {
      "label": "document",
      "confidence": 0.0108
    },
    {
      "label": "money",
      "confidence": 0.0064
    }
  ],
  "inference_latency_ms": 1.15
}
```

#### Response: Unaccepted Low-Confidence Prediction (200 OK)
When top confidence is below the configured threshold:
```json
{
  "label": null,
  "class_id": null,
  "confidence": 0.5421,
  "accepted": false,
  "top_k": [
    {
      "label": "hospital",
      "confidence": 0.5421
    },
    {
      "label": "doctor",
      "confidence": 0.3812
    }
  ],
  "inference_latency_ms": 1.08
}
```

#### Error Response: Invalid Dimension (422 Unprocessable Entity)
```json
{
  "detail": "Invalid landmark tensor structure: Expected 30 frames, got 15 frames"
}
```

---

## 4. Static Handshape Prediction

### `POST /predict/static`
Consumes a single-frame 64-feature hand vector to classify static tokens (specifically Class 17: `letter_a`).

#### Request
```http
POST /predict/static HTTP/1.1
Host: 127.0.0.1:8000
Content-Type: application/json
Accept: application/json

{
  "features": [... 64 float values ...],
  "confidence_threshold": 0.70
}
```

#### Response: Confirmed Token (200 OK)
```json
{
  "label": "letter_a",
  "class_id": 17,
  "confidence": 0.8842,
  "accepted": true,
  "inference_latency_ms": 0.35
}
```

#### Response: Non-Matching Handshape (200 OK)
```json
{
  "label": null,
  "class_id": null,
  "confidence": 0.1215,
  "accepted": false,
  "inference_latency_ms": 0.32
}
```

---

## 5. HTTP Status Code Conventions

| Status Code | Reason |
| :--- | :--- |
| `200 OK` | Valid request processed successfully |
| `422 Unprocessable Entity` | Payload validation failure (incorrect frame count, invalid feature length) |
| `500 Internal Server Error` | Model execution or unhandled runtime failure |
