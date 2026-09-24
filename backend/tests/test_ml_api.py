"""
SignBridge AI - FastAPI ML Backend Contract Tests
Automated API tests for /health, /labels, /predict/sequence, and /predict/static.
Uses deterministic synthetic landmark tensors for API contract and shape validation.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure workspace root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.main import app
from backend.app.ml.inference import model_manager

EXPECTED_18_CLASSES = [
    "help", "doctor", "hospital", "sick", "appointment", "where",
    "bathroom", "yes", "no", "please", "thank_you", "wait",
    "understand", "problem", "money", "pay", "document", "letter_a"
]


@pytest.fixture(scope="module")
def client():
    # Trigger model loading
    model_manager.load_models()
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns loaded status and vocabulary size 18."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["dynamic_model"] == "loaded"
    assert data["static_model"] == "loaded"
    assert data["vocabulary_size"] == 18


def test_labels_endpoint(client):
    """Test GET /labels returns the exact 18-class frozen vocabulary."""
    response = client.get("/labels")
    assert response.status_code == 200
    data = response.json()
    assert data["vocabulary_size"] == 18
    assert len(data["classes"]) == 18
    for cls_name in EXPECTED_18_CLASSES:
        assert cls_name in data["classes"], f"Missing expected class {cls_name}"


def test_predict_sequence_valid_shape(client):
    """Test POST /predict/sequence with valid 30x150 deterministic synthetic features."""
    synthetic_sequence = [[0.05] * 150 for _ in range(30)]
    payload = {
        "frames": synthetic_sequence,
        "confidence_threshold": 0.50
    }
    response = client.post("/predict/sequence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data
    assert "accepted" in data
    assert "top_k" in data
    assert isinstance(data["top_k"], list)
    assert len(data["top_k"]) > 0
    assert "inference_latency_ms" in data


def test_predict_sequence_flat_shape(client):
    """Test POST /predict/sequence with flat 4500 features (30x150)."""
    flat_sequence = [0.02] * 4500
    payload = {
        "frames": flat_sequence,
        "confidence_threshold": 0.50
    }
    response = client.post("/predict/sequence", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data
    assert "top_k" in data


def test_predict_sequence_invalid_shape(client):
    """Test POST /predict/sequence with invalid temporal length (15 frames instead of 30)."""
    invalid_sequence = [[0.1] * 150 for _ in range(15)]
    payload = {"frames": invalid_sequence}
    response = client.post("/predict/sequence", json=payload)
    assert response.status_code == 422


def test_predict_sequence_invalid_feature_dim(client):
    """Test POST /predict/sequence with invalid feature dimension (30 frames x 80 features)."""
    invalid_sequence = [[0.1] * 80 for _ in range(30)]
    payload = {"frames": invalid_sequence}
    response = client.post("/predict/sequence", json=payload)
    assert response.status_code == 422


def test_predict_static_valid_shape(client):
    """Test POST /predict/static with valid 64-feature static hand vector."""
    synthetic_static = [0.0] * 64
    synthetic_static[63] = 1.0  # presence flag
    payload = {
        "features": synthetic_static,
        "confidence_threshold": 0.50
    }
    response = client.post("/predict/static", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "confidence" in data
    assert "accepted" in data
    assert "inference_latency_ms" in data


def test_predict_static_invalid_shape(client):
    """Test POST /predict/static with invalid length (32 instead of 64)."""
    payload = {"features": [0.0] * 32}
    response = client.post("/predict/static", json=payload)
    assert response.status_code == 422


def test_confidence_filtering(client):
    """
    Test confidence thresholding:
    - Extremely high threshold (0.9999) must reject prediction (label=null, accepted=false).
    - Extremely low threshold (0.0001) must accept the prediction.
    """
    synthetic_sequence = [[0.01] * 150 for _ in range(30)]

    # High threshold rejection
    high_thresh_payload = {
        "frames": synthetic_sequence,
        "confidence_threshold": 0.9999
    }
    res_high = client.post("/predict/sequence", json=high_thresh_payload)
    assert res_high.status_code == 200
    data_high = res_high.json()
    assert data_high["accepted"] is False
    assert data_high["label"] is None
    assert data_high["class_id"] is None

    # Low threshold acceptance
    low_thresh_payload = {
        "frames": synthetic_sequence,
        "confidence_threshold": 0.0001
    }
    res_low = client.post("/predict/sequence", json=low_thresh_payload)
    assert res_low.status_code == 200
    data_low = res_low.json()
    assert data_low["accepted"] is True
    assert data_low["label"] is not None
    assert data_low["class_id"] is not None


def test_malformed_input_handling(client):
    """Test that malformed JSON or empty bodies return 422 Unprocessable Entity."""
    res_empty = client.post("/predict/sequence", json={})
    assert res_empty.status_code == 422

    res_wrong_type = client.post("/predict/sequence", json={"frames": "not a list"})
    assert res_wrong_type.status_code == 422


def test_labels_v3_six_sign_endpoint(client):
    """Test GET /labels/v3-six-sign returns exactly 6 classes."""
    res = client.get("/labels/v3-six-sign")
    assert res.status_code == 200
    data = res.json()
    assert data["vocabulary_size"] == 6
    assert len(data["classes"]) == 6
    for cls in ["help", "yes", "no", "thank_you", "please", "hello"]:
        assert cls in data["classes"]


def test_predict_sequence_v3_six_sign(client):
    """Test POST /predict/sequence/v3-six-sign with 30x168 features and rejection of 30x150."""
    valid_seq = [[0.05] * 168 for _ in range(30)]
    res = client.post("/predict/sequence/v3-six-sign", json={"frames": valid_seq})
    assert res.status_code == 200
    data = res.json()
    assert "confidence" in data
    assert "top_k" in data
    assert len(data["top_k"]) == 6

    # Verify rejection of 30x150
    invalid_seq = [[0.05] * 150 for _ in range(30)]
    res_bad = client.post("/predict/sequence/v3-six-sign", json={"frames": invalid_seq})
    assert res_bad.status_code == 422


def test_labels_v6_10_sign_endpoint(client):
    """Test GET /labels/v6-10-sign returns exactly 10 classes."""
    res = client.get("/labels/v6-10-sign")
    assert res.status_code == 200
    data = res.json()
    assert data["vocabulary_size"] == 10
    assert len(data["classes"]) == 10
    expected_v6 = ["hello", "help", "yes", "no", "please", "thank_you", "doctor", "pain", "sick", "where"]
    for cls in expected_v6:
        assert cls in data["classes"]


def test_predict_sequence_v6_10_sign(client):
    """Test POST /predict/sequence/v6-10-sign with 30x168 features and rejection of 30x150."""
    valid_seq = [[0.05] * 168 for _ in range(30)]
    res = client.post("/predict/sequence/v6-10-sign", json={"frames": valid_seq})
    assert res.status_code == 200
    data = res.json()
    assert "confidence" in data
    assert "top_k" in data
    assert len(data["top_k"]) == 10
    assert data["tensor_shapes"]["model_output"] == [1, 10]

    # Verify rejection of 30x150
    invalid_seq = [[0.05] * 150 for _ in range(30)]
    res_bad = client.post("/predict/sequence/v6-10-sign", json={"frames": invalid_seq})
    assert res_bad.status_code == 422

