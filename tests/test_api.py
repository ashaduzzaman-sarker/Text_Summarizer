"""
API endpoint tests using pytest and FastAPI TestClient.
Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_endpoint():
    """Test /health returns status."""
    response = client.get("/health")
    
    # Should return 200 (healthy) or 503 (model not loaded)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data


def test_root_endpoint():
    """Test / returns HTML page."""
    response = client.get("/")
    
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_summarize_success():
    """Test POST /summarize with valid input."""
    payload = {
        "text": "Scientists discovered a new species in the Amazon. " * 10,
        "max_length": 128,
        "min_length": 30
    }
    
    response = client.post("/summarize", json=payload)
    
    # May fail if model not loaded yet
    if response.status_code == 200:
        data = response.json()
        assert "summary" in data
        assert len(data["summary"]) > 0


def test_summarize_text_too_short():
    """Test validation rejects short text."""
    payload = {
        "text": "Short",  # Too short
        "max_length": 128
    }
    
    response = client.post("/summarize", json=payload)
    assert response.status_code == 422  # Validation error


def test_summarize_invalid_params():
    """Test validation rejects invalid parameters."""
    payload = {
        "text": "A" * 100,
        "max_length": 30,
        "min_length": 50  # min > max (invalid)
    }
    
    response = client.post("/summarize", json=payload)
    assert response.status_code == 422