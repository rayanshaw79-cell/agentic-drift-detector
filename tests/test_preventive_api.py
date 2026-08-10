import os
import pytest
from fastapi.testclient import TestClient
from api.main import app
from unittest.mock import patch

@pytest.fixture(autouse=True)
def setup_api_key():
    os.environ["API_SECRET_KEY"] = "test-secret"
    yield
    del os.environ["API_SECRET_KEY"]

client = TestClient(app)

def test_preventive_api_unauthorized():
    response = client.post(
        "/v1/preventive/risk-assess",
        json={"raw_note": "Patient eats pan masala"}
    )
    assert response.status_code == 403

@patch("workflows.preventive_screening.preventive_screening_workflow")
def test_preventive_api_authorized(mock_workflow):
    # Mock the workflow to avoid hitting Gemini API during testing
    mock_workflow.return_value = {
        "patient_id": "test-pat-1",
        "lifestyle_risk_score": 0.85,
        "lifestyle_factors": [{"factor": "tobacco", "context": "chews pan masala"}],
        "preventive_recommendations": "Screening recommended.",
        "execution_time_ms": 150
    }
    
    headers = {"X-API-Key": "test-secret"}
    payload = {
        "raw_note": "Patient eats pan masala.",
        "patient_id": "test-pat-1"
    }
    response = client.post("/v1/preventive/risk-assess", json=payload, headers=headers)
    
    assert response.status_code == 200
    data = response.json()
    assert data["patient_id"] == "test-pat-1"
    assert data["lifestyle_risk_score"] == 0.85
    assert data["preventive_recommendations"] == "Screening recommended."
