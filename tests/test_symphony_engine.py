"""
tests/test_symphony_engine.py — Unit & integration tests for Phase 4 SYMPHONY v2 Engine.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from clinical.tools.symphony_engine import evaluate_recist_response, synthesize_patient_timeline
from clinical.steps.symphony_step import symphony_step
from schemas.clinical_state import ClinicalState

client = TestClient(app)


def test_recist_response_evaluator_partial():
    """Test Partial Response (PR) calculation: >= 30% reduction."""
    measurements = [
        {"date": "2026-01-01", "target_lesion_mm": 50.0},
        {"date": "2026-06-01", "target_lesion_mm": 30.0}  # -40% delta
    ]
    category, delta_pct = evaluate_recist_response(measurements)
    assert category == "PR"
    assert delta_pct == -40.0


def test_recist_response_evaluator_progression():
    """Test Progressive Disease (PD) calculation: >= 20% increase or new lesion."""
    measurements = [
        {"date": "2026-01-01", "target_lesion_mm": 30.0},
        {"date": "2026-06-01", "target_lesion_mm": 40.0, "new_lesions": True}
    ]
    category, delta_pct = evaluate_recist_response(measurements)
    assert category == "PD"


def test_synthesize_patient_timeline():
    """Test multi-visit timeline synthesis."""
    history = [
        {"date": "2026-01-01", "doc_type": "Baseline", "summary": "Baseline lesion 40mm", "target_lesion_mm": 40.0},
        {"date": "2026-05-01", "doc_type": "Follow-up", "summary": "Lesion 24mm", "target_lesion_mm": 24.0}
    ]
    res = synthesize_patient_timeline(history)

    assert res["total_visits"] == 2
    assert res["recist_overall_response"] == "PR"
    assert len(res["chronological_timeline"]) == 2


def test_symphony_step_node():
    """Test SYMPHONY v2 LangGraph state node."""
    state: ClinicalState = {
        "record_id": "test-sym-001",
        "raw_note": "Patient with NSCLC under therapy.",
        "primary_site": "Lung",
        "histology": "Adenocarcinoma",
        "current_step": "raf_audit",
        "step_count": 8,
        "retry_count": 0,
        "path_taken": ["ner", "raf_audit"],
        "execution_time_ms": 130
    }

    result = symphony_step(state)

    assert result["current_step"] == "symphony_longitudinal"
    assert "longitudinal_timeline" in result
    assert "recist_overall_response" in result
    assert len(result["longitudinal_timeline"]) >= 2


def test_fastapi_symphony_endpoint():
    """Integration test for POST /v1/clinical/symphony/timeline."""
    payload = {
        "visit_history": [
            {"date": "2026-01-01", "doc_type": "Baseline", "summary": "Mass 40mm", "target_lesion_mm": 40.0},
            {"date": "2026-06-01", "doc_type": "Follow-up", "summary": "Mass 20mm", "target_lesion_mm": 20.0}
        ]
    }
    response = client.post("/v1/clinical/symphony/timeline", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_visits"] == 2
    assert data["recist_overall_response"] == "PR"


if __name__ == "__main__":
    print("Running test_recist_response_evaluator_partial...")
    test_recist_response_evaluator_partial()
    print("Running test_recist_response_evaluator_progression...")
    test_recist_response_evaluator_progression()
    print("Running test_synthesize_patient_timeline...")
    test_synthesize_patient_timeline()
    print("Running test_symphony_step_node...")
    test_symphony_step_node()
    print("Running test_fastapi_symphony_endpoint...")
    test_fastapi_symphony_endpoint()
    print("\n[SUCCESS] ALL SYMPHONY v2 ENGINE TESTS PASSED SUCCESSFULLY!")
