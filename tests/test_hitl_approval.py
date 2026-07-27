"""
tests/test_hitl_approval.py — Comprehensive unit & integration tests for Human-in-the-Loop (HITL) approval system.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from schemas.clinical_state import ClinicalState
from clinical.steps.clinical_intervention_step import clinical_intervention_step
from workflows.clinical_coding import apply_human_approval
from telemetry.store import (
    init_db,
    save_execution_state,
    save_human_intervention,
    update_execution_human_status,
    get_pending_reviews,
    get_review_history,
)

client = TestClient(app)


def test_clinical_intervention_step_sets_hitl_pending():
    """Verify intervention step purges low-confidence codes and sets human_review_action='pending'."""
    initial_state: ClinicalState = {
        "record_id": "test-hitl-001",
        "raw_note": "Patient with chest pain.",
        "icd10_codes": [
            {"code": "R07.9", "description": "Chest pain", "confidence": 0.8},
            {"code": "UNRESOLVED", "description": "Hallucinated", "confidence": 0.2},
        ],
        "retry_count": 2,
        "overall_confidence": 0.4,
        "current_step": "validation",
        "step_count": 5,
        "path_taken": ["deid", "ner", "validation"],
        "execution_time_ms": 120,
    }

    result = clinical_intervention_step(initial_state)

    assert result["coding_status"] == "requires_clinical_review"
    assert result["human_review_action"] == "pending"
    assert "original_ai_codes" in result
    assert len(result["icd10_codes"]) == 1
    assert result["icd10_codes"][0]["code"] == "R07.9"


def test_apply_human_approval_function():
    """Test state mutation helper for clinician approval and rejection."""
    state = {
        "record_id": "test-hitl-002",
        "coding_status": "requires_clinical_review",
        "icd10_codes": [],
    }

    # Test Approved / Edited
    approved_state = apply_human_approval(
        state,
        action="approved",
        reviewed_by="Dr. Smith",
        notes="Confirmed with chart.",
        final_codes=[{"code": "I10", "description": "Essential hypertension"}]
    )
    assert approved_state["coding_status"] == "approved_by_clinician"
    assert approved_state["human_review_action"] == "approved"
    assert approved_state["reviewed_by"] == "Dr. Smith"
    assert len(approved_state["icd10_codes"]) == 1

    # Test Rejected
    rejected_state = apply_human_approval(
        state,
        action="rejected",
        reviewed_by="Dr. Smith",
        notes="Insufficient documentation."
    )
    assert rejected_state["coding_status"] == "rejected_by_clinician"
    assert rejected_state["human_review_action"] == "rejected"
    assert len(rejected_state["icd10_codes"]) == 0


def test_telemetry_store_hitl_flow(tmp_path, monkeypatch):
    """Test saving execution, retrieving pending reviews, saving intervention, and fetching history."""
    init_db()
    
    test_state = {
        "record_id": "hitl-db-001",
        "severity": "clinical_coding",
        "coding_status": "requires_clinical_review",
        "overall_confidence": 0.3,
        "step_count": 4,
        "retry_count": 2,
        "path_taken": ["ner", "clinical_intervention"],
        "execution_time_ms": 150,
        "sdoh_risk_label": "moderate",
        "sdoh_risk_score": 0.45,
    }
    
    analysis = {"workflow_type": "clinical_coding", "risk_level": "drift"}
    
    # 1. Save state
    save_execution_state(test_state, analysis=analysis)
    
    # 2. Get pending reviews
    pending = get_pending_reviews()
    assert len(pending) > 0
    record_ids = [p["incident_id"] for p in pending]
    assert "hitl-db-001" in record_ids

    # 3. Save intervention
    row_id = save_human_intervention(
        incident_id="hitl-db-001",
        action="approved",
        reviewed_by="Dr. Alex",
        notes="Valid extraction.",
        final_codes=[{"code": "E11.9"}]
    )
    assert row_id > 0

    # 4. Update status
    update_execution_human_status(
        record_id="hitl-db-001",
        new_status="approved_by_clinician",
        human_action="approved",
        notes="Valid extraction.",
        reviewed_by="Dr. Alex"
    )

    # 5. Review history
    history = get_review_history()
    assert len(history) > 0
    hist_rec_ids = [h["incident_id"] for h in history]
    assert "hitl-db-001" in hist_rec_ids


def test_fastapi_hitl_endpoints():
    """Integration test for FastAPI GET review queue and POST approval endpoints."""
    # 1. Seed a review-required record
    test_state = {
        "record_id": "api-hitl-999",
        "severity": "clinical_coding",
        "coding_status": "requires_clinical_review",
        "overall_confidence": 0.2,
        "step_count": 3,
        "retry_count": 2,
        "path_taken": ["ner", "clinical_intervention"],
        "execution_time_ms": 90,
    }
    save_execution_state(test_state, analysis={"workflow_type": "clinical_coding", "risk_level": "drift"})

    # 2. GET /v1/clinical/review-queue
    resp = client.get("/v1/clinical/review-queue")
    assert resp.status_code == 200
    data = resp.json()
    assert "records" in data
    rec_ids = [r["incident_id"] for r in data["records"]]
    assert "api-hitl-999" in rec_ids

    # 3. POST /v1/clinical/approve
    approval_payload = {
        "record_id": "api-hitl-999",
        "action": "approved",
        "reviewed_by": "Dr. House",
        "notes": "Approved after chart review",
        "final_codes": [{"code": "J44.9", "description": "COPD"}]
    }
    post_resp = client.post("/v1/clinical/approve", json=approval_payload)
    assert post_resp.status_code == 200
    post_data = post_resp.json()
    assert post_data["status"] == "success"
    assert post_data["coding_status"] == "approved_by_clinician"

    # 4. GET /v1/clinical/review-history
    hist_resp = client.get("/v1/clinical/review-history")
    assert hist_resp.status_code == 200
    hist_data = hist_resp.json()
    assert "records" in hist_data
    hist_rec_ids = [r["incident_id"] for r in hist_data["records"]]
    assert "api-hitl-999" in hist_rec_ids


if __name__ == "__main__":
    print("Running test_clinical_intervention_step_sets_hitl_pending...")
    test_clinical_intervention_step_sets_hitl_pending()
    print("Running test_apply_human_approval_function...")
    test_apply_human_approval_function()
    print("Running test_telemetry_store_hitl_flow...")
    test_telemetry_store_hitl_flow(None, None)
    print("Running test_fastapi_hitl_endpoints...")
    test_fastapi_hitl_endpoints()
    print("\n[SUCCESS] ALL HITL TESTS PASSED SUCCESSFULLY!")
