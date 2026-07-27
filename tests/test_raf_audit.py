"""
tests/test_raf_audit.py — Unit & integration tests for Phase 3 CMS Financial RAF & RADV Audit Engine.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from clinical.tools.raf_audit_calculator import calculate_raf_audit_metrics
from clinical.steps.raf_audit_step import raf_audit_step
from schemas.clinical_state import ClinicalState

client = TestClient(app)


def test_raf_audit_calculator_logic():
    """Test RAF score calculation and RADV financial clawback exposure."""
    codes = [
        {"code": "E11.40", "hcc_category": "HCC 18", "raf_weight": 0.368, "meat_met": True},
        {"code": "J44.9", "hcc_category": "HCC 111", "raf_weight": 0.335, "meat_met": False},
    ]

    metrics = calculate_raf_audit_metrics(codes, {"age": 70, "gender": "M"})

    assert "total_raf_score" in metrics
    assert "verified_raf_score" in metrics
    assert "unverified_raf_score" in metrics
    assert "radv_financial_exposure_usd" in metrics
    assert metrics["unverified_raf_score"] == 0.335
    assert metrics["radv_financial_exposure_usd"] > 3000.0
    assert metrics["radv_audit_label"] in ("moderate_audit_risk", "high_radv_exposure")


def test_raf_audit_step_node():
    """Test CMS RAF Audit LangGraph state node."""
    state: ClinicalState = {
        "record_id": "test-raf-001",
        "raw_note": "Patient with T2DM and unbacked COPD.",
        "icd10_codes": [
            {"code": "E11.9", "hcc_category": "HCC 19", "raf_weight": 0.105, "meat_met": True},
            {"code": "J44.9", "hcc_category": "HCC 111", "raf_weight": 0.335, "meat_met": False},
        ],
        "current_step": "pharmacovigilance",
        "step_count": 7,
        "retry_count": 0,
        "path_taken": ["ner", "pharmacovigilance"],
        "execution_time_ms": 110
    }

    result = raf_audit_step(state)

    assert result["current_step"] == "raf_audit"
    assert "total_raf_score" in result
    assert "verified_raf_score" in result
    assert "radv_financial_exposure_usd" in result
    assert result["radv_financial_exposure_usd"] > 0.0


def test_fastapi_raf_audit_endpoint():
    """Integration test for POST /v1/clinical/raf-audit/calculate."""
    payload = {
        "icd10_codes": [
            {"code": "I50.9", "hcc_category": "HCC 85", "raf_weight": 0.323, "meat_met": False}
        ],
        "demographics": {"age": 75, "gender": "F"}
    }
    response = client.post("/v1/clinical/raf-audit/calculate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "total_raf_score" in data
    assert "radv_financial_exposure_usd" in data
    assert data["radv_financial_exposure_usd"] > 3000.0


if __name__ == "__main__":
    print("Running test_raf_audit_calculator_logic...")
    test_raf_audit_calculator_logic()
    print("Running test_raf_audit_step_node...")
    test_raf_audit_step_node()
    print("Running test_fastapi_raf_audit_endpoint...")
    test_fastapi_raf_audit_endpoint()
    print("\n[SUCCESS] ALL CMS RAF & RADV AUDIT TESTS PASSED SUCCESSFULLY!")
