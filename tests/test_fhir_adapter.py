"""
tests/test_fhir_adapter.py — Unit & integration tests for Phase 5 SMART-on-FHIR R4 Adapter & Synthetic Generator.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from clinical.tools.fhir_adapter import export_clinical_state_to_fhir, generate_synthetic_patient_chart
from clinical.steps.fhir_step import fhir_step
from schemas.clinical_state import ClinicalState

client = TestClient(app)


def test_fhir_bundle_export():
    """Test exporting ClinicalState to HL7 FHIR R4 Bundle."""
    state = {
        "record_id": "test-fhir-001",
        "demographics": {"age": 72, "gender": "F"},
        "icd10_codes": [
            {"code": "C34.90", "description": "Malignant neoplasm of bronchus or lung", "meat_met": True}
        ],
        "extracted_medications": [
            {"drug_name": "Warfarin", "rxcui": "11289"}
        ],
        "sdoh_risk_label": "high",
        "total_raf_score": 0.741
    }

    bundle = export_clinical_state_to_fhir(state)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "collection"
    assert bundle["total"] >= 4

    resource_types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert "Patient" in resource_types
    assert "Condition" in resource_types
    assert "MedicationStatement" in resource_types
    assert "Observation" in resource_types


def test_synthetic_patient_generator():
    """Test synthetic patient chart generator."""
    patient = generate_synthetic_patient_chart("Breast Cancer")

    assert "patient_id" in patient
    assert "raw_note" in patient
    assert len(patient["medications"]) > 0
    assert len(patient["visit_history"]) >= 2


def test_fhir_step_node():
    """Test FHIR Adapter LangGraph state node."""
    state: ClinicalState = {
        "record_id": "test-fhir-step-001",
        "raw_note": "Patient with cancer.",
        "icd10_codes": [{"code": "C34.11", "description": "Lung cancer"}],
        "current_step": "symphony_longitudinal",
        "step_count": 9,
        "retry_count": 0,
        "path_taken": ["symphony_longitudinal"],
        "execution_time_ms": 140
    }

    result = fhir_step(state)

    assert result["current_step"] == "fhir_export"
    assert "fhir_bundle" in result
    assert result["fhir_bundle"]["resourceType"] == "Bundle"


def test_fastapi_fhir_export_endpoint():
    """Integration test for POST /v1/clinical/fhir/export."""
    state_payload = {
        "record_id": "api-fhir-001",
        "icd10_codes": [{"code": "I50.9", "description": "Heart Failure"}]
    }
    response = client.post("/v1/clinical/fhir/export", json=state_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["resourceType"] == "Bundle"


def test_fastapi_fhir_seed_endpoint():
    """Integration test for POST /v1/clinical/fhir/seed."""
    response = client.post("/v1/clinical/fhir/seed?condition=Colorectal%20Cancer")
    assert response.status_code == 200
    data = response.json()
    assert "patient_id" in data
    assert "raw_note" in data


if __name__ == "__main__":
    print("Running test_fhir_bundle_export...")
    test_fhir_bundle_export()
    print("Running test_synthetic_patient_generator...")
    test_synthetic_patient_generator()
    print("Running test_fhir_step_node...")
    test_fhir_step_node()
    print("Running test_fastapi_fhir_export_endpoint...")
    test_fastapi_fhir_export_endpoint()
    print("Running test_fastapi_fhir_seed_endpoint...")
    test_fastapi_fhir_seed_endpoint()
    print("\n[SUCCESS] ALL SMART-ON-FHIR R4 ADAPTER TESTS PASSED SUCCESSFULLY!")
