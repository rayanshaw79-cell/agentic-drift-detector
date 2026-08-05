"""
tests/test_clinical_adapters.py — Unit & integration tests for Phase 5 Clinical Adapters.

Covers:
  - SMART-on-FHIR R4 Adapter & Synthetic Generator
  - Pharmacovigilance & ADR Safety Scanner
  - CMS Financial RAF & RADV Audit Engine
  - SYMPHONY RECIST Staging Longitudinal Engine
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from schemas.clinical_state import ClinicalState

# SMART-on-FHIR
from clinical.tools.fhir_adapter import export_clinical_state_to_fhir, generate_synthetic_patient_chart
from clinical.steps.fhir_step import fhir_step

# Pharmacovigilance
from clinical.tools.pharmacovigilance_api import check_drug_interactions, get_rxcui_by_name
from clinical.steps.pharmacovigilance_step import pharmacovigilance_step, _extract_adverse_reactions

# CMS RAF
from clinical.tools.raf_audit_calculator import calculate_raf_audit_metrics
from clinical.steps.raf_audit_step import raf_audit_step

# Symphony RECIST
from clinical.tools.symphony_engine import evaluate_recist_response, synthesize_patient_timeline
from clinical.steps.symphony_step import symphony_step

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# SMART-ON-FHIR R4 ADAPTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# PHARMACOVIGILANCE & ADR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_pharmacovigilance_rxcui_lookup():
    """Test RxCUI lookup for standard medications."""
    cui = get_rxcui_by_name("Warfarin")
    assert isinstance(cui, str)
    assert cui != ""


def test_drug_interaction_checker():
    """Test interaction checking between Warfarin and Aspirin."""
    interactions = check_drug_interactions(["Warfarin", "Aspirin"])
    assert isinstance(interactions, list)
    assert len(interactions) > 0
    assert "Warfarin" in interactions[0]["pair"]
    assert interactions[0]["severity"] in ("high", "moderate")


def test_adverse_reaction_extractor():
    """Test extraction of unstructured ADR signals from note text."""
    note = "Patient was prescribed Keytruda. Developed severe maculopapular rash and epistaxis."
    adrs = _extract_adverse_reactions(note, ["Keytruda"])

    assert len(adrs) >= 1
    categories = [a["category"] for a in adrs]
    assert "Cutaneous Reaction" in categories or "Hemorrhagic Reaction" in categories


def test_pharmacovigilance_step_node():
    """Test pharmacovigilance LangGraph state node."""
    state: ClinicalState = {
        "record_id": "test-pharma-001",
        "raw_note": "Patient taking Warfarin 5mg and Aspirin 81mg daily. Reports severe epistaxis.",
        "current_step": "trial_matching",
        "step_count": 6,
        "retry_count": 0,
        "path_taken": ["ner", "trial_matching"],
        "execution_time_ms": 120
    }

    result = pharmacovigilance_step(state)

    assert result["current_step"] == "pharmacovigilance"
    assert "extracted_medications" in result
    assert "drug_interactions" in result
    assert "adverse_drug_reactions" in result
    assert result["drug_safety_risk"] in ("high", "critical")


def test_fastapi_pharmacovigilance_endpoint():
    """Integration test for POST /v1/clinical/pharmacovigilance/check."""
    payload = {
        "medications": ["Warfarin", "Aspirin"],
        "raw_note": "Patient reports severe bleeding and epistaxis."
    }
    response = client.post("/v1/clinical/pharmacovigilance/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["medications"] == ["Warfarin", "Aspirin"]
    assert len(data["interactions"]) > 0
    assert data["drug_safety_risk"] in ("high", "critical")


# ═══════════════════════════════════════════════════════════════════════════════
# CMS FINANCIAL RAF & RADV AUDIT ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# SYMPHONY RECIST LONGITUDINAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════

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
