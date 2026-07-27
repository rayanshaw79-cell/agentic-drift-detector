"""
tests/test_pharmacovigilance.py — Unit & integration tests for Phase 2 Pharmacovigilance & ADR Safety Scanner.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from clinical.tools.pharmacovigilance_api import check_drug_interactions, get_rxcui_by_name
from clinical.steps.pharmacovigilance_step import pharmacovigilance_step, _extract_adverse_reactions, _compute_safety_risk
from schemas.clinical_state import ClinicalState

client = TestClient(app)


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


if __name__ == "__main__":
    print("Running test_pharmacovigilance_rxcui_lookup...")
    test_pharmacovigilance_rxcui_lookup()
    print("Running test_drug_interaction_checker...")
    test_drug_interaction_checker()
    print("Running test_adverse_reaction_extractor...")
    test_adverse_reaction_extractor()
    print("Running test_pharmacovigilance_step_node...")
    test_pharmacovigilance_step_node()
    print("Running test_fastapi_pharmacovigilance_endpoint...")
    test_fastapi_pharmacovigilance_endpoint()
    print("\n[SUCCESS] ALL PHARMACOVIGILANCE TESTS PASSED SUCCESSFULLY!")
