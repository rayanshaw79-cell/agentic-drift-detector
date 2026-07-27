"""
tests/test_trial_matching.py — Unit and integration tests for PRISM v2 Live Clinical Trial Matching Engine.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from clinical.tools.clinical_trials_api import search_recruiting_trials
from clinical.steps.trial_matching_step import trial_matching_step, _evaluate_eligibility
from schemas.clinical_state import ClinicalState

client = TestClient(app)


def test_clinical_trials_api_search():
    """Verify ClinicalTrials.gov API tool returns structured trial list."""
    trials = search_recruiting_trials(condition="Lung Cancer", limit=3)
    assert isinstance(trials, list)
    assert len(trials) > 0
    trial = trials[0]
    assert "nct_id" in trial
    assert "brief_title" in trial
    assert "sponsor" in trial
    assert "phase" in trial
    assert "url" in trial
    assert trial["nct_id"].startswith("NCT")


def test_eligibility_evaluator_scoring():
    """Test eligibility evaluator scoring logic for biomarkers and staging."""
    trial = {
        "nct_id": "NCT05123456",
        "brief_title": "Targeted EGFR Trial",
        "eligibility_criteria": "Inclusion:\n- Stage III Lung Adenocarcinoma\n- EGFR L858R mutation positive"
    }

    score, label, evidence = _evaluate_eligibility(
        trial=trial,
        primary_site="Lung",
        histology="Adenocarcinoma",
        biomarkers=[{"marker": "EGFR", "status": "Positive"}],
        tnm_stage={"overall": "Stage III"}
    )

    assert score >= 0.80
    assert label == "highly_eligible"
    assert "EGFR" in evidence


def test_trial_matching_step_node():
    """Test trial matching LangGraph state node."""
    state: ClinicalState = {
        "record_id": "test-trial-001",
        "raw_note": "Patient with Stage III Lung Adenocarcinoma, EGFR positive.",
        "primary_site": "Lung",
        "histology": "Adenocarcinoma",
        "biomarkers": [{"marker": "EGFR", "status": "Positive"}],
        "tnm_stage": {"overall": "Stage III"},
        "extracted_diagnoses": ["Lung Cancer"],
        "current_step": "sdoh_integration",
        "step_count": 5,
        "retry_count": 0,
        "path_taken": ["ner", "sdoh_integration"],
        "execution_time_ms": 100
    }

    result = trial_matching_step(state)

    assert result["current_step"] == "trial_matching"
    assert "trial_matches" in result
    matches = result["trial_matches"]
    assert len(matches) > 0
    assert matches[0]["nct_id"].startswith("NCT")
    assert "eligibility_score" in matches[0]


def test_fastapi_trials_search_endpoint():
    """Integration test for GET /v1/clinical/trials/search endpoint."""
    response = client.get("/v1/clinical/trials/search?condition=Melanoma&limit=2")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Melanoma"
    assert data["count"] > 0
    assert "trials" in data
    assert len(data["trials"]) <= 2


if __name__ == "__main__":
    print("Running test_clinical_trials_api_search...")
    test_clinical_trials_api_search()
    print("Running test_eligibility_evaluator_scoring...")
    test_eligibility_evaluator_scoring()
    print("Running test_trial_matching_step_node...")
    test_trial_matching_step_node()
    print("Running test_fastapi_trials_search_endpoint...")
    test_fastapi_trials_search_endpoint()
    print("\n[SUCCESS] ALL PRISM V2 TRIAL MATCHING TESTS PASSED SUCCESSFULLY!")
