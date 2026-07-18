import pytest
import os
from unittest.mock import patch, MagicMock

from workflows.oncology_workflows import build_oncology_trial_matching_workflow, run_trial_matching
from schemas.clinical_state import ClinicalState

# Mock Gemini Response to prevent real API calls in CI
MOCK_GEMINI_STAGING_RESPONSE = '''
{
  "primary_site": "lung",
  "histology": "adenocarcinoma",
  "tnm_stage": {
    "T": "T2",
    "N": "N1",
    "M": "M0",
    "overall": "Stage II",
    "evidence_span": "T2N1M0"
  }
}
'''

MOCK_GEMINI_BIOMARKER_RESPONSE = '''
[
  {
    "marker": "EGFR",
    "status": "Positive",
    "evidence_span": "EGFR positive"
  }
]
'''

MOCK_GEMINI_TRIAL_RESPONSE = '''
[
  {
    "nct_id": "NCT01234567",
    "match_confidence": 0.95,
    "evidence": "Patient has lung adenocarcinoma and is EGFR positive."
  }
]
'''

@pytest.fixture
def mock_oncology_state() -> ClinicalState:
    return {
        "record_id": "test-onco-001",
        "raw_note": "Patient presents with lung adenocarcinoma. Staging is T2N1M0. Lab results confirm EGFR positive mutation. No family history.",
        "extracted_diagnoses": None,
        "ner_votes": None,
        "icd10_codes": None,
        "meat_results": None,
        "clinical_record": None,
        "coding_status": None,
        "overall_confidence": None,
        "claims_ready": None,
        "sdoh_risk_label": None,
        "sdoh_risk_score": None,
        "sdoh_shap_factors": None,
        "deid_note": None,
        "phi_detected": False,
        "privacy_leak_risk": 0.0,
        "primary_site": None,
        "histology": None,
        "tnm_stage": None,
        "biomarkers": None,
        "visit_history": None,
        "pre_chart_summary": None,
        "trial_matches": None,
        "current_step": "init",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    }

@patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
def test_oncology_trial_matching_workflow(mock_invoke, mock_oncology_state):
    # Set a dummy API key to bypass the environment check
    os.environ["GEMINI_API_KEY"] = "dummy_key"
    
    # We need to mock the responses for 3 separate LLM calls in sequence
    # 1. Staging Step
    # 2. Biomarker Step
    # 3. Trial Matching Step
    
    mock_response_staging = MagicMock()
    mock_response_staging.content = MOCK_GEMINI_STAGING_RESPONSE
    
    mock_response_biomarker = MagicMock()
    mock_response_biomarker.content = MOCK_GEMINI_BIOMARKER_RESPONSE
    
    mock_response_trial = MagicMock()
    mock_response_trial.content = MOCK_GEMINI_TRIAL_RESPONSE
    
    mock_invoke.side_effect = [
        mock_response_staging,
        mock_response_biomarker,
        mock_response_trial
    ]

    final_state = run_trial_matching(mock_oncology_state)
    
    # Verify path taken
    assert "deid" in final_state["path_taken"]
    assert "oncology_staging" in final_state["path_taken"]
    assert "biomarker_extraction" in final_state["path_taken"]
    assert "trial_matching" in final_state["path_taken"]
    
    # Verify extractions
    assert final_state["primary_site"] == "lung"
    assert final_state["histology"] == "adenocarcinoma"
    assert final_state["tnm_stage"]["T"] == "T2"
    
    # Verify biomarkers
    assert len(final_state["biomarkers"]) == 1
    assert final_state["biomarkers"][0]["marker"] == "EGFR"
    
    # Verify trial match
    assert len(final_state["trial_matches"]) == 1
    assert final_state["trial_matches"][0]["nct_id"] == "NCT01234567"
