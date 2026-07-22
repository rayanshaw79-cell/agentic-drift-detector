import pytest
import os
from unittest.mock import patch, MagicMock

from workflows.oncology_workflows import build_oncology_trial_matching_workflow, run_trial_matching
from schemas.clinical_state import ClinicalState

# ── Harbor Compliance Note ────────────────────────────────────────────────────
# All tests in this file are OUTCOME-VALIDATING (test what the output IS, not
# how the agent produced it). Structural checks are added alongside value
# checks to prevent the structural-deletion exploit (Challenge 4).
# The NOP test at the bottom formally verifies the verifier rejects empty output.
# Mock strategy: content-aware routing (not a fragile ordered call list).
# ─────────────────────────────────────────────────────────────────────────────

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
        "document_type": None,
        "primary_site": None,
        "histology": None,
        "tnm_stage": None,
        "biomarkers": None,
        "visit_history": None,
        "pre_chart_summary": None,
        "trial_matches": None,
        "eval_feedback": None,
        "needs_reextraction": None,
        "current_step": "init",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    }

def _make_smart_llm_mock(staging_content, biomarker_content, trial_content, fallback_content=""):
    """
    Returns a content-aware mock function for ChatGoogleGenerativeAI.invoke.

    Routes each LLM call by the unique agent-identity phrase in the system
    prompt's opening line. Generic keywords (e.g., 'primary_site', 'staging')
    appear in MULTIPLE prompts (e.g., the trial prompt lists 'primary_site' as
    an input field), so we use the unique 'You are a specialized X Agent'
    phrase that is distinct to each step.

      Staging  → "oncology staging agent"
      Biomarker→ "oncology molecular pathology agent"
      Trial    → "clinical trial matching agent"
      Others   → deid, compliance_checker, oncology_router → fallback
    """
    def _smart_invoke(self_or_messages, messages=None):
        if messages is None:
            messages = self_or_messages

        msg_list = messages if isinstance(messages, list) else [messages]
        system_text = getattr(msg_list[0], "content", str(msg_list[0])).lower() if msg_list else ""

        response = MagicMock()
        if "clinical trial matching agent" in system_text:
            response.content = trial_content
        elif "oncology staging agent" in system_text:
            response.content = staging_content
        elif "molecular pathology agent" in system_text:
            response.content = biomarker_content
        else:
            response.content = fallback_content
        return response

    return _smart_invoke


@patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
def test_oncology_trial_matching_workflow(mock_invoke, mock_oncology_state):
    os.environ["GEMINI_API_KEY"] = "dummy_key"

    # Content-aware mock: routes each LLM call by inspecting prompt keywords.
    # This is immune to pipeline step reordering or new steps being added.
    mock_invoke.side_effect = _make_smart_llm_mock(
        staging_content=MOCK_GEMINI_STAGING_RESPONSE,
        biomarker_content=MOCK_GEMINI_BIOMARKER_RESPONSE,
        trial_content=MOCK_GEMINI_TRIAL_RESPONSE,
        fallback_content=mock_oncology_state["raw_note"],  # deid/router: pass note through
    )

    final_state = run_trial_matching(mock_oncology_state)
    
    # ── Structural Checks (Harbor: Challenge 4 — Cheat-Resistance) ────────────
    # Verify the complete set of keys is present BEFORE checking values.
    # This closes the structural-deletion exploit where an agent could delete
    # a key (e.g., "trial_matches") to avoid a failing assertion.
    EXPECTED_OUTPUT_KEYS = {
        "primary_site", "histology", "tnm_stage",
        "biomarkers", "trial_matches", "path_taken",
    }
    missing_keys = EXPECTED_OUTPUT_KEYS - set(final_state.keys())
    assert not missing_keys, (
        f"Structural check failed: output is missing keys {missing_keys}. "
        "An agent may have deleted sections to avoid failing value checks."
    )
    # ─────────────────────────────────────────────────────────────────────────

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


@patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
def test_nop_agent_oncology_workflow_must_fail(mock_invoke, mock_oncology_state):
    """
    Harbor NOP Test (Challenge 3 — Weak Verification).

    This test simulates a NOP agent: an agent that returns empty/garbage
    output for every LLM call. The verifier MUST reject this (reward = 0.0).
    If this test passes with valid outputs, the verifier is too weak.

    A correct verifier must set critical fields (primary_site, trial_matches,
    biomarkers) to None/empty when the LLM returns garbage — it must NOT
    silently pass with whatever junk the agent produced.
    """
    os.environ["GEMINI_API_KEY"] = "dummy_key"

    # NOP agent: returns empty string for ALL LLM calls regardless of count.
    # Using a function instead of a list so it never raises StopIteration,
    # even as new steps are added to the pipeline.
    def _nop_invoke(*args, **kwargs):
        r = MagicMock()
        r.content = ""
        return r
    mock_invoke.side_effect = _nop_invoke

    final_state = run_trial_matching(mock_oncology_state)


    # The verifier MUST fail: critical output fields must be None/empty
    # when the agent produces no useful work. If any of these pass with
    # real values, the pipeline is silently accepting garbage — NOP Passes.
    assert final_state.get("primary_site") is None, (
        "NOP FAILURE: primary_site was populated from empty LLM output. "
        "The extraction step is not validating its LLM response."
    )
    assert final_state.get("trial_matches") is None or final_state.get("trial_matches") == [], (
        "NOP FAILURE: trial_matches was populated from empty LLM output."
    )
    assert final_state.get("biomarkers") is None or final_state.get("biomarkers") == [], (
        "NOP FAILURE: biomarkers was populated from empty LLM output."
    )
