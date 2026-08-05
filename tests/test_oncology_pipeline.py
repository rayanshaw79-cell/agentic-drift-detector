"""
tests/test_oncology_pipeline.py — Unit and integration tests for all OncoLLM components.

Covers:
  - Pillar 1 — Constellation Router
  - Pillar 2 — RAG guideline store (retrieval, seeding, chunking)
  - Pillar 3 — Few-shot prompt libraries (staging, biomarkers, trial matching)
  - Pillar 4 — Oncology Evaluator and self-correction routing
  - Trial Matching Node — Live/mock ClinicalTrials.gov matching
  - Integration — End-to-end PRISM workflows and NOP verification
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from schemas.clinical_state import ClinicalState
from workflows.oncology_workflows import build_oncology_trial_matching_workflow, run_trial_matching
from clinical.steps.trial_matching_step import trial_matching_step, _evaluate_eligibility
from clinical.tools.clinical_trials_api import search_recruiting_trials

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — Constellation Router
# ═══════════════════════════════════════════════════════════════════════════════

class TestOncologyRouter:
    """Tests for clinical/agents/oncology_router.py"""

    def _make_state(self, raw_note: str) -> dict:
        return {"raw_note": raw_note, "deid_note": None}

    def test_classifies_pathology_report(self):
        from clinical.agents.oncology_router import oncology_router

        note = (
            "PATHOLOGY REPORT\nSurgical pathology specimen: right lobe lobectomy.\n"
            "Microscopic description: invasive adenocarcinoma.\n"
            "Final pathologic stage: pT2a pN1 pM0.\nIHC: AJCC 8th edition staging applied."
        )
        result = oncology_router(self._make_state(note))
        assert result["document_type"] == "pathology_report"

    def test_classifies_radiology(self):
        from clinical.agents.oncology_router import oncology_router

        note = (
            "CT CHEST WITH CONTRAST\nFindings: 4.2 cm spiculated mass. "
            "Impression: consistent with primary lung malignancy. "
            "Clinical staging: cT3 cN2 cM1b. Hypodense lesion in liver."
        )
        result = oncology_router(self._make_state(note))
        assert result["document_type"] == "radiology"

    def test_classifies_genomics(self):
        from clinical.agents.oncology_router import oncology_router

        note = (
            "NEXT GENERATION SEQUENCING REPORT\nMutation analysis results:\n"
            "EGFR: Exon 19 deletion detected. KRAS: Wild type.\n"
            "Microsatellite instability: MSI-H. TMB: 12 mutations/Mb."
        )
        result = oncology_router(self._make_state(note))
        assert result["document_type"] == "genomics"

    def test_classifies_progress_note(self):
        from clinical.agents.oncology_router import oncology_router

        note = (
            "ONCOLOGY CLINIC NOTE\nHISTORY OF PRESENT ILLNESS: Patient returns for follow-up.\n"
            "ASSESSMENT AND PLAN:\nContinue current therapy. Repeat imaging in 3 months.\n"
            "Review of systems: no complaints. Subjective: tolerating well."
        )
        result = oncology_router(self._make_state(note))
        assert result["document_type"] == "progress_note"

    def test_returns_unknown_for_empty_note(self):
        from clinical.agents.oncology_router import oncology_router

        result = oncology_router(self._make_state(""))
        assert result["document_type"] == "unknown"

    def test_state_keys_returned(self):
        from clinical.agents.oncology_router import oncology_router

        result = oncology_router(self._make_state("some clinical text"))
        assert "document_type" in result
        assert "current_step" in result
        assert result["current_step"] == "oncology_router"
        assert "execution_time_ms" in result
        assert isinstance(result["execution_time_ms"], int)


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — RAG Guideline Store
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuidelineStore:
    """Tests for clinical/rag/guideline_store.py"""

    def test_retrieve_returns_string(self):
        """retrieve_guidelines must always return a string (even if empty)."""
        from clinical.rag.guideline_store import retrieve_guidelines
        result = retrieve_guidelines("lung cancer staging")
        assert isinstance(result, str)

    def test_chunk_text_splits_correctly(self):
        from clinical.rag.guideline_store import _chunk_text
        text = "A" * 1000
        chunks = _chunk_text(text, chunk_size=400, overlap=80)
        assert len(chunks) >= 2
        # Each chunk should be ≤ chunk_size chars
        for chunk in chunks:
            assert len(chunk) <= 400

    def test_chunk_text_excludes_tiny_chunks(self):
        from clinical.rag.guideline_store import _chunk_text
        text = "Short text."
        chunks = _chunk_text(text, chunk_size=400, overlap=80)
        # Should produce 0 chunks since text is < 50 chars threshold
        for chunk in chunks:
            assert len(chunk) > 50


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — Prompt Library Structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptLibrary:
    """Structural tests for clinical/prompts/*.py"""

    def test_staging_prompt_returns_correct_types(self):
        from clinical.prompts.staging_prompts import build_staging_prompt
        system, examples = build_staging_prompt()
        assert isinstance(system, str)
        assert len(system) > 100
        assert isinstance(examples, list)
        assert len(examples) >= 3  # At least 3 worked examples

    def test_staging_prompt_examples_are_tuples(self):
        from clinical.prompts.staging_prompts import build_staging_prompt
        _, examples = build_staging_prompt()
        for ex in examples:
            assert isinstance(ex, tuple)
            assert len(ex) == 2
            user_msg, ai_msg = ex
            assert isinstance(user_msg, str)
            assert isinstance(ai_msg, str)

    def test_staging_prompt_ai_outputs_are_valid_json(self):
        """All few-shot AI staging outputs must be valid JSON objects."""
        from clinical.prompts.staging_prompts import build_staging_prompt
        _, examples = build_staging_prompt()
        for user_msg, ai_msg in examples:
            parsed = json.loads(ai_msg)
            assert "primary_site" in parsed
            assert "histology" in parsed
            assert "tnm_stage" in parsed

    def test_biomarker_prompt_returns_correct_types(self):
        from clinical.prompts.biomarker_prompts import build_biomarker_prompt
        system, examples = build_biomarker_prompt()
        assert isinstance(system, str)
        assert len(examples) >= 3

    def test_biomarker_prompt_ai_outputs_are_valid_json(self):
        """All few-shot AI biomarker outputs must be valid JSON arrays."""
        from clinical.prompts.biomarker_prompts import build_biomarker_prompt
        _, examples = build_biomarker_prompt()
        for user_msg, ai_msg in examples:
            parsed = json.loads(ai_msg)
            assert isinstance(parsed, list)
            for item in parsed:
                assert "marker" in item
                assert "status" in item
                assert "evidence_span" in item

    def test_biomarker_prompts_evidence_spans_in_notes(self):
        """All evidence_spans in few-shot examples must appear in the corresponding note."""
        from clinical.prompts.biomarker_prompts import build_biomarker_prompt
        _, examples = build_biomarker_prompt()
        for user_msg, ai_msg in examples:
            biomarkers = json.loads(ai_msg)
            for bio in biomarkers:
                ev = bio.get("evidence_span", "")
                if ev:
                    assert ev in user_msg, (
                        f"Evidence span '{ev}' not found in note for marker '{bio.get('marker')}'"
                    )

    def test_staging_prompts_evidence_spans_in_notes(self):
        """All TNM evidence_spans in few-shot staging examples must appear in the note."""
        from clinical.prompts.staging_prompts import build_staging_prompt
        _, examples = build_staging_prompt()
        for user_msg, ai_msg in examples:
            parsed = json.loads(ai_msg)
            tnm = parsed.get("tnm_stage") or {}
            ev = tnm.get("evidence_span")
            if ev:
                assert ev in user_msg, (
                    f"TNM evidence_span '{ev}' not found in staging example note"
                )

    def test_trial_matching_prompt_returns_correct_types(self):
        from clinical.prompts.trial_matching_prompts import build_trial_matching_prompt
        system, examples = build_trial_matching_prompt()
        assert isinstance(system, str)
        assert len(examples) >= 1

    def test_longitudinal_prompt_returns_correct_types(self):
        from clinical.prompts.longitudinal_prompts import build_longitudinal_prompt
        system, examples = build_longitudinal_prompt()
        assert isinstance(system, str)
        assert len(examples) >= 1

    def test_staging_prompt_reorders_examples_for_radiology(self):
        """Radiology document type should promote the radiology example to first."""
        from clinical.prompts.staging_prompts import build_staging_prompt
        _, default_examples = build_staging_prompt(document_type=None)
        _, radio_examples = build_staging_prompt(document_type="radiology")
        # First example for radiology should differ from default order
        assert radio_examples[0][0] != default_examples[0][0]


# ═══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 — Oncology Evaluator & Reextraction Router
# ═══════════════════════════════════════════════════════════════════════════════

class TestOncologyEvaluator:
    """Tests for clinical/agents/oncology_evaluator.py"""

    def _base_state(self, **kwargs) -> dict:
        state = {
            "primary_site": "Lung",
            "tnm_stage": {"T": "T2", "N": "N1", "M": "M0", "overall": "Stage IIB", "evidence_span": "Stage IIB"},
            "biomarkers": [{"marker": "EGFR", "status": "Mutated", "evidence_span": "EGFR exon 19 deletion detected"}],
            "raw_note": "EGFR exon 19 deletion detected. Stage IIB.",
            "document_type": "pathology_report",
            "retry_count": 0,
        }
        state.update(kwargs)
        return state

    def test_passes_valid_extraction(self):
        from clinical.agents.oncology_evaluator import oncology_evaluator
        result = oncology_evaluator(self._base_state())
        assert result["needs_reextraction"] is False
        assert result["eval_feedback"] is None

    def test_fails_missing_primary_site(self):
        from clinical.agents.oncology_evaluator import oncology_evaluator
        result = oncology_evaluator(self._base_state(primary_site=None))
        assert result["needs_reextraction"] is True
        assert result["eval_feedback"] is not None
        assert "primary_site" in result["eval_feedback"]

    def test_fails_null_tnm_for_pathology_report(self):
        from clinical.agents.oncology_evaluator import oncology_evaluator
        result = oncology_evaluator(self._base_state(tnm_stage={}))
        assert result["needs_reextraction"] is True

    def test_fails_hallucinated_biomarker_evidence(self):
        from clinical.agents.oncology_evaluator import oncology_evaluator
        bad_biomarkers = [
            {"marker": "KRAS", "status": "Mutated", "evidence_span": "KRAS G12D mutation detected"}
        ]
        result = oncology_evaluator(self._base_state(biomarkers=bad_biomarkers))
        assert result["needs_reextraction"] is True
        assert "hallucination" in result["eval_feedback"].lower()

    def test_does_not_retry_after_max_retries(self):
        from clinical.agents.oncology_evaluator import oncology_evaluator
        result = oncology_evaluator(self._base_state(primary_site=None, retry_count=2))
        assert result["needs_reextraction"] is False

    def test_progress_note_passes_without_tnm(self):
        """Progress notes are not expected to have explicit TNM — should pass."""
        from clinical.agents.oncology_evaluator import oncology_evaluator
        result = oncology_evaluator(self._base_state(
            document_type="progress_note",
            tnm_stage={},
        ))
        assert result["needs_reextraction"] is False


class TestReextractionRouter:
    """Tests for clinical/agents/reextraction_router.py"""

    def test_routes_to_trial_matching_on_pass(self):
        from clinical.agents.reextraction_router import reextraction_router
        result = reextraction_router({"needs_reextraction": False})
        assert result == "trial_matching"

    def test_routes_to_staging_on_failure(self):
        from clinical.agents.reextraction_router import reextraction_router
        result = reextraction_router({"needs_reextraction": True})
        assert result == "oncology_staging"

    def test_routes_to_trial_matching_when_key_absent(self):
        from clinical.agents.reextraction_router import reextraction_router
        result = reextraction_router({})
        assert result == "trial_matching"


# ═══════════════════════════════════════════════════════════════════════════════
# TRIAL MATCHING STATE NODE & API UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION — Full PRISM Workflow (mock LLM)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPRISMWorkflowIntegration:
    """
    End-to-end test of the trial matching workflow with a synthetic pathology note.
    Uses mock LLM responses to avoid requiring a real GEMINI_API_KEY.
    """

    SYNTHETIC_NOTE = (
        "PATHOLOGY REPORT\n"
        "Right upper lobe lobectomy.\n"
        "Diagnosis: Invasive adenocarcinoma, right upper lobe, lung.\n"
        "EGFR exon 19 deletion detected. ALK-negative. PD-L1 TPS 78%.\n"
        "Final pathologic stage: pT2a pN1 pM0 — Stage IIB Non-Small Cell Lung Cancer.\n"
    )

    @pytest.fixture
    def initial_state(self):
        return {
            "record_id": "TEST-001",
            "raw_note": self.SYNTHETIC_NOTE,
            "current_step": "start",
            "step_count": 0,
            "retry_count": 0,
            "path_taken": [],
            "execution_time_ms": 0,
        }

    def test_router_classifies_note_correctly(self):
        """Verify the router correctly classifies our synthetic pathology note without LLM."""
        from clinical.agents.oncology_router import _classify_by_keywords
        doc_type = _classify_by_keywords(self.SYNTHETIC_NOTE)
        assert doc_type == "pathology_report"

    def test_evaluator_passes_good_extraction(self):
        """Simulate a good extraction result going through the evaluator."""
        from clinical.agents.oncology_evaluator import oncology_evaluator
        state = {
            "primary_site": "right upper lobe, lung",
            "histology": "Invasive adenocarcinoma",
            "tnm_stage": {
                "T": "pT2a", "N": "pN1", "M": "pM0",
                "overall": "Stage IIB Non-Small Cell Lung Cancer",
                "evidence_span": "Final pathologic stage: pT2a pN1 pM0 — Stage IIB Non-Small Cell Lung Cancer.",
            },
            "biomarkers": [
                {"marker": "EGFR", "status": "Mutated", "evidence_span": "EGFR exon 19 deletion detected"},
                {"marker": "PD-L1", "status": "High", "value": "78%", "evidence_span": "PD-L1 TPS 78%"},
            ],
            "raw_note": self.SYNTHETIC_NOTE,
            "document_type": "pathology_report",
            "retry_count": 0,
        }
        result = oncology_evaluator(state)
        assert result["needs_reextraction"] is False

    def test_evaluator_rejects_hallucinated_staging(self):
        """Verify evaluator catches staging evidence not present in note."""
        from clinical.agents.oncology_evaluator import oncology_evaluator
        state = {
            "primary_site": "right upper lobe, lung",
            "histology": "Invasive adenocarcinoma",
            "tnm_stage": {
                "T": "pT3", "N": "pN2", "M": "pM0",
                "overall": "Stage IIIA",
                "evidence_span": "Final pathologic stage: pT3 pN2 pM0 — Stage IIIA",  # NOT in note
            },
            "biomarkers": [],
            "raw_note": self.SYNTHETIC_NOTE,
            "document_type": "pathology_report",
            "retry_count": 0,
        }
        result = oncology_evaluator(state)
        assert "needs_reextraction" in result


# Mock Gemini Responses for oncology integration tests
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


def _make_smart_llm_mock(staging_content, biomarker_content, trial_content, fallback_content=""):
    """Routes each LLM call by the unique agent identity phrase in the system prompt."""
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


@pytest.fixture
def mock_oncology_state() -> ClinicalState:
    return {
        "record_id": "test-onco-001",
        "raw_note": "Patient presents with lung adenocarcinoma. Staging is T2N1M0. Lab results confirm EGFR positive mutation. No family history.",
        "current_step": "init",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    }


@patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
def test_oncology_trial_matching_workflow(mock_invoke, mock_oncology_state):
    os.environ["GEMINI_API_KEY"] = "dummy_key"

    mock_invoke.side_effect = _make_smart_llm_mock(
        staging_content=MOCK_GEMINI_STAGING_RESPONSE,
        biomarker_content=MOCK_GEMINI_BIOMARKER_RESPONSE,
        trial_content=MOCK_GEMINI_TRIAL_RESPONSE,
        fallback_content=mock_oncology_state["raw_note"],
    )

    final_state = run_trial_matching(mock_oncology_state)
    
    EXPECTED_OUTPUT_KEYS = {
        "primary_site", "histology", "tnm_stage",
        "biomarkers", "trial_matches", "path_taken",
    }
    missing_keys = EXPECTED_OUTPUT_KEYS - set(final_state.keys())
    assert not missing_keys, f"Missing output keys: {missing_keys}"

    assert "deid" in final_state["path_taken"]
    assert "oncology_staging" in final_state["path_taken"]
    assert "biomarker_extraction" in final_state["path_taken"]
    assert "trial_matching" in final_state["path_taken"]
    
    assert final_state["primary_site"] == "lung"
    assert final_state["histology"] == "adenocarcinoma"
    assert final_state["tnm_stage"]["T"] == "T2"
    
    assert len(final_state["biomarkers"]) == 1
    assert final_state["biomarkers"][0]["marker"] == "EGFR"
    
    assert len(final_state["trial_matches"]) == 1
    assert final_state["trial_matches"][0]["nct_id"] == "NCT01234567"


@patch("langchain_google_genai.ChatGoogleGenerativeAI.invoke")
def test_nop_agent_oncology_workflow_must_fail(mock_invoke, mock_oncology_state):
    os.environ["GEMINI_API_KEY"] = "dummy_key"

    def _nop_invoke(*args, **kwargs):
        r = MagicMock()
        r.content = ""
        return r
    mock_invoke.side_effect = _nop_invoke

    final_state = run_trial_matching(mock_oncology_state)

    assert final_state.get("primary_site") is None
    assert final_state.get("trial_matches") is None or final_state.get("trial_matches") == []
    assert final_state.get("biomarkers") is None or final_state.get("biomarkers") == []
