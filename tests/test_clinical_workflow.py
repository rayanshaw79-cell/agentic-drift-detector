"""
tests/test_clinical_workflow.py — Test suite for the Clinical Coding Agent.

Tests are grouped into:
  1. Unit tests: individual step functions
  2. Integration tests: full LangGraph workflow
  3. Drift signal tests: clinical-specific drift scoring
"""

import pytest
from unittest.mock import patch, MagicMock


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_clinical_state(**overrides) -> dict:
    """Return a minimal ClinicalState-compatible dict."""
    base = {
        "record_id":           "test-001",
        "raw_note":            "Patient presents with essential hypertension (HTN).",
        "extracted_diagnoses": None,
        "icd10_codes":         None,
        "clinical_record":     None,
        "coding_status":       None,
        "overall_confidence":  None,
        "current_step":        "",
        "step_count":          0,
        "retry_count":         0,
        "path_taken":          [],
        "execution_time_ms":   0,
    }
    base.update(overrides)
    return base


# ── 1. NER Step ───────────────────────────────────────────────────────────────

class TestNerStep:
    def test_regex_fallback_extracts_hypertension(self):
        """Regex extractor should detect 'hypertension' in note."""
        from clinical.steps.ner_step import _regex_ner
        results = _regex_ner("Patient has hypertension and is on medication.")
        assert any("hypertension" in r for r in results), (
            f"Expected hypertension in results, got: {results}"
        )

    def test_regex_fallback_returns_default_on_no_match(self):
        """Regex extractor should return an empty list when nothing matches.
        The 'unspecified condition' placeholder is now added at the ensemble level.
        """
        from clinical.steps.ner_step import _regex_ner
        results = _regex_ner("Routine annual checkup, all normal.")
        assert results == []

    def test_ner_step_populates_state(self):
        """ner_step should return extracted_diagnoses in the result dict."""
        state = _make_clinical_state(raw_note="Patient has diabetes and hypertension.")
        with patch.dict("os.environ", {}, clear=False):
            # Ensure no Gemini key so we hit the regex path
            import os
            os.environ.pop("GEMINI_API_KEY", None)
            from clinical.steps.ner_step import ner_step
            result = ner_step(state)
        assert "extracted_diagnoses" in result
        assert isinstance(result["extracted_diagnoses"], list)
        assert len(result["extracted_diagnoses"]) > 0


# ── 2. NLM API Tool ───────────────────────────────────────────────────────────

class TestNlmApi:
    def test_icd10_lookup_returns_list(self):
        """
        ICD-10 lookup should return a list (even if the network is unavailable,
        it should not raise — it returns []).
        """
        from clinical.tools.nlm_api import lookup_icd10
        result = lookup_icd10("essential hypertension")
        assert isinstance(result, list)

    def test_icd10_lookup_finds_i10(self):
        """
        Live call: essential hypertension should include ICD-10 code I10.
        This test is skipped if the NLM API is unreachable or times out.
        """
        pytest.importorskip("requests")
        from clinical.tools.nlm_api import lookup_icd10
        try:
            result = lookup_icd10("essential hypertension", max_results=5)
        except Exception:
            pytest.skip("NLM API unreachable")

        if not result:
            pytest.skip("NLM API returned no results (likely timeout)")

        codes = [r["code"] for r in result]
        assert any(c.startswith("I1") for c in codes), (
            f"Expected an I1x code for hypertension, got: {codes}"
        )

    def test_icd10_lookup_empty_term_returns_empty(self):
        """Empty term should return [] without raising."""
        from clinical.tools.nlm_api import lookup_icd10
        assert lookup_icd10("") == []


# ── 3. Validation Step ────────────────────────────────────────────────────────

class TestValidationStep:
    def test_valid_codes_produce_complete_status(self):
        """A state with high-confidence codes should result in 'complete'."""
        from clinical.steps.validation_step import validation_step
        state = _make_clinical_state(
            icd10_codes=[
                {"term": "hypertension", "code": "I10",
                 "description": "Essential hypertension", "confidence": 0.92},
            ]
        )
        result = validation_step(state)
        assert result["coding_status"] == "complete"
        assert result["retry_count"] == 0
        assert result["clinical_record"]["total_codes"] == 1

    def test_unresolved_codes_trigger_retry(self):
        """If all codes are UNRESOLVED, retry_count should increment."""
        from clinical.steps.validation_step import validation_step
        state = _make_clinical_state(
            icd10_codes=[
                {"term": "xyzzy", "code": "UNRESOLVED",
                 "description": "No match", "confidence": 0.0},
            ]
        )
        result = validation_step(state)
        assert result["coding_status"] == "requires_clinical_review"
        assert result["retry_count"] == 1

    def test_low_confidence_codes_are_filtered(self):
        """Codes below the confidence threshold should not appear in the record."""
        from clinical.steps.validation_step import validation_step
        state = _make_clinical_state(
            icd10_codes=[
                {"term": "hypertension", "code": "I10",
                 "description": "Essential hypertension", "confidence": 0.3},
            ]
        )
        result = validation_step(state)
        assert result["coding_status"] == "requires_clinical_review"


# ── 4. Clinical Intervention Step ─────────────────────────────────────────────

class TestClinicalInterventionStep:
    def test_intervention_sets_review_status(self):
        """Intervention node must always set coding_status to requires_clinical_review."""
        from clinical.steps.clinical_intervention_step import clinical_intervention_step
        state = _make_clinical_state(
            retry_count=2,
            overall_confidence=0.2,
            icd10_codes=[{"term": "x", "code": "UNRESOLVED",
                          "description": "?", "confidence": 0.0}],
        )
        # trigger_alert is imported lazily inside the function body,
        # so we patch at the source module level.
        with patch("alerts.alert_manager.trigger_alert"):
            result = clinical_intervention_step(state)
        assert result["coding_status"] == "requires_clinical_review"
        assert "clinical_intervention" in result["path_taken"]

    def test_intervention_strips_unresolved_codes(self):
        """Unresolved codes should be removed by the intervention node."""
        from clinical.steps.clinical_intervention_step import clinical_intervention_step
        state = _make_clinical_state(
            retry_count=2,
            icd10_codes=[
                {"term": "x",    "code": "UNRESOLVED", "description": "?", "confidence": 0.0},
                {"term": "htn",  "code": "I10",        "description": "Essential HTN", "confidence": 0.85},
            ],
        )
        # Patch at the actual module where trigger_alert is defined
        with patch("alerts.alert_manager.trigger_alert"):
            result = clinical_intervention_step(state)
        # Only the high-confidence code should survive
        assert all(c["code"] != "UNRESOLVED" for c in result["icd10_codes"])


# ── 5. Clinical Drift Signals ─────────────────────────────────────────────────

class TestClinicalDriftSignals:
    def _baseline(self):
        return {
            "avg_steps": 4.0, "avg_retries": 0.0, "avg_latency": 100.0,
            "escalation_rate": 0.2, "high_severity_rate": 0.2,
            "low_severity_escalation_rate": 0.05,
            "avg_coding_confidence": 0.80,
            "avg_unresolved_rate": 0.05,
        }

    def test_coding_confidence_drift_healthy(self):
        from drift.drift_detector import coding_confidence_drift
        state = _make_clinical_state(overall_confidence=0.85)
        assert coding_confidence_drift(state, self._baseline()) == 0

    def test_coding_confidence_drift_fires(self):
        from drift.drift_detector import coding_confidence_drift
        state = _make_clinical_state(overall_confidence=0.30)
        assert coding_confidence_drift(state, self._baseline()) == 35

    def test_unresolved_entity_drift_healthy(self):
        from drift.drift_detector import unresolved_entity_drift
        state = _make_clinical_state(icd10_codes=[
            {"code": "I10", "confidence": 0.9},
        ])
        assert unresolved_entity_drift(state, self._baseline()) == 0

    def test_unresolved_entity_drift_fires(self):
        from drift.drift_detector import unresolved_entity_drift
        state = _make_clinical_state(icd10_codes=[
            {"code": "UNRESOLVED", "confidence": 0.0},
            {"code": "UNRESOLVED", "confidence": 0.0},
            {"code": "UNRESOLVED", "confidence": 0.0},
        ])
        assert unresolved_entity_drift(state, self._baseline()) > 0


# ── 6. Full Workflow Integration ──────────────────────────────────────────────

class TestClinicalWorkflowIntegration:
    def _initial_state(self, note: str) -> dict:
        import uuid
        return {
            "record_id":           str(uuid.uuid4())[:8],
            "raw_note":            note,
            "extracted_diagnoses": None,
            "icd10_codes":         None,
            "clinical_record":     None,
            "coding_status":       None,
            "overall_confidence":  None,
            "current_step":        "",
            "step_count":          0,
            "retry_count":         0,
            "path_taken":          [],
            "execution_time_ms":   0,
        }

    def test_full_workflow_produces_output(self, tmp_path, monkeypatch):
        """
        Integration: the full LangGraph graph should run without exception
        and populate coding_status and path_taken.
        Uses the regex NER path (no GEMINI_API_KEY) and the live NLM API.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        # Redirect DB to a temp file so we don't pollute the real DB
        monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))

        from workflows.clinical_coding import clinical_coding_workflow
        state = self._initial_state("Patient with essential hypertension and diabetes.")
        result = clinical_coding_workflow(state)

        assert result.get("coding_status") in ("complete", "requires_clinical_review")
        assert "ner" in result.get("path_taken", [])
        assert "clinical_output" in result.get("path_taken", [])

    def test_workflow_triggers_intervention_on_low_confidence(self, monkeypatch):
        """
        If the disambiguation step forces overall_confidence to 0, the workflow
        should route through clinical_intervention before clinical_output.
        """
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with patch("clinical.steps.disambiguation_step._fallback_select") as mock_dis, \
             patch("alerts.alert_manager.trigger_alert"):
            # Simulate disambiguation always returning confidence=0
            mock_dis.return_value = [{"term": "x", "code": "UNRESOLVED",
                                      "description": "?", "confidence": 0.0}]
            from workflows.clinical_coding import clinical_coding_workflow as wf
            state = self._initial_state("Unresolvable gibberish xyzzy qwerty.")
            result = wf(state)

        assert result.get("coding_status") == "requires_clinical_review"
