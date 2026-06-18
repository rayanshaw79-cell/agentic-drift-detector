"""
tests/test_meat_validation.py — Unit and integration tests for MEAT Criteria Validation.

Tests cover:
  - Regex fallback correctly identifies MEAT evidence for treated conditions
  - Regex fallback returns meat_met=False for historical-only mentions
  - MEAT fields are correctly attached to icd10_codes in state
  - Validation step respects meat_met flags (zeroes RAF weight when MEAT fails)
  - Full pipeline correctly separates active vs. historical conditions
"""

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_code(code: str, term: str, confidence: float = 0.85, raf_weight: float = 0.302) -> dict:
    return {
        "code":         code,
        "term":         term,
        "description":  f"{term} (mock description)",
        "confidence":   confidence,
        "hcc":          37,
        "hcc_category": "Test",
        "raf_weight":   raf_weight,
    }


# ── Regex Fallback Tests ───────────────────────────────────────────────────────

class TestMeatRegexFallback:

    def test_treated_condition_detected(self):
        """
        When a note contains 'adjusted' near a condition term,
        the regex fallback should classify it as Treated.
        """
        from clinical.steps.meat_validation_step import _regex_meat_validate

        note  = "Patient has hypertension. We adjusted lisinopril dosage to 10mg today."
        codes = [_make_code("I10", "hypertension")]
        results = _regex_meat_validate(note, codes)

        assert len(results) == 1
        r = results[0]
        assert r["meat_met"]      is True,       f"Expected meat_met=True, got {r}"
        assert r["meat_category"] == "Treated",  f"Expected Treated, got {r['meat_category']}"
        assert "lisinopril" in r["meat_evidence"].lower(), \
            f"Expected evidence to contain 'lisinopril', got: {r['meat_evidence']}"

    def test_monitored_condition_detected(self):
        """Notes using 'monitoring' should produce meat_category=Monitored."""
        from clinical.steps.meat_validation_step import _regex_meat_validate

        note  = "Patient with diabetes mellitus. Monitoring HbA1c levels closely."
        codes = [_make_code("E11.9", "diabetes mellitus")]
        results = _regex_meat_validate(note, codes)

        r = results[0]
        assert r["meat_met"] is True
        assert r["meat_category"] in ("Monitored", "Assessed", "Evaluated")

    def test_historical_only_fails_meat(self):
        """
        Past Medical History mentions alone should NOT satisfy MEAT.
        Note: regex is keyword-based so it won't catch all nuance, but a
        pure history-with-no-action note should return False.
        """
        from clinical.steps.meat_validation_step import _regex_meat_validate

        note  = "PMH: Patient has a history of asthma. No current complaints."
        codes = [_make_code("J45.998", "asthma")]
        results = _regex_meat_validate(note, codes)

        r = results[0]
        # No MEAT keywords in the note — should fail
        assert r["meat_met"]      is False,  f"Expected meat_met=False, got {r}"
        assert r["meat_category"] == "None", f"Expected None, got {r['meat_category']}"

    def test_multiple_conditions_partially_pass(self):
        """
        One treated condition and one historical condition.
        MEAT should pass for treated, fail for historical.
        """
        from clinical.steps.meat_validation_step import _regex_meat_validate

        note = (
            "Patient's mother had breast cancer. "
            "Patient presents with hypertension — we increased Amlodipine to 10mg."
        )
        codes = [
            _make_code("I10",   "hypertension"),
            _make_code("C50.9", "breast cancer"),
        ]
        results = _regex_meat_validate(note, codes)

        by_code = {r["code"]: r for r in results}
        assert by_code["I10"]["meat_met"]   is True,  "Hypertension should pass (treated)"
        assert by_code["C50.9"]["meat_met"] is False, "Breast cancer should fail (family hx)"


# ── MEAT Application Tests ────────────────────────────────────────────────────

class TestMeatApplication:

    def test_raf_weight_zeroed_for_failed_meat(self):
        """
        _apply_meat_results should zero out raf_weight if meat_met is False.
        """
        from clinical.steps.meat_validation_step import _apply_meat_results

        codes = [_make_code("E10", "diabetes", raf_weight=0.302)]
        meat_results = [{
            "code":          "E10",
            "meat_met":      False,
            "meat_category": "None",
            "meat_evidence": "",
        }]
        enriched = _apply_meat_results(codes, meat_results)
        assert enriched[0]["raf_weight"]   == 0.0,  "RAF weight should be zeroed"
        assert enriched[0]["meat_met"]     is False
        assert enriched[0]["meat_category"] == "None"

    def test_raf_weight_preserved_for_passed_meat(self):
        """
        _apply_meat_results should preserve raf_weight if meat_met is True.
        """
        from clinical.steps.meat_validation_step import _apply_meat_results

        codes = [_make_code("I48.0", "atrial fibrillation", raf_weight=0.270)]
        meat_results = [{
            "code":          "I48.0",
            "meat_met":      True,
            "meat_category": "Treated",
            "meat_evidence": "Started warfarin anticoagulation.",
        }]
        enriched = _apply_meat_results(codes, meat_results)
        assert enriched[0]["raf_weight"]   == 0.270, "RAF weight should be preserved"
        assert enriched[0]["meat_met"]     is True
        assert enriched[0]["meat_evidence"] == "Started warfarin anticoagulation."


# ── Validation Step MEAT Integration ─────────────────────────────────────────

class TestValidationMeatIntegration:

    def _run_validation(self, codes: list[dict]) -> dict:
        from clinical.steps.validation_step import validation_step
        state = {
            "icd10_codes":  codes,
            "record_id":    "test-001",
            "raw_note":     "Test note",
            "ner_votes":    [],
            "retry_count":  0,
        }
        return validation_step(state)

    def test_claims_ready_true_when_all_meat_pass(self):
        """claims_ready should be True when all valid codes have meat_met=True and high confidence."""
        codes = [{
            "code":         "I48.0",
            "term":         "atrial fibrillation",
            "description":  "Atrial fibrillation",
            "confidence":   0.90,
            "hcc":          96,
            "hcc_category": "Atrial Fibrillation",
            "raf_weight":   0.270,
            "meat_met":     True,
            "meat_category": "Treated",
            "meat_evidence": "Started warfarin.",
        }]
        result = self._run_validation(codes)
        assert result["claims_ready"] is True

    def test_claims_ready_false_when_meat_fails(self):
        """claims_ready should be False when all codes failed MEAT validation."""
        codes = [{
            "code":         "E11.9",
            "term":         "diabetes",
            "description":  "Type 2 diabetes mellitus",
            "confidence":   0.90,
            "hcc":          38,
            "hcc_category": "Diabetes",
            "raf_weight":   0.0,      # Already zeroed by meat_validation_step
            "meat_met":     False,
            "meat_category": "None",
            "meat_evidence": "",
        }]
        result = self._run_validation(codes)
        assert result["claims_ready"] is False

    def test_meat_summary_in_clinical_record(self):
        """The clinical_record should contain a meat_summary block."""
        codes = [
            {
                "code": "I10", "term": "hypertension", "description": "Hypertension",
                "confidence": 0.90, "hcc": None, "hcc_category": "Hypertension",
                "raf_weight": 0.0, "meat_met": True, "meat_category": "Treated",
                "meat_evidence": "Adjusted Lisinopril.",
            },
            {
                "code": "E11.9", "term": "diabetes", "description": "Diabetes",
                "confidence": 0.90, "hcc": 38, "hcc_category": "Diabetes",
                "raf_weight": 0.0, "meat_met": False, "meat_category": "None",
                "meat_evidence": "",
            },
        ]
        result = self._run_validation(codes)
        record = result.get("clinical_record", {})
        summary = record.get("meat_summary", {})
        assert summary["meat_passed"]  == 1, f"Expected 1 passed, got {summary}"
        assert summary["meat_failed"]  == 1, f"Expected 1 failed, got {summary}"
        assert summary["meat_pass_rate"] == 0.5
