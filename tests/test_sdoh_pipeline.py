"""
tests/test_sdoh_pipeline.py — Test suite for the SDOH Longitudinal Risk Agent.

Tests:
  1. Patient simulator generates valid records.
  2. Risk trajectory step produces per-visit scores.
  3. Intervention flag fires on accelerating risk delta.
  4. End-to-end SDOH workflow produces a complete report.
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_visits():
    """A minimal 5-visit patient history with increasing risk signals."""
    base = {
        "patient_id": "PT-TEST",
        "age": 55, "gender": "M", "race": "Hispanic",
        "zip_code": "73301",
        "smoking_flag": 1, "alcohol_flag": 0, "exercise_score": 0.2,
        "food_risk_score": 0.7, "env_aqi": 140.0, "env_poverty_rate": 0.30,
        "icd10_codes": "E11.9", "icd10_code_count": 1, "chain_stage": 0,
    }
    visits = []
    for i in range(5):
        v = {**base,
             "visit_number":   i + 1,
             "visit_date":     f"2024-0{i+1}-15",
             "hcc_score":      round(0.2 + i * 0.15, 2),
             "chain_stage":    i,
             "sdoh_risk_score": round(0.20 + i * 0.12, 3),
             "icd10_code_count": i + 1}
        visits.append(v)
    return visits


@pytest.fixture
def minimal_sdoh_state(sample_visits):
    return {
        "patient_id":        "PT-TEST",
        "visit_history":     sample_visits,
        "sdoh_profile":      None,
        "risk_trajectory":   None,
        "risk_delta":        None,
        "predicted_risk_label": None,
        "shap_factors":      None,
        "intervention_flag": None,
        "intervention_reason": None,
        "sdoh_report":       None,
        "current_step":      "start",
        "path_taken":        [],
        "execution_time_ms": 0,
    }


# ── Test 1: Synthetic Data Generator ─────────────────────────────────────────

class TestPatientSimulator:
    def test_generates_expected_columns(self):
        from clinical.sdoh.patient_simulator import generate_dataset
        df = generate_dataset(n_patients=10, n_months=4)
        required = {"patient_id", "visit_number", "smoking_flag", "hcc_score",
                    "sdoh_risk_score", "sdoh_risk_label", "icd10_codes"}
        assert required.issubset(set(df.columns))

    def test_generates_correct_row_count(self):
        from clinical.sdoh.patient_simulator import generate_dataset
        df = generate_dataset(n_patients=20, n_months=6)
        assert len(df) == 20 * 6

    def test_risk_labels_are_valid(self):
        from clinical.sdoh.patient_simulator import generate_dataset
        df = generate_dataset(n_patients=50, n_months=3)
        valid_labels = {"low", "moderate", "high", "critical"}
        assert set(df["sdoh_risk_label"].unique()).issubset(valid_labels)

    def test_hcc_scores_are_positive(self):
        from clinical.sdoh.patient_simulator import generate_dataset
        df = generate_dataset(n_patients=10, n_months=3)
        assert (df["hcc_score"] >= 0).all()


# ── Test 2: SDOH Extraction Step ─────────────────────────────────────────────

class TestSdohExtractionStep:
    def test_profile_built_from_visit_history(self, minimal_sdoh_state):
        from clinical.sdoh.sdoh_extraction_step import sdoh_extraction_step
        result = sdoh_extraction_step(minimal_sdoh_state)
        assert result["sdoh_profile"] is not None
        assert result["sdoh_profile"]["patient_id"] == "PT-TEST"
        assert result["sdoh_profile"]["smoking_flag"] == 1
        assert "env_aqi" in result["sdoh_profile"]

    def test_empty_history_returns_empty_profile(self):
        from clinical.sdoh.sdoh_extraction_step import sdoh_extraction_step
        state = {
            "patient_id": "PT-EMPTY", "visit_history": [],
            "current_step": "start", "path_taken": [], "execution_time_ms": 0,
        }
        result = sdoh_extraction_step(state)
        assert result["sdoh_profile"] == {}


# ── Test 3: Risk Trajectory Step (with mocked model) ─────────────────────────

class TestRiskTrajectoryStep:
    def test_produces_trajectory_list(self, minimal_sdoh_state, sample_visits):
        """Verify trajectory is generated with one score per visit."""
        from clinical.sdoh.risk_trajectory_step import risk_trajectory_step

        mock_bundle = MagicMock()
        mock_proba  = ("moderate", 0.45, [{"feature": "hcc_score", "contribution": 0.12}])

        with patch("clinical.sdoh.risk_trajectory_step.load", return_value=mock_bundle), \
             patch("clinical.sdoh.risk_trajectory_step.predict_proba", return_value=mock_proba):
            state = {**minimal_sdoh_state, "sdoh_profile": sample_visits[-1]}
            result = risk_trajectory_step(state)

        assert "risk_trajectory" in result
        assert len(result["risk_trajectory"]) == len(sample_visits)

    def test_risk_delta_computed(self, minimal_sdoh_state, sample_visits):
        from clinical.sdoh.risk_trajectory_step import risk_trajectory_step

        mock_bundle = MagicMock()
        # Simulate increasing scores
        call_count = {"n": 0}
        scores = [0.15, 0.25, 0.40, 0.55, 0.70]
        label_map = {0.15: "low", 0.25: "low", 0.40: "moderate",
                     0.55: "moderate", 0.70: "high"}

        def mock_predict(bundle, visit):
            n = call_count["n"] % len(scores)
            call_count["n"] += 1
            s = scores[n]
            return label_map[s], s, []

        with patch("clinical.sdoh.risk_trajectory_step.load", return_value=mock_bundle), \
             patch("clinical.sdoh.risk_trajectory_step.predict_proba", side_effect=mock_predict):
            state = {**minimal_sdoh_state, "sdoh_profile": sample_visits[-1]}
            result = risk_trajectory_step(state)

        assert result["risk_delta"] != 0.0


# ── Test 4: Intervention Check Step ──────────────────────────────────────────

class TestInterventionCheckStep:
    def test_flags_critical_label(self):
        from clinical.sdoh.intervention_check_step import intervention_check_step
        state = {
            "patient_id": "PT-001",
            "risk_trajectory": [0.2, 0.4, 0.95],
            "risk_delta": 0.05,
            "predicted_risk_label": "critical",
            "current_step": "start", "path_taken": [], "execution_time_ms": 0,
        }
        result = intervention_check_step(state)
        assert result["intervention_flag"] is True
        assert "CRITICAL" in result["intervention_reason"]

    def test_flags_large_delta(self):
        from clinical.sdoh.intervention_check_step import intervention_check_step
        state = {
            "patient_id": "PT-002",
            "risk_trajectory": [0.2, 0.45],
            "risk_delta": 0.25,
            "predicted_risk_label": "high",
            "current_step": "start", "path_taken": [], "execution_time_ms": 0,
        }
        result = intervention_check_step(state)
        assert result["intervention_flag"] is True

    def test_no_flag_for_stable_patient(self):
        from clinical.sdoh.intervention_check_step import intervention_check_step
        state = {
            "patient_id": "PT-003",
            "risk_trajectory": [0.15, 0.14, 0.16, 0.15],
            "risk_delta": 0.01,
            "predicted_risk_label": "low",
            "current_step": "start", "path_taken": [], "execution_time_ms": 0,
        }
        result = intervention_check_step(state)
        assert result["intervention_flag"] is False


# ── Test 5: End-to-End Workflow ───────────────────────────────────────────────

class TestSdohWorkflowIntegration:
    def test_workflow_produces_sdoh_report(self, minimal_sdoh_state, sample_visits):
        """
        Full pipeline test using mocked model.
        Verifies the final state contains a populated sdoh_report.
        """
        from clinical.sdoh.sdoh_workflow import build_sdoh_graph

        mock_bundle = MagicMock()
        mock_proba  = ("high", 0.72, [{"feature": "hcc_score", "contribution": 0.18}])

        with patch("clinical.sdoh.risk_trajectory_step.load", return_value=mock_bundle), \
             patch("clinical.sdoh.risk_trajectory_step.predict_proba", return_value=mock_proba):
            graph  = build_sdoh_graph()
            result = graph.invoke(minimal_sdoh_state)

        assert result["sdoh_report"] is not None
        report = result["sdoh_report"]
        assert report["patient_id"]    == "PT-TEST"
        assert report["predicted_risk_label"] in {"low", "moderate", "high", "critical"}
        assert isinstance(report["risk_trajectory"], list)
        assert len(report["risk_trajectory"]) == len(sample_visits)
