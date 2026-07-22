import pytest
from drift.drift_detector import (
    escalation_bias,
    classification_bias,
    latency_drift,
    step_count_drift,
    retry_drift
)

# ── Harbor Compliance Note ────────────────────────────────────────────────────
# Structural checks guard against key-deletion exploits (Challenge 4).
# NOP test (test_nop_empty_state_must_not_be_healthy) verifies that a
# completely empty/zero state is not silently passed as "healthy".
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_baseline():
    return {
        "avg_steps": 4.0,
        "avg_retries": 0.0,
        "avg_latency": 1000.0,
        "escalation_rate": 0.2,
        "high_severity_rate": 0.1,  # Rare to have high severity
        "low_severity_escalation_rate": 0.05  # Very rare to escalate low severity
    }

def test_escalation_bias_anomalous(healthy_baseline):
    # Agent escalates a low severity incident when normally it doesn't
    state = {
        "severity": "low",
        "decision": "escalate"
    }
    
    score = escalation_bias(state, healthy_baseline)
    assert score == 25, "Expected high penalty for anomalously escalating low severity"

def test_escalation_bias_normal(healthy_baseline):
    # Agent auto-resolves a low severity incident
    state = {
        "severity": "low",
        "decision": "auto_resolve"
    }
    
    score = escalation_bias(state, healthy_baseline)
    assert score == 0, "Expected no penalty for healthy resolution"

def test_classification_bias_anomalous(healthy_baseline):
    # Agent categorizes as high severity when normally it's 10%
    state = {"severity": "high"}
    
    score = classification_bias(state, healthy_baseline)
    assert score == 20, "Expected penalty for anomalous high severity classification"

def test_latency_drift_high(healthy_baseline):
    # Baseline is 1000ms. Expected threshold is max(500, 1500ms).
    # Actual is 5500ms. Overflow is 4000ms (4 seconds).
    # penalty is min(20, int(4 * 5)) = 20.
    state = {"execution_time_ms": 5500}
    
    score = latency_drift(state, healthy_baseline)
    assert score == 20, "Expected max penalty for 4 seconds of latency overflow"

def test_step_count_drift(healthy_baseline):
    # Max steps is max(5, 4+1) = 5. Actual is 7. Overflow is 2.
    # penalty is min(30, 20) = 20.
    state = {"step_count": 7}
    
    score = step_count_drift(state, healthy_baseline)
    assert score == 20


def test_retry_drift_normal(healthy_baseline):
    """No penalty when retries are within baseline tolerance."""
    state = {"retry_count": 0}
    assert retry_drift(state, healthy_baseline) == 0


def test_retry_drift_anomalous(healthy_baseline):
    """Penalty applied when retries significantly exceed baseline."""
    # Baseline avg_retries=0 → max_retries=1; actual=3 → overflow=2 → min(25, 30)=25
    state = {"retry_count": 3}
    score = retry_drift(state, healthy_baseline)
    assert score == 25, f"Expected 25, got {score}"


# ─── Empty-DB baseline tests ─────────────────────────────────────────────────

def test_empty_db_returns_default_baseline():
    """
    get_historical_metrics() must return safe defaults when the database is
    empty — it must not crash or return None values.
    The temp_db fixture (conftest.py) ensures we start with a fresh, empty DB.
    """
    from telemetry.store import get_historical_metrics

    baseline = get_historical_metrics()

    assert isinstance(baseline, dict), "Baseline must be a dict"
    assert baseline["avg_steps"] == 4.0
    assert baseline["avg_retries"] == 0.0
    assert baseline["avg_latency"] == 100.0
    assert 0 <= baseline["escalation_rate"] <= 1
    assert 0 <= baseline["high_severity_rate"] <= 1
    assert 0 <= baseline["low_severity_escalation_rate"] <= 1


def test_empty_db_drift_scoring_does_not_crash():
    """
    analyze_workflow() should complete without error even with an empty
    telemetry database (falls back to default baseline).
    """
    from drift.drift_detector import analyze_workflow

    state = {
        "incident_id": "test-empty",
        "severity": "medium",
        "decision": "auto_resolve",
        "confidence": 0.85,
        "step_count": 4,
        "retry_count": 0,
        "path_taken": ["triage", "investigation", "decision", "notification"],
        "execution_time_ms": 300,
    }

    result = analyze_workflow(state)

    # ── Structural Check (Harbor: Challenge 4) ─────────────────────────────
    # Validate the full key structure of the analysis result exists BEFORE
    # checking individual values, preventing the structural-deletion exploit.
    EXPECTED_ANALYSIS_KEYS = {"drift_score", "risk_level"}
    missing_keys = EXPECTED_ANALYSIS_KEYS - set(result.keys())
    assert not missing_keys, (
        f"Structural check failed: analysis result is missing keys {missing_keys}."
    )
    # ─────────────────────────────────────────────────────────────────────────

    assert "drift_score" in result
    assert "risk_level" in result
    assert result["risk_level"] in ("healthy", "drift_detected", "high_risk")
    assert isinstance(result["drift_score"], int)


def test_nop_empty_state_must_not_be_healthy():
    """
    Harbor NOP Test (Challenge 3 — Weak Verification).

    Simulates a NOP agent that submits a completely hollow/zeroed-out state.
    The drift analyzer must NOT classify this as "healthy". If it does,
    the verifier has a NOP-pass bug: doing nothing passes the benchmark.

    A zero-step, zero-latency, zero-retry "ghost" execution should never
    be awarded a clean bill of health by a correctly written verifier.
    """
    from drift.drift_detector import analyze_workflow

    # NOP state: agent did nothing, all values are empty/zero
    nop_state = {
        "incident_id": "nop-test-ghost",
        "severity": "high",         # High severity incident
        "decision": "auto_resolve",  # Claimed to auto-resolve with no work
        "confidence": 0.0,
        "step_count": 0,             # No steps taken
        "retry_count": 0,
        "path_taken": [],            # No path recorded
        "execution_time_ms": 0,      # Zero latency = no work done
    }

    result = analyze_workflow(nop_state)

    # A ghost execution (0 steps, 0 latency, empty path) auto-resolving a
    # high-severity incident should trigger at least some drift signal.
    # If risk_level is "healthy" here, the verifier is too weak (NOP passes).
    assert result["risk_level"] != "healthy", (
        "NOP FAILURE: A zero-step, zero-latency ghost state was classified as "
        "'healthy'. The verifier must detect this as anomalous drift, not a "
        "valid resolution. The verifier is too weak."
    )
