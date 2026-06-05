import pytest
from drift.drift_detector import (
    escalation_bias,
    classification_bias,
    latency_drift,
    step_count_drift,
    retry_drift
)

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
