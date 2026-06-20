import pytest
from drift.drift_detector import calculate_drift_score

def test_ml_drift_detector_healthy():
    state = {
        "step_count": 3,
        "retry_count": 0,
        "execution_time_ms": 300,
        "severity": "low",
        "decision": "auto_resolve"
    }
    baseline = {}
    score = calculate_drift_score(state, baseline)
    # Healthy states should have a low anomaly score
    assert score < 30

def test_ml_drift_detector_anomalous():
    state = {
        "step_count": 8,
        "retry_count": 4,
        "execution_time_ms": 2500,
        "severity": "high",
        "decision": "auto_resolve" # Anomalous decision for high severity
    }
    baseline = {}
    score = calculate_drift_score(state, baseline)
    # Extremely anomalous states should be flagged as drift or high risk
    assert score >= 60
