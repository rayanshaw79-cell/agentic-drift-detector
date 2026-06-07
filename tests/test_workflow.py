import pytest
from workflows.incident_triage import incident_triage_workflow
from schemas.incident_state import IncidentState


def _base_state(incident_id="test-123"):
    return {
        "incident_id": incident_id,
        "incident_text": "Test incident",
        "severity": None,
        "investigation_summary": None,
        "decision": None,
        "confidence": None,
        "current_step": "",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0
    }


# ─── Shared mock helpers ─────────────────────────────────────────────────────

def _mock_triage(state):
    return {"current_step": "triage", "step_count": 1, "path_taken": ["triage"], "severity": "low", "execution_time_ms": 100}

def _mock_investigation(state):
    return {"current_step": "investigation", "step_count": 1, "path_taken": ["investigation"], "investigation_summary": "mocked", "execution_time_ms": 100}

def _mock_notification(state):
    return {"current_step": "notification", "step_count": 1, "path_taken": ["notification"], "execution_time_ms": 100}

def _mock_intervention(state):
    return {"current_step": "intervention", "step_count": 1, "path_taken": ["intervention"], "confidence": 0.7, "decision": "escalate", "execution_time_ms": 100}


# ─── Test 1: Healthy auto-resolve ───────────────────────────────────────────

def test_workflow_healthy_auto_resolve(monkeypatch):
    """A high-confidence decision should skip retry and go straight to notification."""
    def mock_decision(state):
        return {"current_step": "decision", "step_count": 1, "path_taken": ["decision"], "confidence": 0.9, "decision": "auto_resolve", "retry_count": 0, "execution_time_ms": 100}

    monkeypatch.setattr("workflows.incident_triage.triage_step", _mock_triage)
    monkeypatch.setattr("workflows.incident_triage.investigation_step", _mock_investigation)
    monkeypatch.setattr("workflows.incident_triage.decision_step", mock_decision)
    monkeypatch.setattr("workflows.incident_triage.notification_step", _mock_notification)
    monkeypatch.setattr("workflows.incident_triage.intervention_step", _mock_intervention)

    from workflows.incident_triage import build_workflow
    graph = build_workflow()
    final_state = graph.invoke(_base_state())

    assert final_state["step_count"] == 4
    assert final_state["retry_count"] == 0
    assert final_state["path_taken"] == ["triage", "investigation", "decision", "notification"]
    assert final_state["decision"] == "auto_resolve"
    assert final_state["execution_time_ms"] == 400


# ─── Test 2: Single retry on low confidence ─────────────────────────────────

def test_workflow_low_confidence_single_retry(monkeypatch):
    """A single low-confidence run should trigger one retry, then proceed to notification."""
    call_count = {"n": 0}

    def mock_decision(state):
        call_count["n"] += 1
        retry_increment = 1 if call_count["n"] == 1 else 0
        return {
            "current_step": "decision",
            "step_count": 1,
            "path_taken": ["decision"],
            "confidence": 0.2,
            "decision": "escalate",
            "retry_count": retry_increment,
            "execution_time_ms": 100
        }

    monkeypatch.setattr("workflows.incident_triage.triage_step", _mock_triage)
    monkeypatch.setattr("workflows.incident_triage.investigation_step", _mock_investigation)
    monkeypatch.setattr("workflows.incident_triage.decision_step", mock_decision)
    monkeypatch.setattr("workflows.incident_triage.notification_step", _mock_notification)
    monkeypatch.setattr("workflows.incident_triage.intervention_step", _mock_intervention)

    from workflows.incident_triage import build_workflow
    graph = build_workflow()
    final_state = graph.invoke(_base_state("test-retry"))

    # triage + investigation + decision + decision + notification = 5
    assert final_state["step_count"] == 5
    assert final_state["retry_count"] == 1
    assert final_state["path_taken"].count("decision") == 2
    assert "intervention" not in final_state["path_taken"]


# ─── Test 3: Intervention triggered on persistent drift loop ─────────────────

def test_workflow_intervention_on_drift_loop(monkeypatch):
    """If retry_count >= 2 and confidence is still low, the healing node must be triggered."""
    call_count = {"n": 0}

    def mock_decision_always_low(state):
        call_count["n"] += 1
        return {
            "current_step": "decision",
            "step_count": 1,
            "path_taken": ["decision"],
            "confidence": 0.1,
            "decision": "escalate",
            "retry_count": 1,   # Each call adds 1, so after 2 calls total = 2
            "execution_time_ms": 100
        }

    monkeypatch.setattr("workflows.incident_triage.triage_step", _mock_triage)
    monkeypatch.setattr("workflows.incident_triage.investigation_step", _mock_investigation)
    monkeypatch.setattr("workflows.incident_triage.decision_step", mock_decision_always_low)
    monkeypatch.setattr("workflows.incident_triage.notification_step", _mock_notification)
    monkeypatch.setattr("workflows.incident_triage.intervention_step", _mock_intervention)

    from workflows.incident_triage import build_workflow
    graph = build_workflow()
    final_state = graph.invoke(_base_state("test-intervention"))

    # Intervention node must appear in the path
    assert "intervention" in final_state["path_taken"]
    # After intervention, notification must still follow
    assert final_state["path_taken"][-1] == "notification"
    # Final decision should be forced to escalate by intervention
    assert final_state["decision"] == "escalate"
