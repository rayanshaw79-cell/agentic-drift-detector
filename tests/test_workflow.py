import pytest
from workflows.incident_triage import incident_triage_workflow
from schemas.incident_state import IncidentState

def test_workflow_healthy_auto_resolve(monkeypatch):
    # Mock the steps so we don't rely on random.py for tests
    def mock_triage(state):
        return {"current_step": "triage", "step_count": 1, "path_taken": ["triage"], "severity": "low", "execution_time_ms": 100}
    def mock_investigation(state):
        return {"current_step": "investigation", "step_count": 1, "path_taken": ["investigation"], "investigation_summary": "mocked", "execution_time_ms": 100}
    def mock_decision(state):
        return {"current_step": "decision", "step_count": 1, "path_taken": ["decision"], "confidence": 0.9, "decision": "auto_resolve", "retry_count": 0, "execution_time_ms": 100}
    def mock_notification(state):
        return {"current_step": "notification", "step_count": 1, "path_taken": ["notification"], "execution_time_ms": 100}

    monkeypatch.setattr("workflows.incident_triage.triage_step", mock_triage)
    monkeypatch.setattr("workflows.incident_triage.investigation_step", mock_investigation)
    monkeypatch.setattr("workflows.incident_triage.decision_step", mock_decision)
    monkeypatch.setattr("workflows.incident_triage.notification_step", mock_notification)

    # We must rebuild the graph after mocking the nodes
    from workflows.incident_triage import build_workflow
    graph = build_workflow()

    initial_state = {
        "incident_id": "test-123",
        "incident_text": "Test",
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

    final_state = graph.invoke(initial_state)

    assert final_state["step_count"] == 4
    assert final_state["retry_count"] == 0
    assert final_state["path_taken"] == ["triage", "investigation", "decision", "notification"]
    assert final_state["decision"] == "auto_resolve"
    assert final_state["execution_time_ms"] == 400

def test_workflow_low_confidence_retry(monkeypatch):
    # This mock simulates a decision node that ALWAYS returns low confidence and increments retry_count
    # The LangGraph edge should detect this, loop once, and then proceed.
    def mock_decision(state):
        retry_increment = 0
        if state.get("retry_count", 0) < 1:
            retry_increment = 1
            
        return {
            "current_step": "decision", 
            "step_count": 1, 
            "path_taken": ["decision"], 
            "confidence": 0.2, 
            "decision": "escalate", 
            "retry_count": retry_increment, 
            "execution_time_ms": 100
        }

    # We just need to replace decision_step. But we need to rebuild the graph cleanly.
    # To avoid import state issues, we use patch.
    monkeypatch.setattr("workflows.incident_triage.decision_step", mock_decision)

    from workflows.incident_triage import build_workflow
    graph = build_workflow()

    initial_state = {
        "incident_id": "test-retry",
        "incident_text": "Test",
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

    final_state = graph.invoke(initial_state)

    # 1 triage + 1 investigate + 2 decision + 1 notification = 5 steps
    assert final_state["step_count"] == 5
    assert final_state["retry_count"] == 1
    # Check that decision appears exactly twice in the path
    assert final_state["path_taken"].count("decision") == 2
