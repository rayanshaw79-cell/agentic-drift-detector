from langgraph.graph import StateGraph, END
from schemas.incident_state import IncidentState
from steps.triage import triage_step
from steps.investigate import investigation_step
from steps.decision import decision_step
from steps.notification import notification_step
from steps.intervention import intervention_step


def should_retry(state: IncidentState) -> str:
    """
    Conditional edge after the decision node.
    
    - If confidence is low AND we haven't retried yet → retry decision
    - If confidence is low AND we've already retried → trigger intervention (agentic healing)
    - Otherwise → proceed to notification
    """
    path_taken = state.get("path_taken", [])
    decision_count = path_taken.count("decision")
    confidence = state.get("confidence", 1.0)
    retry_count = state.get("retry_count", 0)

    if confidence < 0.6 and decision_count < 2 and retry_count < 2:
        return "decision"
    elif confidence < 0.6 and retry_count >= 2:
        # Drift loop detected — send to intervention/healing
        return "intervention"
    return "notification"


def build_workflow() -> StateGraph:
    workflow = StateGraph(IncidentState)

    # Add nodes
    workflow.add_node("triage", triage_step)
    workflow.add_node("investigation", investigation_step)
    workflow.add_node("decision", decision_step)
    workflow.add_node("notification", notification_step)
    workflow.add_node("intervention", intervention_step)

    # Add edges
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "investigation")
    workflow.add_edge("investigation", "decision")

    # Conditional edge: retry → intervention → notification
    workflow.add_conditional_edges(
        "decision",
        should_retry,
        {
            "decision": "decision",
            "intervention": "intervention",
            "notification": "notification"
        }
    )

    # Intervention always terminates cleanly to notification
    workflow.add_edge("intervention", "notification")
    workflow.add_edge("notification", END)

    return workflow.compile()


# For backwards compatibility with run.py
def incident_triage_workflow(initial_state: IncidentState) -> IncidentState:
    graph = build_workflow()
    final_state = graph.invoke(initial_state)
    return final_state
