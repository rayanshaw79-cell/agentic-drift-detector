from langgraph.graph import StateGraph, END
from schemas.incident_state import IncidentState
from steps.triage import triage_step
from steps.investigate import investigation_step
from steps.decision import decision_step
from steps.notification import notification_step

def should_retry(state: IncidentState) -> str:
    # We allow 1 retry. This means the decision node can execute a maximum of 2 times.
    path_taken = state.get("path_taken", [])
    decision_count = path_taken.count("decision")
    
    if state.get("confidence", 1.0) < 0.6 and decision_count < 2:
        return "decision"
    return "notification"

def build_workflow() -> StateGraph:
    workflow = StateGraph(IncidentState)

    # Add nodes
    workflow.add_node("triage", triage_step)
    workflow.add_node("investigation", investigation_step)
    workflow.add_node("decision", decision_step)
    workflow.add_node("notification", notification_step)

    # Add edges
    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "investigation")
    workflow.add_edge("investigation", "decision")
    
    # Conditional edge for decision retry loop
    workflow.add_conditional_edges(
        "decision",
        should_retry,
        {
            "decision": "decision",
            "notification": "notification"
        }
    )
    
    workflow.add_edge("notification", END)

    return workflow.compile()

# For backwards compatibility with run.py during refactoring
def incident_triage_workflow(initial_state: IncidentState) -> IncidentState:
    graph = build_workflow()
    final_state = graph.invoke(initial_state)
    return final_state
