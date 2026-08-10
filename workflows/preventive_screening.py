from langgraph.graph import StateGraph, END
from schemas.clinical_state import ClinicalState
from clinical.steps.deid_step import deid_step
from clinical.steps.lifestyle_ner_step import lifestyle_ner_step
from clinical.steps.preventive_rag_step import preventive_rag_step
from clinical.steps.fhir_step import fhir_step

def route_by_risk(state: ClinicalState) -> str:
    score = state.get("lifestyle_risk_score", 0.0)
    if score == 0.0:
        return "healthy"
    elif score > 0.7:
        return "critical"
    else:
        return "risk_detected"

def build_preventive_workflow() -> StateGraph:
    workflow = StateGraph(ClinicalState)

    # Nodes
    workflow.add_node("deid", deid_step)
    workflow.add_node("lifestyle_ner", lifestyle_ner_step)
    workflow.add_node("preventive_rag", preventive_rag_step)
    workflow.add_node("fhir_export", fhir_step)

    workflow.set_entry_point("deid")
    workflow.add_edge("deid", "lifestyle_ner")
    
    workflow.add_conditional_edges(
        "lifestyle_ner",
        route_by_risk,
        {
            "healthy": END,
            "risk_detected": "preventive_rag",
            "critical": "fhir_export"
        }
    )

    workflow.add_edge("preventive_rag", END)
    workflow.add_edge("fhir_export", END)

    return workflow.compile()

def preventive_screening_workflow(initial_state: ClinicalState) -> ClinicalState:
    graph = build_preventive_workflow()
    return graph.invoke(initial_state)
