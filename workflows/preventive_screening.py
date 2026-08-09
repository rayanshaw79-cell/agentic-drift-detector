from langgraph.graph import StateGraph, END
from schemas.clinical_state import ClinicalState
from clinical.steps.deid_step import deid_step
from clinical.steps.lifestyle_ner_step import lifestyle_ner_step
from clinical.steps.preventive_rag_step import preventive_rag_step

def build_preventive_workflow() -> StateGraph:
    workflow = StateGraph(ClinicalState)

    # Nodes
    workflow.add_node("deid", deid_step)
    workflow.add_node("lifestyle_ner", lifestyle_ner_step)
    workflow.add_node("preventive_rag", preventive_rag_step)

    # Simple linear graph for MVP
    workflow.set_entry_point("deid")
    workflow.add_edge("deid", "lifestyle_ner")
    workflow.add_edge("lifestyle_ner", "preventive_rag")
    workflow.add_edge("preventive_rag", END)

    return workflow.compile()

def preventive_screening_workflow(initial_state: ClinicalState) -> ClinicalState:
    graph = build_preventive_workflow()
    return graph.invoke(initial_state)
