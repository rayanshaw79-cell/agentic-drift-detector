"""
clinical/sdoh/sdoh_workflow.py — SDOH Longitudinal Risk LangGraph Agent.

Compiles a 4-node StateGraph that:
  1. sdoh_extraction     → builds structured SDOH profile from visit history
  2. risk_trajectory     → scores each visit and computes risk delta
  3. intervention_check  → flags accelerating or critical risk
  4. sdoh_output         → assembles the final SDOH report dict

Usage:
    from clinical.sdoh.sdoh_workflow import build_sdoh_graph
    from clinical.sdoh.patient_store import get_patient_history

    graph  = build_sdoh_graph()
    visits = get_patient_history("PT-0001")
    result = graph.invoke({
        "patient_id":    "PT-0001",
        "visit_history": visits,
        "current_step":  "start",
        "path_taken":    [],
        "execution_time_ms": 0,
    })
    print(result["sdoh_report"])
"""

from langgraph.graph import StateGraph, END

from schemas.sdoh_state import SdohState
from clinical.sdoh.sdoh_extraction_step    import sdoh_extraction_step
from clinical.sdoh.risk_trajectory_step    import risk_trajectory_step
from clinical.sdoh.intervention_check_step import intervention_check_step
from clinical.sdoh.sdoh_output_step        import sdoh_output_step


def build_sdoh_graph() -> StateGraph:
    """Build and compile the SDOH longitudinal risk LangGraph agent."""
    graph = StateGraph(SdohState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("sdoh_extraction",    sdoh_extraction_step)
    graph.add_node("risk_trajectory",    risk_trajectory_step)
    graph.add_node("intervention_check", intervention_check_step)
    graph.add_node("sdoh_output",        sdoh_output_step)

    # ── Edges (linear pipeline) ───────────────────────────────────────────────
    graph.set_entry_point("sdoh_extraction")
    graph.add_edge("sdoh_extraction",    "risk_trajectory")
    graph.add_edge("risk_trajectory",    "intervention_check")
    graph.add_edge("intervention_check", "sdoh_output")
    graph.add_edge("sdoh_output",        END)

    return graph.compile()
