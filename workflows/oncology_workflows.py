"""
workflows/oncology_workflows.py — Triomics-aligned specialized oncology workflows.

v2: Fully upgraded to OncoLLM Pillars 1–4.

PRISM workflow topology (build_oncology_trial_matching_workflow):
  deid
    → compliance_checker
    → [should_retry_deid] → deid (loop) | oncology_router
    → oncology_router          (Pillar 1: document classification)
    → oncology_staging         (Pillar 1+2+3: RAG-grounded, few-shot)
    → biomarker_extraction     (Pillar 1+3: few-shot, document-type aware)
    → oncology_evaluator       (Pillar 4: self-correction check)
    → [reextraction_router] → oncology_staging (loop) | trial_matching
    → trial_matching           (Pillar 3 + live ClinicalTrials.gov)
    → clinical_output

SYMPHONY workflow topology (build_oncology_pre_chart_workflow):
  oncology_router
    → longitudinal_summary     (Pillar 3: few-shot narrative synthesis)
    → clinical_output
"""

from langgraph.graph import StateGraph, END
from schemas.clinical_state import ClinicalState

# Standard steps
from clinical.steps.deid_step import deid_step
from clinical.steps.compliance_checker_step import compliance_checker_step
from clinical.steps.clinical_output_step import clinical_output_step

# Upgraded oncology steps (Pillars 1–3)
from clinical.steps.oncology_staging_step import oncology_staging_step
from clinical.steps.biomarker_extraction_step import biomarker_extraction_step
from clinical.steps.longitudinal_summary_step import longitudinal_summary_step
from clinical.steps.trial_matching_step import trial_matching_step

# New OncoLLM agents (Pillars 1 and 4)
from clinical.agents.oncology_router import oncology_router
from clinical.agents.oncology_evaluator import oncology_evaluator
from clinical.agents.reextraction_router import reextraction_router


# ── Conditional Edge Helpers ──────────────────────────────────────────────────

def should_retry_deid(state: ClinicalState) -> str:
    """Decide what happens after the Compliance Checker node."""
    phi_detected = state.get("phi_detected", False)
    retry_count = state.get("retry_count", 0)
    if phi_detected and retry_count < 3:
        return "deid"
    return "oncology_router"


# ── PRISM Workflow — Trial Matching ───────────────────────────────────────────

def build_oncology_trial_matching_workflow() -> StateGraph:
    """
    Simulates Triomics PRISM + Harmony workflows.

    Full OncoLLM pipeline:
      DeID → Router → Staging → Biomarkers → Evaluator → Trial Matching
    """
    workflow = StateGraph(ClinicalState)

    # Standard steps
    workflow.add_node("deid",                deid_step)
    workflow.add_node("compliance_checker",  compliance_checker_step)

    # Pillar 1: Constellation Router
    workflow.add_node("oncology_router",     oncology_router)

    # Pillar 1+2+3: Specialist extraction nodes
    workflow.add_node("oncology_staging",    oncology_staging_step)
    workflow.add_node("biomarker_extraction", biomarker_extraction_step)

    # Pillar 4: Self-Correction Evaluator
    workflow.add_node("oncology_evaluator",  oncology_evaluator)

    # Pillar 3 + live API: Trial Matching
    workflow.add_node("trial_matching",      trial_matching_step)
    workflow.add_node("clinical_output",     clinical_output_step)

    # ── Edges ─────────────────────────────────────────────────────────────────
    workflow.set_entry_point("deid")
    workflow.add_edge("deid", "compliance_checker")

    # PHI retry loop
    workflow.add_conditional_edges(
        "compliance_checker",
        should_retry_deid,
        {
            "deid":            "deid",
            "oncology_router": "oncology_router",
        },
    )

    # Pillar 1 → extraction pipeline
    workflow.add_edge("oncology_router",    "oncology_staging")
    workflow.add_edge("oncology_staging",   "biomarker_extraction")
    workflow.add_edge("biomarker_extraction", "oncology_evaluator")

    # Pillar 4 self-correction loop
    workflow.add_conditional_edges(
        "oncology_evaluator",
        reextraction_router,
        {
            "oncology_staging": "oncology_staging",   # loop back for correction
            "trial_matching":   "trial_matching",      # proceed on pass
        },
    )

    workflow.add_edge("trial_matching",  "clinical_output")
    workflow.add_edge("clinical_output", END)

    return workflow.compile()


# ── SYMPHONY Workflow — Pre-Chart Longitudinal Summary ────────────────────────

def build_oncology_pre_chart_workflow() -> StateGraph:
    """
    Simulates Triomics Symphony workflow.
    Router → Longitudinal Summary (few-shot, Pillar 3) → Output.
    """
    workflow = StateGraph(ClinicalState)

    workflow.add_node("oncology_router",      oncology_router)
    workflow.add_node("longitudinal_summary", longitudinal_summary_step)
    workflow.add_node("clinical_output",      clinical_output_step)

    workflow.set_entry_point("oncology_router")
    workflow.add_edge("oncology_router",      "longitudinal_summary")
    workflow.add_edge("longitudinal_summary", "clinical_output")
    workflow.add_edge("clinical_output",      END)

    return workflow.compile()


# ── Public Entry Points ───────────────────────────────────────────────────────

def run_trial_matching(initial_state: ClinicalState) -> ClinicalState:
    graph = build_oncology_trial_matching_workflow()
    return graph.invoke(initial_state)


def run_pre_chart_summary(initial_state: ClinicalState) -> ClinicalState:
    graph = build_oncology_pre_chart_workflow()
    return graph.invoke(initial_state)
