"""
workflows/clinical_coding.py — LangGraph state machine for Clinical Coding Agent.

Graph topology:
    NER → Ontology Lookup → Disambiguation → MEAT Validation → Validation
                                                                        ↓
                          [conditional: retry / clinical_intervention / clinical_output]
                                                                        ↓
                                                           Clinical Output (END)

The MEAT Validation node (new) ensures every ICD-10 code is backed by
documented clinical action before it can contribute to HCC Risk Adjustment.
This prevents audit failures under Medicare RADV (Risk Adjustment Data
Validation) rules.
"""

from langgraph.graph import StateGraph, END

from schemas.clinical_state import ClinicalState
from clinical.steps.ner_step import ner_step
from clinical.steps.ontology_lookup_step import ontology_lookup_step
from clinical.steps.disambiguation_step import disambiguation_step
from clinical.steps.meat_validation_step import meat_validation_step
from clinical.steps.validation_step import validation_step
from clinical.steps.clinical_intervention_step import clinical_intervention_step
from clinical.steps.clinical_output_step import clinical_output_step


# ── Conditional Edge Logic ────────────────────────────────────────────────────

def should_recode(state: ClinicalState) -> str:
    """
    Decide what happens after the Validation node.

    - First-pass failure (retry_count < 2): retry disambiguation with extra
      context so the agent has a second chance to resolve ambiguous terms.
    - Persistent failure (retry_count >= 2): send to clinical intervention
      (agentic healing → human review queue).
    - Success: proceed to output.
    """
    confidence  = state.get("overall_confidence") or 0.0
    retry_count = state.get("retry_count", 0)
    status      = state.get("coding_status", "complete")

    if status == "requires_clinical_review" and retry_count < 2:
        return "disambiguation"   # retry with the existing note context
    elif status == "requires_clinical_review" and retry_count >= 2:
        return "clinical_intervention"
    return "clinical_output"


# ── Graph Factory ─────────────────────────────────────────────────────────────

def build_clinical_workflow() -> StateGraph:
    workflow = StateGraph(ClinicalState)

    # Nodes
    workflow.add_node("ner",                   ner_step)
    workflow.add_node("ontology_lookup",        ontology_lookup_step)
    workflow.add_node("disambiguation",         disambiguation_step)
    workflow.add_node("meat_validation",        meat_validation_step)   # NEW
    workflow.add_node("validation",             validation_step)
    workflow.add_node("clinical_intervention",  clinical_intervention_step)
    workflow.add_node("clinical_output",        clinical_output_step)

    # Linear path: NER → Lookup → Disambiguation → MEAT Validation → Validation
    workflow.set_entry_point("ner")
    workflow.add_edge("ner",             "ontology_lookup")
    workflow.add_edge("ontology_lookup", "disambiguation")
    workflow.add_edge("disambiguation",  "meat_validation")   # NEW
    workflow.add_edge("meat_validation", "validation")         # NEW

    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validation",
        should_recode,
        {
            "disambiguation":        "disambiguation",
            "clinical_intervention": "clinical_intervention",
            "clinical_output":       "clinical_output",
        },
    )

    # Intervention always terminates at clinical_output
    workflow.add_edge("clinical_intervention", "clinical_output")
    workflow.add_edge("clinical_output",        END)

    return workflow.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def clinical_coding_workflow(initial_state: ClinicalState) -> ClinicalState:
    graph = build_clinical_workflow()
    return graph.invoke(initial_state)
