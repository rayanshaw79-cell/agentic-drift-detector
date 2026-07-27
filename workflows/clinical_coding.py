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
from clinical.steps.deid_step import deid_step
from clinical.steps.compliance_checker_step import compliance_checker_step
from clinical.steps.sdoh_integration_step import sdoh_integration_step
from clinical.steps.trial_matching_step import trial_matching_step
from clinical.steps.pharmacovigilance_step import pharmacovigilance_step
from clinical.steps.raf_audit_step import raf_audit_step
from clinical.steps.symphony_step import symphony_step
from clinical.steps.fhir_step import fhir_step

# ── Conditional Edge Logic ────────────────────────────────────────────────────

def should_retry_deid(state: ClinicalState) -> str:
    """
    Decide what happens after the Compliance Checker node.
    - If phi_detected is True and retry_count < 3: loop back to deid
    - Otherwise: proceed to ner
    """
    phi_detected = state.get("phi_detected", False)
    retry_count = state.get("retry_count", 0)
    
    if phi_detected and retry_count < 3:
        return "deid"
    return "ner"

def should_recode(state: ClinicalState) -> str:
    """
    Decide what happens after the Validation node.

    - First-pass failure (retry_count < 2): retry disambiguation with extra
      context so the agent has a second chance to resolve ambiguous terms.
    - Persistent failure (retry_count >= 2): send to clinical intervention
      (agentic healing → human review queue).
    - Success: proceed to sdoh_integration.
    """
    retry_count = state.get("retry_count", 0)
    status      = state.get("coding_status", "complete")

    if status == "requires_clinical_review" and retry_count < 2:
        return "disambiguation"   # retry with the existing note context
    elif status == "requires_clinical_review" and retry_count >= 2:
        return "clinical_intervention"
    return "sdoh_integration"


# ── Graph Factory ─────────────────────────────────────────────────────────────

def build_clinical_workflow() -> StateGraph:
    workflow = StateGraph(ClinicalState)

    # Nodes
    workflow.add_node("deid",                   deid_step)
    workflow.add_node("compliance_checker",     compliance_checker_step)
    workflow.add_node("ner",                   ner_step)
    workflow.add_node("ontology_lookup",        ontology_lookup_step)
    workflow.add_node("disambiguation",         disambiguation_step)
    workflow.add_node("meat_validation",        meat_validation_step)
    workflow.add_node("validation",             validation_step)
    workflow.add_node("sdoh_integration",       sdoh_integration_step)
    workflow.add_node("trial_matching",         trial_matching_step)
    workflow.add_node("pharmacovigilance",      pharmacovigilance_step)
    workflow.add_node("raf_audit",              raf_audit_step)
    workflow.add_node("symphony_longitudinal",  symphony_step)
    workflow.add_node("fhir_export",            fhir_step)  # Phase 5 SMART-on-FHIR R4 Adapter Node
    workflow.add_node("clinical_intervention",  clinical_intervention_step)
    workflow.add_node("clinical_output",        clinical_output_step)

    # Linear path: De-ID → Compliance → NER → Lookup → Disambiguation → MEAT Validation → Validation
    workflow.set_entry_point("deid")
    workflow.add_edge("deid", "compliance_checker")
    workflow.add_conditional_edges(
        "compliance_checker",
        should_retry_deid,
        {
            "deid": "deid",
            "ner": "ner",
        },
    )
    workflow.add_edge("ner",             "ontology_lookup")
    workflow.add_edge("ontology_lookup", "disambiguation")
    workflow.add_edge("disambiguation",  "meat_validation")
    workflow.add_edge("meat_validation", "validation")

    # Conditional routing after validation
    workflow.add_conditional_edges(
        "validation",
        should_recode,
        {
            "disambiguation":        "disambiguation",
            "clinical_intervention": "clinical_intervention",
            "sdoh_integration":      "sdoh_integration",
        },
    )

    # SDOH integration proceeds to trial matching → pharmacovigilance → raf_audit → symphony → fhir → output
    workflow.add_edge("sdoh_integration",    "trial_matching")
    workflow.add_edge("trial_matching",      "pharmacovigilance")
    workflow.add_edge("pharmacovigilance",    "raf_audit")
    workflow.add_edge("raf_audit",           "symphony_longitudinal")
    workflow.add_edge("symphony_longitudinal", "fhir_export")
    workflow.add_edge("fhir_export",          "clinical_output")

    # Intervention always terminates at clinical_output
    workflow.add_edge("clinical_intervention", "clinical_output")
    workflow.add_edge("clinical_output",        END)

    return workflow.compile()


# ── Public entry point ────────────────────────────────────────────────────────

def clinical_coding_workflow(initial_state: ClinicalState) -> ClinicalState:
    graph = build_clinical_workflow()
    return graph.invoke(initial_state)


def apply_human_approval(
    state: dict,
    action: str,
    reviewed_by: str = "clinician",
    notes: str = "",
    final_codes: list | None = None
) -> dict:
    """
    Applies clinician approval or modification to a flagged clinical state.
    """
    import datetime
    updated_state = dict(state)
    updated_state["human_review_action"] = action
    updated_state["reviewed_by"] = reviewed_by
    updated_state["reviewed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    updated_state["human_notes"] = notes

    if action in ("approved", "edited"):
        updated_state["coding_status"] = "approved_by_clinician"
        if final_codes is not None:
            updated_state["icd10_codes"] = final_codes
        updated_state["overall_confidence"] = 1.0
    elif action == "rejected":
        updated_state["coding_status"] = "rejected_by_clinician"
        updated_state["icd10_codes"] = []
        updated_state["overall_confidence"] = 0.0

    return updated_state

