"""
clinical/agents/reextraction_router.py — Conditional edge function for Pillar 4.

Reads state["needs_reextraction"] and routes back to "oncology_staging"
if correction is needed, or forward to "trial_matching" if extraction passed.
"""

from schemas.clinical_state import ClinicalState


def reextraction_router(state: ClinicalState) -> str:
    """
    Conditional edge: oncology_evaluator → (oncology_staging | trial_matching)

    Returns:
        "oncology_staging"  if needs_reextraction is True
        "trial_matching"    if extraction passed evaluation
    """
    if state.get("needs_reextraction"):
        return "oncology_staging"
    return "trial_matching"
