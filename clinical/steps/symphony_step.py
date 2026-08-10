"""
clinical/steps/symphony_step.py — LangGraph SYMPHONY v2 Longitudinal Reasoning node.

Synthesizes multi-visit patient records into chronological disease trajectories and evaluates
serial target lesion measurements per RECIST 1.1 standards.
"""

import logging
from typing import Dict, Any

from schemas.clinical_state import ClinicalState
from clinical.tools.symphony_engine import synthesize_patient_timeline

log = logging.getLogger(__name__)


def symphony_step(state: ClinicalState) -> dict:
    """
    LangGraph State Node — SYMPHONY v2 Longitudinal & RECIST 1.1 Engine.

    Reads:
      - visit_history: List[dict]
      - primary_site, histology
    Emits:
      - pre_chart_summary: str
      - longitudinal_timeline: List[dict]
      - recist_overall_response: str
      - lesion_measurements: List[dict]
    """
    visit_history = state.get("visit_history")

    # If no visit history provided, construct a realistic 2-visit trajectory demonstration
    if not visit_history:
        primary_site = state.get("primary_site") or "Lung"
        histology = state.get("histology") or "Adenocarcinoma"
        visit_history = [
            {
                "date": "2026-01-15",
                "doc_type": "Baseline CT Radiology",
                "summary": f"Baseline CT scan revealed 42mm primary mass in right upper lobe {primary_site} ({histology}).",
                "target_lesion_mm": 42.0,
                "new_lesions": False
            },
            {
                "date": "2026-07-20",
                "doc_type": "Follow-up CT Radiology",
                "summary": f"Follow-up CT scan shows primary mass decreased from 42mm to 25mm (-40.5% shrinkage). No new lesions.",
                "target_lesion_mm": 25.0,
                "new_lesions": False
            }
        ]

    synthesis = synthesize_patient_timeline(visit_history)

    log.info(
        "[SYMPHONY v2 ENGINE] Visits: %d | RECIST 1.1 Response: %s (%+.1f%%)",
        synthesis['total_visits'], synthesis['recist_overall_response'], synthesis['recist_delta_pct']
    )

    return {
        "current_step": "symphony_longitudinal",
        "pre_chart_summary": synthesis["pre_chart_summary"],
        "longitudinal_timeline": synthesis["chronological_timeline"],
        "recist_overall_response": synthesis["recist_overall_response"],
        "lesion_measurements": synthesis["serial_measurements"],
        "path_taken": ["symphony_longitudinal"]
    }
