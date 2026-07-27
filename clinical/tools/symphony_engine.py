"""
clinical/tools/symphony_engine.py — SYMPHONY v2 Longitudinal Patient Timeline & RECIST 1.1 Engine.

Synthesizes multi-visit patient records into chronological disease trajectories and evaluates
serial oncology tumor measurements per RECIST 1.1 standards (CR, PR, SD, PD).
"""

import logging
from typing import List, Dict, Any, Tuple

log = logging.getLogger(__name__)


def synthesize_patient_timeline(visit_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Synthesizes visit records into a structured chronological timeline.

    Args:
        visit_history: List of visit dicts:
            [{date, doc_type, summary, target_lesion_mm, new_lesions}]

    Returns:
        Structured timeline summary object.
    """
    if not visit_history:
        return {
            "chronological_timeline": [],
            "pre_chart_summary": "Single visit record; no prior longitudinal history provided.",
            "total_visits": 0
        }

    # Sort visits by date ascending
    sorted_visits = sorted(visit_history, key=lambda v: v.get("date", "1970-01-01"))

    timeline = []
    serial_measurements = []

    for visit in sorted_visits:
        v_date = visit.get("date", "N/A")
        doc_type = visit.get("doc_type", "Progress Note")
        summary = visit.get("summary", "Routine clinical follow-up.")
        lesion_mm = visit.get("target_lesion_mm")
        new_lesions = visit.get("new_lesions", False)

        timeline_item = {
            "date": v_date,
            "doc_type": doc_type,
            "summary": summary,
            "target_lesion_mm": lesion_mm,
            "new_lesions": new_lesions
        }
        timeline.append(timeline_item)

        if lesion_mm is not None:
            serial_measurements.append({
                "date": v_date,
                "target_lesion_mm": float(lesion_mm),
                "new_lesions": new_lesions
            })

    # Calculate RECIST 1.1 Response Category
    recist_response, recist_delta_pct = evaluate_recist_response(serial_measurements)

    summary_text = (
        f"Longitudinal synthesis of {len(sorted_visits)} visits ({sorted_visits[0].get('date')} to {sorted_visits[-1].get('date')}). "
        f"RECIST 1.1 Response: {recist_response} ({recist_delta_pct:+.1f}% target lesion diameter change)."
    )

    return {
        "chronological_timeline": timeline,
        "pre_chart_summary": summary_text,
        "serial_measurements": serial_measurements,
        "recist_overall_response": recist_response,
        "recist_delta_pct": recist_delta_pct,
        "total_visits": len(sorted_visits)
    }


def evaluate_recist_response(serial_measurements: List[Dict[str, Any]]) -> Tuple[str, float]:
    """
    Evaluates serial target lesion sum measurements according to RECIST 1.1 standards.

    RECIST 1.1 Criteria:
      - Complete Response (CR): Disappearance of all target lesions (100% reduction -> 0mm).
      - Partial Response (PR): At least 30% decrease in sum of target lesion diameters from baseline.
      - Progressive Disease (PD): At least 20% increase in sum of target lesion diameters from nadir, OR appearance of new lesions.
      - Stable Disease (SD): Neither sufficient shrinkage to qualify for PR nor sufficient increase to qualify for PD.

    Returns:
        (recist_category: str, delta_pct: float)
    """
    if not serial_measurements or len(serial_measurements) < 2:
        return "UNEVALUABLE", 0.0

    baseline_mm = serial_measurements[0]["target_lesion_mm"]
    latest_visit = serial_measurements[-1]
    latest_mm = latest_visit["target_lesion_mm"]
    has_new_lesions = latest_visit.get("new_lesions", False)

    if baseline_mm <= 0:
        return "UNEVALUABLE", 0.0

    delta_pct = ((latest_mm - baseline_mm) / baseline_mm) * 100.0

    # Evaluate RECIST 1.1 Category
    if latest_mm == 0:
        category = "CR"  # Complete Response
    elif has_new_lesions or delta_pct >= 20.0:
        category = "PD"  # Progressive Disease
    elif delta_pct <= -30.0:
        category = "PR"  # Partial Response
    else:
        category = "SD"  # Stable Disease

    log.info("[SYMPHONY RECIST 1.1] Baseline: %.1fmm -> Latest: %.1fmm (Delta: %+.1f%%) => %s", baseline_mm, latest_mm, delta_pct, category)

    return category, round(delta_pct, 1)
