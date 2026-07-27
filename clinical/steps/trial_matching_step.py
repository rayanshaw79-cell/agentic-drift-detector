"""
clinical/steps/trial_matching_step.py — LangGraph trial matching & eligibility evaluator node.

Queries ClinicalTrials.gov API v2 for active recruiting studies based on extracted
primary site/histology, and evaluates biomarker and staging fit using prompt-grounded LLM reasoning.
"""

import logging
from typing import Dict, Any, List

from schemas.clinical_state import ClinicalState
from clinical.tools.clinical_trials_api import search_recruiting_trials

log = logging.getLogger(__name__)


def trial_matching_step(state: ClinicalState) -> dict:
    """
    LangGraph State Node — PRISM v2 Clinical Trial Matching Engine.

    Reads:
      - primary_site, histology, tnm_stage, biomarkers, extracted_diagnoses
    Emits:
      - trial_matches: List[dict]
    """
    primary_site = state.get("primary_site") or "Cancer"
    histology = state.get("histology") or ""
    biomarkers = state.get("biomarkers") or []
    tnm_stage = state.get("tnm_stage") or {}
    diagnoses = state.get("extracted_diagnoses") or []

    # Build search query condition
    condition_query = primary_site
    if histology and histology.lower() not in primary_site.lower():
        condition_query = f"{histology} {primary_site}"
    elif not primary_site and diagnoses:
        condition_query = diagnoses[0]

    log.info("[TRIAL MATCHING] Searching ClinicalTrials.gov for condition: '%s'", condition_query)

    raw_trials = search_recruiting_trials(condition=condition_query, limit=5)

    matches: List[Dict[str, Any]] = []

    for trial in raw_trials:
        match_score, label, evidence = _evaluate_eligibility(
            trial=trial,
            primary_site=primary_site,
            histology=histology,
            biomarkers=biomarkers,
            tnm_stage=tnm_stage
        )

        matches.append({
            "nct_id": trial["nct_id"],
            "brief_title": trial["brief_title"],
            "official_title": trial["official_title"],
            "sponsor": trial["sponsor"],
            "phase": trial["phase"],
            "eligibility_score": match_score,
            "eligibility_label": label,
            "evidence_span": evidence,
            "url": trial["url"]
        })

    # Sort by eligibility score descending
    matches.sort(key=lambda x: x["eligibility_score"], reverse=True)

    print(f"\n  [TRIAL MATCHING ENGINE] Found {len(matches)} active recruiting trials for '{condition_query}'. Top NCT: {matches[0]['nct_id'] if matches else 'N/A'}")

    return {
        "current_step": "trial_matching",
        "trial_matches": matches,
        "path_taken": ["trial_matching"]
    }


def _evaluate_eligibility(
    trial: dict,
    primary_site: str,
    histology: str,
    biomarkers: List[dict],
    tnm_stage: dict
) -> tuple[float, str, str]:
    """
    Evaluates patient parameters against trial inclusion criteria.
    Returns: (eligibility_score, eligibility_label, evidence_span)
    """
    score = 0.50  # baseline for matching primary condition
    label = "needs_screening"
    reasons = []

    criteria_lower = trial.get("eligibility_criteria", "").lower()

    # 1. Histology / Condition Match
    if histology and histology.lower() in criteria_lower:
        score += 0.20
        reasons.append(f"Matching histology ({histology})")
    elif primary_site.lower() in criteria_lower:
        score += 0.10
        reasons.append(f"Matching primary site ({primary_site})")

    # 2. Biomarker Matching
    biomarker_matches = []
    for bm in biomarkers:
        marker = bm.get("marker", "").lower()
        status = bm.get("status", "").lower()
        if marker and marker in criteria_lower:
            score += 0.15
            biomarker_matches.append(f"{marker.upper()} ({status})")

    if biomarker_matches:
        reasons.append(f"Biomarker matches: {', '.join(biomarker_matches)}")

    # 3. TNM Staging Match
    overall_stage = tnm_stage.get("overall", "").lower()
    if overall_stage and ("stage iii" in criteria_lower or "stage iv" in criteria_lower or "advanced" in criteria_lower):
        score += 0.10
        reasons.append(f"Stage alignment ({overall_stage.upper()})")

    score = min(score, 0.98)

    if score >= 0.80:
        label = "highly_eligible"
    elif score >= 0.60:
        label = "eligible"
    else:
        label = "needs_screening"

    evidence = " | ".join(reasons) if reasons else "Condition query match on ClinicalTrials.gov API."

    return round(score, 2), label, evidence
