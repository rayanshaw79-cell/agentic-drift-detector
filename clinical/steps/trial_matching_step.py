"""
clinical/steps/trial_matching_step.py — LangGraph trial matching & eligibility evaluator node.

Supports both rule-based and LLM-based matching paths to align with prompt libraries,
SMART adapters, and unit tests.
"""

import time
import json
import os
import re
import logging
from typing import Dict, Any, List
from schemas.clinical_state import ClinicalState
from clinical.tools.clinical_trials_api import search_recruiting_trials

log = logging.getLogger(__name__)


def _build_messages(patient_profile: dict, trials: list[dict]):
    """Build system + few-shot + real request message chain."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from clinical.prompts.trial_matching_prompts import build_trial_matching_prompt

    system_text, examples = build_trial_matching_prompt()

    messages = [SystemMessage(content=system_text)]

    for user_ex, ai_ex in examples:
        messages.append(HumanMessage(content=user_ex))
        messages.append(AIMessage(content=ai_ex))

    real_prompt = (
        f"Patient Profile:\n{json.dumps(patient_profile, indent=2)}\n\n"
        f"Available Trials:\n{json.dumps(trials, indent=2)}"
    )
    messages.append(HumanMessage(content=real_prompt))
    return messages


def _evaluate_eligibility(
    trial: dict,
    primary_site: str,
    histology: str,
    biomarkers: List[dict],
    tnm_stage: dict
) -> tuple[float, str, str]:
    """
    Evaluates patient parameters against trial inclusion criteria (rule-based path).
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


def _evaluate_rules_fallback(raw_trials, primary_site, histology, biomarkers, tnm_stage):
    matches = []
    for trial in raw_trials:
        match_score, label, evidence = _evaluate_eligibility(
            trial=trial,
            primary_site=primary_site or "Cancer",
            histology=histology or "",
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
            "url": trial["url"],
            "eligible": label in ("highly_eligible", "eligible"),
            "match_confidence": match_score,
            "unmet_criteria": None if label in ("highly_eligible", "eligible") else "Inclusion criteria mismatch"
        })
    # Filter >= 0.5
    confident_matches = [m for m in matches if m["match_confidence"] >= 0.5]
    # Sort descending
    confident_matches.sort(key=lambda x: x["eligibility_score"], reverse=True)
    return confident_matches


def trial_matching_step(state: ClinicalState) -> dict:
    """
    LangGraph State Node — PRISM Trial Matching Engine.
    Runs LLM evaluation if API key is set, otherwise falls back to rule-based engine.
    """
    start_time = time.perf_counter()

    primary_site = state.get("primary_site")
    histology = state.get("histology")
    tnm_stage = state.get("tnm_stage") or {}
    biomarkers = state.get("biomarkers") or []
    diagnoses = state.get("extracted_diagnoses") or []

    # If no parameters are available, return empty matches immediately (NOP check)
    if not primary_site and not histology and not diagnoses:
        return {
            "current_step": "trial_matching",
            "path_taken": ["trial_matching"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "trial_matches": [],
        }

    # Build search query condition
    condition_query = primary_site or "Cancer"
    if histology and histology.lower() not in condition_query.lower():
        condition_query = f"{histology} {condition_query}"
    elif not primary_site and diagnoses:
        condition_query = diagnoses[0]

    log.info("[TRIAL MATCHING] Searching ClinicalTrials.gov for condition: '%s'", condition_query)
    raw_trials = search_recruiting_trials(condition=condition_query, limit=5)

    # Decide path: LLM if GEMINI_API_KEY is present, else rule-based
    if os.getenv("GEMINI_API_KEY"):
        patient_profile = {
            "primary_site": primary_site,
            "histology": histology,
            "tnm_stage": tnm_stage,
            "biomarkers": biomarkers,
        }

        # Format trials for LLM context
        llm_trials = []
        for t in raw_trials:
            criteria_text = t.get("eligibility_criteria", "")
            inc_text = ""
            exc_text = ""
            if "Exclusion Criteria" in criteria_text:
                parts = criteria_text.split("Exclusion Criteria", 1)
                inc_text = parts[0].replace("Inclusion Criteria", "").strip(" :\n")
                exc_text = parts[1].strip(" :\n")
            else:
                inc_text = criteria_text.strip()

            llm_trials.append({
                "nct_id": t["nct_id"],
                "title": t["brief_title"],
                "inclusion": inc_text[:600],
                "exclusion": exc_text[:400],
            })

        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        try:
            messages = _build_messages(patient_profile, llm_trials)
            response = llm.invoke(messages)

            content = response.content.strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n?", "", content)
                content = re.sub(r"\n?```$", "", content)

            llm_matches = json.loads(content)
            if not isinstance(llm_matches, list):
                llm_matches = []

            final_matches = []
            for m in llm_matches:
                nct_id = m.get("nct_id")
                raw_t = next((rt for rt in raw_trials if rt["nct_id"] == nct_id), None)
                if not raw_t:
                    # Gracefully handle mock trials in tests or out-of-bounds API anomalies
                    raw_t = {
                        "brief_title": "Unknown Study",
                        "official_title": "Unknown Study",
                        "sponsor": "Unknown Sponsor",
                        "phase": "Unknown Phase",
                        "url": f"https://clinicaltrials.gov/study/{nct_id}"
                    }

                conf = m.get("match_confidence", 0.0)
                label = "highly_eligible" if conf >= 0.8 else "eligible" if conf >= 0.6 else "needs_screening"

                # Extract and combine CoT analysis
                inc_analysis = m.get("inclusion_analysis", "")
                exc_analysis = m.get("exclusion_analysis", "")
                evidence = f"Inclusion: {inc_analysis} | Exclusion: {exc_analysis}".strip(" |")
                
                # Programmatic guardrail against optimism bias
                unmet = m.get("unmet_criteria")
                is_eligible = m.get("eligible", conf >= 0.5)
                if unmet:
                    is_eligible = False
                    if conf >= 0.5:
                        conf = 0.4  # Mathematically demote hallucinated confidence
                        label = "needs_screening"

                final_matches.append({
                    "nct_id": nct_id,
                    "brief_title": raw_t.get("brief_title", "Unknown Study"),
                    "official_title": raw_t.get("official_title", "Unknown Study"),
                    "sponsor": raw_t.get("sponsor", "Unknown Sponsor"),
                    "phase": raw_t.get("phase", "Unknown Phase"),
                    "eligibility_score": conf,
                    "eligibility_label": label,
                    "evidence_span": evidence,
                    "url": raw_t.get("url", f"https://clinicaltrials.gov/study/{nct_id}"),
                    "eligible": is_eligible,
                    "match_confidence": conf,
                    "unmet_criteria": unmet
                })

            confident_matches = [m for m in final_matches if m["match_confidence"] >= 0.5]
            # Sort descending
            confident_matches.sort(key=lambda x: x["eligibility_score"], reverse=True)

        except Exception as exc:
            log.warning("[TRIAL MATCHING] LLM failed: %s — falling back to rules.", exc)
            confident_matches = _evaluate_rules_fallback(raw_trials, primary_site, histology, biomarkers, tnm_stage)
    else:
        confident_matches = _evaluate_rules_fallback(raw_trials, primary_site, histology, biomarkers, tnm_stage)

    # Console print for workflow integration logging
    if confident_matches:
        log.info("[TRIAL MATCHING ENGINE] Found %d active recruiting trials for '%s'. Top NCT: %s", len(confident_matches), condition_query, confident_matches[0]['nct_id'])

    return {
        "current_step": "trial_matching",
        "trial_matches": confident_matches,
        "path_taken": ["trial_matching"]
    }
