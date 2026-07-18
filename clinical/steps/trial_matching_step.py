"""
clinical/steps/trial_matching_step.py — Proactive Clinical Trial Matching (PRISM).

v2: Upgraded to OncoLLM Pillar 1 + 3 + Live ClinicalTrials.gov API.
  - Live Data: Fetches real NCT trials from ClinicalTrials.gov API v2
               instead of the previous 2-item mock list.
  - Pillar 3:  Uses the few-shot trial matching prompt library with
               explicit eligibility reasoning rules.
  - Pillar 1:  Uses document_type to inform query construction.
  - Fallback:  If ClinicalTrials.gov is unreachable, falls back to the
               original mock trials to keep the pipeline running.
"""

import time
import json
import os
import re
import logging
import requests
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

# ── ClinicalTrials.gov API v2 ─────────────────────────────────────────────────
_CTGOV_BASE = "https://clinicaltrials.gov/api/v2/studies"
_CTGOV_TIMEOUT = 8  # seconds
_MAX_TRIALS = 10    # limit trial list sent to LLM

# ── Fallback mock trials (used if API is unavailable) ─────────────────────────
_MOCK_TRIALS_FALLBACK = [
    {
        "nct_id": "NCT01234567",
        "title": "Targeted Therapy for EGFR+ Non-Small Cell Lung Cancer",
        "inclusion": "Stage III or IV Non-Small Cell Lung Cancer (NSCLC). EGFR mutation positive.",
        "exclusion": "Prior treatment with EGFR inhibitors.",
    },
    {
        "nct_id": "NCT09876543",
        "title": "Immunotherapy for PD-L1 High Solid Tumors",
        "inclusion": "Advanced solid tumors. PD-L1 expression > 50%.",
        "exclusion": "Active autoimmune disease.",
    },
]


def _fetch_trials(primary_site: str, histology: str | None, biomarkers: list[dict]) -> list[dict]:
    """
    Query ClinicalTrials.gov API v2 for relevant open trials.

    Constructs a query from primary_site + top biomarker markers and returns
    a normalised list of trial dicts ready for LLM consumption.
    """
    # Build query terms
    query_terms = [primary_site]
    if histology:
        query_terms.append(histology.split()[0])  # e.g., "Invasive" → skip, "Adenocarcinoma" → keep
    for bio in biomarkers[:2]:  # Add top 2 biomarker names
        if bio.get("status") in ("Mutated", "Positive", "Amplified", "High"):
            query_terms.append(bio.get("marker", ""))

    query = " ".join(q for q in query_terms if q)
    log.info("[TRIAL MATCHING] Querying ClinicalTrials.gov for: '%s'", query)

    params = {
        "query.cond": query,
        "filter.overallStatus": "RECRUITING",
        "fields": "NCTId,BriefTitle,EligibilityModule",
        "pageSize": _MAX_TRIALS,
        "format": "json",
    }

    try:
        response = requests.get(_CTGOV_BASE, params=params, timeout=_CTGOV_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except Exception as exc:
        log.warning("[TRIAL MATCHING] ClinicalTrials.gov request failed: %s — using fallback.", exc)
        return _MOCK_TRIALS_FALLBACK

    trials = []
    for study in data.get("studies", []):
        try:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            elig = proto.get("eligibilityModule", {})
            criteria_text = elig.get("eligibilityCriteria", "")

            # Split inclusion/exclusion from the criteria text
            inc_text = ""
            exc_text = ""
            if "Exclusion Criteria" in criteria_text:
                parts = criteria_text.split("Exclusion Criteria", 1)
                inc_text = parts[0].replace("Inclusion Criteria", "").strip(" :\n")
                exc_text = parts[1].strip(" :\n")
            else:
                inc_text = criteria_text.strip()

            # Truncate to keep the prompt manageable
            trials.append({
                "nct_id": ident.get("nctId", ""),
                "title": ident.get("briefTitle", ""),
                "inclusion": inc_text[:600],
                "exclusion": exc_text[:400],
            })
        except Exception:
            continue

    if not trials:
        log.warning("[TRIAL MATCHING] No trials returned from API — using fallback.")
        return _MOCK_TRIALS_FALLBACK

    log.info("[TRIAL MATCHING] Fetched %d trials from ClinicalTrials.gov.", len(trials))
    return trials


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


def trial_matching_step(state: ClinicalState) -> dict:
    """
    Evaluates the patient's structured oncology profile against live NCT trials.

    Reads:  state["primary_site"], state["histology"], state["tnm_stage"],
            state["biomarkers"]
    Writes: state["trial_matches"]
    """
    start_time = time.perf_counter()

    primary_site = state.get("primary_site")
    histology = state.get("histology")
    tnm_stage = state.get("tnm_stage")
    biomarkers = state.get("biomarkers") or []

    if not primary_site or not os.getenv("GEMINI_API_KEY"):
        return {
            "current_step": "trial_matching",
            "path_taken": ["trial_matching"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "trial_matches": [],
        }

    # ── Step 1: Fetch live trials from ClinicalTrials.gov ──────────────────────
    trials = _fetch_trials(primary_site, histology, biomarkers)

    # ── Step 2: Build patient profile for LLM ─────────────────────────────────
    patient_profile = {
        "primary_site": primary_site,
        "histology": histology,
        "tnm_stage": tnm_stage,
        "biomarkers": biomarkers,
    }

    # ── Step 3: LLM matching with few-shot prompts ─────────────────────────────
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    try:
        messages = _build_messages(patient_profile, trials)
        response = llm.invoke(messages)

        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        matches = json.loads(content)
        if not isinstance(matches, list):
            matches = []

        # Filter to only confident matches for display; keep all for audit
        confident_matches = [m for m in matches if m.get("match_confidence", 0) >= 0.5]
        log.info(
            "[TRIAL MATCHING] %d/%d trials are potential matches (≥0.5 confidence).",
            len(confident_matches), len(matches),
        )

    except Exception as exc:
        log.warning("[TRIAL MATCHING] LLM failed: %s", exc)
        matches = []
        confident_matches = []

    return {
        "current_step": "trial_matching",
        "path_taken": ["trial_matching"],
        "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
        "trial_matches": confident_matches,  # High-confidence matches surfaced to clinician
    }
