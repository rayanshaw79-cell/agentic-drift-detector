"""
clinical/sdoh/sdoh_extraction_step.py — SDOH Profile Extraction Node.

Reads the patient's visit_history from ClinicalState and synthesises a
structured SDOH profile dict for the *latest* visit.

The profile merges:
  • Structured fields already in the visit record (AQI, poverty rate, etc.)
  • LLM-extracted signals from any free-text clinical note in the visit
    (if a 'raw_note' key is present on the visit dict).
"""

import logging
import os
import time

from schemas.sdoh_state import SdohState

log = logging.getLogger(__name__)


def _llm_extract_from_note(raw_note: str) -> dict:
    """
    Use Gemini to extract SDOH signals from unstructured clinical note text.
    Returns a dict with boolean / float SDOH signal overrides.
    Falls back to empty dict on any error or missing API key.
    """
    if not os.getenv("GEMINI_API_KEY") or not raw_note:
        return {}

    import re, json
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    prompt = (
        "You are a clinical social determinants analyst. "
        "Extract SDOH signals from the note below. "
        "Return ONLY a valid JSON object with these keys "
        "(use null if not mentioned):\n"
        '{"smoking": true|false|null, "alcohol": true|false|null, '
        '"exercise_frequency": "none"|"low"|"moderate"|"high"|null, '
        '"food_insecurity_mentioned": true|false|null, '
        '"housing_instability_mentioned": true|false|null, '
        '"social_isolation_mentioned": true|false|null}'
    )

    try:
        response = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=f"Clinical note:\n{raw_note}"),
        ])
        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)
        return json.loads(content)
    except Exception as exc:
        log.warning("[SDOH EXTRACTION] LLM note extraction failed: %s", exc)
        return {}


def _exercise_score_from_freq(freq: str | None) -> float:
    return {"none": 0.0, "low": 0.25, "moderate": 0.60, "high": 1.0}.get(freq or "", 0.5)


# ── LangGraph Node ────────────────────────────────────────────────────────────

def sdoh_extraction_step(state: SdohState) -> dict:
    """
    LangGraph node — SDOH Profile Extraction.

    Reads:  state["visit_history"]
    Writes: state["sdoh_profile"]
    """
    start         = time.perf_counter()
    visit_history = state.get("visit_history") or []

    if not visit_history:
        log.warning("[SDOH EXTRACTION] No visit history found for patient %s", state.get("patient_id"))
        return {
            "current_step":      "sdoh_extraction",
            "path_taken":        ["sdoh_extraction"],
            "execution_time_ms": int((time.perf_counter() - start) * 1000),
            "sdoh_profile":      {},
        }

    # Use the most recent visit as the baseline
    latest_visit = visit_history[-1]

    # Start with structured fields directly from the record
    profile = {
        "patient_id":        latest_visit.get("patient_id"),
        "visit_number":      latest_visit.get("visit_number", len(visit_history)),
        "age":               latest_visit.get("age", 50),
        "gender":            latest_visit.get("gender", "Unknown"),
        "race":              latest_visit.get("race", "Unknown"),
        "zip_code":          latest_visit.get("zip_code", "00000"),
        "smoking_flag":      int(latest_visit.get("smoking_flag", 0)),
        "alcohol_flag":      int(latest_visit.get("alcohol_flag", 0)),
        "exercise_score":    latest_visit.get("exercise_score", 0.5),
        "food_risk_score":   latest_visit.get("food_risk_score", 0.0),
        "env_aqi":           latest_visit.get("env_aqi", 80.0),
        "env_poverty_rate":  latest_visit.get("env_poverty_rate", 0.15),
        "hcc_score":         latest_visit.get("hcc_score", 0.0),
        "icd10_codes":       latest_visit.get("icd10_codes", ""),
        "icd10_code_count":  latest_visit.get("icd10_code_count", 0),
        "chain_stage":       latest_visit.get("chain_stage", 0),
        "sdoh_risk_score":   latest_visit.get("sdoh_risk_score", 0.0),
        # Derived context flags
        "food_insecurity_mentioned":    False,
        "housing_instability_mentioned": False,
        "social_isolation_mentioned":   False,
    }

    # Optionally enrich from a free-text note on the latest visit
    raw_note = latest_visit.get("raw_note", "")
    if raw_note:
        llm_signals = _llm_extract_from_note(raw_note)
        log.info("[SDOH EXTRACTION] LLM signals: %s", llm_signals)

        if llm_signals.get("smoking") is not None:
            profile["smoking_flag"] = int(llm_signals["smoking"])
        if llm_signals.get("alcohol") is not None:
            profile["alcohol_flag"] = int(llm_signals["alcohol"])
        if llm_signals.get("exercise_frequency"):
            profile["exercise_score"] = _exercise_score_from_freq(llm_signals["exercise_frequency"])
        for flag in ("food_insecurity_mentioned", "housing_instability_mentioned", "social_isolation_mentioned"):
            if llm_signals.get(flag) is not None:
                profile[flag] = bool(llm_signals[flag])
                if flag == "food_insecurity_mentioned" and llm_signals[flag]:
                    profile["food_risk_score"] = max(profile["food_risk_score"], 0.6)

    latency = int((time.perf_counter() - start) * 1000) + 5
    log.info("[SDOH EXTRACTION] Profile built for patient %s (visit %d)",
             profile.get("patient_id"), profile.get("visit_number", 0))

    return {
        "current_step":      "sdoh_extraction",
        "path_taken":        ["sdoh_extraction"],
        "execution_time_ms": latency,
        "sdoh_profile":      profile,
    }
