"""
clinical/steps/disambiguation_step.py — Code Disambiguation & Confidence Scoring node.

Resolves ambiguous ICD-10 candidates (e.g., "MS" → Multiple Sclerosis vs
Mitral Stenosis) by re-reading the original clinical note and scoring each
candidate using Google Gemini.

Fallback (no API key): selects the first candidate returned by the API
with a moderate confidence of 0.60.
"""

import json
import logging
import os
import re
import time

from schemas.clinical_state import ClinicalState
from tenacity import retry, wait_exponential, stop_after_attempt

log = logging.getLogger(__name__)


@retry(wait=wait_exponential(min=1, max=10), stop=stop_after_attempt(3))
def _gemini_disambiguate(raw_note: str, candidates: list[dict]) -> list[dict]:
    """
    Use Gemini to select the single best ICD-10 code per term and assign
    a confidence score.

    candidates: list of {"term", "code", "description", "confidence"}
    Returns:    same list but with confidence filled in; only the best
                code per term is retained.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    if not candidates:
        return []

    # Build a grouped candidate summary for the prompt
    by_term: dict[str, list[dict]] = {}
    for c in candidates:
        by_term.setdefault(c["term"], []).append(c)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    selected: list[dict] = []
    for term, group in by_term.items():
        if group[0]["code"] == "UNRESOLVED":
            group[0]["confidence"] = 0.0
            selected.append(group[0])
            continue

        options_text = "\n".join(
            f"  - Code: {c['code']}, Description: {c['description']}"
            for c in group
        )
        response = llm.invoke([
            SystemMessage(content=(
                "You are a senior clinical coding specialist. "
                "Given the clinical note and a list of candidate ICD-10-CM codes, "
                "select the SINGLE most accurate code and assign a confidence score (0.0–1.0). "
                "Return ONLY a JSON object with keys: 'code', 'description', 'confidence'. "
                'Example: {"code": "I10", "description": "Essential hypertension", "confidence": 0.92}'
            )),
            HumanMessage(content=(
                f"Clinical note:\n{raw_note}\n\n"
                f"Condition being coded: {term}\n\n"
                f"Candidate ICD-10 codes:\n{options_text}"
            )),
        ])

        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        try:
            result = json.loads(content)
            best = {
                "term":        term,
                "code":        result.get("code", group[0]["code"]),
                "description": result.get("description", group[0]["description"]),
                "confidence":  float(result.get("confidence", 0.6)),
            }
        except (json.JSONDecodeError, ValueError):
            best = {**group[0], "confidence": 0.6}

        selected.append(best)

    return selected


def _fallback_select(candidates: list[dict]) -> list[dict]:
    """
    Fallback: pick the first candidate per term with confidence 0.60.
    """
    seen: set[str] = set()
    selected: list[dict] = []
    for c in candidates:
        if c["term"] not in seen:
            seen.add(c["term"])
            selected.append({**c, "confidence": 0.60})
    return selected


# ── LangGraph Node ────────────────────────────────────────────────────────────

def disambiguation_step(state: ClinicalState) -> dict:
    """
    LangGraph node — Disambiguation & Confidence Scoring.

    Reads:  state["icd10_codes"], state["raw_note"]
    Writes: state["icd10_codes"] (refined), state["overall_confidence"],
            step metadata
    """
    raw_note   = state.get("raw_note", "")
    candidates = state.get("icd10_codes") or []
    ner_votes  = state.get("ner_votes") or []
    use_gemini = bool(os.getenv("GEMINI_API_KEY"))

    start = time.perf_counter()
    refined: list[dict] = []

    if use_gemini:
        try:
            refined = _gemini_disambiguate(raw_note, candidates)
            log.info("[DISAMBIGUATION] Gemini selected %d codes.", len(refined))
        except Exception as exc:
            log.warning("[DISAMBIGUATION] Gemini failed (%s) — using fallback.", exc)

    if not refined:
        refined = _fallback_select(candidates)
        log.info("[DISAMBIGUATION] Fallback selection: %d codes.", len(refined))

    # ── Apply Bayesian Confidence Adjustment ──────────────────────────────────
    term_to_posterior = {v["term"]: v.get("posterior", 0.5) for v in ner_votes}
    
    for code in refined:
        base_conf = code.get("confidence", 0.6)
        posterior = term_to_posterior.get(code["term"], 0.5)
        # Weighted average: 40% LLM, 60% Bayesian ensemble
        adjusted = round((base_conf * 0.4) + (posterior * 0.6), 2)
        
        log.info(
            "[DISAMBIGUATION] Adjusting '%s' confidence: LLM=%.2f, Bayes=%.2f -> Final=%.2f", 
            code["term"], base_conf, posterior, adjusted
        )
        code["confidence"] = adjusted

    # Overall confidence = mean of individual code confidences
    if refined:
        overall = round(
            sum(c["confidence"] for c in refined) / len(refined), 2
        )
    else:
        overall = 0.0

    latency = int((time.perf_counter() - start) * 1000) + 30

    return {
        "current_step":      "disambiguation",
        "step_count":        1,
        "path_taken":        ["disambiguation"],
        "icd10_codes":       refined,
        "overall_confidence": overall,
        "execution_time_ms": latency,
    }
