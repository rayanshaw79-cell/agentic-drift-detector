"""
clinical/steps/meat_validation_step.py — MEAT Criteria Validation Node.

MEAT = Monitored, Evaluated, Assessed, Treated.

Medicare Risk Adjustment (HCC) coding is only valid if the provider
demonstrated they ADDRESSED a condition during the current encounter.
Simply noting "Patient has a history of diabetes" in the Past Medical
History section does NOT qualify for HCC Risk Weight credit.

This node:
  1. Takes the list of disambiguated icd10_codes from state.
  2. For each condition, asks Gemini to find direct textual evidence of
     clinical action (M/E/A/T) in the raw clinical note.
  3. Attaches three new keys to each code dict:
       - meat_met      (bool)  — was there documented clinical action?
       - meat_category (str)   — "Monitored"|"Evaluated"|"Assessed"|"Treated"|"None"
       - meat_evidence (str)   — the exact quoted sentence from the note
  4. If meat_met is False the HCC RAF weight is zeroed out in the
     subsequent validation_step so it doesn't contribute to fraudulent
     risk adjustment.
"""

import json
import logging
import os
import time

from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

# MEAT category keywords used as the fallback heuristic
_MEAT_KEYWORDS: dict[str, list[str]] = {
    "Monitored": [
        "monitoring", "monitor", "tracked", "tracking", "follow-up",
        "checked", "checking", "watch", "watched", "observed", "observing",
        "repeat labs", "repeat test",
    ],
    "Evaluated": [
        "evaluated", "evaluation", "reviewed", "assess", "assessed",
        "assessment", "examined", "examination", "work-up", "workup",
        "ordered labs", "lab results", "test results",
    ],
    "Assessed": [
        "well-controlled", "poorly controlled", "stable", "unstable",
        "worsening", "improving", "in remission", "exacerbated",
        "controlled", "uncontrolled",
    ],
    "Treated": [
        "prescribed", "started", "adjusted", "increased", "decreased",
        "refilled", "continue", "continued", "discontinued", "initiated",
        "administered", "injected", "dose", "dosage", "medication",
        "mg", "therapy", "treatment", "surgery", "procedure",
    ],
}

_FALLBACK_MSG = "No direct MEAT evidence found in note."


def _llm_meat_validate(raw_note: str, codes: list[dict]) -> list[dict]:
    """
    Call Gemini 2.0 Flash to perform structured MEAT validation.
    Returns a list of result dicts keyed by ICD-10 code.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    conditions_list = "\n".join(
        f"  - {c['code']}: {c.get('description', c.get('term', ''))}"
        for c in codes
    )

    system_prompt = """You are a certified clinical coding auditor specialising in
Medicare Risk Adjustment (RADV). Your task is to determine whether a provider
has documented clinical action (Monitoring, Evaluation, Assessment, or Treatment)
for each listed condition in a clinical note.

Respond ONLY with a valid JSON array. Each element must have:
  "code"         — the ICD-10 code (unchanged)
  "meat_met"     — true if ANY clinical action is documented, false otherwise
  "meat_category"— one of: "Monitored", "Evaluated", "Assessed", "Treated", "None"
  "meat_evidence"— the EXACT verbatim sentence or phrase from the note that proves
                   the action. If meat_met is false, set this to "".

IMPORTANT RULES:
  - Past Medical History (PMH) mentions alone do NOT count as clinical action.
  - Family history mentions do NOT count.
  - "Patient has a history of X" does NOT count.
  - The action must be related to the CURRENT ENCOUNTER.
  - Quote verbatim — do not paraphrase."""

    user_prompt = f"""CLINICAL NOTE:
---
{raw_note}
---

CONDITIONS TO VALIDATE:
{conditions_list}

Return the JSON array now."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # Strip markdown fences if Gemini wraps the JSON
    raw_text = response.content.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    return json.loads(raw_text)


def _regex_meat_validate(raw_note: str, codes: list[dict]) -> list[dict]:
    """
    Fast regex fallback — scan the note for MEAT keyword families.

    Strategy:
      Pass 1: look for a MEAT keyword in the SAME sentence as the condition term.
      Pass 2: if no co-occurrence, broaden search to the whole note — BUT only
              if the term doesn't appear exclusively in family-history sentences.
              This prevents attributing another condition's treatment to an item
              that is only a family-history mention (e.g. "mother had breast cancer").
    """
    results = []

    # Family-history / negation phrase indicators
    _FAMILY = {"mother", "father", "sibling", "sister", "brother",
               "grandmother", "grandfather", "family history", "fh:", "fhx",
               "denies", "no history of", "rules out", "ruled out"}

    # Pre-split the note into sentences once
    sentences = [s.strip() for s in raw_note.replace("\n", ". ").split(".") if s.strip()]

    for code in codes:
        term = code.get("term", code.get("description", "")).lower()
        found_cat  = "None"
        found_snip = ""

        # ── Pass 1: look for MEAT keyword in a sentence that also contains the term ──
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if term not in sentence_lower:
                continue
            for category, keywords in _MEAT_KEYWORDS.items():
                if any(kw in sentence_lower for kw in keywords):
                    found_cat  = category
                    found_snip = sentence.strip()
                    break
            if found_cat != "None":
                break

        # ── Pass 2: broader search across whole note ──────────────────────────
        # Only runs if Pass 1 failed AND the term is not exclusively inside
        # family-history or negation-flavoured sentences.
        if found_cat == "None" and term in raw_note.lower():
            term_sentences = [s for s in sentences if term in s.lower()]
            only_family = term_sentences and all(
                any(indicator in s.lower() for indicator in _FAMILY)
                for s in term_sentences
            )
            if not only_family:
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    for category, keywords in _MEAT_KEYWORDS.items():
                        if any(kw in sentence_lower for kw in keywords):
                            found_cat  = category
                            found_snip = sentence.strip()
                            break
                    if found_cat != "None":
                        break

        results.append({
            "code":          code["code"],
            "meat_met":      found_cat != "None",
            "meat_category": found_cat,
            "meat_evidence": found_snip or _FALLBACK_MSG,
        })

    return results


def _apply_meat_results(codes: list[dict], meat_results: list[dict]) -> list[dict]:
    """
    Merge MEAT validation results back onto the code dicts.
    If meat_met is False, zero out the RAF weight so it doesn't
    contribute to fraudulent risk adjustment scoring.
    """
    meat_by_code = {r["code"]: r for r in meat_results}

    enriched = []
    for code in codes:
        icd = code.get("code", "UNRESOLVED")
        meat = meat_by_code.get(icd, {
            "meat_met":      False,
            "meat_category": "None",
            "meat_evidence": _FALLBACK_MSG,
        })

        enriched_code = {**code}
        enriched_code["meat_met"]      = meat["meat_met"]
        enriched_code["meat_category"] = meat["meat_category"]
        enriched_code["meat_evidence"] = meat["meat_evidence"]

        # Fraud-prevention: zero out RAF weight if MEAT not satisfied
        if not meat["meat_met"]:
            enriched_code["raf_weight"] = 0.0
            log.warning(
                "[MEAT] %-10s %-50s → MEAT NOT MET — RAF weight zeroed.",
                icd,
                code.get("description", "")[:50],
            )
        else:
            log.info(
                "[MEAT] %-10s → %-10s | Evidence: %s",
                icd,
                meat["meat_category"],
                meat["meat_evidence"][:80],
            )

        enriched.append(enriched_code)  # ← was missing!

    return enriched


def meat_validation_step(state: ClinicalState) -> dict:
    """
    LangGraph node — MEAT Criteria Validation.

    Reads:  state["icd10_codes"], state["raw_note"]
    Writes: state["icd10_codes"]  (enriched with MEAT fields)
            state["meat_results"] (raw LLM output for audit trail)
    """
    codes    = state.get("icd10_codes") or []
    raw_note = state.get("raw_note", "")
    start    = time.perf_counter()

    if not codes:
        log.info("[MEAT] No codes to validate — skipping.")
        return {
            "current_step":      "meat_validation",
            "step_count":        1,
            "path_taken":        ["meat_validation"],
            "meat_results":      [],
            "execution_time_ms": 0,
        }

    # ── Attempt LLM validation, fall back to regex ────────────────────────────
    meat_results: list[dict] = []
    try:
        meat_results = _llm_meat_validate(raw_note, codes)
        log.info("[MEAT] LLM validated %d codes.", len(meat_results))
    except Exception as exc:
        log.warning("[MEAT] LLM failed (%s) — using regex fallback.", exc)
        meat_results = _regex_meat_validate(raw_note, codes)

    # ── Merge results back onto icd10_codes ──────────────────────────────────
    enriched_codes = _apply_meat_results(codes, meat_results)

    meat_met_count = sum(1 for c in enriched_codes if c.get("meat_met"))
    log.info(
        "[MEAT] %d/%d codes satisfied MEAT criteria.",
        meat_met_count, len(enriched_codes),
    )

    latency = int((time.perf_counter() - start) * 1000) + 5

    return {
        "current_step":      "meat_validation",
        "step_count":        1,
        "path_taken":        ["meat_validation"],
        "icd10_codes":       enriched_codes,
        "meat_results":      meat_results,
        "execution_time_ms": latency,
    }
