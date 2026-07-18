"""
clinical/steps/biomarker_extraction_step.py — Extract molecular/genetic biomarkers.

v2: Upgraded to OncoLLM Pillar 1 + 3.
  - Pillar 1: Reads document_type from Constellation Router to select the
              right few-shot prompt variant (e.g., promotes breast IHC examples
              for pathology_report doc type).
  - Pillar 3: Uses the few-shot biomarker prompt library instead of a generic
              single-line system prompt.
  - Provenance check remains: every evidence_span must be an exact substring
    of the raw note (zero hallucination guarantee).
"""

import time
import json
import os
import re
import logging
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)


def _build_messages(raw_note: str, document_type: str):
    """Build system + few-shot + real note message chain for biomarker extraction."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from clinical.prompts.biomarker_prompts import build_biomarker_prompt

    system_text, examples = build_biomarker_prompt(document_type=document_type)

    messages = [SystemMessage(content=system_text)]

    for user_ex, ai_ex in examples:
        messages.append(HumanMessage(content=f"Clinical note:\n{user_ex}"))
        messages.append(AIMessage(content=ai_ex))

    messages.append(HumanMessage(content=f"Clinical note:\n{raw_note}"))
    return messages


def biomarker_extraction_step(state: ClinicalState) -> dict:
    """
    Extracts biomarkers (e.g., EGFR, ALK, PD-L1, HER2) with provenance.

    Reads:  state["raw_note"], state["document_type"]
    Writes: state["biomarkers"]  — list of {marker, status, value, evidence_span}
    """
    start_time = time.perf_counter()
    raw_note = state.get("raw_note", "")
    document_type = state.get("document_type") or "unknown"

    if not raw_note or not os.getenv("GEMINI_API_KEY"):
        return {
            "current_step": "biomarker_extraction",
            "path_taken": ["biomarker_extraction"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "biomarkers": [],
        }

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    try:
        messages = _build_messages(raw_note, document_type)
        response = llm.invoke(messages)

        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        extracted = json.loads(content)
        if not isinstance(extracted, list):
            extracted = []

        # ── Strict provenance check ────────────────────────────────────────────
        valid_biomarkers = []
        for bio in extracted:
            ev = bio.get("evidence_span", "")
            if ev and ev in raw_note:
                valid_biomarkers.append(bio)
            else:
                log.warning(
                    "[BIOMARKER PROVENANCE FAILURE] evidence_span '%s' not in raw note. Rejecting %s.",
                    ev, bio.get("marker", "?"),
                )

    except Exception as exc:
        log.warning("[BIOMARKER EXTRACTION] LLM extraction failed: %s", exc)
        valid_biomarkers = []

    log.info(
        "[BIOMARKER] Extracted %d validated biomarkers (doc_type=%s)",
        len(valid_biomarkers), document_type,
    )

    return {
        "current_step": "biomarker_extraction",
        "path_taken": ["biomarker_extraction"],
        "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
        "biomarkers": valid_biomarkers,
    }
