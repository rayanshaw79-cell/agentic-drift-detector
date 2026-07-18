"""
clinical/agents/oncology_router.py — Constellation Router (OncoLLM Pillar 1).

Classifies the incoming clinical document into one of four types so that
downstream specialist agents can load the optimal prompt variant.

Document types:
  pathology_report — structured path report with explicit T/N/M, IHC results
  radiology        — CT/PET/MRI report, staging implied from imaging findings
  genomics         — NGS / molecular pathology panel report
  progress_note    — narrative oncologist/clinic note
  unknown          — cannot determine type

The classification runs BEFORE oncology_staging and biomarker_extraction.
It adds `document_type` to state so those nodes can load matching prompts.
"""

import os
import time
import logging
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

# ── Keyword-based fast-path classifier ───────────────────────────────────────
# Used as a cheap heuristic before calling the LLM. If confident (≥ 2 hits),
# skips the LLM call entirely to save tokens and latency.

_KEYWORD_SIGNALS: dict[str, list[str]] = {
    "pathology_report": [
        "pathology report", "surgical pathology", "core needle biopsy",
        "histologic", "histological", "microscopic description",
        "final pathologic stage", "pT", "pN", "pM", "ajcc",
        "immunohistochemistry", "IHC", "hematoxylin", "eosin",
    ],
    "radiology": [
        "CT ", "ct chest", "ct abdomen", "MRI ", "PET ", "pet scan",
        "x-ray", "radiograph", "findings:", "impression:", "hypodense",
        "hyperintense", "spiculated", "nodule", "lesion", "lymph node",
        "clinical staging", "cT", "cN", "cM",
    ],
    "genomics": [
        "next generation sequencing", "NGS", "molecular pathology",
        "mutation analysis", "FISH", "gene panel", "variant",
        "exon 19", "exon 21", "deletion detected", "amplification",
        "microsatellite", "MSI", "TMB", "tumor mutational burden",
    ],
    "progress_note": [
        "history of present illness", "HPI", "assessment and plan",
        "plan:", "follow-up", "clinic note", "oncology note",
        "current medications", "review of systems", "chief complaint",
        "subjective:", "objective:", "assessment:", "plan:",
    ],
}

_MIN_HITS_FOR_FAST_PATH = 2


def _classify_by_keywords(text: str) -> str | None:
    """Return document type if keyword heuristic is confident, else None."""
    text_lower = text.lower()
    scores = {}
    for doc_type, keywords in _KEYWORD_SIGNALS.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        scores[doc_type] = hits

    best_type = max(scores, key=lambda k: scores[k])
    if scores[best_type] >= _MIN_HITS_FOR_FAST_PATH:
        log.debug(
            "[ROUTER] Keyword fast-path: %s (score=%d)", best_type, scores[best_type]
        )
        return best_type
    return None


def _classify_by_llm(raw_note: str) -> str:
    """Fallback LLM classifier for ambiguous documents."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    system = (
        "You are a medical document classifier. Classify the following clinical "
        "document into EXACTLY ONE of these categories:\n"
        "  pathology_report, radiology, genomics, progress_note, unknown\n\n"
        "Respond with ONLY the single category word. No explanation."
    )

    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Document (first 800 chars):\n{raw_note[:800]}"),
    ])
    result = response.content.strip().lower().replace('"', "").replace("'", "")

    valid = {"pathology_report", "radiology", "genomics", "progress_note", "unknown"}
    return result if result in valid else "unknown"


def oncology_router(state: ClinicalState) -> dict:
    """
    LangGraph node — Constellation Router.

    Reads:  state["raw_note"] (or state["deid_note"] if available)
    Writes: state["document_type"]
    """
    start_time = time.perf_counter()

    # Prefer de-identified note if available
    note = state.get("deid_note") or state.get("raw_note", "")

    if not note:
        log.warning("[ROUTER] No note available for classification.")
        doc_type = "unknown"
    else:
        # Fast path: keyword heuristic
        doc_type = _classify_by_keywords(note)

        # Slow path: LLM fallback (only if API key present and heuristic failed)
        if doc_type is None:
            if os.getenv("GEMINI_API_KEY"):
                try:
                    doc_type = _classify_by_llm(note)
                    log.info("[ROUTER] LLM classified document as: %s", doc_type)
                except Exception as exc:
                    log.warning("[ROUTER] LLM classification failed: %s — defaulting to unknown", exc)
                    doc_type = "unknown"
            else:
                doc_type = "unknown"

    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    log.info("[ROUTER] document_type=%s  latency=%dms", doc_type, elapsed_ms)

    return {
        "current_step": "oncology_router",
        "document_type": doc_type,
        "path_taken": ["oncology_router"],
        "execution_time_ms": elapsed_ms,
    }
