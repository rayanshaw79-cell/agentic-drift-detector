"""
clinical/steps/oncology_staging_step.py — Extract primary site, histology,
and TNM staging.

v2: Upgraded to OncoLLM Pillar 1 + 2 + 3.
  - Pillar 1: Reads document_type from state (set by Constellation Router)
              to select the best-fit few-shot prompt variant.
  - Pillar 2: Queries the RAG guideline store for NCCN/NCI staging rules
              relevant to the note's cancer type; injects them as grounding
              context in the system prompt.
  - Pillar 3: Uses the few-shot staging prompt library instead of a generic
              one-liner, dramatically reducing hallucination rate.
  - Pillar 4: Accepts eval_feedback from the Evaluator node and appends it
              to the prompt on re-extraction passes.
"""

import time
import json
import os
import re
import logging
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)


def _build_messages(
    raw_note: str,
    document_type: str,
    rag_context: str | None,
    eval_feedback: str | None,
):
    """Construct the full message list: system + few-shot pairs + real note."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from clinical.prompts.staging_prompts import build_staging_prompt

    system_text, examples = build_staging_prompt(
        document_type=document_type,
        rag_context=rag_context,
    )

    messages = [SystemMessage(content=system_text)]

    # Inject few-shot examples as alternating Human/AI turns
    for user_ex, ai_ex in examples:
        messages.append(HumanMessage(content=f"Clinical note:\n{user_ex}"))
        messages.append(AIMessage(content=ai_ex))

    # Optionally inject evaluator feedback from a previous failed pass
    note_content = f"Clinical note:\n{raw_note}"
    if eval_feedback:
        note_content += (
            f"\n\n[EVALUATOR FEEDBACK — CORRECTION REQUIRED]\n{eval_feedback}\n"
            f"Please re-examine the note and correct the issues above."
        )

    messages.append(HumanMessage(content=note_content))
    return messages


def oncology_staging_step(state: ClinicalState) -> dict:
    """
    Extracts TNM staging, primary site, and histology from raw_note.

    Reads:
      state["raw_note"], state["document_type"], state["eval_feedback"]
    Writes:
      state["primary_site"], state["histology"], state["tnm_stage"]
    """
    start_time = time.perf_counter()
    raw_note = state.get("raw_note", "")
    document_type = state.get("document_type") or "unknown"
    eval_feedback = state.get("eval_feedback")

    if not raw_note or not os.getenv("GEMINI_API_KEY"):
        return {
            "current_step": "oncology_staging",
            "path_taken": ["oncology_staging"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "tnm_stage": None,
            "primary_site": None,
            "histology": None,
            "needs_reextraction": False,
        }

    # ── Pillar 2: RAG guideline retrieval ─────────────────────────────────────
    rag_context: str | None = None
    try:
        from clinical.rag.guideline_store import retrieve_guidelines
        # Use a rough query — the site may not be extracted yet
        rag_query = document_type if document_type != "unknown" else "cancer staging TNM AJCC"
        rag_context = retrieve_guidelines(query=rag_query, k=2)
        if rag_context:
            log.debug("[STAGING] RAG context retrieved (%d chars)", len(rag_context))
    except Exception as exc:
        log.debug("[STAGING] RAG unavailable: %s — proceeding without.", exc)

    # ── LLM call ──────────────────────────────────────────────────────────────
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    try:
        messages = _build_messages(raw_note, document_type, rag_context, eval_feedback)
        response = llm.invoke(messages)

        content = response.content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        extracted = json.loads(content)

        # ── Strict provenance check ────────────────────────────────────────────
        for key in ["primary_site", "histology"]:
            val = extracted.get(key)
            if val and val not in raw_note:
                log.warning(
                    "[STAGING PROVENANCE FAILURE] '%s' value '%s' not in raw note. Rejecting.",
                    key, val,
                )
                extracted[key] = None

        if extracted.get("tnm_stage") and extracted["tnm_stage"].get("evidence_span"):
            ev = extracted["tnm_stage"]["evidence_span"]
            if ev and ev not in raw_note:
                log.warning(
                    "[STAGING PROVENANCE FAILURE] TNM evidence_span '%s' not in raw note. Rejecting.",
                    ev,
                )
                extracted["tnm_stage"] = None

    except Exception as exc:
        log.warning("[ONCOLOGY STAGING] LLM extraction failed: %s", exc)
        extracted = {"primary_site": None, "histology": None, "tnm_stage": None}

    return {
        "current_step": "oncology_staging",
        "path_taken": ["oncology_staging"],
        "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
        "primary_site": extracted.get("primary_site"),
        "histology": extracted.get("histology"),
        "tnm_stage": extracted.get("tnm_stage"),
        # Reset evaluator flags — evaluator will re-set if needed
        "needs_reextraction": False,
        "eval_feedback": None,
    }
