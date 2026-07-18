"""
clinical/steps/longitudinal_summary_step.py — Generate chronological pre-chart summaries.

v2: Upgraded to OncoLLM Pillar 3.
  - Uses the few-shot longitudinal prompt library with 2 worked examples
    enforcing a structured narrative format (Diagnosis → Treatment → Course → Status).
  - Simulates Triomics Symphony product.
"""

import time
import json
import os
import logging
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)


def _build_messages(history_text: str):
    """Build system + few-shot + real history message chain."""
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from clinical.prompts.longitudinal_prompts import build_longitudinal_prompt

    system_text, examples = build_longitudinal_prompt()

    messages = [SystemMessage(content=system_text)]

    for user_ex, ai_ex in examples:
        messages.append(HumanMessage(content=user_ex))
        messages.append(AIMessage(content=ai_ex))

    messages.append(HumanMessage(content=f"Visit History:\n{history_text}"))
    return messages


def longitudinal_summary_step(state: ClinicalState) -> dict:
    """
    Analyses visit_history to build a chronological pre-chart summary.

    Reads:  state["visit_history"]
    Writes: state["pre_chart_summary"]
    """
    start_time = time.perf_counter()
    visit_history = state.get("visit_history", [])

    if not visit_history or not os.getenv("GEMINI_API_KEY"):
        return {
            "current_step": "longitudinal_summary",
            "path_taken": ["longitudinal_summary"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "pre_chart_summary": "Insufficient visit history to generate longitudinal summary.",
        }

    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Format visit history — support both dict and plain string entries
    if isinstance(visit_history, list) and visit_history and isinstance(visit_history[0], dict):
        history_text = json.dumps(visit_history, indent=2)
    else:
        history_text = str(visit_history)

    try:
        messages = _build_messages(history_text)
        response = llm.invoke(messages)
        summary = response.content.strip()
    except Exception as exc:
        log.warning("[LONGITUDINAL SUMMARY] LLM failed: %s", exc)
        summary = "Failed to generate longitudinal summary."

    log.info("[LONGITUDINAL] Summary generated (%d chars)", len(summary))

    return {
        "current_step": "longitudinal_summary",
        "path_taken": ["longitudinal_summary"],
        "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
        "pre_chart_summary": summary,
    }
