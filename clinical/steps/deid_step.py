"""
clinical/steps/deid_step.py — De-identification Sub-Agent.

Scrub PHI/PII from unstructured clinical notes before they enter the main pipeline.
Combines fast regex matching for explicit patterns (SSNs, dates) with an LLM
pass for contextual identifiers (names, locations).
"""

import os
import re
import time
import logging
from typing import Dict, Any

from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

def _regex_deid(text: str) -> str:
    """Fast regex pass for standard PHI patterns."""
    # SSN pattern
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", text)
    # Phone numbers (simple US formats)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    # Dates (MM/DD/YYYY or similar)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", "[DATE]", text)
    # DOB specific indicator
    text = re.sub(r"(?i)\bDOB:\s*\S+", "DOB: [DATE]", text)
    return text

def deid_step(state: ClinicalState) -> Dict[str, Any]:
    """
    LangGraph node — PHI De-identification.
    
    Reads:  state["raw_note"] (on first pass) or state["deid_note"] (if looping)
    Writes: state["deid_note"]
    """
    start = time.perf_counter()
    
    # If we are looping back from the compliance checker, we refine the already partially scrubbed note
    # Otherwise, we start fresh with the raw note.
    text_to_scrub = state.get("deid_note")
    if not text_to_scrub:
        text_to_scrub = state.get("raw_note", "")
        
    # 1. Regex Pass
    regex_scrubbed = _regex_deid(text_to_scrub)
    
    # 2. LLM Pass for Contextual PHI
    use_gemini = bool(os.getenv("GEMINI_API_KEY"))
    final_scrubbed = regex_scrubbed
    
    if use_gemini:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import SystemMessage, HumanMessage
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash",
                temperature=0,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
            response = llm.invoke([
                SystemMessage(content=(
                    "You are a strict HIPAA compliance officer. Your job is to redact all "
                    "Protected Health Information (PHI) from the following clinical note. "
                    "Replace any patient names, doctor names, specific hospital names, "
                    "addresses, and other unique identifiers with placeholders like "
                    "[PATIENT_NAME], [DOCTOR_NAME], [HOSPITAL], [LOCATION]. "
                    "Preserve all clinical information, symptoms, medical history, and diagnoses exactly as written. "
                    "Return ONLY the redacted clinical note."
                )),
                HumanMessage(content=f"Clinical note:\n{regex_scrubbed}"),
            ])
            final_scrubbed = response.content.strip()
            log.info("[DeID] Completed LLM redaction pass.")
        except Exception as exc:
            log.error(f"[DeID] LLM pass failed: {exc}. Falling back to Regex only.")
    
    latency = int((time.perf_counter() - start) * 1000)
    
    return {
        "current_step": "deid",
        "step_count": 1,
        "path_taken": ["deid"],
        "deid_note": final_scrubbed,
        "execution_time_ms": latency,
    }
