"""
clinical/steps/compliance_checker_step.py — Compliance Checker Sub-Agent.

Acts as a strict auditor for the De-identification step. Evaluates the scrubbed
note to ensure no PHI/PII remains. If it finds leaks, it flags them so the 
workflow can loop back for another De-ID pass.
"""

import os
import time
import logging
from typing import Dict, Any

from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

def compliance_checker_step(state: ClinicalState) -> Dict[str, Any]:
    """
    LangGraph node — PHI Compliance Check.
    
    Reads:  state["deid_note"]
    Writes: state["phi_detected"], state["privacy_leak_risk"], state["retry_count"]
    """
    start = time.perf_counter()
    deid_note = state.get("deid_note", "")
    use_gemini = bool(os.getenv("GEMINI_API_KEY"))
    
    phi_detected = False
    risk_increment = 0.0
    
    if use_gemini and deid_note:
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
                    "You are a strict HIPAA compliance auditor. Review the following "
                    "redacted clinical note. Return ONLY 'PASS' if absolutely NO "
                    "identifying information (names, real dates, specific locations, SSNs, phone numbers) "
                    "remains in the text. If you find ANY identifying information that was missed, "
                    "return 'FAIL' followed by a brief reason."
                )),
                HumanMessage(content=f"Redacted note:\n{deid_note}"),
            ])
            
            content = response.content.strip().upper()
            if content.startswith("FAIL"):
                phi_detected = True
                risk_increment = 0.5  # Add risk penalty for failing compliance
                log.warning(f"[Compliance] PHI leak detected! Reason: {content}")
            else:
                log.info("[Compliance] De-ID passed successfully.")
                
        except Exception as exc:
            log.error(f"[Compliance] LLM pass failed: {exc}.")
            
    latency = int((time.perf_counter() - start) * 1000)
    
    update = {
        "current_step": "compliance_checker",
        "step_count": 1,
        "path_taken": ["compliance_checker"],
        "execution_time_ms": latency,
        "phi_detected": phi_detected,
    }
    
    # We only increment retry/risk if it failed
    if phi_detected:
        update["retry_count"] = 1
        update["privacy_leak_risk"] = risk_increment
        
    return update
