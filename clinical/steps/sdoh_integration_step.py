"""
clinical/steps/sdoh_integration_step.py — Unifies SDOH Risk Model into Clinical Workflow.

Runs the SDOH gradient boosting classifier on the outputs of the clinical coding
pipeline (ICD-10 codes, HCC scores, and MEAT validation) to predict patient 
risk trajectory and compute SHAP explanations.
"""

import logging
import time
import re

from schemas.clinical_state import ClinicalState
from clinical.sdoh.risk_model import load, predict_proba

log = logging.getLogger(__name__)

def _extract_age(text: str) -> int:
    """Fallback simple regex for age if not explicitly passed."""
    match = re.search(r'(\d{1,3})[- ]?year[- ]?old', text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 45  # Default

def sdoh_integration_step(state: ClinicalState) -> dict:
    """
    LangGraph node — SDOH Population Health Integration.

    Reads:  state["raw_note"], state["icd10_codes"]
    Writes: state["sdoh_risk_label"], state["sdoh_risk_score"], state["sdoh_shap_factors"]
    """
    start = time.perf_counter()

    codes = state.get("icd10_codes") or []
    icd10_code_count = len(codes)
    
    # Calculate total RAF weight for MEAT-verified conditions only
    hcc_score = sum(c.get("raf_weight", 0.0) for c in codes if c.get("meat_met", False))
    
    age = _extract_age(state.get("raw_note", ""))

    visit = {
        "age": age,
        "visit_number": 1,
        "hcc_score": hcc_score,
        "env_aqi": 60,                # Mock environmental/SDOH default
        "env_poverty_rate": 0.10,     # Mock environmental/SDOH default
        "food_risk_score": 0.0,
        "smoking_flag": 0,            # Would ideally come from NER
        "alcohol_flag": 0,
        "exercise_score": 0.5,
        "icd10_code_count": icd10_code_count,
        "chain_stage": 0,
        "sdoh_risk_score": 0.0,
    }

    try:
        bundle = load()
        label, proba, shap_factors = predict_proba(bundle, visit)
    except FileNotFoundError as exc:
        log.error("[SDOH INTEGRATION] Risk model not found: %s", exc)
        label = "unknown"
        proba = 0.0
        shap_factors = []
    except Exception as exc:
        log.error("[SDOH INTEGRATION] Error during prediction: %s", exc)
        label = "unknown"
        proba = 0.0
        shap_factors = []

    latency = int((time.perf_counter() - start) * 1000) + 10
    
    log.info("[SDOH INTEGRATION] Risk: %s (%.2f) | HCC: %.3f | Codes: %d", 
             label, proba, hcc_score, icd10_code_count)

    return {
        "current_step": "sdoh_integration",
        "step_count": 1,
        "path_taken": ["sdoh_integration"],
        "execution_time_ms": latency,
        "sdoh_risk_label": label,
        "sdoh_risk_score": proba,
        "sdoh_shap_factors": shap_factors,
    }
