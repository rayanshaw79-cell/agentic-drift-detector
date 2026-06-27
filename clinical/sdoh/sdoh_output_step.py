"""
clinical/sdoh/sdoh_output_step.py — SDOH Final Report Assembly Node.

Assembles the completed SDOH report dict from all upstream step outputs.
"""

import logging
import time

from schemas.sdoh_state import SdohState

log = logging.getLogger(__name__)

LABEL_COLORS = {
    "low":      "#3fb950",
    "moderate": "#d29922",
    "high":     "#f85149",
    "critical": "#ff0000",
}


def sdoh_output_step(state: SdohState) -> dict:
    """
    LangGraph node — SDOH Report Assembly.

    Reads:  all upstream state fields
    Writes: state["sdoh_report"]
    """
    start   = time.perf_counter()
    profile = state.get("sdoh_profile") or {}

    top_factors = state.get("shap_factors") or []
    top_factor_summary = ", ".join(
        f"{f['feature']} ({'+' if f['contribution'] > 0 else ''}{f['contribution']:.3f})"
        for f in top_factors[:3]
    ) if top_factors else "N/A"

    label = state.get("predicted_risk_label", "low")
    report = {
        # Identity
        "patient_id":            state.get("patient_id"),
        "visit_count":           len(state.get("visit_history") or []),
        # Demographics
        "age":                   profile.get("age"),
        "gender":                profile.get("gender"),
        "race":                  profile.get("race"),
        "zip_code":              profile.get("zip_code"),
        # SDOH Factors
        "smoking":               bool(profile.get("smoking_flag", 0)),
        "alcohol":               bool(profile.get("alcohol_flag", 0)),
        "exercise_score":        profile.get("exercise_score"),
        "food_risk_score":       profile.get("food_risk_score"),
        "env_aqi":               profile.get("env_aqi"),
        "env_poverty_rate":      profile.get("env_poverty_rate"),
        # Clinical
        "hcc_score":             profile.get("hcc_score"),
        "icd10_codes":           profile.get("icd10_codes", ""),
        # Risk Output
        "risk_trajectory":       state.get("risk_trajectory", []),
        "predicted_risk_label":  label,
        "risk_delta":            state.get("risk_delta", 0.0),
        "risk_label_color":      LABEL_COLORS.get(label, "#8b949e"),
        # Explainability
        "shap_factors":          top_factors,
        "top_factor_summary":    top_factor_summary,
        # Intervention
        "intervention_flag":     state.get("intervention_flag", False),
        "intervention_reason":   state.get("intervention_reason"),
    }

    latency = int((time.perf_counter() - start) * 1000) + 2
    log.info("[SDOH OUTPUT] Report assembled for patient %s (risk=%s, intervention=%s)",
             report["patient_id"], label, report["intervention_flag"])

    return {
        "current_step":      "sdoh_output",
        "path_taken":        ["sdoh_output"],
        "execution_time_ms": latency,
        "sdoh_report":       report,
    }
