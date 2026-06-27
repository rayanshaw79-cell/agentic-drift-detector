"""
clinical/sdoh/risk_trajectory_step.py — Risk Trajectory Scoring Node.

Loads the pre-trained GradientBoostingClassifier and scores every visit
in the patient's history to produce a list of risk probabilities over time.
Also computes risk_delta (change from previous to current visit).
"""

import logging
import time

from clinical.sdoh.risk_model import load, predict_proba
from schemas.sdoh_state import SdohState

log = logging.getLogger(__name__)


# ── LangGraph Node ────────────────────────────────────────────────────────────

def risk_trajectory_step(state: SdohState) -> dict:
    """
    LangGraph node — Risk Trajectory Scoring.

    Reads:  state["visit_history"], state["sdoh_profile"]
    Writes: state["risk_trajectory"], state["risk_delta"],
            state["predicted_risk_label"], state["shap_factors"]
    """
    start         = time.perf_counter()
    visit_history = state.get("visit_history") or []
    sdoh_profile  = state.get("sdoh_profile") or {}

    if not visit_history:
        log.warning("[RISK TRAJECTORY] Empty visit history — skipping.")
        return {
            "current_step":        "risk_trajectory",
            "path_taken":          ["risk_trajectory"],
            "execution_time_ms":   int((time.perf_counter() - start) * 1000),
            "risk_trajectory":     [],
            "risk_delta":          0.0,
            "predicted_risk_label": "low",
            "shap_factors":        [],
        }

    # Load the trained model bundle
    try:
        bundle = load()
    except FileNotFoundError as exc:
        log.error("[RISK TRAJECTORY] %s", exc)
        return {
            "current_step":        "risk_trajectory",
            "path_taken":          ["risk_trajectory"],
            "execution_time_ms":   int((time.perf_counter() - start) * 1000),
            "risk_trajectory":     [],
            "risk_delta":          0.0,
            "predicted_risk_label": "unknown",
            "shap_factors":        [],
        }

    # Score every historical visit to build the trajectory
    trajectory: list[float] = []
    for visit in visit_history:
        label, proba, _ = predict_proba(bundle, visit)
        # Map label to a normalised 0-1 risk score for charting
        label_to_score = {"low": 0.15, "moderate": 0.40, "high": 0.70, "critical": 0.95}
        trajectory.append(label_to_score.get(label, proba))

    # Score the current (latest) visit with full SHAP explanation
    current_visit = {**visit_history[-1], **sdoh_profile}
    current_label, current_proba, shap_factors = predict_proba(bundle, current_visit)

    # Risk delta: change from 2nd-last to last visit
    risk_delta = round(trajectory[-1] - trajectory[-2], 3) if len(trajectory) >= 2 else 0.0

    latency = int((time.perf_counter() - start) * 1000) + 10
    log.info(
        "[RISK TRAJECTORY] Patient %s | label=%s | delta=%.3f | visits=%d",
        state.get("patient_id"), current_label, risk_delta, len(visit_history)
    )

    return {
        "current_step":         "risk_trajectory",
        "path_taken":           ["risk_trajectory"],
        "execution_time_ms":    latency,
        "risk_trajectory":      trajectory,
        "risk_delta":           risk_delta,
        "predicted_risk_label": current_label,
        "shap_factors":         shap_factors,
    }
