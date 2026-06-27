"""
clinical/sdoh/intervention_check_step.py — SDOH Intervention Flag Node.

Fires intervention_flag = True when a patient's risk is accelerating:
  • risk_delta > DELTA_THRESHOLD for the most recent visit, OR
  • The last two consecutive deltas are both positive (sustained rise), OR
  • predicted_risk_label is "critical"
"""

import logging
import time

from schemas.sdoh_state import SdohState

log = logging.getLogger(__name__)

DELTA_THRESHOLD = 0.15      # Single-visit acceleration threshold
CRITICAL_LABEL  = "critical"


# ── LangGraph Node ────────────────────────────────────────────────────────────

def intervention_check_step(state: SdohState) -> dict:
    """
    LangGraph node — SDOH Intervention Flagging.

    Reads:  state["risk_trajectory"], state["risk_delta"],
            state["predicted_risk_label"]
    Writes: state["intervention_flag"], state["intervention_reason"]
    """
    start      = time.perf_counter()
    trajectory = state.get("risk_trajectory") or []
    risk_delta = state.get("risk_delta", 0.0)
    label      = state.get("predicted_risk_label", "low")

    intervention_flag   = False
    intervention_reason = None

    if label == CRITICAL_LABEL:
        intervention_flag   = True
        intervention_reason = (
            f"Patient has reached CRITICAL risk level (predicted label: {label}). "
            "Immediate clinical intervention recommended."
        )
    elif risk_delta > DELTA_THRESHOLD:
        intervention_flag   = True
        intervention_reason = (
            f"Risk score accelerated by {risk_delta:.2f} in the most recent visit "
            f"(threshold: {DELTA_THRESHOLD}). Preventive intervention recommended."
        )
    elif len(trajectory) >= 3:
        # Check if the last two consecutive deltas are both positive (sustained upward trend)
        d1 = trajectory[-2] - trajectory[-3]
        d2 = trajectory[-1] - trajectory[-2]
        if d1 > 0 and d2 > 0:
            intervention_flag   = True
            intervention_reason = (
                f"Sustained risk escalation over the last 3 visits "
                f"(+{d1:.2f}, +{d2:.2f}). Risk trajectory is trending upward."
            )

    if intervention_flag:
        log.warning("[INTERVENTION CHECK] Flag raised for patient %s: %s",
                    state.get("patient_id"), intervention_reason)
    else:
        log.info("[INTERVENTION CHECK] No intervention required for patient %s",
                 state.get("patient_id"))

    latency = int((time.perf_counter() - start) * 1000) + 2

    return {
        "current_step":        "intervention_check",
        "path_taken":          ["intervention_check"],
        "execution_time_ms":   latency,
        "intervention_flag":   intervention_flag,
        "intervention_reason": intervention_reason,
    }
