import random
from schemas.incident_state import IncidentState
from telemetry.logger import log_step


def decision_step(state: IncidentState) -> IncidentState:
    state["current_step"] = "decision"
    state["step_count"] += 1
    state["path_taken"].append("decision")

    # Simulated confidence score
    confidence = round(random.uniform(0.4, 0.95), 2)
    state["confidence"] = confidence

    # Allow ONE retry if confidence is low
    if confidence < 0.6 and state["retry_count"] < 1:
        state["retry_count"] += 1
        log_step(state)
        return state

    # Final decision logic
    if state["severity"] == "low" and confidence >= 0.6:
        state["decision"] = "auto_resolve"
    else:
        state["decision"] = "escalate"

    log_step(state)
    return state
