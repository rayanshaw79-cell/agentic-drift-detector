import random
from schemas.incident_state import IncidentState
from telemetry.logger import log_step


def triage_step(state: IncidentState) -> IncidentState:
    state["current_step"] = "triage"
    state["step_count"] += 1
    state["path_taken"].append("triage")

    # Simulated severity classification
    state["severity"] = random.choice(["low", "medium", "high"])

    log_step(state)
    return state
