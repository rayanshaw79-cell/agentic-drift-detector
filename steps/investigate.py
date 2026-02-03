from schemas.incident_state import IncidentState
from telemetry.logger import log_step


def investigation_step(state: IncidentState) -> IncidentState:
    state["current_step"] = "investigation"
    state["step_count"] += 1
    state["path_taken"].append("investigation")

    severity = state.get("severity", "unknown")

    state["investigation_summary"] = (
        f"Reviewed logs and metrics for {severity} severity incident. "
        "Detected elevated response times."
    )

    log_step(state)
    return state
