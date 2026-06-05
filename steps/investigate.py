import random
from schemas.incident_state import IncidentState

def investigation_step(state: IncidentState) -> dict:
    severity = state.get("severity", "unknown")
    latency = random.randint(300, 800)

    summary = (
        f"Reviewed logs and metrics for {severity} severity incident. "
        "Detected elevated response times."
    )

    return {
        "current_step": "investigation",
        "step_count": 1,
        "path_taken": ["investigation"],
        "investigation_summary": summary,
        "execution_time_ms": latency
    }
