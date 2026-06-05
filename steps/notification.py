import random
from schemas.incident_state import IncidentState

def notification_step(state: IncidentState) -> dict:
    latency = random.randint(50, 150)
    
    message = (
        f"INCIDENT {state.get('incident_id')} | "
        f"Severity: {state.get('severity')} | "
        f"Decision: {state.get('decision')} | "
        f"Confidence: {state.get('confidence')}"
    )

    print("\n[NOTIFICATION SENT]")
    print(message)

    return {
        "current_step": "notification",
        "step_count": 1,
        "path_taken": ["notification"],
        "execution_time_ms": latency
    }
