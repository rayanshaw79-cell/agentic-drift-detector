from schemas.incident_state import IncidentState
from telemetry.logger import log_step


def notification_step(state: IncidentState) -> IncidentState:
    state["current_step"] = "notification"
    state["step_count"] += 1
    state["path_taken"].append("notification")

    message = (
        f"INCIDENT {state['incident_id']} | "
        f"Severity: {state['severity']} | "
        f"Decision: {state['decision']} | "
        f"Confidence: {state['confidence']}"
    )

    print("\n📣 NOTIFICATION SENT")
    print(message)

    log_step(state)
    return state
