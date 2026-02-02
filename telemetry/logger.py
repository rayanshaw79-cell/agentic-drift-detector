import json
from datetime import datetime
from schemas.incident_state import IncidentState


def log_step(state: IncidentState):
    log = {
        "incident_id": state["incident_id"],
        "step": state["current_step"],
        "step_count": state["step_count"],
        "retry_count": state["retry_count"],
        "path_taken": state["path_taken"],
        "timestamp": datetime.utcnow().isoformat()
    }

    with open("telemetry.jsonl", "a") as f:
        f.write(json.dumps(log) + "\n")
