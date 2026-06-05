import random
import os
from schemas.incident_state import IncidentState

def triage_step(state: IncidentState) -> dict:
    if os.getenv("SIMULATE_BIAS") == "true":
        # 80% chance to panic and set severity to high
        severity = random.choices(["low", "medium", "high"], weights=[0.1, 0.1, 0.8])[0]
    else:
        # Normal healthy distribution
        severity = random.choices(["low", "medium", "high"], weights=[0.6, 0.3, 0.1])[0]
        
    latency = random.randint(100, 300)
    
    return {
        "current_step": "triage",
        "step_count": 1,
        "path_taken": ["triage"],
        "severity": severity,
        "execution_time_ms": latency
    }
