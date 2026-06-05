import random
import os
from schemas.incident_state import IncidentState

# Check if we are simulating bias via env var
def is_bias_simulated():
    return os.getenv("SIMULATE_BIAS") == "true"

def decision_step(state: IncidentState) -> dict:
    # If bias is simulated, drop confidence heavily
    if is_bias_simulated():
        confidence = round(random.uniform(0.1, 0.45), 2)
    else:
        # We still have the old SIMULATE_RETRY_DRIFT fallback, but let's just 
        # assume normal healthy confidence here otherwise
        confidence = round(random.uniform(0.6, 0.95), 2)

    latency = random.randint(500, 1500)
    
    # We will determine the actual routing (retry vs escalate vs resolve)
    # inside the LangGraph conditional edge, NOT here.
    # We just record the decision output based on confidence.
    
    decision = "escalate"
    if state.get("severity") == "low" and confidence >= 0.6:
        decision = "auto_resolve"
        
    retry_increment = 0
    if confidence < 0.6 and state.get("retry_count", 0) < 1:
        retry_increment = 1
        
    return {
        "current_step": "decision",
        "step_count": 1,
        "path_taken": ["decision"],
        "confidence": confidence,
        "decision": decision,
        "retry_count": retry_increment,
        "execution_time_ms": latency
    }
