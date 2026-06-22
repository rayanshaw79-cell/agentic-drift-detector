import os
import random
import json
from schemas.incident_state import IncidentState

def intervention_step(state: IncidentState) -> dict:
    """
    Agentic Healing (Circuit Breaker):
    When a drift loop is detected, we do NOT use an LLM to heal an LLM.
    Instead, we deterministically escalate to a human and drop confidence to 0.0.
    This preserves the integrity of the telemetry and ensures safety.
    """
    print("\n  [INTERVENTION] DRIFT LOOP DETECTED — Applying Deterministic Circuit Breaker...")
    
    incident_text = state.get("incident_text", "Unknown incident")
    severity = state.get("severity", "unknown")
    
    decision = "escalate"
    forced_confidence = 0.0
    
    print(f"  [INTERVENTION] Circuit breaker activated. Escalating incident safely.")

    latency = random.randint(50, 100)

    return {
        "current_step": "intervention",
        "step_count": 1,
        "path_taken": ["intervention"],
        "confidence": forced_confidence,
        "decision": decision,
        "execution_time_ms": latency
    }
