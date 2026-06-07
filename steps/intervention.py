import random
from schemas.incident_state import IncidentState


def intervention_step(state: IncidentState) -> dict:
    """
    Agentic Healing Step: Triggered when a drift loop is detected (retry_count > 1).
    This node forcibly stabilizes the agent by:
    - Resetting confidence to a stable value
    - Forcing an escalation decision (safest action under uncertainty)
    - Logging the intervention for telemetry
    """
    print("\n  [INTERVENTION] DRIFT LOOP DETECTED — Applying agentic healing...")
    print(f"  [INTERVENTION] Agent had retried {state.get('retry_count', 0)} times with low confidence.")
    print("  [INTERVENTION] Forcing escalation to restore autonomy quality.")

    # Stable forced confidence and decision
    forced_confidence = round(random.uniform(0.65, 0.75), 2)
    latency = random.randint(100, 300)

    return {
        "current_step": "intervention",
        "step_count": 1,
        "path_taken": ["intervention"],
        "confidence": forced_confidence,
        "decision": "escalate",
        "execution_time_ms": latency
    }
