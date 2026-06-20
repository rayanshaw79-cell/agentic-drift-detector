import os
import random
import json
from schemas.incident_state import IncidentState

def _llm_reviewer(incident_text: str, severity: str) -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "You are a Senior SRE Reviewer acting as a circuit breaker for a looping agent. "
            "Your job is to definitively resolve ambiguity. "
            "Return ONLY a JSON object with 'decision' (either 'escalate' or 'auto_resolve') "
            "and 'confidence' (a float between 0.0 and 1.0). "
            "If in doubt, escalate. Example: {\"decision\": \"escalate\", \"confidence\": 0.95}"
        )),
        HumanMessage(content=f"Incident: {incident_text}\nSeverity: {severity}")
    ])
    
    try:
        return json.loads(response.content.strip())
    except:
        return {"decision": "escalate", "confidence": 1.0}

def intervention_step(state: IncidentState) -> dict:
    """
    Smarter Agentic Healing: 
    When a drift loop is detected, we instantiate a Senior Reviewer Agent to break the tie.
    If the reviewer still lacks confidence, we default to escalation.
    """
    print("\n  [INTERVENTION] DRIFT LOOP DETECTED — Applying Smarter Agentic Healing...")
    
    incident_text = state.get("incident_text", "Unknown incident")
    severity = state.get("severity", "unknown")
    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    
    decision = "escalate"
    forced_confidence = 1.0
    
    if use_llm:
        try:
            print("  [INTERVENTION] Summoning Senior Reviewer Agent...")
            result = _llm_reviewer(incident_text, severity)
            decision = result.get("decision", "escalate")
            forced_confidence = float(result.get("confidence", 1.0))
            print(f"  [INTERVENTION] Reviewer decided: {decision} (Confidence: {forced_confidence})")
            
            # If the reviewer still isn't confident, we hard-escalate.
            if forced_confidence < 0.7:
                print("  [INTERVENTION] Reviewer lacks confidence. Hard-escalating.")
                decision = "escalate"
                forced_confidence = 1.0
                
        except Exception as e:
            print(f"  [INTERVENTION] Reviewer failed ({e}). Forcing escalation.")
    else:
        print("  [INTERVENTION] Simulation mode. Forcing escalation to restore autonomy quality.")
        forced_confidence = round(random.uniform(0.85, 0.95), 2)

    latency = random.randint(200, 500)

    return {
        "current_step": "intervention",
        "step_count": 1,
        "path_taken": ["intervention"],
        "confidence": forced_confidence,
        "decision": decision,
        "execution_time_ms": latency
    }
