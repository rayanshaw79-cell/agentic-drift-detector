import random
import os
import json
from schemas.incident_state import IncidentState


def is_bias_simulated():
    return os.getenv("SIMULATE_BIAS") == "true"


def _llm_decision(incident_text: str, severity: str, investigation_summary: str) -> dict:
    """Use ChatOpenAI to make an escalation decision."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert SRE making a final incident response decision. "
            "Based on the severity and investigation, decide whether to 'escalate' to on-call engineers "
            "or 'auto_resolve' the incident automatically. "
            "Also provide a confidence score between 0.0 and 1.0. "
            "Return ONLY a JSON object with keys 'decision' and 'confidence'. "
            "Example: {\"decision\": \"escalate\", \"confidence\": 0.85}"
        )),
        HumanMessage(content=(
            f"Incident: {incident_text}\n"
            f"Severity: {severity}\n"
            f"Investigation: {investigation_summary}"
        ))
    ])

    result = json.loads(response.content.strip())
    decision = result.get("decision", "escalate").lower()
    confidence = float(result.get("confidence", 0.7))

    if decision not in ["escalate", "auto_resolve"]:
        decision = "escalate"
    confidence = max(0.0, min(1.0, confidence))

    return {"decision": decision, "confidence": confidence}


def decision_step(state: IncidentState) -> dict:
    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    decision_result = None

    if use_llm and not is_bias_simulated():
        try:
            decision_result = _llm_decision(
                state.get("incident_text", "Unknown incident"),
                state.get("severity", "unknown"),
                state.get("investigation_summary", "No investigation data")
            )
            print(f"  [LLM DECISION] Decision: {decision_result['decision']} (confidence: {decision_result['confidence']})")
        except Exception as e:
            print(f"  [LLM DECISION] Falling back to simulation (error: {e})")

    if decision_result is None:
        if is_bias_simulated():
            confidence = round(random.uniform(0.1, 0.45), 2)
        else:
            confidence = round(random.uniform(0.6, 0.95), 2)

        decision = "escalate"
        if state.get("severity") == "low" and confidence >= 0.6:
            decision = "auto_resolve"

        decision_result = {"decision": decision, "confidence": confidence}

    confidence = decision_result["confidence"]
    decision = decision_result["decision"]

    latency = random.randint(500, 1500)

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
