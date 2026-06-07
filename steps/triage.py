import random
import os
import json
from schemas.incident_state import IncidentState

def _llm_triage(incident_text: str, severity_context: str) -> str:
    """Use ChatOpenAI to classify incident severity."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert SRE (Site Reliability Engineer) performing incident triage. "
            "Your job is to classify the severity of an incident as exactly one of: low, medium, high. "
            "Return ONLY a JSON object with a single key 'severity'. Example: {\"severity\": \"high\"}"
        )),
        HumanMessage(content=f"Incident: {incident_text}\nContext: {severity_context}")
    ])

    result = json.loads(response.content.strip())
    severity = result.get("severity", "medium").lower()
    if severity not in ["low", "medium", "high"]:
        severity = "medium"
    return severity


def triage_step(state: IncidentState) -> dict:
    incident_text = state.get("incident_text", "Unknown incident")
    use_llm = bool(os.getenv("OPENAI_API_KEY"))
    severity = None

    if use_llm and os.getenv("SIMULATE_BIAS") != "true":
        try:
            severity = _llm_triage(incident_text, "Production environment, critical infrastructure")
            print(f"  [LLM TRIAGE] Severity classified as: {severity}")
        except Exception as e:
            print(f"  [LLM TRIAGE] Falling back to simulation (error: {e})")

    if severity is None:
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
