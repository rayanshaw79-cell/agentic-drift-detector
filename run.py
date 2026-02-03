import uuid
from workflows.incident_triage import incident_triage_workflow
from schemas.incident_state import IncidentState
from drift.drift_detector import analyze_workflow


def main():
    initial_state: IncidentState = {
        "incident_id": str(uuid.uuid4())[:8],
        "incident_text": "Auth service latency spike",

        "severity": None,
        "investigation_summary": None,
        "decision": None,
        "confidence": None,

        "current_step": "",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],

        "execution_time_ms": 0
    }

    # 1️⃣ Run workflow
    final_state = incident_triage_workflow(initial_state)

    print("\n✅ FINAL WORKFLOW STATE")
    print(final_state)

    # 2️⃣ Analyze drift (ONLY place this happens)
    analysis = analyze_workflow(final_state)

    print("\n🚨 DRIFT ANALYSIS")
    print(analysis)


if __name__ == "__main__":
    main()
