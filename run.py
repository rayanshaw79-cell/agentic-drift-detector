import uuid
import sys
import os
from dotenv import load_dotenv

from workflows.incident_triage import incident_triage_workflow
from schemas.incident_state import IncidentState
from drift.drift_detector import analyze_workflow
from telemetry.store import init_db, save_execution_state

# We keep this mock alert manager
from alerts.alert_manager import trigger_alert

def create_initial_state() -> IncidentState:
    return {
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

def main():
    load_dotenv()
    init_db()

    # Clear SIMULATE_BIAS by default
    os.environ["SIMULATE_BIAS"] = "false"

    simulate_count = 1
    if len(sys.argv) > 1:
        if "--simulate-batch" in sys.argv:
            try:
                idx = sys.argv.index("--simulate-batch")
                simulate_count = int(sys.argv[idx + 1])
                print(f"Simulating {simulate_count} executions to build baseline...")
            except:
                simulate_count = 50
                
        if "--bias" in sys.argv:
            print("[WARNING] RUNNING WITH ARTIFICIAL BIAS ENABLED")
            os.environ["SIMULATE_BIAS"] = "true"

    for i in range(simulate_count):
        initial_state = create_initial_state()
        final_state = incident_triage_workflow(initial_state)
        save_execution_state(final_state)
        
        if simulate_count > 1:
            print(f"Completed simulation {i+1}/{simulate_count}")

    if simulate_count == 1:
        print("\n[FINAL WORKFLOW STATE]")
        print(final_state)

        # Analyze drift against dynamic baseline
        analysis = analyze_workflow(final_state)

        print("\n[DRIFT ANALYSIS]")
        print(analysis)

        trigger_alert(analysis, final_state)

if __name__ == "__main__":
    main()
