from schemas.incident_state import IncidentState
from steps.triage import triage_step
from steps.investigate import investigation_step
from steps.decision import decision_step
from steps.notification import notification_step


def incident_triage_workflow(initial_state: IncidentState) -> IncidentState:
    state = initial_state

    state = triage_step(state)
    state = investigation_step(state)

    previous_retries = state["retry_count"]
    state = decision_step(state)

    if state["retry_count"] > previous_retries:
        state = decision_step(state)

    state = notification_step(state)
    return state
