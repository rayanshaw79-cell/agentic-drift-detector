from typing import TypedDict, Optional, List


class IncidentState(TypedDict):
    # --- Identity ---
    incident_id: str
    incident_text: str

    # --- Agent outputs ---
    severity: Optional[str]
    investigation_summary: Optional[str]
    decision: Optional[str]
    confidence: Optional[float]

    # --- Execution metadata ---
    current_step: str
    step_count: int
    retry_count: int
    path_taken: List[str]

    # --- Performance ---
    execution_time_ms: int
