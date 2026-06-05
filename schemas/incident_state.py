import operator
from typing import TypedDict, Optional, List, Annotated

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
    step_count: Annotated[int, operator.add]
    retry_count: Annotated[int, operator.add]
    path_taken: Annotated[List[str], operator.add]

    # --- Performance ---
    execution_time_ms: Annotated[int, operator.add]
