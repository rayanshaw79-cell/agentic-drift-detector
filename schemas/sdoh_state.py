"""
schemas/sdoh_state.py — TypedDict state schema for the SDOH Longitudinal Risk Agent.

Carries patient visit history, extracted social determinants, computed risk
trajectory scores, and intervention flags through the LangGraph pipeline.
"""

import operator
from typing import TypedDict, Optional, List, Annotated


class SdohState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    patient_id: str

    # ── Input Data ────────────────────────────────────────────────────────────
    # Raw visit records fetched from patient_store (list of dicts)
    visit_history: Optional[List[dict]]

    # ── SDOH Extraction Output ────────────────────────────────────────────────
    # Structured social determinants profile for the latest visit
    sdoh_profile: Optional[dict]

    # ── Risk Trajectory Output ────────────────────────────────────────────────
    # Probability score (0.0–1.0) per historical visit
    risk_trajectory: Optional[List[float]]

    # Change in risk from the previous visit to the current one
    risk_delta: Optional[float]

    # Predicted label for the current visit: "low" | "moderate" | "high" | "critical"
    predicted_risk_label: Optional[str]

    # Top SHAP feature contributions for the current visit
    shap_factors: Optional[List[dict]]   # [{"feature": str, "contribution": float}]

    # ── Intervention ──────────────────────────────────────────────────────────
    # True if risk is accelerating and preventive action is recommended
    intervention_flag: Optional[bool]
    intervention_reason: Optional[str]

    # ── Final Output ──────────────────────────────────────────────────────────
    sdoh_report: Optional[dict]

    # ── Execution Metadata (accumulated via LangGraph Annotated reducers) ──────
    current_step: str
    path_taken:        Annotated[List[str], operator.add]
    execution_time_ms: Annotated[int, operator.add]
