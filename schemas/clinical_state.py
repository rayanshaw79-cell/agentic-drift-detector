"""
schemas/clinical_state.py — TypedDict state schema for the Clinical Coding Agent.

Mirrors the IncidentState pattern but carries medical domain fields.
The execution metadata fields (step_count, retry_count, path_taken,
execution_time_ms) are annotated with operator.add so LangGraph
accumulates them correctly across nodes.
"""

import operator
from typing import TypedDict, Optional, List, Annotated


class ClinicalState(TypedDict):
    # ── Identity ──────────────────────────────────────────────────────────────
    record_id: str
    raw_note: str                        # Unstructured clinical text input

    # ── Extracted Entities (NER step output) ──────────────────────────────────
    extracted_diagnoses: Optional[List[str]]    # e.g. ["hypertension", "T2DM"]

    # ── Bayesian Ensemble votes (from upgraded NER step) ─────────────────────
    # Each dict: {"term": str, "posterior": float, "votes": {gemini, regex, nlm}}
    ner_votes: Optional[List[dict]]

    # ── Coded Outputs (Ontology Lookup + Disambiguation) ──────────────────────
    icd10_codes: Optional[List[dict]]
    # Each dict: {term, code, description, confidence, hcc, hcc_category, raf_weight}

    # ── Final Output ──────────────────────────────────────────────────────────
    clinical_record: Optional[dict]      # Structured Python dict output
    coding_status: Optional[str]         # "complete" | "requires_clinical_review"
    overall_confidence: Optional[float]  # Aggregate confidence across all codes
    claims_ready: Optional[bool]         # True if all codes are claims-submittable

    # ── Execution Metadata (accumulated via LangGraph Annotated reducers) ──────
    current_step: str
    step_count:        Annotated[int, operator.add]
    retry_count:       Annotated[int, operator.add]
    path_taken:        Annotated[List[str], operator.add]
    execution_time_ms: Annotated[int, operator.add]

