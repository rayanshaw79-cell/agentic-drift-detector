"""
schemas/clinical_state.py — TypedDict state schema for the Clinical Coding Agent.

Mirrors the IncidentState pattern but carries medical domain fields.
The execution metadata fields (step_count, retry_count, path_taken,
execution_time_ms) are annotated with operator.add so LangGraph
accumulates them correctly across nodes.
"""

import operator
from typing import Optional, List, Annotated
from typing import TypedDict


class ClinicalState(TypedDict, total=False):
    # ── Identity ──────────────────────────────────────────────────────────────
    record_id: str
    raw_note: str                        # Unstructured clinical text input
    
    # ── De-identification (PHI/PII Scrubbing) ─────────────────────────────────
    deid_note: Optional[str]             # Scrubbed version of the clinical text
    phi_detected: Optional[bool]         # True if the compliance checker found leaks
    privacy_leak_risk: Optional[float]   # Severity/Risk of privacy leaks (0.0 to 1.0)

    # ── Extracted Entities (NER step output) ──────────────────────────────────
    extracted_diagnoses: Optional[List[str]]    # e.g. ["hypertension", "T2DM"]

    # ── Bayesian Ensemble votes (from upgraded NER step) ─────────────────────
    # Each dict: {"term": str, "posterior": float, "votes": {gemini, regex, nlm}}
    ner_votes: Optional[List[dict]]

    # ── Coded Outputs (Ontology Lookup + Disambiguation) ──────────────────────
    icd10_codes: Optional[List[dict]]
    # Each dict after MEAT validation:
    #   {term, code, description, confidence, hcc, hcc_category, raf_weight,
    #    meat_met, meat_category, meat_evidence}

    # ── MEAT Validation (raw audit trail from meat_validation_step) ───────────
    # List of raw MEAT decisions from the LLM (one per ICD-10 code).
    # Preserved separately for audit logging and dashboard display.
    meat_results: Optional[List[dict]]

    # ── Final Output ──────────────────────────────────────────────────────────
    clinical_record: Optional[dict]      # Structured Python dict output
    coding_status: Optional[str]         # "complete" | "requires_clinical_review" | "approved_by_clinician"
    overall_confidence: Optional[float]  # Aggregate confidence across all codes
    claims_ready: Optional[bool]         # True if all codes are claims-submittable

    # ── Human-in-the-Loop (HITL) Review Metadata ─────────────────────────────
    human_review_action: Optional[str]   # "approved" | "edited" | "rejected" | "pending"
    reviewed_by: Optional[str]           # Clinician identifier
    reviewed_at: Optional[str]           # ISO timestamp string
    human_notes: Optional[str]            # Clinician review notes / rationale
    original_ai_codes: Optional[List[dict]] # Pre-human-override AI extraction snapshot

    # ── SDOH Population Health (Added during Clinical Unification) ────────────
    sdoh_risk_label: Optional[str]       # "low", "moderate", "high", "critical"
    sdoh_risk_score: Optional[float]     # Underlying float probability from ML
    sdoh_shap_factors: Optional[List[dict]] # Top drivers from SHAP explanation

    # ── Preventive Oncology (Project ASHA-AI MVP) ─────────────────────────────
    patient_id: Optional[str]
    lifestyle_factors: Optional[List[dict]] # Extracted from ASHA worker notes (e.g., tobacco, arsenic)
    lifestyle_risk_score: Optional[float]   # Calculated drift risk
    preventive_recommendations: Optional[str] # LLM generated output based on guidelines

    # ── Oncology Specific Fields (Harmony & PRISM) ─────────────────────────────
    # Extracted from pathology / clinical notes
    document_type: Optional[str]         # From Constellation Router: pathology_report | radiology | genomics | progress_note | unknown
    primary_site: Optional[str]          # e.g., "Lung", "Breast"
    histology: Optional[str]             # e.g., "Adenocarcinoma"
    tnm_stage: Optional[dict]            # {"T": "T2", "N": "N1", "M": "M0", "overall": "Stage II"}
    biomarkers: Optional[List[dict]]     # e.g., [{"marker": "EGFR", "status": "Positive", "evidence_span": "..."}]
    
    # ── Longitudinal Reasoning (Symphony) ──────────────────────────────────────
    visit_history: Optional[List[dict]]  # History of past visits to build timeline
    pre_chart_summary: Optional[str]     # Chronological summary of disease progression
    
    # ── Clinical Trial Matching (PRISM) ───────────────────────────────────────
    trial_matches: Optional[List[dict]]  # [{"nct_id": "NCT123", "match_confidence": 0.9, "evidence": "..."}]

    # ── Pharmacovigilance & Drug Safety (Phase 2 Expansion) ───────────────────
    extracted_medications: Optional[List[dict]]  # [{"drug_name": "Warfarin", "rxcui": "11289"}]
    drug_interactions: Optional[List[dict]]      # [{"pair": "Warfarin + Aspirin", "severity": "high"}]
    adverse_drug_reactions: Optional[List[dict]] # [{"category": "Hemorrhagic", "symptom": "Epistaxis"}]
    drug_safety_risk: Optional[str]              # "low" | "moderate" | "high" | "critical"

    # ── CMS Financial RAF & RADV Audit (Phase 3 Expansion) ────────────────────
    total_raf_score: Optional[float]             # Aggregated demographic + disease RAF
    verified_raf_score: Optional[float]          # MEAT-verified compliant RAF score
    unverified_raf_score: Optional[float]        # Unverified RAF score lacking evidence
    radv_financial_exposure_usd: Optional[float] # Estimated Medicare audit clawback ($ USD)
    radv_audit_label: Optional[str]              # "low_audit_risk" | "moderate_audit_risk" | "high_radv_exposure"

    # ── Longitudinal Reasoning & RECIST 1.1 (Phase 4 Expansion) ────────────────
    longitudinal_timeline: Optional[List[dict]]   # Chronological visit summaries
    recist_overall_response: Optional[str]        # "CR" | "PR" | "SD" | "PD" | "UNEVALUABLE"
    lesion_measurements: Optional[List[dict]]     # Serial target lesion measurements (mm)

    # ── SMART-on-FHIR R4 Adapter (Phase 5 Expansion) ─────────────────────────
    fhir_bundle: Optional[dict]                  # HL7 FHIR R4 JSON Bundle object

    # ── Self-Correction Evaluator Loop (OncoLLM Pillar 4) ────────────────────
    eval_feedback: Optional[str]         # Evaluator's critique — injected into re-extraction prompt
    needs_reextraction: Optional[bool]   # True triggers the correction loop back to staging

    # ── Execution Metadata (accumulated via LangGraph Annotated reducers) ──────
    current_step: str
    step_count:        Annotated[int, operator.add]
    retry_count:       Annotated[int, operator.add]
    path_taken:        Annotated[List[str], operator.add]
    execution_time_ms: Annotated[int, operator.add]

