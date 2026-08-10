import uuid
import time
import os
import logging
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

log = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(key: str = Depends(api_key_header)):
    expected_key = os.getenv("API_SECRET_KEY")
    if expected_key and key != expected_key:
        raise HTTPException(status_code=403, detail="Forbidden")

from workflows.clinical_coding import clinical_coding_workflow, apply_human_approval
from telemetry.queue import enqueue
from telemetry.store import (
    get_pending_reviews,
    get_review_history,
    save_human_intervention,
    update_execution_human_status,
)
from clinical.config.cache import init_semantic_cache

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_semantic_cache()
    yield

app = FastAPI(
    title="Clinical Coding & SDOH API",
    description="Agentic workflow for clinical extraction, MEAT validation, and SDOH risk prediction.",
    version="1.0.0",
    dependencies=[Depends(verify_api_key)],
    lifespan=lifespan
)

class ClinicalRequest(BaseModel):
    raw_note: str
    record_id: Optional[str] = None
    tenant_id: Optional[str] = "default"

class ClinicalResponse(BaseModel):
    record_id: str
    coding_status: str
    overall_confidence: float
    icd10_codes: List[Dict[str, Any]]
    sdoh_risk_label: Optional[str]
    sdoh_risk_score: Optional[float]
    sdoh_shap_factors: Optional[List[Dict[str, Any]]]
    trial_matches: Optional[List[Dict[str, Any]]] = None
    extracted_medications: Optional[List[Dict[str, Any]]] = None
    drug_interactions: Optional[List[Dict[str, Any]]] = None
    adverse_drug_reactions: Optional[List[Dict[str, Any]]] = None
    drug_safety_risk: Optional[str] = None
    total_raf_score: Optional[float] = None
    verified_raf_score: Optional[float] = None
    unverified_raf_score: Optional[float] = None
    radv_financial_exposure_usd: Optional[float] = None
    radv_audit_label: Optional[str] = None
    longitudinal_timeline: Optional[List[Dict[str, Any]]] = None
    recist_overall_response: Optional[str] = None
    lesion_measurements: Optional[List[Dict[str, Any]]] = None
    fhir_bundle: Optional[Dict[str, Any]] = None
    path_taken: List[str]
    execution_time_ms: int

class PharmaCheckRequest(BaseModel):
    medications: List[str]
    raw_note: Optional[str] = ""

class PreventiveRequest(BaseModel):
    raw_note: str
    patient_id: Optional[str] = None
    tenant_id: Optional[str] = "default"

class RafAuditRequest(BaseModel):
    icd10_codes: List[Dict[str, Any]]
    demographics: Optional[Dict[str, Any]] = None

class SymphonyTimelineRequest(BaseModel):
    visit_history: List[Dict[str, Any]]

class ApprovalRequest(BaseModel):
    record_id: str
    action: str  # "approved" | "edited" | "rejected"
    reviewed_by: Optional[str] = "clinician"
    notes: Optional[str] = ""
    final_codes: Optional[List[Dict[str, Any]]] = None

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/v1/clinical/extract", response_model=ClinicalResponse)
def extract_clinical_data(req: ClinicalRequest):
    record_id = req.record_id or f"api-{uuid.uuid4().hex[:8]}"
    
    initial_state = {
        "record_id": record_id,
        "raw_note": req.raw_note,
        "extracted_diagnoses": None,
        "icd10_codes": None,
        "clinical_record": None,
        "coding_status": None,
        "overall_confidence": None,
        "current_step": "init",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
        "sdoh_risk_label": None,
        "sdoh_risk_score": None,
        "sdoh_shap_factors": None
    }
    
    start_time = time.perf_counter()
    try:
        final_state = clinical_coding_workflow(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
        
    execution_time = int((time.perf_counter() - start_time) * 1000)
    final_state["execution_time_ms"] = execution_time
    
    # Analysis summary for telemetry
    analysis = {
        "workflow_type": "clinical_coding",
        "risk_level": "healthy" if final_state.get("coding_status") == "complete" else "drift",
    }
    
    # Save to telemetry store asynchronously or synchronously
    try:
        enqueue(final_state, analysis=analysis, tenant_id=req.tenant_id)
    except Exception as e:
        log.error("Telemetry enqueue failed: %s", e)
        
    return ClinicalResponse(
        record_id=final_state["record_id"],
        coding_status=final_state.get("coding_status", "unknown"),
        overall_confidence=final_state.get("overall_confidence") or 0.0,
        icd10_codes=final_state.get("icd10_codes", []),
        sdoh_risk_label=final_state.get("sdoh_risk_label"),
        sdoh_risk_score=final_state.get("sdoh_risk_score"),
        sdoh_shap_factors=final_state.get("sdoh_shap_factors"),
        trial_matches=final_state.get("trial_matches"),
        extracted_medications=final_state.get("extracted_medications"),
        drug_interactions=final_state.get("drug_interactions"),
        adverse_drug_reactions=final_state.get("adverse_drug_reactions"),
        drug_safety_risk=final_state.get("drug_safety_risk"),
        total_raf_score=final_state.get("total_raf_score"),
        verified_raf_score=final_state.get("verified_raf_score"),
        unverified_raf_score=final_state.get("unverified_raf_score"),
        radv_financial_exposure_usd=final_state.get("radv_financial_exposure_usd"),
        radv_audit_label=final_state.get("radv_audit_label"),
        longitudinal_timeline=final_state.get("longitudinal_timeline"),
        recist_overall_response=final_state.get("recist_overall_response"),
        lesion_measurements=final_state.get("lesion_measurements"),
        fhir_bundle=final_state.get("fhir_bundle"),
        path_taken=final_state.get("path_taken", []),
        execution_time_ms=execution_time
    )

# ── Preventive Oncology (Project ASHA-AI) Endpoints ─────────────────────────

@app.post("/v1/preventive/risk-assess")
def assess_preventive_risk(req: PreventiveRequest):
    from workflows.preventive_screening import preventive_screening_workflow
    
    patient_id = req.patient_id or f"pat-{uuid.uuid4().hex[:8]}"
    
    initial_state = {
        "record_id": patient_id, # using record_id for consistency in state
        "patient_id": patient_id,
        "raw_note": req.raw_note,
        "current_step": "init",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
        "lifestyle_factors": [],
        "lifestyle_risk_score": 0.0,
        "preventive_recommendations": None
    }
    
    start_time = time.perf_counter()
    try:
        final_state = preventive_screening_workflow(initial_state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
        
    execution_time = int((time.perf_counter() - start_time) * 1000)
    final_state["execution_time_ms"] = execution_time
    
    analysis = {
        "workflow_type": "preventive_screening",
        "risk_level": "high_risk" if final_state.get("lifestyle_risk_score", 0) > 0.5 else "healthy",
        "drift_score": int(final_state.get("lifestyle_risk_score", 0) * 100)
    }
    
    try:
        enqueue(final_state, analysis=analysis, tenant_id=req.tenant_id)
    except Exception as e:
        log.error("Telemetry enqueue failed: %s", e)
        
    return {
        "patient_id": final_state["patient_id"],
        "lifestyle_risk_score": final_state.get("lifestyle_risk_score"),
        "lifestyle_factors": final_state.get("lifestyle_factors"),
        "preventive_recommendations": final_state.get("preventive_recommendations"),
        "execution_time_ms": execution_time
    }

# ── SMART-on-FHIR R4 Adapter Endpoints ─────────────────────────────────────

@app.post("/v1/clinical/fhir/export")
def export_fhir_bundle(state_data: Dict[str, Any]):
    """Export extracted clinical state into HL7 FHIR R4 JSON Bundle."""
    from clinical.tools.fhir_adapter import export_clinical_state_to_fhir
    return export_clinical_state_to_fhir(state_data)

@app.post("/v1/preventive/fhir/export")
def export_preventive_fhir_bundle(state_data: Dict[str, Any]):
    """Export extracted ASHA preventive oncology state into HL7 FHIR R4 JSON Bundle."""
    from clinical.tools.fhir_adapter import export_preventive_state_to_fhir
    return export_preventive_state_to_fhir(state_data)

@app.post("/v1/clinical/fhir/seed")
def seed_synthetic_patient(condition: Optional[str] = "Non-Small Cell Lung Cancer"):
    """Generate synthetic oncology patient chart for evaluation and testing."""
    from clinical.tools.fhir_adapter import generate_synthetic_patient_chart
    return generate_synthetic_patient_chart(condition=condition or "Non-Small Cell Lung Cancer")

# ── SYMPHONY v2 Longitudinal Disease Timeline Endpoint ──────────────────────

@app.post("/v1/clinical/symphony/timeline")
def generate_symphony_timeline(req: SymphonyTimelineRequest):
    """Synthesize multi-visit patient records and calculate RECIST 1.1 treatment response."""
    from clinical.tools.symphony_engine import synthesize_patient_timeline
    return synthesize_patient_timeline(req.visit_history)

# ── CMS Financial RAF & RADV Audit Endpoint ─────────────────────────────────

@app.post("/v1/clinical/raf-audit/calculate")
def calculate_raf_audit(req: RafAuditRequest):
    """Calculate CMS RAF scores and Medicare RADV audit financial exposure ($ USD)."""
    from clinical.tools.raf_audit_calculator import calculate_raf_audit_metrics
    return calculate_raf_audit_metrics(req.icd10_codes, req.demographics)

# ── Pharmacovigilance & Drug Safety Endpoint ────────────────────────────────

@app.post("/v1/clinical/pharmacovigilance/check")
def check_pharmacovigilance(req: PharmaCheckRequest):
    """Check drug-drug interactions and extract adverse drug reaction signals."""
    from clinical.tools.pharmacovigilance_api import check_drug_interactions
    from clinical.steps.pharmacovigilance_step import _extract_adverse_reactions, _compute_safety_risk

    interactions = check_drug_interactions(req.medications) if len(req.medications) >= 2 else []
    adrs = _extract_adverse_reactions(req.raw_note or "", req.medications)
    risk_level = _compute_safety_risk(interactions, adrs)

    return {
        "medications": req.medications,
        "interactions": interactions,
        "adverse_drug_reactions": adrs,
        "drug_safety_risk": risk_level
    }

# ── PRISM v2 Live Clinical Trial Matching Endpoint ──────────────────────────

@app.get("/v1/clinical/trials/search")
def search_clinical_trials(condition: str, location: Optional[str] = None, limit: int = 5):
    """Search live recruiting trials from ClinicalTrials.gov API v2."""
    from clinical.tools.clinical_trials_api import search_recruiting_trials
    trials = search_recruiting_trials(condition=condition, location=location, limit=limit)
    return {"query": condition, "count": len(trials), "trials": trials}

# ── Human-in-the-Loop (HITL) Endpoints ─────────────────────────────────────

@app.get("/v1/clinical/review-queue")
def fetch_review_queue(limit: int = 50):
    """Fetch pending records requiring clinical review."""
    pending = get_pending_reviews(limit=limit)
    return {"pending_count": len(pending), "records": pending}

@app.post("/v1/clinical/approve")
def submit_human_approval(req: ApprovalRequest):
    """Submit a clinician's approval, edit, or rejection decision for a flagged record."""
    if req.action not in ("approved", "edited", "rejected"):
        raise HTTPException(status_code=400, detail="Action must be 'approved', 'edited', or 'rejected'")
    
    new_status = "approved_by_clinician" if req.action in ("approved", "edited") else "rejected_by_clinician"
    
    # Save intervention audit record
    intervention_id = save_human_intervention(
        incident_id=req.record_id,
        action=req.action,
        reviewed_by=req.reviewed_by or "clinician",
        notes=req.notes or "",
        original_codes=[],
        final_codes=req.final_codes or []
    )
    
    # Update execution telemetry record
    update_execution_human_status(
        record_id=req.record_id,
        new_status=new_status,
        human_action=req.action,
        notes=req.notes or "",
        reviewed_by=req.reviewed_by or "clinician",
        final_codes=req.final_codes or []
    )
    
    return {
        "status": "success",
        "record_id": req.record_id,
        "intervention_id": intervention_id,
        "action": req.action,
        "coding_status": new_status
    }

@app.get("/v1/clinical/review-history")
def fetch_review_history(limit: int = 50):
    """Fetch past clinician review interventions for auditing."""
    history = get_review_history(limit=limit)
    return {"history_count": len(history), "records": history}

