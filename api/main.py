import uuid
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from workflows.clinical_coding import clinical_coding_workflow
from telemetry.store import save_execution_state
from clinical.config.cache import init_semantic_cache

app = FastAPI(
    title="Clinical Coding & SDOH API",
    description="Agentic workflow for clinical extraction, MEAT validation, and SDOH risk prediction.",
    version="1.0.0"
)

@app.on_event("startup")
def on_startup():
    init_semantic_cache()

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
    path_taken: List[str]
    execution_time_ms: int

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
        save_execution_state(final_state, analysis=analysis, tenant_id=req.tenant_id)
    except Exception as e:
        print(f"Telemetry save failed: {e}")
        
    return ClinicalResponse(
        record_id=final_state["record_id"],
        coding_status=final_state.get("coding_status", "unknown"),
        overall_confidence=final_state.get("overall_confidence") or 0.0,
        icd10_codes=final_state.get("icd10_codes", []),
        sdoh_risk_label=final_state.get("sdoh_risk_label"),
        sdoh_risk_score=final_state.get("sdoh_risk_score"),
        sdoh_shap_factors=final_state.get("sdoh_shap_factors"),
        path_taken=final_state.get("path_taken", []),
        execution_time_ms=execution_time
    )
