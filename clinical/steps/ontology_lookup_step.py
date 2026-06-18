"""
clinical/steps/ontology_lookup_step.py — Ontology Lookup node (ICD-10-CM only).

For each diagnosis extracted by the NER step, queries the free NLM
ICD-10-CM clinical tables API and returns candidate code objects.

The lookup is scoped to ICD-10 only (per scope decision).
"""

import logging
import time

from clinical.tools.nlm_api import lookup_icd10
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)


def ontology_lookup_step(state: ClinicalState) -> dict:
    """
    LangGraph node — ICD-10 Ontology Lookup.

    Reads:  state["extracted_diagnoses"]
    Writes: state["icd10_codes"], step metadata
    """
    diagnoses = state.get("extracted_diagnoses") or []
    all_codes: list[dict] = []
    total_latency = 0

    for term in diagnoses:
        start = time.perf_counter()
        candidates = lookup_icd10(term, max_results=5)
        elapsed = int((time.perf_counter() - start) * 1000)
        total_latency += elapsed

        if candidates:
            # Attach a preliminary confidence placeholder (disambiguation will refine it)
            for c in candidates:
                c["confidence"] = 0.0   # will be set by disambiguation step
            all_codes.extend(candidates)
            log.info("[LOOKUP] '%s' → %d candidates (%d ms)", term, len(candidates), elapsed)
        else:
            # Unresolved: keep a placeholder so the agent knows it failed
            all_codes.append({
                "term":        term,
                "code":        "UNRESOLVED",
                "description": f"No ICD-10 match found for '{term}'",
                "confidence":  0.0,
            })
            log.warning("[LOOKUP] No ICD-10 match for term '%s'", term)

    return {
        "current_step":    "ontology_lookup",
        "step_count":      1,
        "path_taken":      ["ontology_lookup"],
        "icd10_codes":     all_codes,
        "execution_time_ms": max(total_latency, 20),
    }
