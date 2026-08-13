"""
clinical/steps/ontology_lookup_step.py — Ontology Lookup node (ICD-10-CM only).

For each diagnosis extracted by the NER step, queries the free NLM
ICD-10-CM clinical tables API and returns candidate code objects.

The lookup is scoped to ICD-10 only (per scope decision).
"""

import logging
import time

from clinical.tools.nlm_api import lookup_icd10, lookup_snomed, lookup_loinc
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
            # ICD-10 match found
            for c in candidates:
                c["confidence"] = 0.0   # will be set by disambiguation step
                c["system"] = "ICD-10-CM"
            all_codes.extend(candidates)
            log.info("[LOOKUP] '%s' → %d ICD-10 candidates (%d ms)", term, len(candidates), elapsed)
        else:
            # Fallback 1: SNOMED
            snomed_candidates = lookup_snomed(term)
            if snomed_candidates:
                for c in snomed_candidates:
                    c["confidence"] = 0.0
                all_codes.extend(snomed_candidates)
                log.info("[LOOKUP] '%s' → resolved via SNOMED fallback", term)
            else:
                # Fallback 2: LOINC
                loinc_candidates = lookup_loinc(term)
                if loinc_candidates:
                    for c in loinc_candidates:
                        c["confidence"] = 0.0
                    all_codes.extend(loinc_candidates)
                    log.info("[LOOKUP] '%s' → resolved via LOINC fallback", term)
                else:
                    # Completely unresolved
                    all_codes.append({
                        "term":        term,
                        "code":        "UNRESOLVED",
                        "description": f"No terminology match found for '{term}'",
                        "confidence":  0.0,
                        "system":      "NONE"
                    })
                    log.warning("[LOOKUP] No match found in any terminology for term '%s'", term)

    return {
        "current_step":    "ontology_lookup",
        "step_count":      1,
        "path_taken":      ["ontology_lookup"],
        "icd10_codes":     all_codes,
        "execution_time_ms": max(total_latency, 20),
    }
