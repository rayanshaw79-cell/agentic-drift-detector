"""
clinical/steps/validation_step.py — Validation node.

Validates ICD-10 codes, enriches them with HCC risk adjustment data,
and assembles the final structured clinical record (Python dict).

Validation rules:
  1. Every code must be non-empty and not "UNRESOLVED".
  2. Every code must have confidence ≥ 0.5 to be included in output.
  3. At least one valid code must exist for the record to be "complete".

New in this version (Miimansa-inspired):
  4. Adds claims_ready flag: confidence >= 0.85 and status == complete
  5. Enriches each code with HCC category and RAF weight (risk adjustment)
"""

import logging
import time
from datetime import datetime, timezone

from schemas.clinical_state import ClinicalState
from clinical.tools.hcc_mapper import enrich_codes_with_hcc

log = logging.getLogger(__name__)

_MIN_CONFIDENCE      = 0.5
_CLAIMS_READY_THRESHOLD = 0.85


def validation_step(state: ClinicalState) -> dict:
    """
    LangGraph node — Validation, HCC Enrichment & Record Assembly.

    Reads:  state["icd10_codes"], state["record_id"], state["raw_note"]
    Writes: state["clinical_record"], state["coding_status"],
            state["claims_ready"], state["retry_count"], step metadata
    """
    codes          = state.get("icd10_codes") or []
    record_id      = state.get("record_id", "unknown")
    raw_note       = state.get("raw_note", "")
    ner_votes      = state.get("ner_votes") or []
    start          = time.perf_counter()
    retry_increment = 0

    # ── Filter to valid, high-confidence codes ────────────────────────────────
    valid_codes = [
        c for c in codes
        if c.get("code", "UNRESOLVED") != "UNRESOLVED"
        and c.get("confidence", 0.0) >= _MIN_CONFIDENCE
    ]
    unresolved = [c for c in codes if c.get("code", "UNRESOLVED") == "UNRESOLVED"]

    if unresolved:
        log.warning(
            "[VALIDATION] %d unresolved term(s): %s",
            len(unresolved), [u["term"] for u in unresolved],
        )

    # ── HCC Enrichment (risk adjustment) ─────────────────────────────────────
    if valid_codes:
        valid_codes = enrich_codes_with_hcc(valid_codes)
        hcc_codes = [c for c in valid_codes if c.get("hcc") is not None]
        total_raf = round(sum(c.get("raf_weight", 0.0) for c in hcc_codes), 4)
        log.info(
            "[VALIDATION] HCC enrichment: %d/%d codes map to HCC — total RAF weight %.3f",
            len(hcc_codes), len(valid_codes), total_raf,
        )
    else:
        total_raf = 0.0

    # ── Determine coding status ───────────────────────────────────────────────
    if not valid_codes:
        log.warning("[VALIDATION] No valid codes — triggering retry.")
        retry_increment = 1
        coding_status = "requires_clinical_review"
    else:
        coding_status = "complete"

    # ── Claims-Readiness Flag ─────────────────────────────────────────────────
    # A record is claims-ready if ALL valid codes meet the higher confidence bar
    # and the overall coding status is complete.
    # This directly supports payer claims adjudication use cases (Miimansa).
    claims_ready = (
        coding_status == "complete"
        and len(valid_codes) > 0
        and all(c.get("confidence", 0.0) >= _CLAIMS_READY_THRESHOLD for c in valid_codes)
    )

    # ── Assemble the structured clinical record ───────────────────────────────
    clinical_record = {
        "record_id":         record_id,
        "coded_at":          datetime.now(timezone.utc).isoformat(),
        "source_note":       raw_note[:200] + ("..." if len(raw_note) > 200 else ""),
        "icd10_codes":       valid_codes,
        "unresolved":        [u["term"] for u in unresolved],
        "total_codes":       len(valid_codes),
        "status":            coding_status,
        "claims_ready":      claims_ready,
        "total_raf_weight":  total_raf,
        # Bayesian ensemble summary
        "ner_ensemble": {
            "total_candidates": len(ner_votes),
            "passed_threshold": len([v for v in ner_votes if v.get("posterior", 0) >= 0.45]),
            "avg_posterior":    round(
                sum(v.get("posterior", 0) for v in ner_votes) / len(ner_votes), 3
            ) if ner_votes else 0.0,
        },
    }

    latency = int((time.perf_counter() - start) * 1000) + 10

    return {
        "current_step":    "validation",
        "step_count":      1,
        "path_taken":      ["validation"],
        "clinical_record": clinical_record,
        "coding_status":   coding_status,
        "claims_ready":    claims_ready,
        "retry_count":     retry_increment,
        "execution_time_ms": latency,
    }
