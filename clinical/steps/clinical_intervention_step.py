"""
clinical/steps/clinical_intervention_step.py — Clinical Agentic Healing node.

Triggered when the conditional edge detects a persistent low-confidence
loop (retry_count >= 2) or when validation cannot produce any valid codes.

Actions:
  1. Strips any hallucinated / low-confidence codes.
  2. Sets coding_status to "requires_clinical_review" (safe failure mode).
  3. Fires a Slack/Discord alert via the existing alert_manager.
  4. Returns a clean state so the output node can emit a safe record.
"""

import logging

from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)


def clinical_intervention_step(state: ClinicalState) -> dict:
    """
    LangGraph node — Clinical Agentic Healing.

    This node NEVER writes to the medical records database.
    It always escalates to human review.
    """
    retry_count = state.get("retry_count", 0)
    confidence  = state.get("overall_confidence", 0.0)
    record_id   = state.get("record_id", "unknown")

    conf_display = confidence if confidence is not None else 0.0
    log.warning(
        "[CLINICAL INTERVENTION] Drift loop on record '%s': "
        "retry_count=%d, confidence=%.2f — routing to clinical review queue.",
        record_id, retry_count, conf_display,
    )
    print(f"\n  [CLINICAL INTERVENTION] Record '{record_id}' sent to human review.")
    print(f"     Reason: {retry_count} retries with confidence {conf_display:.2f}")

    original_draft_codes = state.get("icd10_codes") or []

    # Purge any low-confidence / hallucinated codes for default output
    safe_codes = [
        c for c in original_draft_codes
        if c.get("code", "UNRESOLVED") != "UNRESOLVED"
        and c.get("confidence", 0.0) >= 0.6
    ]

    # Attempt to fire Slack/Discord alert (best-effort — don't crash on failure)
    try:
        from alerts.alert_manager import trigger_alert
        mock_analysis = {
            "risk_level":  "high_risk",
            "drift_score": 75,
        }
        mock_state = {
            "incident_id": record_id,
            "decision":    "requires_clinical_review",
            "retry_count": retry_count,
            "severity":    "clinical_coding",
            "path_taken":  state.get("path_taken", []) + ["clinical_intervention"],
        }
        trigger_alert(mock_analysis, mock_state)
    except Exception as exc:
        log.debug("[CLINICAL INTERVENTION] Alert skipped: %s", exc)

    latency = 50

    return {
        "current_step":       "clinical_intervention",
        "step_count":         1,
        "path_taken":         ["clinical_intervention"],
        "icd10_codes":        safe_codes,
        "original_ai_codes":  original_draft_codes,
        "coding_status":      "requires_clinical_review",
        "human_review_action": "pending",
        "overall_confidence": 0.0,
        "execution_time_ms":  latency,
    }
