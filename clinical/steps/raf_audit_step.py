"""
clinical/steps/raf_audit_step.py — LangGraph CMS RAF & RADV Audit Node.

Audits extracted medical codes against Medicare Risk Adjustment Data Validation (RADV)
rules, computing verified vs unverified RAF weights and financial clawback exposure ($ USD).
"""

import logging
from typing import Dict, Any

from schemas.clinical_state import ClinicalState
from clinical.tools.raf_audit_calculator import calculate_raf_audit_metrics

log = logging.getLogger(__name__)


def raf_audit_step(state: ClinicalState) -> dict:
    """
    LangGraph State Node — CMS Financial RAF & RADV Audit Engine.

    Reads:
      - icd10_codes, meat_results
    Emits:
      - total_raf_score: float
      - verified_raf_score: float
      - unverified_raf_score: float
      - radv_financial_exposure_usd: float
      - radv_audit_label: str ("low_audit_risk" | "moderate_audit_risk" | "high_radv_exposure")
    """
    icd10_codes = state.get("icd10_codes") or []

    metrics = calculate_raf_audit_metrics(icd10_codes)

    log.info(
        "[RAF & RADV AUDIT ENGINE] Total RAF: %.3f | Verified RAF: %.3f | RADV Exposure: $%,.2f | Audit Risk: %s",
        metrics['total_raf_score'], metrics['verified_raf_score'], metrics['radv_financial_exposure_usd'], metrics['radv_audit_label'].upper()
    )

    return {
        "current_step": "raf_audit",
        "total_raf_score": metrics["total_raf_score"],
        "verified_raf_score": metrics["verified_raf_score"],
        "unverified_raf_score": metrics["unverified_raf_score"],
        "radv_financial_exposure_usd": metrics["radv_financial_exposure_usd"],
        "radv_audit_label": metrics["radv_audit_label"],
        "path_taken": ["raf_audit"]
    }
