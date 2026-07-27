"""
clinical/tools/raf_audit_calculator.py — CMS-HCC Risk Adjustment Factor (RAF) & RADV Audit Engine.

Calculates demographic base RAF scores, MEAT-verified vs unverified disease RAF weights,
and estimates financial clawback exposure ($ USD) under Medicare RADV (Risk Adjustment Data Validation) rules.
"""

import logging
from typing import List, Dict, Any

log = logging.getLogger(__name__)

# Base CMS Demographic Coefficients (Community Non-Dual Aged)
DEMOGRAPHIC_RAF_BASE = {
    "M65_69": 0.305, "M70_74": 0.389, "M75_79": 0.477, "M80_84": 0.582, "M85_PLUS": 0.741,
    "F65_69": 0.271, "F70_74": 0.345, "F75_79": 0.432, "F80_84": 0.541, "F85_PLUS": 0.710,
    "DEFAULT": 0.350
}

# Average Annualized Medicare Advantage Dollar Value per RAF Point (CMS Standard ~$10,200)
ANNUAL_DOLLAR_VALUE_PER_RAF_POINT = 10200.00


def calculate_raf_audit_metrics(
    icd10_codes: List[Dict[str, Any]],
    demographics: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Calculates total patient RAF score, verified vs unverified RAF weights,
    and RADV financial clawback risk ($ USD).

    Args:
        icd10_codes: List of extracted code objects containing:
            {code, description, hcc_category, raf_weight, meat_met, confidence}
        demographics: Optional dict {age, gender, medicaid}

    Returns:
        Structured RAF audit dictionary.
    """
    demographics = demographics or {}
    age = demographics.get("age", 72)
    gender = demographics.get("gender", "M").upper()

    # 1. Base Demographic RAF
    demo_key = f"{gender}{'65_69' if 65 <= age <= 69 else '70_74' if 70 <= age <= 74 else '75_79' if 75 <= age <= 79 else '80_84' if 80 <= age <= 84 else '85_PLUS' if age >= 85 else 'DEFAULT'}"
    base_demo_raf = DEMOGRAPHIC_RAF_BASE.get(demo_key, DEMOGRAPHIC_RAF_BASE["DEFAULT"])

    # 2. Disease RAF Breakdown (Verified vs Unverified)
    verified_disease_raf = 0.0
    unverified_disease_raf = 0.0
    code_audit_details = []

    seen_hcc = set()

    for item in icd10_codes or []:
        hcc = item.get("hcc_category") or item.get("hcc")
        raf_weight = float(item.get("raf_weight") or 0.0)
        meat_met = bool(item.get("meat_met", True))
        code_str = item.get("code", "")

        # CMS HCC hierarchies only count top weight per category
        is_unique_hcc = hcc and hcc not in seen_hcc
        if is_unique_hcc:
            seen_hcc.add(hcc)

        if meat_met:
            if is_unique_hcc:
                verified_disease_raf += raf_weight
            audit_status = "MEAT Verified (Compliant)"
        else:
            if is_unique_hcc:
                unverified_disease_raf += raf_weight
            audit_status = "Missing MEAT Proof (RADV Risk)"

        code_audit_details.append({
            "code": code_str,
            "hcc_category": hcc or "N/A",
            "raf_weight": raf_weight,
            "meat_met": meat_met,
            "audit_status": audit_status
        })

    total_raf_score = round(base_demo_raf + verified_disease_raf + unverified_disease_raf, 3)
    verified_raf_score = round(base_demo_raf + verified_disease_raf, 3)
    unverified_disease_raf = round(unverified_disease_raf, 3)

    # 3. Calculate RADV Financial Exposure ($ USD Clawback Risk)
    radv_exposure_usd = round(unverified_disease_raf * ANNUAL_DOLLAR_VALUE_PER_RAF_POINT, 2)

    # 4. Determine RADV Audit Risk Label
    if unverified_disease_raf > 0.4 or radv_exposure_usd >= 4000.0:
        radv_audit_label = "high_radv_exposure"
    elif unverified_disease_raf > 0.0:
        radv_audit_label = "moderate_audit_risk"
    else:
        radv_audit_label = "low_audit_risk"

    log.info(
        "[RAF AUDIT ENGINE] Total RAF: %.3f | Verified: %.3f | Unverified: %.3f | RADV Exposure: $%.2f",
        total_raf_score, verified_raf_score, unverified_disease_raf, radv_exposure_usd
    )

    return {
        "base_demographic_raf": base_demo_raf,
        "total_raf_score": total_raf_score,
        "verified_raf_score": verified_raf_score,
        "unverified_raf_score": unverified_disease_raf,
        "radv_financial_exposure_usd": radv_exposure_usd,
        "radv_audit_label": radv_audit_label,
        "code_audit_details": code_audit_details
    }
