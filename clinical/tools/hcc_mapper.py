"""
clinical/tools/hcc_mapper.py — ICD-10 → HCC (Hierarchical Condition Category) mapper.

Maps ICD-10-CM codes to CMS HCC categories used for risk adjustment (RAF scores).
This directly supports the "risk adjustment" use case from Miimansa's payer pipeline.

HCC categories are used by:
  - CMS Medicare Advantage (Part C) risk adjustment
  - Payer risk adjustment models
  - Population health stratification

Coverage: The most clinically significant ICD-10 → HCC mappings.
Full mapping: https://www.cms.gov/medicare/payment/medicare-advantage-rates-statistics/risk-adjustment
"""

import logging

log = logging.getLogger(__name__)

# ── ICD-10 → HCC Mapping Table ────────────────────────────────────────────────
# Format: "ICD10_PREFIX": {"hcc": int, "category": str, "raf_weight": float}
# RAF weights are approximate relative risk scores from CMS V28 model.
# A prefix match is used (e.g. "E11" matches "E11", "E110", "E11.9", etc.)

_HCC_MAP: dict[str, dict] = {
    # ── Diabetes ──────────────────────────────────────────────────────────────
    "E10":  {"hcc": 37,  "category": "Diabetes with Chronic Complications",    "raf_weight": 0.302},
    "E11":  {"hcc": 38,  "category": "Diabetes without Complication",          "raf_weight": 0.105},
    "E13":  {"hcc": 37,  "category": "Diabetes with Chronic Complications",    "raf_weight": 0.302},
    "P70":  {"hcc": 38,  "category": "Neonatal/Perinatal Diabetes",            "raf_weight": 0.105},

    # ── Cardiovascular ────────────────────────────────────────────────────────
    "I10":  {"hcc": None, "category": "Essential Hypertension",                "raf_weight": 0.0},
    "I11":  {"hcc": 136, "category": "Hypertensive Heart Disease",             "raf_weight": 0.272},
    "I12":  {"hcc": 138, "category": "Hypertensive Chronic Kidney Disease",    "raf_weight": 0.289},
    "I13":  {"hcc": 136, "category": "Hypertensive Heart & CKD",               "raf_weight": 0.272},
    "I15":  {"hcc": 136, "category": "Secondary Hypertension",                 "raf_weight": 0.272},
    "I20":  {"hcc": 89,  "category": "Angina Pectoris",                        "raf_weight": 0.149},
    "I21":  {"hcc": 88,  "category": "Acute Myocardial Infarction",            "raf_weight": 0.199},
    "I22":  {"hcc": 88,  "category": "Subsequent MI",                          "raf_weight": 0.199},
    "I25":  {"hcc": 89,  "category": "Coronary Artery Disease",                "raf_weight": 0.149},
    "I48":  {"hcc": 96,  "category": "Atrial Fibrillation",                    "raf_weight": 0.270},
    "I50":  {"hcc": 85,  "category": "Congestive Heart Failure",               "raf_weight": 0.331},

    # ── Kidney Disease ────────────────────────────────────────────────────────
    "N18":  {"hcc": 138, "category": "Chronic Kidney Disease Stage 3-5",       "raf_weight": 0.289},
    "N19":  {"hcc": 135, "category": "Acute Renal Failure",                    "raf_weight": 0.395},
    "Z99":  {"hcc": 134, "category": "Dependence on Renal Dialysis",           "raf_weight": 0.395},

    # ── Respiratory ───────────────────────────────────────────────────────────
    "J44":  {"hcc": 111, "category": "COPD",                                   "raf_weight": 0.335},
    "J45":  {"hcc": 112, "category": "Asthma",                                 "raf_weight": 0.060},
    "J18":  {"hcc": 115, "category": "Pneumonia",                              "raf_weight": 0.256},

    # ── Neurological ──────────────────────────────────────────────────────────
    "G20":  {"hcc": 78,  "category": "Parkinson's Disease",                    "raf_weight": 0.386},
    "G30":  {"hcc": 52,  "category": "Alzheimer's Disease",                    "raf_weight": 0.346},
    "G35":  {"hcc": 77,  "category": "Multiple Sclerosis",                     "raf_weight": 0.418},
    "I63":  {"hcc": 99,  "category": "Ischemic Stroke",                        "raf_weight": 0.247},
    "I64":  {"hcc": 99,  "category": "Stroke NOS",                             "raf_weight": 0.247},

    # ── Mental Health ─────────────────────────────────────────────────────────
    "F32":  {"hcc": 59,  "category": "Major Depressive Disorder",              "raf_weight": 0.309},
    "F33":  {"hcc": 59,  "category": "Recurrent Depressive Disorder",          "raf_weight": 0.309},
    "F31":  {"hcc": 58,  "category": "Bipolar Disorder",                       "raf_weight": 0.309},
    "F20":  {"hcc": 57,  "category": "Schizophrenia",                          "raf_weight": 0.421},
    "F41":  {"hcc": None, "category": "Anxiety (no HCC)",                      "raf_weight": 0.0},

    # ── Obesity ───────────────────────────────────────────────────────────────
    "E66":  {"hcc": None, "category": "Obesity (no direct HCC assignment)",    "raf_weight": 0.0},

    # ── Infections ───────────────────────────────────────────────────────────
    "A41":  {"hcc": 2,   "category": "Septicemia, Sepsis",                     "raf_weight": 0.514},
    "N39":  {"hcc": None, "category": "UTI (no HCC — generally acute/episodic)","raf_weight": 0.0},

    # ── Thyroid ───────────────────────────────────────────────────────────────
    "E03":  {"hcc": None, "category": "Hypothyroidism (no HCC)",               "raf_weight": 0.0},
    "E05":  {"hcc": None, "category": "Hyperthyroidism (no HCC)",              "raf_weight": 0.0},

    # ── Anaemia ───────────────────────────────────────────────────────────────
    "D50":  {"hcc": None, "category": "Iron Deficiency Anaemia (no HCC)",      "raf_weight": 0.0},
    "D63":  {"hcc": 47,  "category": "Anaemia in Chronic Disease",             "raf_weight": 0.179},
}



def map_icd10_to_hcc(icd10_code: str) -> dict:
    """
    Map a single ICD-10-CM code to its HCC category.

    Uses prefix matching — "E11.9" matches the "E11" entry.

    Args:
        icd10_code: ICD-10-CM code string (e.g. "E11", "E11.9", "I50.9")

    Returns:
        {
          "hcc":      int | None,   # HCC category number (None if not in model)
          "category": str,          # Human-readable category name
          "raf_weight": float,      # Relative risk weight (0.0 if no HCC)
          "mapped":   bool,         # True if a mapping was found
        }
    """
    if not icd10_code or icd10_code == "UNRESOLVED":
        return {"hcc": None, "category": "Unknown", "raf_weight": 0.0, "mapped": False}

    # Normalise: remove dots and uppercase
    code = icd10_code.upper().replace(".", "")

    # Try progressively shorter prefixes (longest match first)
    for length in range(min(len(code), 5), 2, -1):
        prefix = code[:length]
        if prefix in _HCC_MAP:
            mapping = _HCC_MAP[prefix].copy()
            mapping["mapped"] = True
            log.debug("[HCC] %s → HCC %s (%s)", icd10_code,
                      mapping["hcc"], mapping["category"])
            return mapping

    log.debug("[HCC] %s → no HCC mapping found", icd10_code)
    return {"hcc": None, "category": "Not in HCC model", "raf_weight": 0.0, "mapped": False}


def enrich_codes_with_hcc(codes: list[dict]) -> list[dict]:
    """
    Enrich a list of ICD-10 code dicts with HCC mapping fields.

    Args:
        codes: List of dicts with at minimum a "code" key

    Returns:
        Same list with added keys: hcc, hcc_category, raf_weight
    """
    enriched = []
    for c in codes:
        hcc_info = map_icd10_to_hcc(c.get("code", ""))
        enriched.append({
            **c,
            "hcc":         hcc_info["hcc"],
            "hcc_category": hcc_info["category"],
            "raf_weight":  hcc_info["raf_weight"],
        })
    return enriched
