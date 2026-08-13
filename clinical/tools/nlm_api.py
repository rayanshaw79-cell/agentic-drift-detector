"""
clinical/tools/nlm_api.py — Free NLM (National Library of Medicine) API wrapper.

Provides tool functions for the clinical agent nodes:
  - lookup_icd10(term)  → ICD-10-CM codes via clinicaltables.nlm.nih.gov
  - lookup_rxnorm(drug) → RxNorm CUI codes via rxnav.nlm.nih.gov

Both APIs are completely free, open, and require no API key.
"""

import logging
import time

import requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_ICD10_BASE  = "https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search"
_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST/drugs.json"
_TIMEOUT     = 8   # seconds


# ── ICD-10 Lookup ─────────────────────────────────────────────────────────────

def lookup_icd10(term: str, max_results: int = 5) -> list[dict]:
    """
    Query the NLM ICD-10-CM clinical tables API.

    Args:
        term:        Clinical diagnosis term (e.g. "essential hypertension")
        max_results: Maximum candidate codes to return

    Returns:
        List of dicts: [{"term": str, "code": str, "description": str}]
        Returns [] on network errors or no matches.
    """
    if not term or not term.strip():
        return []

    params = {
        "sf":      "code,name",
        "df":      "code,name",
        "terms":   term.strip(),
        "maxList": max_results,
    }

    start = time.perf_counter()
    try:
        response = requests.get(_ICD10_BASE, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        elapsed = int((time.perf_counter() - start) * 1000)
        log.debug("ICD-10 lookup '%s' → %d ms", term, elapsed)
    except requests.RequestException as exc:
        log.warning("ICD-10 API error for '%s': %s", term, exc)
        return []

    # Response format: [total, codes_list, null, [[code, name], ...]]
    if not data or len(data) < 4 or not data[3]:
        log.debug("ICD-10: no results for '%s'", term)
        return []

    results = []
    for item in data[3]:
        if len(item) >= 2:
            results.append({
                "term":        term,
                "code":        item[0],
                "description": item[1],
            })
    return results[:max_results]


# ── RxNorm Lookup (kept for future use) ───────────────────────────────────────

def lookup_rxnorm(drug_name: str, max_results: int = 3) -> list[dict]:
    """
    Query the NLM RxNorm drug concepts API.

    Args:
        drug_name:   Drug/medication name (e.g. "metformin 500 mg")
        max_results: Maximum candidate RxCUIs to return

    Returns:
        List of dicts: [{"drug": str, "rxcui": str, "name": str}]
        Returns [] on network errors or no matches.
    """
    if not drug_name or not drug_name.strip():
        return []

    params = {"name": drug_name.strip()}

    start = time.perf_counter()
    try:
        response = requests.get(_RXNORM_BASE, params=params, timeout=_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        elapsed = int((time.perf_counter() - start) * 1000)
        log.debug("RxNorm lookup '%s' → %d ms", drug_name, elapsed)
    except requests.RequestException as exc:
        log.warning("RxNorm API error for '%s': %s", drug_name, exc)
        return []

    drug_groups = data.get("drugGroup", {}).get("conceptGroup", [])
    results = []
    for group in drug_groups:
        for concept in group.get("conceptProperties", []):
            results.append({
                "drug":  drug_name,
                "rxcui": concept.get("rxcui", ""),
                "name":  concept.get("name", ""),
            })
            if len(results) >= max_results:
                return results
    return results


# ── Offline Fallback Mappings (SNOMED & LOINC) ──────────────────────────────

# In an enterprise setting, SNOMED and LOINC lookups require an authenticated UMLS API key.
# For high-frequency, low-latency BPO mapping, local fallback dictionaries are commonly
# used to catch known "data mismatches" (e.g. vitals parsed as conditions).

_FALLBACK_SNOMED = {
    "malignant neoplastic disease": {"code": "363346000", "description": "Malignant neoplastic disease (disorder)"},
    "cancer": {"code": "363346000", "description": "Malignant neoplastic disease (disorder)"},
    "chronic pain": {"code": "82423001", "description": "Chronic pain (finding)"},
    "fever": {"code": "386661006", "description": "Fever (finding)"},
    "headache": {"code": "25064002", "description": "Headache (finding)"},
    "fatigue": {"code": "84229001", "description": "Fatigue (finding)"},
    "nausea": {"code": "422587007", "description": "Nausea (finding)"},
}

_FALLBACK_LOINC = {
    "systolic blood pressure": {"code": "8480-6", "description": "Systolic blood pressure"},
    "blood pressure": {"code": "85354-9", "description": "Blood pressure panel with all children optional"},
    "diastolic blood pressure": {"code": "8462-4", "description": "Diastolic blood pressure"},
    "fasting glucose": {"code": "14771-0", "description": "Fasting glucose [Mass/volume] in Serum or Plasma"},
    "heart rate": {"code": "8867-4", "description": "Heart rate"},
    "weight": {"code": "29463-7", "description": "Body weight"},
    "temperature": {"code": "8310-5", "description": "Body temperature"},
}

def lookup_snomed(term: str, max_results: int = 1) -> list[dict]:
    """
    Offline fallback lookup for SNOMED-CT terms.
    """
    if not term:
        return []
    
    clean_term = term.strip().lower()
    
    # Exact or substring match in our fallback dictionary
    for key, data in _FALLBACK_SNOMED.items():
        if key in clean_term or clean_term in key:
            return [{
                "term": term,
                "code": data["code"],
                "description": data["description"],
                "system": "SNOMED"
            }]
    return []

def lookup_loinc(term: str, max_results: int = 1) -> list[dict]:
    """
    Offline fallback lookup for LOINC terms (labs and vitals).
    """
    if not term:
        return []
    
    clean_term = term.strip().lower()
    
    for key, data in _FALLBACK_LOINC.items():
        if key in clean_term or clean_term in key:
            return [{
                "term": term,
                "code": data["code"],
                "description": data["description"],
                "system": "LOINC"
            }]
    return []

