"""
clinical/steps/pharmacovigilance_step.py — LangGraph Pharmacovigilance & ADR Safety Scanner node.

Extracts prescribed medications and unstructured Adverse Drug Reaction (ADR) signals
from clinical notes, checks NLM RxNav interactions, and assigns safety risk levels.
"""

import logging
import re
from typing import List, Dict, Any

from schemas.clinical_state import ClinicalState
from clinical.tools.pharmacovigilance_api import check_drug_interactions, get_rxcui_by_name

log = logging.getLogger(__name__)

# Common medication patterns for extraction
KNOWN_DRUG_PATTERNS = [
    r"\b(warfarin|coumadin)\b", r"\b(aspirin|bayer)\b", r"\b(ibuprofen|advil|motrin)\b",
    r"\b(keytruda|pembrolizumab)\b", r"\b(opdivo|nivolumab)\b", r"\b(prednisone|dexamethasone)\b",
    r"\b(metformin|glucophage)\b", r"\b(lisinopril|zestril)\b", r"\b(atorvastatin|lipitor)\b",
    r"\b(heparin|apixaban|eliquis|rivaroxaban|xarelto)\b"
]

# Common Adverse Drug Reaction (ADR) signal keywords
ADR_KEYWORD_PATTERNS = [
    (r"\b(maculopapular rash|severe rash|skin eruption|erythema)\b", "Cutaneous Reaction"),
    (r"\b(epistaxis|gastrointestinal bleeding|hematuria|bruising|hemorrhage)\b", "Hemorrhagic Reaction"),
    (r"\b(hepatotoxicity|elevated alt|elevated ast|jaundice)\b", "Hepatotoxicity"),
    (r"\b(nephrotoxicity|elevated creatinine|acute kidney injury)\b", "Nephrotoxicity"),
    (r"\b(pneumonitis|shortness of breath|dyspnea)\b", "Pulmonary Toxicity"),
    (r"\b(anaphylaxis|angioedema|hives)\b", "Severe Hypersensitivity")
]


def pharmacovigilance_step(state: ClinicalState) -> dict:
    """
    LangGraph State Node — Pharmacovigilance & ADR Scanner.

    Reads:
      - raw_note
    Emits:
      - extracted_medications: List[dict]
      - drug_interactions: List[dict]
      - adverse_drug_reactions: List[dict]
      - drug_safety_risk: str ("low" | "moderate" | "high" | "critical")
    """
    raw_note = state.get("raw_note", "")

    # 1. Extract Prescribed Medications
    extracted_meds = _extract_medications(raw_note)
    med_names = [m["drug_name"] for m in extracted_meds]

    log.info("[PHARMACOVIGILANCE] Extracted %d prescribed medications: %s", len(med_names), med_names)

    # 2. Check Drug-Drug & Drug-Condition Interactions via NLM RxNav
    interactions = check_drug_interactions(med_names) if len(med_names) >= 2 else []

    # 3. Extract Unstructured Adverse Drug Reaction (ADR) signals from note
    adverse_reactions = _extract_adverse_reactions(raw_note, med_names)

    # 4. Compute overall Drug Safety Risk
    risk_level = _compute_safety_risk(interactions, adverse_reactions)

    print(f"\n  [PHARMACOVIGILANCE SCANNER] Meds: {len(med_names)} | Interactions: {len(interactions)} | ADR Signals: {len(adverse_reactions)} | Risk: {risk_level.upper()}")

    return {
        "current_step": "pharmacovigilance",
        "extracted_medications": extracted_meds,
        "drug_interactions": interactions,
        "adverse_drug_reactions": adverse_reactions,
        "drug_safety_risk": risk_level,
        "path_taken": ["pharmacovigilance"]
    }


def _extract_medications(text: str) -> List[Dict[str, Any]]:
    """Regex & NLP heuristic medication extractor."""
    meds = []
    seen = set()

    for pattern in KNOWN_DRUG_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for m in matches:
            drug_str = m.group(0).title()
            if drug_str.lower() not in seen:
                seen.add(drug_str.lower())
                rxcui = get_rxcui_by_name(drug_str)
                meds.append({
                    "drug_name": drug_str,
                    "rxcui": rxcui,
                    "evidence_span": m.group(0)
                })

    return meds


def _extract_adverse_reactions(text: str, med_names: List[str]) -> List[Dict[str, Any]]:
    """Extracts unstructured ADR symptom signals and maps them to suspect drugs."""
    adrs = []
    text_lower = text.lower()

    for pattern, category in ADR_KEYWORD_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            symptom_text = match.group(0).title()

            # Identify suspect drug nearby in sentence
            suspect_drug = med_names[0] if med_names else "Unspecified Agent"
            for d in med_names:
                if d.lower() in text_lower:
                    suspect_drug = d
                    break

            adrs.append({
                "category": category,
                "symptom": symptom_text,
                "suspected_drug": suspect_drug,
                "evidence_span": match.group(0),
                "severity": "high" if category in ("Hemorrhagic Reaction", "Anaphylaxis", "Hepatotoxicity") else "moderate"
            })

    return adrs


def _compute_safety_risk(interactions: List[dict], adrs: List[dict]) -> str:
    """Computes categorical safety risk level."""
    high_interactions = [i for i in interactions if i.get("severity") == "high"]
    high_adrs = [a for a in adrs if a.get("severity") == "high"]

    if high_interactions and high_adrs:
        return "critical"
    elif high_interactions or high_adrs:
        return "high"
    elif interactions or adrs:
        return "moderate"
    return "low"
