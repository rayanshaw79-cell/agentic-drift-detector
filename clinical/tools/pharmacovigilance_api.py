"""
clinical/tools/pharmacovigilance_api.py — NLM RxNav Drug Interaction & Safety API Client.

Queries the official NLM RxNav REST API (https://rxnav.nlm.nih.gov/REST/)
to convert drug names to RxNorm Concept Unique Identifiers (RxCUIs)
and fetch high-severity drug-drug interactions.
Includes graceful offline fallback generation.
"""

import logging
import urllib.parse
import urllib.request
import json
from typing import List, Dict, Any

log = logging.getLogger(__name__)

RXNAV_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


def get_rxcui_by_name(drug_name: str) -> str:
    """Fetch RxNorm Concept Unique Identifier (RxCUI) for a drug name."""
    url = f"{RXNAV_BASE_URL}/rxcui.json?name={urllib.parse.quote(drug_name)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticDriftDetector/2.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                id_group = data.get("idGroup", {})
                rxnorm_ids = id_group.get("rxnormId", [])
                if rxnorm_ids:
                    return rxnorm_ids[0]
    except Exception as exc:
        log.debug("[Pharmacovigilance API] RxCUI lookup failed for '%s': %s", drug_name, exc)
    return "UNKNOWN_RXCUI"


def check_drug_interactions(drug_list: List[str]) -> List[Dict[str, Any]]:
    """
    Checks high-severity drug-drug interactions for a list of prescribed medications.

    Args:
        drug_list: List of medication names (e.g. ["Warfarin", "Aspirin", "Keytruda"])

    Returns:
        List of interaction alert objects: {pair, severity, description, source}
    """
    if not drug_list or len(drug_list) < 2:
        return []

    # Map names to RxCUIs
    rxcuis = []
    rxcui_map = {}
    for d in drug_list:
        cui = get_rxcui_by_name(d)
        if cui != "UNKNOWN_RXCUI":
            rxcuis.append(cui)
            rxcui_map[cui] = d

    if len(rxcuis) >= 2:
        cui_str = "+".join(rxcuis)
        url = f"{RXNAV_BASE_URL}/interaction/list.json?rxcuis={cui_str}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AgenticDriftDetector/2.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return _parse_rxnav_interactions(data, rxcui_map)
        except Exception as exc:
            log.warning("[Pharmacovigilance API] Live interaction query failed (%s). Using fallback scanner.", exc)

    # Fallback offline interaction detection rules
    return _fallback_interaction_checker(drug_list)


def _parse_rxnav_interactions(data: dict, rxcui_map: dict) -> List[Dict[str, Any]]:
    """Parse raw NLM RxNav interaction list JSON."""
    alerts = []
    full_type_group = data.get("fullInteractionTypeGroup", [])
    for group in full_type_group:
        full_types = group.get("fullInteractionType", [])
        for ftype in full_types:
            min_concepts = ftype.get("minConcept", [])
            drug_pair_names = [c.get("name", "Unknown") for c in min_concepts]
            pair_str = " + ".join(drug_pair_names)
            
            inter_pairs = ftype.get("interactionPair", [])
            for pair in inter_pairs:
                desc = pair.get("description", "Potential interaction detected.")
                severity = pair.get("severity", "N/A")
                alerts.append({
                    "pair": pair_str,
                    "severity": "high" if "high" in severity.lower() or "severe" in desc.lower() else "moderate",
                    "description": desc,
                    "source": "NLM RxNav Live API"
                })
    return alerts


def _fallback_interaction_checker(drug_list: List[str]) -> List[Dict[str, Any]]:
    """Offline rule-based fallback scanner for known severe drug interactions."""
    drugs_lower = [d.lower() for d in drug_list]
    alerts = []

    # Rule 1: Anticoagulants + NSAIDs/Aspirin
    anticoagulants = ["warfarin", "heparin", "apixaban", "rivaroxaban", "dabigatran"]
    nsaids = ["aspirin", "ibuprofen", "naproxen", "ketorolac"]

    has_anticoag = any(a in drugs_lower for a in anticoagulants)
    has_nsaid = any(n in drugs_lower for n in nsaids)

    if has_anticoag and has_nsaid:
        found_anticoag = [a.title() for a in anticoagulants if a in drugs_lower][0]
        found_nsaid = [n.title() for n in nsaids if n in drugs_lower][0]
        alerts.append({
            "pair": f"{found_anticoag} + {found_nsaid}",
            "severity": "high",
            "description": f"Concomitant use of {found_anticoag} and {found_nsaid} significantly increases the risk of severe gastrointestinal hemorrhage and major bleeding.",
            "source": "Pharmacovigilance Safety Engine"
        })

    # Rule 2: Immunotherapy + High-dose Corticosteroids
    immuno = ["pembrolizumab", "keytruda", "nivolumab", "opdivo", "atezolizumab"]
    steroids = ["prednisone", "dexamethasone", "methylprednisolone"]

    has_immuno = any(i in drugs_lower for i in immuno)
    has_steroid = any(s in drugs_lower for s in steroids)

    if has_immuno and has_steroid:
        found_immuno = [i.title() for i in immuno if i in drugs_lower][0]
        found_steroid = [s.title() for s in steroids if s in drugs_lower][0]
        alerts.append({
            "pair": f"{found_immuno} + {found_steroid}",
            "severity": "moderate",
            "description": f"High-dose {found_steroid} may attenuate the therapeutic efficacy of anti-PD-1/PD-L1 checkpoint inhibitor {found_immuno}.",
            "source": "Pharmacovigilance Safety Engine"
        })

    return alerts
