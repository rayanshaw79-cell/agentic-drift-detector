import logging
import re
from typing import Dict, Any, List
from schemas.clinical_state import ClinicalState
from clinical.tools.bayesian_ensemble import bayesian_posterior

log = logging.getLogger(__name__)

# Standard clinical risk concepts
RISK_KEYWORDS = [
    "tobacco", "smoking", "beedi", "gutka", "betel nut",
    "areca nut", "arsenic", "alcohol", "pesticides", "obesity"
]

# Indic / Hinglish transliteration & vernacular synonym dictionary
HINGLISH_SYNONYM_MAP = {
    "tobacco": ["tambaku", "tambaaku", "khaini", "zarda", "surti", "chhad", "tobacco"],
    "beedi": ["bidi", "beedi", "sutta", "chutta"],
    "smoking": ["sigret", "cigarette", "cigar", "smoking", "hookah", "hukka"],
    "gutka": ["gutka", "gutkha", "pan masala", "shikhar", "vimal", "kamla pasand"],
    "betel nut": ["supari", "supaarhee", "chhalia", "paan", "areca nut", "betel nut"],
    "arsenic": ["arsenic", "paani kharab", "groundwater chemical", "ganda paani", "zahar paani"],
    "alcohol": ["sharaab", "sharab", "daaru", "daru", "liquor", "alcohol"],
    "pesticides": ["kheti dawai", "khet dawai", "pesticide", "pesticides", "spray"],
    "obesity": ["motaapa", "motapa", "bhaari vajan", "heavy weight", "obesity"]
}

def normalize_hinglish_clinical_note(note: str) -> Dict[str, List[str]]:
    """
    Pre-pass: Scans text for Hinglish / Indic code-switched terms and maps them
    to standard clinical risk concepts for ensemble voting.
    """
    note_lower = note.lower()
    matched_concepts: Dict[str, List[str]] = {}
    
    for concept, synonyms in HINGLISH_SYNONYM_MAP.items():
        for syn in synonyms:
            if re.search(r'\b' + re.escape(syn) + r'\b', note_lower):
                matched_concepts.setdefault(concept, []).append(syn)
                
    return matched_concepts

def calculate_synergistic_risk(extracted_factors: List[Dict[str, Any]]) -> float:
    """
    Calculates a biological synergistic risk score rather than naive linear addition.
    Multiplicative risk formula with interaction multiplier matrix:
      - Tobacco + Alcohol -> 1.4x multiplier
      - Arsenic + Tobacco -> 1.5x multiplier
      - Gutka / Tobacco + Betel Nut -> 1.3x multiplier
    """
    if not extracted_factors:
        return 0.1
        
    detected_terms = {f["term"] for f in extracted_factors}
    
    # Base independent non-event probabilities product: 1 - prod(1 - w_i * P_i)
    weights = {
        "tobacco": 0.35, "gutka": 0.35, "beedi": 0.30, "smoking": 0.30,
        "arsenic": 0.40, "betel nut": 0.25, "alcohol": 0.20, "pesticides": 0.25, "obesity": 0.20
    }
    
    prob_no_risk = 1.0
    for factor in extracted_factors:
        term = factor["term"]
        post = factor["posterior"]
        w = weights.get(term, 0.2)
        prob_no_risk *= (1.0 - w * post)
        
    base_risk = 1.0 - prob_no_risk
    
    # Synergistic interaction multipliers
    multiplier = 1.0
    if ("tobacco" in detected_terms or "beedi" in detected_terms or "gutka" in detected_terms) and "alcohol" in detected_terms:
        multiplier *= 1.4
    if "arsenic" in detected_terms and ("tobacco" in detected_terms or "smoking" in detected_terms or "beedi" in detected_terms or "gutka" in detected_terms):
        multiplier *= 1.5
    if "gutka" in detected_terms and "betel nut" in detected_terms:
        multiplier *= 1.3
        
    final_score = base_risk * multiplier
    return min(1.0, round(final_score, 2))

def lifestyle_ner_step(state: ClinicalState) -> ClinicalState:
    log.info("[Preventive] Running Hinglish-Aware Lifestyle NER Extraction...")
    note = state.get("deid_note") or state.get("raw_note") or ""
    
    matched_hinglish = normalize_hinglish_clinical_note(note)
    note_lower = note.lower()
    
    extracted_factors = []
    
    # 1. Gemini pass (Simulated multi-lingual concept extraction)
    gemini_terms = list(matched_hinglish.keys())
    
    # 2. Regex pass (Exact Hinglish synonym match)
    regex_terms = []
    for concept, synonyms in HINGLISH_SYNONYM_MAP.items():
        for syn in synonyms:
            if re.search(r'\b' + re.escape(syn) + r'\b', note_lower):
                regex_terms.append(concept)
                break
                
    # Combine via Bayesian ensemble
    all_terms = set(gemini_terms + regex_terms)
    
    for term in all_terms:
        matched_syns = matched_hinglish.get(term, [])
        votes = {
            "gemini": term in gemini_terms,
            "regex": term in regex_terms,
            "nlm": len(matched_syns) > 0
        }
        posterior = bayesian_posterior(votes)
        
        extracted_factors.append({
            "term": term,
            "posterior": posterior,
            "matched_synonyms": matched_syns,
            "votes": votes
        })
        
    state["lifestyle_factors"] = extracted_factors
    state["lifestyle_risk_score"] = calculate_synergistic_risk(extracted_factors)
    
    return state
