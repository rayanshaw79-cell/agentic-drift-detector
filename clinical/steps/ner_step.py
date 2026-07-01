"""
clinical/steps/ner_step.py — Named Entity Recognition (NER) node.

UPGRADED: Now uses a Bayesian Ensemble of 3 extractors, implementing
Vibhu Agarwal's (Miimansa) "Precision Without Losing Recall" model:

  P(Entity Exists | Votes) via Bayes' Rule with independence assumption.

Extractor 1 — Gemini LLM   (high recall, moderate precision)
Extractor 2 — Regex dict   (very high precision, lower recall)
Extractor 3 — NLM API      (codebook-grounded, good precision)

The Bayesian posterior determines which terms pass to the lookup step
and seeds the initial confidence estimate.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from schemas.clinical_state import ClinicalState
from clinical.tools.bayesian_ensemble import ensemble_vote

log = logging.getLogger(__name__)

# ── Common medical condition keywords for the regex extractor ─────────────────
_CONDITION_PATTERNS = [
    r"\b(hypertension|htn|high blood pressure)\b",
    r"\b(diabetes(?: mellitus)?|t[12]dm|diabetic)\b",
    r"\b(asthma|copd|bronchitis)\b",
    r"\b(heart failure|chf|cardiac failure)\b",
    r"\b(myocardial infarction|mi|heart attack)\b",
    r"\b(atrial fibrillation|afib|a-?fib)\b",
    r"\b(stroke|tia|cerebrovascular)\b",
    r"\b(pneumonia|pneumonitis)\b",
    r"\b(urinary tract infection|uti)\b",
    r"\b(sepsis|septicemia)\b",
    r"\b(anxiety|depression|bipolar|schizophrenia)\b",
    r"\b(obesity|bmi)\b",
    r"\b(chronic kidney disease|ckd|renal failure)\b",
    r"\b(hypothyroidism|hyperthyroidism|thyroid disease)\b",
    r"\b(anemia|anaemia)\b",
    r"\b(parkinson(?:'s)?|alzheimer(?:'s)?|dementia)\b",
    r"\b(copd|emphysema|chronic obstructive)\b",
    r"\b(angina|coronary artery disease|cad)\b",
    r"\b(multiple sclerosis|ms)\b",
]

# Minimum Bayesian posterior to forward a term to the ontology lookup step
_MIN_POSTERIOR = 0.45


# ── Extractor 1: Gemini LLM ───────────────────────────────────────────────────

def _gemini_ner(raw_note: str) -> list[str]:
    """
    Use Google Gemini to extract diagnoses from the clinical note.
    Returns a deduplicated list of condition strings.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
    response = llm.invoke([
        SystemMessage(content=(
            "You are a clinical NLP specialist. Extract all medical diagnoses, "
            "conditions, and diseases from the clinical note provided. "
            "Return ONLY a JSON object with a single key 'diagnoses' containing "
            "a list of strings, each being one distinct condition in plain English. "
            "Do NOT include medications, symptoms without a diagnosis, or procedures. "
            'Example: {"diagnoses": ["essential hypertension", "type 2 diabetes mellitus"]}'
        )),
        HumanMessage(content=f"Clinical note:\n{raw_note}"),
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

    result = json.loads(content)
    diagnoses = result.get("diagnoses", [])
    return [d.strip().lower() for d in diagnoses if d.strip()]


# ── Extractor 2: Regex dictionary ────────────────────────────────────────────

def _regex_ner(raw_note: str) -> list[str]:
    """
    Lightweight regex-based extraction.
    Very high specificity — almost never produces false positives.
    """
    note_lower = raw_note.lower()
    found: set[str] = set()
    for pattern in _CONDITION_PATTERNS:
        match = re.search(pattern, note_lower, re.IGNORECASE)
        if match:
            found.add(match.group(0).lower().strip())
    return sorted(found)


# ── Extractor 3: NLM API term search ─────────────────────────────────────────

def _nlm_ner(raw_note: str) -> list[str]:
    """
    Use the NLM ICD-10 API as a third extractor.

    Strategy: extract short noun phrases that look medical and check
    which ones the NLM codebook recognises. Capped at 8 candidates
    so this always completes within the 15s thread timeout.
    """
    from clinical.tools.nlm_api import lookup_icd10

    note_lower = raw_note.lower()

    # Extract candidate bigrams and trigrams using a targeted regex
    # (avoids submitting every word pair — only medically plausible phrases)
    candidates: list[str] = []

    # Single medical-looking words (>5 chars, not a stop word)
    _STOP = {"patient", "presents", "history", "started", "noted", "taking",
             "current", "reports", "denies", "years", "weeks", "months",
             "diagnosed", "treated", "managed", "follow", "recent", "known"}
    words = re.findall(r"\b[a-z]{5,}\b", note_lower)
    for w in words:
        if w not in _STOP:
            candidates.append(w)

    # Bigrams and trigrams from the full note
    all_words = re.findall(r"\b[a-z]{3,}\b", note_lower)
    for i in range(len(all_words) - 1):
        bigram = f"{all_words[i]} {all_words[i+1]}"
        candidates.append(bigram)
    for i in range(len(all_words) - 2):
        trigram = f"{all_words[i]} {all_words[i+1]} {all_words[i+2]}"
        candidates.append(trigram)

    # Deduplicate and cap to avoid timeout
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for c in candidates:
        if c not in seen and c not in _STOP:
            seen.add(c)
            unique_candidates.append(c)
        if len(unique_candidates) >= 8:
            break

    confirmed: list[str] = []
    for candidate in unique_candidates:
        results = lookup_icd10(candidate, max_results=1)
        if results:
            confirmed.append(candidate)

    return confirmed


# ── LangGraph Node ────────────────────────────────────────────────────────────

def ner_step(state: ClinicalState) -> dict:
    """
    LangGraph node — Bayesian Ensemble NER.

    Runs 3 extractors (Gemini, Regex, NLM API) in parallel, then applies
    Bayesian voting to compute P(Entity Exists | Votes) per term.

    Reads:  state["deid_note"] (fallback to state["raw_note"])
    Writes: state["extracted_diagnoses"] — list of high-confidence terms
            state["ner_votes"]            — per-term extractor votes & posteriors
    """
    # Use the scrubbed note from the De-ID step if available
    raw_note   = state.get("deid_note") or state.get("raw_note", "")
    use_gemini = bool(os.getenv("GEMINI_API_KEY"))
    start      = time.perf_counter()

    gemini_terms: list[str] = []
    regex_terms:  list[str] = _regex_ner(raw_note)

    # ── Run Gemini and NLM concurrently ──────────────────────────────────────
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {}

        if use_gemini:
            futures["gemini"] = pool.submit(_gemini_ner, raw_note)

        futures["nlm"] = pool.submit(_nlm_ner, raw_note)

        for name, future in futures.items():
            try:
                result = future.result(timeout=15)
                if name == "gemini":
                    gemini_terms = result
                    log.info("[NER] Gemini: %d terms", len(gemini_terms))
                else:
                    nlm_terms_result = result
                    log.info("[NER] NLM: %d terms", len(nlm_terms_result))
            except Exception as exc:
                log.warning("[NER] %s extractor failed: %s", name, exc)
                nlm_terms_result = []

    # Ensure nlm_terms_result is always defined
    try:
        nlm_terms_result
    except NameError:
        nlm_terms_result = []

    log.info("[NER] Regex: %d terms — %s", len(regex_terms), regex_terms)

    # ── Apply Bayesian Ensemble ───────────────────────────────────────────────
    voted = ensemble_vote(gemini_terms, regex_terms, nlm_terms_result)

    # Filter to terms above the posterior threshold
    confirmed = [v for v in voted if v["posterior"] >= _MIN_POSTERIOR]

    # If nothing passed the threshold, fall back to regex hits (safety net)
    if not confirmed and regex_terms:
        log.warning("[NER] No terms above threshold — using regex as safety net.")
        confirmed = [{
            "term":      t,
            "posterior": 0.55,
            "votes":     {"gemini": False, "regex": True, "nlm": False},
            "n_votes":   1,
        } for t in regex_terms]

    # If still nothing, use a placeholder
    if not confirmed:
        confirmed = [{
            "term":      "unspecified condition",
            "posterior": 0.40,
            "votes":     {"gemini": False, "regex": False, "nlm": False},
            "n_votes":   0,
        }]

    # Extract just the term strings for downstream steps
    diagnoses = [v["term"] for v in confirmed]

    latency = int((time.perf_counter() - start) * 1000) + 50

    log.info(
        "[NER] Ensemble result: %d terms (posteriors: %s)",
        len(confirmed),
        {v["term"]: v["posterior"] for v in confirmed},
    )

    return {
        "current_step":        "ner",
        "step_count":          1,
        "path_taken":          ["ner"],
        "extracted_diagnoses": diagnoses,
        "ner_votes":           confirmed,   # Full Bayesian vote data
        "execution_time_ms":   latency,
    }
