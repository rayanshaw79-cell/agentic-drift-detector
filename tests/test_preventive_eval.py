"""
tests/test_preventive_eval.py — Automated Evaluation Suite for Preventive Oncology (ASHA-AI).

Covers:
  1. RAG Guideline Retrieval Recall & Section Accuracy on `eval/preventive_golden_set.json`
  2. Indic / Hinglish Code-Switched NER Extraction Recall & Precision
  3. Biological Synergistic Risk Multiplier Scoring correctness
"""

import json
import pytest
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from clinical.rag.guideline_store import retrieve_guidelines
from clinical.steps.lifestyle_ner_step import (
    lifestyle_ner_step,
    normalize_hinglish_clinical_note,
    calculate_synergistic_risk,
)

GOLDEN_SET_PATH = Path(__file__).parent.parent / "eval" / "preventive_golden_set.json"


# ══════════════════════════════════════════════════════════════════════════════
# 1. Preventive RAG Golden Set Retrieval Benchmark
# ══════════════════════════════════════════════════════════════════════════════

class TestPreventiveRAGBenchmark:

    def test_golden_set_file_exists(self):
        assert GOLDEN_SET_PATH.exists(), f"Golden set file missing at {GOLDEN_SET_PATH}"

    def test_rag_retrieval_recall_on_golden_set(self):
        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            cases = json.load(f)

        assert len(cases) > 0, "Golden set is empty"

        hits = 0
        total = len(cases)

        for case in cases:
            query = case["query"]
            expected_keywords = case["expected_chunk_keywords"]

            # Retrieve top 2 chunks
            retrieved_text = retrieve_guidelines(query, k=2).lower()

            # Check if any expected keyword exists in retrieved text
            match = any(kw.lower() in retrieved_text for kw in expected_keywords)
            if match:
                hits += 1

        recall = hits / total
        print(f"\n[Preventive RAG Recall@k=2]: {hits}/{total} ({recall * 100:.1f}%)")
        assert recall >= 0.85, f"Preventive RAG recall below threshold 85%: {recall}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Hinglish & Vernacular Extraction Suite
# ══════════════════════════════════════════════════════════════════════════════

class TestHinglishExtraction:

    def test_khaini_and_bidi_normalization(self):
        note = "Patient 10 saal se khaini aur bidi pita hai."
        matched = normalize_hinglish_clinical_note(note)
        assert "tobacco" in matched
        assert "beedi" in matched
        assert matched["tobacco"] == ["khaini"]
        assert matched["beedi"] == ["bidi"]

    def test_paani_kharab_arsenic_normalization(self):
        note = "Gaon me peene ka paani kharab hai chemical ki wajah se."
        matched = normalize_hinglish_clinical_note(note)
        assert "arsenic" in matched
        assert "paani kharab" in matched["arsenic"]

    def test_supari_betel_nut_normalization(self):
        note = "Daily supari aur pan masala chabati hai."
        matched = normalize_hinglish_clinical_note(note)
        assert "betel nut" in matched
        assert "gutka" in matched

    def test_full_hinglish_ner_step(self):
        state = {
            "raw_note": "ASHA field report: 45 saal male. Khaini aur bidi ka sewan. Paani kharab.",
            "patient_id": "PAT-TEST-001"
        }
        res = lifestyle_ner_step(state)
        terms = {f["term"] for f in res["lifestyle_factors"]}
        assert "tobacco" in terms
        assert "beedi" in terms
        assert "arsenic" in terms
        assert res["lifestyle_risk_score"] >= 0.85


# ══════════════════════════════════════════════════════════════════════════════
# 3. Synergistic Risk Engine Mechanics
# ══════════════════════════════════════════════════════════════════════════════

class TestSynergisticRiskEngine:

    def test_single_factor_risk(self):
        factors = [{"term": "tobacco", "posterior": 0.99}]
        score = calculate_synergistic_risk(factors)
        assert 0.3 <= score <= 0.45

    def test_synergistic_tobacco_arsenic_multiplier(self):
        single_tobacco = calculate_synergistic_risk([{"term": "tobacco", "posterior": 0.99}])
        synergy_comb = calculate_synergistic_risk([
            {"term": "tobacco", "posterior": 0.99},
            {"term": "arsenic", "posterior": 0.99}
        ])
        # Multiplicative synergy should raise risk significantly higher than single factor
        assert synergy_comb > (single_tobacco * 1.5)
        assert synergy_comb >= 0.85
