"""
tests/test_rag_eval.py — Unit tests for the RAG evaluation harness.

Tests the metric computers and faithfulness scorer in isolation — no LLM
calls, no golden set file required, no ChromaDB access needed.

Covers:
  - Recall@k:              perfect hit, clean miss
  - MRR:                   rank-1 and rank-2 positions
  - Context Precision:     all-relevant, mixed, all-noise cases
  - Faithfulness Tier 1:   grounded pass, partial, hard failure gate
"""

import pytest
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.rag_eval import (
    compute_recall_at_k,
    compute_mrr,
    compute_context_precision,
    score_faithfulness_deterministic,
    FaithfulnessError,
)


# ══════════════════════════════════════════════════════════════════════════════
# Recall@k
# ══════════════════════════════════════════════════════════════════════════════

class TestRecallAtK:

    def test_perfect_recall_single_keyword(self):
        """Keyword present in retrieved text → 1.0."""
        retrieved = "T2a: Tumor > 3 cm but ≤ 4 cm, Stage IIB"
        assert compute_recall_at_k(retrieved, ["T2a"]) == 1.0

    def test_perfect_recall_second_keyword_matches(self):
        """First keyword absent, second present → still 1.0 (any match wins)."""
        retrieved = "Stage IIB Non-Small Cell Lung Cancer T2a pN1"
        assert compute_recall_at_k(retrieved, ["T3", "Stage IIB"]) == 1.0

    def test_recall_miss_no_keywords_found(self):
        """No expected keywords in text → 0.0."""
        retrieved = "Stage IIIA: T4 N0 M0 colorectal adenocarcinoma"
        assert compute_recall_at_k(retrieved, ["T2b", "Stage IIB"]) == 0.0

    def test_recall_is_case_insensitive(self):
        """Keyword match should be case-insensitive."""
        retrieved = "t2b tumor greater than 4 cm"
        assert compute_recall_at_k(retrieved, ["T2b"]) == 1.0

    def test_recall_empty_retrieved_text(self):
        """Empty retrieved text → 0.0 regardless of keywords."""
        assert compute_recall_at_k("", ["T2a", "Stage IIB"]) == 0.0

    def test_recall_empty_keywords(self):
        """Empty keyword list → 0.0 (no criteria = no match)."""
        assert compute_recall_at_k("T2a pN1 Stage IIB", []) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MRR (Mean Reciprocal Rank)
# ══════════════════════════════════════════════════════════════════════════════

class TestMRR:

    def test_mrr_rank_1(self):
        """Relevant keyword in first chunk → MRR = 1.0."""
        chunks = [
            "T2a: Tumor > 3 cm but ≤ 4 cm. Stage IIB.",
            "N1: ipsilateral hilar lymph nodes.",
        ]
        assert compute_mrr(chunks, ["T2a"]) == 1.0

    def test_mrr_rank_2(self):
        """Relevant keyword in second chunk → MRR = 0.5."""
        chunks = [
            "N1: ipsilateral hilar lymph nodes and intrapulmonary nodes.",
            "T2b: Tumor > 4 cm but ≤ 5 cm.",
        ]
        assert compute_mrr(chunks, ["T2b"]) == pytest.approx(0.5)

    def test_mrr_rank_3(self):
        """Relevant keyword in third chunk → MRR = 1/3."""
        chunks = [
            "Stage IIIA grouping definitions.",
            "N2: ipsilateral mediastinal nodes.",
            "T2a: Tumor > 3 cm but ≤ 4 cm.",
        ]
        assert compute_mrr(chunks, ["T2a"]) == pytest.approx(1 / 3)

    def test_mrr_no_relevant_chunk(self):
        """No chunk contains the keyword → MRR = 0.0."""
        chunks = [
            "Stage IIIA definitions.",
            "N1 lymph node criteria.",
        ]
        assert compute_mrr(chunks, ["HER2", "EGFR"]) == 0.0

    def test_mrr_empty_chunks(self):
        """Empty chunk list → MRR = 0.0."""
        assert compute_mrr([], ["T2a"]) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Context Precision@k
# ══════════════════════════════════════════════════════════════════════════════

class TestContextPrecision:

    def test_all_chunks_relevant(self):
        """Both chunks contain keywords → precision = 1.0."""
        chunks = [
            "T2a: Tumor > 3 cm but ≤ 4 cm.",
            "Stage IIB: T2a N1 M0; T2b N1 M0.",
        ]
        assert compute_context_precision(chunks, ["T2a", "Stage IIB"]) == 1.0

    def test_half_chunks_relevant(self):
        """One of two chunks is relevant → precision = 0.5."""
        chunks = [
            "T2a: Tumor > 3 cm but ≤ 4 cm.",
            "M1c: Peritoneal surface metastasis.",  # irrelevant to lung T2a query
        ]
        assert compute_context_precision(chunks, ["T2a"]) == pytest.approx(0.5)

    def test_no_chunks_relevant(self):
        """Neither chunk contains a keyword → precision = 0.0."""
        chunks = [
            "M1c: Peritoneal surface metastasis.",
            "Tis: Carcinoma in situ, lamina propria invasion.",
        ]
        assert compute_context_precision(chunks, ["T2b", "Stage IIB"]) == 0.0

    def test_empty_chunks_list(self):
        """Empty chunk list → precision = 0.0."""
        assert compute_context_precision([], ["T2a"]) == 0.0

    def test_single_chunk_relevant(self):
        """Single chunk list with match → 1.0."""
        assert compute_context_precision(["T2b lung NSCLC"], ["T2b"]) == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# Faithfulness Tier 1 — Deterministic String Grounding
# ══════════════════════════════════════════════════════════════════════════════

class TestFaithfulnessDeterministic:

    def test_fully_grounded_returns_1(self):
        """All extracted values present in retrieved context → 1.0."""
        extracted = {
            "primary_site": "Right upper lobe, lung",
            "histology": "Invasive adenocarcinoma",
            "tnm_stage": {
                "T": "pT2a",
                "N": "pN1",
                "M": "pM0",
                "overall": "Stage IIB Non-Small Cell Lung Cancer",
            },
        }
        # Build a context that contains all values
        context = (
            "Right upper lobe, lung: Invasive adenocarcinoma. "
            "pT2a pN1 pM0 — Stage IIB Non-Small Cell Lung Cancer."
        )
        score = score_faithfulness_deterministic(extracted, context)
        assert score == 1.0

    def test_partially_grounded_returns_partial_score(self):
        """Half the values are in context → score = 0.5. No error raised."""
        extracted = {
            "primary_site": "Right upper lobe, lung",  # present
            "histology": "Invasive adenocarcinoma",      # present
            "tnm_stage": {
                "T": "pT4",       # NOT in context
                "N": None,
                "M": None,
                "overall": None,
            },
        }
        context = "Right upper lobe, lung: Invasive adenocarcinoma, acinar predominant."
        score = score_faithfulness_deterministic(extracted, context)
        # 2 out of 3 non-null values grounded = 0.666...
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_below_threshold_raises_faithfulness_error(self):
        """Score < 0.50 must raise FaithfulnessError (hard failure gate)."""
        extracted = {
            "primary_site": "Hallucinated Site",    # NOT in context
            "histology": "Hallucinated Histology",  # NOT in context
            "tnm_stage": {
                "T": "pT99",   # NOT in context
                "N": None,
                "M": None,
                "overall": None,
            },
        }
        context = "Stage IIB Non-Small Cell Lung Cancer: pT2a pN1 pM0."
        with pytest.raises(FaithfulnessError):
            score_faithfulness_deterministic(extracted, context)

    def test_no_extracted_values_returns_1(self):
        """All extracted values are null → score = 1.0 (undefined, not a failure)."""
        extracted = {
            "primary_site": None,
            "histology": None,
            "tnm_stage": {"T": None, "N": None, "M": None, "overall": None},
        }
        score = score_faithfulness_deterministic(extracted, "any context text")
        assert score == 1.0

    def test_empty_retrieved_context_returns_0(self):
        """
        Empty context means RAG returned nothing. Score = 0.0.
        Note: this does NOT raise FaithfulnessError since the failure
        is in retrieval (Recall = 0), not in hallucination.
        """
        extracted = {
            "primary_site": "lung",
            "histology": None,
            "tnm_stage": {"T": "pT2a", "N": None, "M": None, "overall": None},
        }
        score = score_faithfulness_deterministic(extracted, "")
        assert score == 0.0

    def test_tnm_only_extraction_grounded(self):
        """Only T/N/M values extracted (no primary_site/histology) → still grounded."""
        extracted = {
            "primary_site": None,
            "histology": None,
            "tnm_stage": {
                "T": "cT3",
                "N": "cN2",
                "M": "cM1b",
                "overall": None,
            },
        }
        context = "Clinical staging: cT3 cN2 cM1b. Hepatic and mediastinal involvement."
        score = score_faithfulness_deterministic(extracted, context)
        assert score == 1.0
