"""
eval/rag_eval.py — RAG Evaluation Harness (OncoLLM Pillar 2 Quality Gate)

Measures four metrics against the golden dataset in eval/rag_golden_set.json:
  1. Recall@k            — did we retrieve the right guideline chunk?
  2. MRR                 — was the best chunk ranked #1 or buried?
  3. Context Precision@k — are all retrieved chunks relevant, or is there noise?
  4. Answer Faithfulness — are extracted clinical values grounded in RAG context?
     Tier 1: Deterministic string-grounding (always runs, no API key required)
     Tier 2: LLM-as-judge semantic check (runs if GEMINI_API_KEY is set)

Usage:
  python eval/rag_eval.py --chunker naive    # baseline (current system)
  python eval/rag_eval.py --chunker section  # Phase 1 section chunker
  python eval/rag_eval.py --chunker hybrid   # Phase 3 BM25 + dense

Results are written to eval/results/<chunker>_YYYYMMDD_HHMMSS.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("rag_eval")

GOLDEN_SET_PATH = Path(__file__).parent / "rag_golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"


# ── Custom Exceptions ─────────────────────────────────────────────────────────

class FaithfulnessError(Exception):
    """
    Raised when Tier 1 deterministic faithfulness falls below the minimum
    acceptable threshold (0.50). This is a hard failure gate — it means
    fewer than half of extracted clinical values are traceable to the
    RAG context we injected.
    """


# ── Metric Computers ──────────────────────────────────────────────────────────

def compute_recall_at_k(retrieved_text: str, expected_keywords: list[str]) -> float:
    """
    Recall@k: 1.0 if ANY expected keyword appears in the full retrieved text.
    0.0 otherwise. Binary per-sample metric, averaged across the golden set.
    """
    text_lower = retrieved_text.lower()
    for kw in expected_keywords:
        if kw.lower() in text_lower:
            return 1.0
    return 0.0


def compute_mrr(chunks: list[str], expected_keywords: list[str]) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first chunk that contains a keyword.
    Returns 0.0 if no chunk is relevant.
    """
    for rank, chunk in enumerate(chunks, start=1):
        chunk_lower = chunk.lower()
        for kw in expected_keywords:
            if kw.lower() in chunk_lower:
                return 1.0 / rank
    return 0.0


def compute_context_precision(chunks: list[str], expected_keywords: list[str]) -> float:
    """
    Context Precision@k: fraction of retrieved chunks that contain at least
    one expected keyword (i.e., are relevant). A chunk is relevant if it
    contains ANY of the expected keywords.
    """
    if not chunks:
        return 0.0
    relevant = sum(
        1 for chunk in chunks
        if any(kw.lower() in chunk.lower() for kw in expected_keywords)
    )
    return relevant / len(chunks)


def score_faithfulness_deterministic(
    extracted: dict[str, Any],
    retrieved_context: str,
) -> float:
    """
    Tier 1 — Deterministic Faithfulness Scorer (always runs, no API key).

    For each non-null extracted clinical value (primary_site, histology,
    T, N, M, overall stage), checks whether that value appears verbatim
    in the retrieved RAG context.

    This is stricter than the existing provenance check in the staging step
    (which checks against the raw note). This checks against the RAG chunks
    specifically — answering: 'Did RAG actually provide the grounding
    for this extraction, or did the LLM ignore the retrieved context?'

    Args:
        extracted: Dict with keys: primary_site, histology, tnm_stage
                   (tnm_stage is a dict with T, N, M, overall, evidence_span)
        retrieved_context: Full string returned by retrieve_guidelines()

    Returns:
        Float in [0.0, 1.0]. Raises FaithfulnessError if score < 0.50.
    """
    if not retrieved_context:
        # No context was retrieved — faithfulness cannot be computed.
        # This itself is a RAG failure (Recall would also be 0), but
        # faithfulness is undefined here, not a hallucination.
        return 0.0

    values_to_check: list[str] = []

    if extracted.get("primary_site"):
        values_to_check.append(extracted["primary_site"])
    if extracted.get("histology"):
        values_to_check.append(extracted["histology"])

    tnm = extracted.get("tnm_stage") or {}
    for key in ("T", "N", "M", "overall"):
        val = tnm.get(key)
        if val:
            values_to_check.append(val)

    if not values_to_check:
        # Nothing was extracted — not a faithfulness failure, just no output.
        return 1.0

    grounded_count = sum(
        1 for val in values_to_check
        if val in retrieved_context
    )
    score = grounded_count / len(values_to_check)

    if score < 0.50:
        raise FaithfulnessError(
            f"Deterministic faithfulness FAILED: {score:.2f} "
            f"({grounded_count}/{len(values_to_check)} values grounded in RAG context). "
            f"Values checked: {values_to_check}"
        )

    return score


def score_faithfulness_llm(
    query: str,
    retrieved_context: str,
    expected_keywords: list[str],
) -> dict[str, Any]:
    """
    Tier 2 — LLM-as-Judge Faithfulness Scorer.

    Runs only when GEMINI_API_KEY is set. Uses gemini-2.0-flash with a
    structured prompt to evaluate whether the retrieved context semantically
    supports the expected clinical facts.

    Returns:
        dict with keys: faithful (bool), score (float), evidence (str)
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "faithful": None,
            "score": None,
            "evidence": "GEMINI_API_KEY not set — LLM judge skipped (Tier 1 passed).",
            "tier": "skipped",
        }

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=api_key,
        )

        system = """\
You are a clinical RAG faithfulness evaluator. You will be given:
  1. A retrieved guideline context (from a vector store)
  2. A list of expected clinical keywords that should be present in the context

Your task: Determine whether the retrieved context is faithful and relevant —
i.e., whether it actually contains the clinical facts needed to answer the query.

Respond with ONLY a JSON object. No explanation outside the JSON.
Schema:
{
  "faithful": true | false,
  "score": 0.0 to 1.0,
  "evidence": "one sentence explaining your verdict"
}"""

        user_msg = f"""\
RETRIEVED CONTEXT:
{retrieved_context[:1500]}

EXPECTED CLINICAL KEYWORDS:
{json.dumps(expected_keywords)}

Is the retrieved context faithful and sufficient to ground these clinical facts?"""

        response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user_msg)])
        content = response.content.strip()

        # Strip markdown fences if present
        import re
        content = re.sub(r"^```(?:json)?\n?", "", content)
        content = re.sub(r"\n?```$", "", content)

        result = json.loads(content)
        result["tier"] = "llm_judge"
        return result

    except Exception as exc:
        log.warning("[FAITHFULNESS LLM] Judge call failed: %s", exc)
        return {
            "faithful": None,
            "score": None,
            "evidence": f"LLM judge call failed: {exc}",
            "tier": "error",
        }


# ── Retrieval Adapter ─────────────────────────────────────────────────────────

def _get_retrieve_fn(chunker: str):
    """
    Returns the retrieve_guidelines function for the given chunker mode.
    All three modes share the same public API — the chunker flag controls
    which backend is active (set via environment variable before import).
    """
    os.environ["RAG_CHUNKER"] = chunker  # consumed by guideline_store in Phase 1+

    from clinical.rag.guideline_store import retrieve_guidelines

    def retrieve_with_chunks(query: str, k: int = 2) -> tuple[str, list[str]]:
        """Returns (full_text, list_of_chunks)."""
        full_text = retrieve_guidelines(query=query, k=k)
        # Split on the separator we use in guideline_store
        chunks = [c.strip() for c in full_text.split("\n\n---\n\n") if c.strip()]
        return full_text, chunks

    return retrieve_with_chunks


# ── Report Printer ────────────────────────────────────────────────────────────

def _print_report(chunker: str, metrics: dict, sample_results: list[dict]) -> None:
    """Pretty-print a summary table to stdout."""
    width = 57
    print(f"\n{'=' * width}")
    print(f"  RAG Evaluation Report -- chunker: {chunker.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * width}")
    print(f"  {'Metric':<30} {'Score':>10}  {'Samples':>8}")
    print(f"  {'-' * 50}")
    print(f"  {'Recall@2':<30} {metrics['recall_at_k']:>10.3f}  {metrics['n_samples']:>8}")
    print(f"  {'MRR':<30} {metrics['mrr']:>10.3f}  {metrics['n_samples']:>8}")
    print(f"  {'Context Precision@2':<30} {metrics['context_precision']:>10.3f}  {metrics['n_samples']:>8}")
    print(f"  {'Faithfulness (Tier 1 Det.)':<30} {metrics['faithfulness_t1']:>10.3f}  {metrics['n_faithfulness']:>8}")

    llm_score = metrics.get("faithfulness_t2_llm")
    if llm_score is not None:
        print(f"  {'Faithfulness (Tier 2 LLM)':<30} {llm_score:>10.3f}  {metrics['n_faithfulness']:>8}")
    else:
        print(f"  {'Faithfulness (Tier 2 LLM)':<30} {'N/A (no key)':>10}  {metrics['n_faithfulness']:>8}")

    print(f"{'=' * width}\n")

    # Show per-sample failures
    failures = [r for r in sample_results if not r.get("recall_hit")]
    if failures:
        print(f"  [!] Recall MISSES ({len(failures)}):")
        for f in failures:
            print(f"     [{f['query_id']}] {f['query'][:60]}")
            print(f"         Expected keywords: {f['expected_keywords']}")
        print()


# ── Main Evaluation Loop ───────────────────────────────────────────────────────

def run_evaluation(chunker: str, k: int = 2) -> dict:
    """
    Full evaluation pipeline against the golden set.

    Args:
        chunker: One of 'naive', 'section', 'hybrid'
        k:       Number of chunks to retrieve per query

    Returns:
        Dict of aggregated metrics + per-sample results
    """
    golden = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    retrieve = _get_retrieve_fn(chunker)

    sample_results = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    precision_scores: list[float] = []
    faith_t1_scores: list[float] = []
    faith_t2_scores: list[float] = []

    print(f"\n[RAG EVAL] Running {len(golden)} samples — chunker={chunker}, k={k}")

    for sample in golden:
        qid = sample["query_id"]
        query = sample["query"]
        keywords = sample["expected_chunk_keywords"]

        t0 = time.perf_counter()
        full_text, chunks = retrieve(query=query, k=k)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        # ── Retrieval Metrics ─────────────────────────────────────────────────
        recall = compute_recall_at_k(full_text, keywords)
        mrr = compute_mrr(chunks, keywords)
        precision = compute_context_precision(chunks, keywords)

        recall_scores.append(recall)
        mrr_scores.append(mrr)
        precision_scores.append(precision)

        # ── Faithfulness Tier 1 — Deterministic (always runs) ────────────────
        # For golden set eval we simulate a partial extraction using expected
        # keywords as proxies for extracted values (since we don't call the LLM
        # during eval). This tests whether those values appear in RAG output.
        simulated_extraction = {
            "primary_site": keywords[0] if keywords else None,
            "histology": None,
            "tnm_stage": {"T": keywords[1] if len(keywords) > 1 else None},
        }

        faith_t1 = 0.0
        faith_t1_error = None
        try:
            faith_t1 = score_faithfulness_deterministic(simulated_extraction, full_text)
        except FaithfulnessError as e:
            faith_t1_error = str(e)
            faith_t1 = 0.0

        faith_t1_scores.append(faith_t1)

        # ── Faithfulness Tier 2 — LLM Judge (optional, additive) ─────────────
        faith_t2_result = score_faithfulness_llm(query, full_text, keywords)
        if faith_t2_result.get("score") is not None:
            faith_t2_scores.append(float(faith_t2_result["score"]))

        result = {
            "query_id": qid,
            "query": query,
            "cancer_type": sample["cancer_type"],
            "expected_section": sample["expected_section"],
            "expected_keywords": keywords,
            "recall_hit": recall == 1.0,
            "recall": recall,
            "mrr": mrr,
            "context_precision": precision,
            "faithfulness_t1": faith_t1,
            "faithfulness_t1_error": faith_t1_error,
            "faithfulness_t2": faith_t2_result,
            "latency_ms": latency_ms,
            "retrieved_preview": full_text[:300] if full_text else "",
        }
        sample_results.append(result)

        status = "PASS" if recall == 1.0 else "FAIL"
        print(f"  {status} [{qid}] recall={recall:.1f} mrr={mrr:.2f} prec={precision:.2f} faith_t1={faith_t1:.2f} ({latency_ms}ms)")

    # ── Aggregated Metrics ────────────────────────────────────────────────────
    n = len(golden)
    metrics = {
        "chunker": chunker,
        "k": k,
        "n_samples": n,
        "n_faithfulness": len(faith_t1_scores),
        "recall_at_k": sum(recall_scores) / n,
        "mrr": sum(mrr_scores) / n,
        "context_precision": sum(precision_scores) / n,
        "faithfulness_t1": sum(faith_t1_scores) / len(faith_t1_scores) if faith_t1_scores else 0.0,
        "faithfulness_t2_llm": sum(faith_t2_scores) / len(faith_t2_scores) if faith_t2_scores else None,
        "timestamp": datetime.now().isoformat(),
    }

    # ── Save Results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{chunker}_{ts}.json"
    out_data = {"metrics": metrics, "samples": sample_results}
    out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\n[RAG EVAL] Results saved -> {out_path}")

    _print_report(chunker, metrics, sample_results)

    return out_data


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation Harness")
    parser.add_argument(
        "--chunker",
        choices=["naive", "section", "hybrid"],
        default="naive",
        help="Which chunker backend to evaluate (default: naive)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=2,
        help="Number of chunks to retrieve per query (default: 2)",
    )
    args = parser.parse_args()

    result = run_evaluation(chunker=args.chunker, k=args.k)
    metrics = result["metrics"]

    # Hard failure gate: Tier 1 faithfulness is the clinical safety floor
    if metrics["faithfulness_t1"] < 0.50:
        print(
            f"\n[RAG EVAL] HARD FAILURE: Tier 1 faithfulness {metrics['faithfulness_t1']:.2f} < 0.50 threshold.\n"
            f"           This chunker configuration is NOT safe for clinical use.\n"
        )
        sys.exit(1)

    print("[RAG EVAL] All metrics computed. Review results above.\n")
