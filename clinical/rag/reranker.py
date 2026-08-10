import logging
import math
from typing import Any

log = logging.getLogger(__name__)

_cross_encoder = None

def get_reranker():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            # Lightweight cross-encoder for MS MARCO
            _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            log.info("[RERANKER] Loaded cross-encoder/ms-marco-MiniLM-L-6-v2")
        except ImportError:
            log.warning("[RERANKER] sentence-transformers not installed. Reranking disabled.")
            _cross_encoder = "DISABLED"
        except Exception as e:
            log.warning(f"[RERANKER] Failed to load CrossEncoder: {e}")
            _cross_encoder = "DISABLED"
    return _cross_encoder

def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))

def rerank_results(query: str, chunks: list[dict], threshold: float = 0.3) -> list[dict]:
    """
    Reranks a list of chunks based on a query using a Cross-Encoder.
    Only returns chunks with a score >= threshold.
    """
    model = get_reranker()
    if model == "DISABLED" or not chunks:
        # Fallback: just return the chunks if reranking isn't available
        return chunks

    pairs = [[query, chunk["text"]] for chunk in chunks]
    try:
        scores = model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = sigmoid(float(score))
            
        # Filter by absolute threshold and sort
        passed = [c for c in chunks if c.get("rerank_score", 1.0) >= threshold]
        passed.sort(key=lambda x: x.get("rerank_score", 1.0), reverse=True)
        return passed
    except Exception as e:
        log.warning(f"[RERANKER] Reranking prediction failed: {e}")
        return chunks
