"""
clinical/rag/bm25_index.py — In-memory BM25 index for sparse retrieval (Phase 3).

BM25 complements dense vector search (ChromaDB) by excelling at exact keyword matches,
which is critical for oncology staging (e.g. matching 'T2b' or 'Stage IIB' exactly,
where dense vectors often conflate 'T2a' and 'T2b' as just 'tumor size').

This index is built lazily on first access by loading all chunks currently stored
in the ChromaDB collection.
"""

import re
import logging
from typing import TypedDict
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

log = logging.getLogger(__name__)

class BM25Document(TypedDict):
    id: str
    text: str
    cancer_type: str
    source: str
    section: str

_bm25_index = None
_documents: list[BM25Document] = []


def _tokenize(text: str) -> list[str]:
    """
    Simple word tokenizer for BM25.
    Lowercases and splits on non-alphanumeric characters.
    """
    return [w for w in re.split(r'\W+', text.lower()) if w]


def init_bm25_from_chroma(collection):
    """
    Initialize the BM25 index from a populated ChromaDB collection.
    
    Args:
        collection: The ChromaDB collection object.
    """
    global _bm25_index, _documents
    if BM25Okapi is None:
        log.warning("[RAG] rank_bm25 not installed. BM25 search disabled.")
        return

    # Fetch all documents from ChromaDB
    # Note: In a production system with millions of chunks, you'd serialize the BM25 index 
    # to disk. But for NCI guidelines (~30 chunks), building it in-memory takes < 1ms.
    all_data = collection.get(include=["documents", "metadatas"])
    
    docs = all_data.get("documents", [])
    ids = all_data.get("ids", [])
    metas = all_data.get("metadatas", [])
    
    if not docs:
        log.warning("[RAG] ChromaDB is empty; cannot build BM25 index.")
        return

    _documents = []
    tokenized_corpus = []
    
    for doc_id, text, meta in zip(ids, docs, metas):
        _documents.append(
            BM25Document(
                id=doc_id,
                text=text,
                cancer_type=meta.get("cancer_type", ""),
                source=meta.get("source", ""),
                section=meta.get("section", "")
            )
        )
        tokenized_corpus.append(_tokenize(text))

    _bm25_index = BM25Okapi(tokenized_corpus)
    log.info("[RAG] BM25 index built over %d chunks.", len(_documents))


def search_bm25(query: str, cancer_type: str | None = None, k: int = 2) -> list[tuple[BM25Document, float]]:
    """
    Search the BM25 index for the top-k chunks.
    
    Args:
        query: Free-text query.
        cancer_type: Optional pre-filter to match ChromaDB behaviour.
        k: Number of chunks to retrieve.
        
    Returns:
        List of (BM25Document, bm25_score) tuples, sorted by descending score.
    """
    if _bm25_index is None:
        return []
        
    query_tokens = _tokenize(query)
    scores = _bm25_index.get_scores(query_tokens)
    
    # Filter and sort results
    results = []
    for i, score in enumerate(scores):
        if score <= 0:
            continue
            
        doc = _documents[i]
        
        # Apply cancer type pre-filter if requested
        if cancer_type and doc["cancer_type"] != cancer_type:
            continue
            
        results.append((doc, score))
        
    # Sort descending by score and take top-k
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]
