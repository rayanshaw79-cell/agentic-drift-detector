"""
clinical/rag/guideline_store.py — In-process ChromaDB vector store for NCI/NCCN
oncology staging guidelines (OncoLLM Pillar 2).

The store is initialised ONCE at import time (lazy singleton pattern).
If the ChromaDB collection is empty, it automatically seeds from the
text files in docs/guidelines/.

Changes from v1 (naive chunker):
  - Phase 1: Uses section_chunker.chunk_guideline_file() — structure-aware
             chunking that splits on ━━━ section separators and ALL CAPS
             subsection headers. Stores section + chunk_index in metadata.
  - Phase 2: Cancer type inference and soft-fallback.
  - Phase 3: BM25 hybrid search. Dense search (ChromaDB) and Sparse search (BM25)
             are combined using Reciprocal Rank Fusion (RRF). This fixes the dense
             retriever's weakness at matching exact stage codes (e.g. 'Stage IIB').

Public API (unchanged):
  retrieve_guidelines(query: str, k: int = 2) -> str
    Returns the top-k most relevant guideline chunks joined as a string,
    ready to inject into a system prompt. Returns empty string if store
    is unavailable or empty.
"""

import logging
import os
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)

# Persistent storage directory (inside workspace so it survives restarts)
_CHROMA_PERSIST_DIR = str(
    Path(__file__).parent.parent.parent / "artifacts" / "chroma_guidelines"
)

# v2 collection name forces a fresh seed using the section chunker.
_COLLECTION_NAME = "oncology_guidelines_v2"
_GUIDELINES_DIR = Path(__file__).parent.parent.parent / "docs" / "guidelines"

# Singleton client and collection references
_client = None
_collection = None


# ── Cancer-type inference ──────────────────────────────────────────────────────

_nlp = None

def _get_ner_pipeline():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Load a lightweight NER model, disable unnecessary pipes for speed
            _nlp = spacy.load("en_core_web_sm", disable=["parser", "attribute_ruler", "lemmatizer"])
            # Add basic rules to catch cancer terminology if the standard model misses them
            ruler = _nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "CANCER_TYPE", "pattern": [{"LOWER": {"IN": ["lung", "breast", "colorectal", "brain", "skin", "prostate", "pancreatic"]}}, {"LOWER": "cancer", "OP": "?"}]},
                {"label": "CANCER_TYPE", "pattern": [{"LOWER": {"IN": ["melanoma", "glioblastoma", "carcinoma", "nsclc", "sclc"]}}]},
            ]
            ruler.add_patterns(patterns)
        except Exception as e:
            log.warning(f"[RAG] Failed to initialize NER pipeline: {e}")
            _nlp = "FAILED"
    return _nlp

def _infer_cancer_type(query: str) -> str | None:
    # 1. Attempt NER Extraction
    nlp = _get_ner_pipeline()
    if nlp and nlp != "FAILED":
        doc = nlp(query)
        for ent in doc.ents:
            if ent.label_ == "CANCER_TYPE":
                text = ent.text.lower()
                # Map extracted entity to standardized types
                if any(kw in text for kw in ("lung", "nsclc", "sclc")): return "Lung"
                if any(kw in text for kw in ("breast", "mammary")): return "Breast"
                if any(kw in text for kw in ("colorectal", "colon", "rectal")): return "Colorectal"
                if any(kw in text for kw in ("brain", "glioblastoma")): return "Brain"
                if any(kw in text for kw in ("skin", "melanoma")): return "Skin"
                # If it's another cancer type found by NER, return it capitalized
                return ent.text.title()

    # 2. Fallback to Simple Heuristic
    q = query.lower()
    if any(kw in q for kw in ("lung", "nsclc", "sclc", "pulmonary", "bronch")):
        return "Lung"
    if any(kw in q for kw in ("breast", "mammary", "her2", "brca", "t4d")):
        return "Breast"
    if any(kw in q for kw in ("colorectal", "colon", "rectal", "sigmoid", "kras", "nras", "msi")):
        return "Colorectal"
    
    return None


# ── Collection initialisation ──────────────────────────────────────────────────

def _get_collection():
    """Lazy-initialise ChromaDB client and collection (singleton)."""
    global _client, _collection

    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        log.warning("[RAG] chromadb not installed. Guideline grounding disabled.")
        return None

    os.makedirs(_CHROMA_PERSIST_DIR, exist_ok=True)
    _client = chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)

    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        ef = embedding_functions.DefaultEmbeddingFunction()

    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine", "chunker": "section_v1"},
    )

    if _collection.count() == 0:
        log.info("[RAG] Collection empty — seeding from %s", _GUIDELINES_DIR)
        _seed_collection(_collection)
    else:
        log.info("[RAG] Collection ready (%d chunks loaded).", _collection.count())

    # Initialise BM25 (Phase 3)
    try:
        from clinical.rag.bm25_index import init_bm25_from_chroma
        init_bm25_from_chroma(_collection)
    except Exception as e:
        log.warning("[RAG] Failed to initialize BM25: %s", e)

    return _collection


# ── Seeding ────────────────────────────────────────────────────────────────────

def _seed_collection(collection) -> None:
    if not _GUIDELINES_DIR.exists():
        log.warning("[RAG] Guidelines directory not found: %s", _GUIDELINES_DIR)
        return

    try:
        from clinical.rag.section_chunker import chunk_guideline_file
    except ImportError:
        log.error("[RAG] section_chunker not found — seeding aborted.")
        return

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for txt_file in sorted(_GUIDELINES_DIR.glob("*.txt")):
        cancer_type = txt_file.stem.replace("_staging", "").replace("_", " ").title()
        chunks = chunk_guideline_file(file_path=txt_file, cancer_type=cancer_type)

        for chunk in chunks:
            all_texts.append(chunk["text"])
            all_ids.append(f"{txt_file.stem}_s{chunk['chunk_index']:03d}")
            all_meta.append(
                {
                    "cancer_type": chunk["cancer_type"],
                    "source": chunk["source"],
                    "section": chunk["section"],
                    "chunk_index": chunk["chunk_index"],
                }
            )

    if not all_texts:
        log.warning("[RAG] No guideline text files found in %s", _GUIDELINES_DIR)
        return

    batch_size = 100
    for i in range(0, len(all_texts), batch_size):
        collection.add(
            documents=all_texts[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
            metadatas=all_meta[i : i + batch_size],
        )

    n_files = len(list(_GUIDELINES_DIR.glob("*.txt")))
    log.info("[RAG] Seeded %d section chunks from %d guideline files.", len(all_texts), n_files)


# ── Retrieval ──────────────────────────────────────────────────────────────────

class RRFResult(TypedDict):
    id: str
    text: str
    source: str
    section: str
    score: float


def retrieve_guidelines(query: str, k: int = 2) -> str:
    """
    Retrieve the top-k most relevant NCI staging guideline chunks for a query.
    Phase 3: Uses Reciprocal Rank Fusion (RRF) to merge dense and sparse results.
    """
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return ""

    cancer_type = _infer_cancer_type(query)
    # We fetch more chunks initially to allow RRF to do its job
    fetch_k = min(10, collection.count())
    
    rrf_scores: dict[str, RRFResult] = {}
    
    # RRF Constant (usually 60)
    rrf_k = 60

    # ── 1. Dense Search (ChromaDB) ────────────────────────────────────────────
    try:
        dense_ids, dense_docs, dense_metas = [], [], []
        if cancer_type:
            res = collection.query(
                query_texts=[query],
                where={"cancer_type": cancer_type},
                n_results=fetch_k,
                include=["documents", "metadatas"]
            )
            dense_ids = res.get("ids", [[]])[0]
            dense_docs = res.get("documents", [[]])[0]
            dense_metas = res.get("metadatas", [[]])[0]
            
        if len(dense_ids) == 0:
            log.debug("[RAG] Filtered dense query returned 0 chunks — falling back to unfiltered.")
            res = collection.query(
                query_texts=[query],
                n_results=fetch_k,
                include=["documents", "metadatas"]
            )
            dense_ids = res.get("ids", [[]])[0]
            dense_docs = res.get("documents", [[]])[0]
            dense_metas = res.get("metadatas", [[]])[0]

        for rank, (doc_id, text, meta) in enumerate(zip(dense_ids, dense_docs, dense_metas)):
            rrf_scores[doc_id] = RRFResult(
                id=doc_id,
                text=text,
                source=meta.get("source", "unknown"),
                section=meta.get("section", "unknown"),
                score=1.0 / (rrf_k + rank + 1)
            )
    except Exception as exc:
        log.warning("[RAG] Dense retrieval failed: %s", exc)

    # ── 2. Sparse Search (BM25) ───────────────────────────────────────────────
    try:
        from clinical.rag.bm25_index import search_bm25
        # Also fall back to unfiltered if sparse fetch returns nothing
        bm25_results = search_bm25(query, cancer_type=cancer_type, k=fetch_k)
        if len(bm25_results) == 0:
            log.debug("[RAG] Filtered sparse query returned 0 chunks — falling back to unfiltered.")
            bm25_results = search_bm25(query, cancer_type=None, k=fetch_k)
            
        for rank, (doc, _score) in enumerate(bm25_results):
            doc_id = doc["id"]
            score_add = 1.0 / (rrf_k + rank + 1)
            
            if doc_id in rrf_scores:
                rrf_scores[doc_id]["score"] += score_add
            else:
                rrf_scores[doc_id] = RRFResult(
                    id=doc_id,
                    text=doc["text"],
                    source=doc["source"],
                    section=doc["section"],
                    score=score_add
                )
    except Exception as exc:
        log.warning("[RAG] Sparse retrieval failed: %s", exc)

    # ── 3. Merge and Format ───────────────────────────────────────────────────
    if not rrf_scores:
        return ""
        
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
    top_results = sorted_results[:k]
    
    formatted: list[str] = []
    for res in top_results:
        header = f"[SOURCE: {res['source']} | SECTION: {res['section']}]"
        formatted.append(f"{header}\n{res['text']}")

    return "\n\n---\n\n".join(formatted)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Legacy helper maintained to keep unit tests passing."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]

