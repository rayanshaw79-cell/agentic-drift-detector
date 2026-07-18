"""
clinical/rag/guideline_store.py — In-process ChromaDB vector store for NCI/NCCN
oncology staging guidelines (OncoLLM Pillar 2).

The store is initialised ONCE at import time (lazy singleton pattern).
If the ChromaDB collection is empty, it automatically seeds from the
text files in docs/guidelines/.

Public API:
  retrieve_guidelines(query: str, k: int = 2) -> str
    Returns the top-k most relevant guideline chunks joined as a string,
    ready to inject into a system prompt. Returns empty string if store
    is unavailable or empty.
"""

import logging
import os
from pathlib import Path
from functools import lru_cache

log = logging.getLogger(__name__)

# Persistent storage directory (inside workspace so it survives restarts)
_CHROMA_PERSIST_DIR = str(
    Path(__file__).parent.parent.parent / "artifacts" / "chroma_guidelines"
)
_COLLECTION_NAME = "oncology_guidelines"
_GUIDELINES_DIR = Path(__file__).parent.parent.parent / "docs" / "guidelines"

# Singleton client and collection references
_client = None
_collection = None


def _get_collection():
    """Lazy-initialise ChromaDB client and collection (singleton)."""
    global _client, _collection

    if _collection is not None:
        return _collection

    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError:
        log.warning(
            "[RAG] chromadb not installed. Run: pip install chromadb. "
            "Guideline grounding disabled."
        )
        return None

    os.makedirs(_CHROMA_PERSIST_DIR, exist_ok=True)

    _client = chromadb.PersistentClient(path=_CHROMA_PERSIST_DIR)

    # Use a simple sentence-transformer embedding (no API key required)
    # Falls back gracefully if sentence-transformers not installed.
    try:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
    except Exception:
        ef = embedding_functions.DefaultEmbeddingFunction()

    _collection = _client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Auto-seed if collection is empty
    if _collection.count() == 0:
        log.info("[RAG] Collection empty — seeding from %s", _GUIDELINES_DIR)
        _seed_collection(_collection)
    else:
        log.info("[RAG] Collection ready (%d chunks loaded).", _collection.count())

    return _collection


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]


def _seed_collection(collection) -> None:
    """Load all .txt files from docs/guidelines/ into ChromaDB."""
    if not _GUIDELINES_DIR.exists():
        log.warning("[RAG] Guidelines directory not found: %s", _GUIDELINES_DIR)
        return

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_meta: list[dict] = []

    for txt_file in sorted(_GUIDELINES_DIR.glob("*.txt")):
        cancer_type = txt_file.stem.replace("_staging", "").replace("_", " ").title()
        text = txt_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{txt_file.stem}_chunk_{i:03d}")
            all_meta.append({"cancer_type": cancer_type, "source": txt_file.name})

    if not all_chunks:
        log.warning("[RAG] No guideline text files found in %s", _GUIDELINES_DIR)
        return

    # ChromaDB add in batches of 100
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            documents=all_chunks[i : i + batch_size],
            ids=all_ids[i : i + batch_size],
            metadatas=all_meta[i : i + batch_size],
        )

    log.info("[RAG] Seeded %d chunks from %d guideline files.", len(all_chunks), len(list(_GUIDELINES_DIR.glob("*.txt"))))


def retrieve_guidelines(query: str, k: int = 2) -> str:
    """
    Retrieve the top-k most relevant NCI staging guideline chunks for a query.

    Args:
        query: Free-text query — typically the cancer type or document_type.
        k:     Number of chunks to retrieve.

    Returns:
        A concatenated string of guideline chunks ready for system prompt injection.
        Returns an empty string if the store is unavailable.
    """
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        return ""

    try:
        results = collection.query(query_texts=[query], n_results=min(k, collection.count()))
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "\n\n---\n\n".join(docs)
    except Exception as exc:
        log.warning("[RAG] Retrieval failed: %s", exc)
        return ""
