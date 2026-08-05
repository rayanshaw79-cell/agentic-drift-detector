"""
clinical/rag/section_chunker.py — Structure-aware guideline chunker (Phase 1).

Replaces the naive character-window _chunk_text function in guideline_store.py
with a chunker that respects the semantic structure of NCI/AJCC guideline files.

Chunking strategy (three-pass):
  Pass 1: Split on lines containing 5+ BOX DRAWINGS HEAVY HORIZONTAL (━) chars.
           These are the major section separators present in all three guideline
           files. Each block becomes a named section with a title.

  Pass 2: Within each major section, split on ALL CAPS subsection headers that
           end with a colon (e.g. "PRIMARY TUMOR (T):", "DISTANT METASTASIS (M):").
           This separates T-staging, N-staging, and stage groupings into distinct,
           atomically-retrievable chunks — fixing the key baseline failure on
           lung_002 and lung_004.

  Pass 3: If any resulting chunk exceeds MAX_CHUNK_CHARS (700), split further
           using langchain RecursiveCharacterTextSplitter with sentence-boundary
           separators. Pure-Python fallback (blank-line split) if langchain
           text splitters are not available.

Each chunk carries rich metadata:
  - section:     human-readable section name (e.g. "STAGE GROUPINGS (NSCLC)")
  - chunk_index: global sequential index within the file
  - cancer_type: inferred from file stem (e.g. "Lung")
  - source:      filename (e.g. "lung_staging.txt")
"""

import re
import logging
from pathlib import Path
from typing import TypedDict

log = logging.getLogger(__name__)

# ━ = U+2501 BOX DRAWINGS HEAVY HORIZONTAL — the separator in all guideline files
_SEP_PATTERN = re.compile(r"^\u2501{5,}\s*$", re.MULTILINE)

# Subsection headers inside a major ━━━ block:
#   - Must start with an uppercase letter
#   - Must contain ≥ 5 characters before the colon (rules out "T1:", "N0:", "M1a:")
#   - Must end with a colon followed only by optional whitespace
# Examples that MATCH:   "PRIMARY TUMOR (T):", "REGIONAL LYMPH NODES (N):", "DISTANT METASTASIS (M):"
# Examples that DON'T:   "T1:", "N0:", "HER2 Status:", "Hormone Receptor (HR) Status:"
_SUBSECTION_HEADER = re.compile(r"^([A-Z][A-Z\s/\(\)\-\.&]{4,}):\s*$", re.MULTILINE)

_MAX_CHUNK_CHARS = 700  # sub-split chunks larger than this
_MIN_CHUNK_CHARS = 50   # discard chunks smaller than this (preamble fragments, etc.)


class ChunkResult(TypedDict):
    text: str
    section: str
    chunk_index: int
    cancer_type: str
    source: str


# ── Public API ─────────────────────────────────────────────────────────────────

def chunk_guideline_file(file_path: Path, cancer_type: str) -> list[ChunkResult]:
    """
    Parse a guideline .txt file into structured chunks with rich metadata.

    Args:
        file_path:   Path to the guideline text file.
        cancer_type: Human-readable cancer type string (e.g. "Lung").

    Returns:
        List of ChunkResult TypedDicts, one per atomic retrievable chunk.
    """
    text = file_path.read_text(encoding="utf-8")
    source = file_path.name
    results: list[ChunkResult] = []
    global_idx = 0

    # ── Pass 1: Split on ━━━ separator lines ──────────────────────────────────
    major_sections = _split_major_sections(text)
    log.debug("[CHUNKER] %s: %d major sections identified", source, len(major_sections))

    for section_name, section_content in major_sections:
        if not section_content.strip():
            continue

        # ── Pass 2: Split on ALL CAPS subsection headers ending with `:` ──────
        sub_blocks = _split_subsections(section_name, section_content)

        for sub_name, sub_text in sub_blocks:
            sub_text = sub_text.strip()
            if len(sub_text) < _MIN_CHUNK_CHARS:
                continue

            # ── Pass 3: Fine-split oversized subsection blocks ─────────────
            if len(sub_text) > _MAX_CHUNK_CHARS:
                fine_chunks = _split_by_lines(sub_text, max_chars=_MAX_CHUNK_CHARS)
            else:
                fine_chunks = [sub_text]

            for chunk_text in fine_chunks:
                chunk_text = chunk_text.strip()
                if len(chunk_text) < _MIN_CHUNK_CHARS:
                    continue

                results.append(
                    ChunkResult(
                        text=chunk_text,
                        section=sub_name,
                        chunk_index=global_idx,
                        cancer_type=cancer_type,
                        source=source,
                    )
                )
                global_idx += 1

    log.info(
        "[CHUNKER] %s: %d chunks produced (cancer_type=%s)",
        source,
        len(results),
        cancer_type,
    )
    return results


# ── Private helpers ────────────────────────────────────────────────────────────

def _split_major_sections(text: str) -> list[tuple[str, str]]:
    """
    Split text into (section_name, section_content) pairs using ━━━ separators.

    Guideline files alternate between separator-title-separator blocks and
    content blocks. This function identifies the pattern and pairs each title
    with its following content.

    Example structure (lung_staging.txt):
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      NON-SMALL CELL LUNG CANCER (NSCLC) — TNM DEFINITIONS
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      PRIMARY TUMOR (T):
      ...

      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
      STAGE GROUPINGS (NSCLC)
      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

      Stage IA1: T1a N0 M0
      ...
    """
    parts = _SEP_PATTERN.split(text)
    sections: list[tuple[str, str]] = []
    i = 0

    while i < len(parts):
        part = parts[i].strip()
        if not part:
            i += 1
            continue

        non_empty_lines = [ln for ln in part.splitlines() if ln.strip()]
        if not non_empty_lines:
            i += 1
            continue

        # Heuristic: a part with ≤ 2 non-empty lines and a longer following part
        # is a section TITLE; the next part is its CONTENT.
        if len(non_empty_lines) <= 2 and i + 1 < len(parts):
            next_part = parts[i + 1].strip()
            next_non_empty = [ln for ln in next_part.splitlines() if ln.strip()]
            if len(next_non_empty) > 2:
                section_name = non_empty_lines[0].strip().rstrip(":")
                sections.append((section_name, next_part))
                i += 2
                continue

        # Multi-line part: treat first line as section name, rest as content.
        section_name = non_empty_lines[0].strip().rstrip(":")
        content = "\n".join(ln for ln in part.splitlines() if ln.strip() != non_empty_lines[0].strip())
        if content.strip():
            sections.append((section_name, content))
        i += 1

    # Fallback: no separators found — treat entire text as one section.
    if not sections:
        lines = text.strip().splitlines()
        name = lines[0].strip() if lines else "General"
        content = "\n".join(lines[1:]) if len(lines) > 1 else text
        sections.append((name, content))

    return sections


def _split_subsections(
    parent_section: str,
    content: str,
) -> list[tuple[str, str]]:
    """
    Within a major section, split on ALL CAPS subsection headers ending with `:`.

    Example:
      Input parent_section = "NON-SMALL CELL LUNG CANCER (NSCLC) — TNM DEFINITIONS"
      Input content contains:
        PRIMARY TUMOR (T):
        T1: ...
        ...
        REGIONAL LYMPH NODES (N):
        N0: ...

      Output:
        [("PRIMARY TUMOR (T)", "PRIMARY TUMOR (T):\nT1: ..."),
         ("REGIONAL LYMPH NODES (N)", "REGIONAL LYMPH NODES (N):\nN0: ...")]

    If no subsection headers found, returns the whole block under the parent name.
    """
    headers = list(_SUBSECTION_HEADER.finditer(content))

    if not headers:
        return [(parent_section, content)]

    sub_sections: list[tuple[str, str]] = []

    # Text before the first subsection header → belongs to parent section
    pre_content = content[: headers[0].start()].strip()
    if len(pre_content) >= _MIN_CHUNK_CHARS:
        sub_sections.append((parent_section, pre_content))

    for idx, match in enumerate(headers):
        sub_name = match.group(1).strip()
        start = match.end()
        end = headers[idx + 1].start() if idx + 1 < len(headers) else len(content)
        sub_content = content[start:end].strip()
        # Re-attach the header line into the chunk text for retrieval context
        chunk_text = f"{match.group(0).strip()}\n{sub_content}"
        sub_sections.append((sub_name, chunk_text))

    return sub_sections


def _split_by_lines(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks not exceeding max_chars, using sentence/line boundaries.

    Preferred: langchain RecursiveCharacterTextSplitter for clean sentence splits.
    Fallback:  blank-line paragraph grouping (pure Python, no dependencies).
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", " "],
        )
        return splitter.split_text(text)
    except ImportError:
        pass

    # Pure-Python fallback: split on blank lines, merge until max_chars
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = (current + "\n\n" + para).strip() if current else para
    if current.strip():
        chunks.append(current.strip())
    return chunks
