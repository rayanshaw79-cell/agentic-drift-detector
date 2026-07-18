"""
clinical/agents/oncology_evaluator.py — Self-Correction Evaluator (OncoLLM Pillar 4).

After oncology_staging and biomarker_extraction run, this node reviews the
outputs and decides whether they are complete and trustworthy enough to
proceed to trial matching.

Checks performed:
  1. PRIMARY SITE CHECK: primary_site must be non-null.
  2. TNM COMPLETENESS CHECK: tnm_stage must exist and have at least ONE
     non-null component (T, N, M, or overall). A completely null tnm_stage
     for a pathology report is suspicious.
  3. BIOMARKER PROVENANCE RE-CHECK: All biomarker evidence_spans must
     appear verbatim in raw_note (defence-in-depth; staging step already
     checks this, but we double-check here).
  4. STAGING EVIDENCE CHECK: tnm_stage.evidence_span must be non-null for
     pathology_report and radiology doc types.

If ANY check fails:
  - Sets needs_reextraction = True
  - Sets eval_feedback with a specific critique string
  - The workflow's conditional edge routes back to oncology_staging

If all checks pass:
  - Sets needs_reextraction = False, eval_feedback = None
  - Workflow proceeds to trial_matching
"""

import logging
import time
from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

# Document types where we expect explicit staging evidence
_EXPLICIT_STAGE_TYPES = {"pathology_report", "radiology"}

# Max number of re-extraction cycles allowed (mirrors retry_count pattern)
MAX_REEXTRACTION_RETRIES = 2


def oncology_evaluator(state: ClinicalState) -> dict:
    """
    LangGraph node — Oncology Extraction Evaluator.

    Reads:  state["primary_site"], state["tnm_stage"], state["biomarkers"],
            state["raw_note"], state["document_type"], state["retry_count"]
    Writes: state["needs_reextraction"], state["eval_feedback"]
    """
    start_time = time.perf_counter()

    primary_site = state.get("primary_site")
    tnm_stage = state.get("tnm_stage") or {}
    biomarkers = state.get("biomarkers") or []
    raw_note = state.get("raw_note", "")
    document_type = state.get("document_type") or "unknown"
    retry_count = state.get("retry_count", 0)

    issues: list[str] = []

    # ── Check 1: Primary site ─────────────────────────────────────────────────
    if not primary_site:
        issues.append(
            "primary_site is null. Look more carefully for the anatomical cancer origin "
            "(e.g., 'lung', 'breast', 'colon') in the note."
        )

    # ── Check 2: TNM completeness (only strict for path/radiology reports) ────
    if document_type in _EXPLICIT_STAGE_TYPES:
        t_val = tnm_stage.get("T")
        n_val = tnm_stage.get("N")
        m_val = tnm_stage.get("M")
        overall = tnm_stage.get("overall")

        if not any([t_val, n_val, m_val, overall]):
            issues.append(
                "tnm_stage is completely null for a document that should contain staging "
                "information. Look for explicit T/N/M notation or a stage group statement "
                "(e.g., 'Stage IIB', 'pT2a pN1 pM0'). If truly absent, output null for "
                "individual components but do not omit the object entirely."
            )

        # ── Check 3: Staging evidence_span for high-confidence doc types ──────
        ev = tnm_stage.get("evidence_span") if tnm_stage else None
        if not ev and any([t_val, n_val, overall]):
            issues.append(
                "tnm_stage has component values but evidence_span is null. "
                "You must quote the EXACT sentence from the note where you found the staging."
            )

    # ── Check 4: Biomarker provenance re-verification ─────────────────────────
    bad_biomarkers = []
    for bio in biomarkers:
        ev = bio.get("evidence_span", "")
        if ev and ev not in raw_note:
            bad_biomarkers.append(bio.get("marker", "?"))

    if bad_biomarkers:
        issues.append(
            f"The following biomarkers have evidence_spans NOT found verbatim in the "
            f"raw note (hallucination detected): {bad_biomarkers}. "
            f"Re-extract and only output exact substrings from the note."
        )

    # ── Decision ──────────────────────────────────────────────────────────────
    if issues and retry_count < MAX_REEXTRACTION_RETRIES:
        feedback = (
            f"EVALUATION FAILED — {len(issues)} issue(s) found:\n"
            + "\n".join(f"  {i+1}. {issue}" for i, issue in enumerate(issues))
        )
        log.warning(
            "[EVALUATOR] Re-extraction triggered (retry %d/%d). Issues:\n%s",
            retry_count + 1, MAX_REEXTRACTION_RETRIES, feedback,
        )
        return {
            "current_step": "oncology_evaluator",
            "path_taken": ["oncology_evaluator"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "needs_reextraction": True,
            "eval_feedback": feedback,
            "retry_count": 1,  # Increment for the LangGraph Annotated reducer
        }
    else:
        if issues:
            log.warning(
                "[EVALUATOR] Max retries reached (%d). Proceeding with imperfect extraction.",
                MAX_REEXTRACTION_RETRIES,
            )
        else:
            log.info("[EVALUATOR] All checks passed. Proceeding to trial matching.")

        return {
            "current_step": "oncology_evaluator",
            "path_taken": ["oncology_evaluator"],
            "execution_time_ms": int((time.perf_counter() - start_time) * 1000),
            "needs_reextraction": False,
            "eval_feedback": None,
        }
