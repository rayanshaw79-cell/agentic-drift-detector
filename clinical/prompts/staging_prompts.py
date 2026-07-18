"""
clinical/prompts/staging_prompts.py — Few-shot prompt templates for oncology staging.

Covers TNM extraction from:
  - Pathology reports  (explicit T/N/M notation)
  - Radiology notes    (descriptive staging implied from imaging)
  - Progress notes     (stage documented by oncologist in narrative)

Each example enforces the PROVENANCE RULE: every extracted value must be
an exact substring of the original clinical note.
"""

from typing import Optional


# ── System Prompt ─────────────────────────────────────────────────────────────

STAGING_SYSTEM_PROMPT = """\
You are a specialized Oncology Staging Agent trained to extract cancer staging
information from clinical documents with zero hallucination.

Your ONLY job is to identify:
  1. primary_site — the anatomical cancer origin
  2. histology    — the cell type / histological classification
  3. tnm_stage    — AJCC 8th edition T/N/M components and the overall stage group

CRITICAL PROVENANCE RULE:
  - Every value you output MUST be an exact verbatim substring copied from
    the input clinical note.
  - Do NOT infer, rephrase, or synthesize. If a value cannot be found as an
    exact string in the note, output null.
  - The evidence_span field for tnm_stage must also be an exact quote.

HALLUCINATION PREVENTION:
  - Do NOT output a stage if there is no explicit staging statement.
  - "History of lung cancer" alone does NOT give you a stage.
  - Look for explicit T/N/M notation OR an explicit "Stage IIA / Stage IV" statement.
  - If only partial staging is available (e.g., only T2 is stated), populate what
    you can find and leave the rest null.

Output ONLY a single JSON object. No explanation text. No markdown fences.
Schema:
{
  "primary_site":  "exact text snippet | null",
  "histology":     "exact text snippet | null",
  "tnm_stage": {
    "T":            "T-component string | null",
    "N":            "N-component string | null",
    "M":            "M-component string | null",
    "overall":      "stage group string | null",
    "evidence_span":"exact quoted sentence from note | null"
  }
}
"""

# ── Few-Shot Examples ─────────────────────────────────────────────────────────
# Format: list of (user_message, expected_assistant_output) pairs.
# These are injected as alternating HumanMessage / AIMessage before the real note.

STAGING_FEW_SHOT_EXAMPLES = [
    # ── Example 1: Explicit pathology report, NSCLC ──────────────────────────
    (
        """\
PATHOLOGY REPORT
Patient: [REDACTED]
Specimen: Right upper lobe lobectomy

DIAGNOSIS:
Right upper lobe, lung: Invasive adenocarcinoma, acinar predominant.
Tumor size: 3.2 cm greatest dimension.
Pleural invasion: Present (PL1).
Lymph nodes: 2 of 14 lymph nodes positive for metastatic carcinoma
(levels 4R and 7).
Distant metastasis: None identified on pre-operative imaging.

PATHOLOGIC STAGING (AJCC 8th Edition):
pT2a pN1 pM0 — Stage IIB Non-Small Cell Lung Cancer
""",
        """\
{
  "primary_site": "Right upper lobe, lung",
  "histology": "Invasive adenocarcinoma, acinar predominant",
  "tnm_stage": {
    "T": "pT2a",
    "N": "pN1",
    "M": "pM0",
    "overall": "Stage IIB Non-Small Cell Lung Cancer",
    "evidence_span": "pT2a pN1 pM0 — Stage IIB Non-Small Cell Lung Cancer"
  }
}""",
    ),
    # ── Example 2: Radiology note, no explicit TNM — implied Stage IV ─────────
    (
        """\
CT CHEST/ABDOMEN/PELVIS WITH CONTRAST — 2024-11-03

FINDINGS:
Lungs: 4.1 cm spiculated mass in the right upper lobe, highly suspicious
for primary lung malignancy. Multiple bilateral pulmonary nodules, largest
measuring 1.2 cm in the left lower lobe, consistent with intrapulmonary
metastases.

Mediastinum: Multiple enlarged mediastinal lymph nodes at stations 4R and 7,
the largest measuring 2.3 cm in short axis, suspicious for nodal involvement.

Liver: Three hypodense lesions in segments V and VI, largest 2.8 cm,
consistent with hepatic metastases.

Adrenal glands: No adrenal lesions.

IMPRESSION:
Findings are consistent with advanced right upper lobe lung cancer
with bilateral pulmonary, mediastinal nodal, and hepatic metastatic disease.
Clinical staging: cT3 cN2 cM1b.
""",
        """\
{
  "primary_site": "right upper lobe lung",
  "histology": null,
  "tnm_stage": {
    "T": "cT3",
    "N": "cN2",
    "M": "cM1b",
    "overall": null,
    "evidence_span": "Clinical staging: cT3 cN2 cM1b."
  }
}""",
    ),
    # ── Example 3: Progress note — stage mentioned narratively ────────────────
    (
        """\
ONCOLOGY FOLLOW-UP NOTE — 2025-01-15
Dr. [REDACTED]

HISTORY OF PRESENT ILLNESS:
Ms. [REDACTED] is a 58-year-old woman with a known diagnosis of Stage IIIA
Her2-positive invasive ductal carcinoma of the left breast, diagnosed in
October 2024. She is currently on cycle 4 of neoadjuvant dose-dense
AC-T chemotherapy. She reports mild fatigue and nausea, otherwise tolerating
treatment well.

PLAN:
Continue neoadjuvant chemotherapy. Repeat staging CT after cycle 6.
""",
        """\
{
  "primary_site": "left breast",
  "histology": "Her2-positive invasive ductal carcinoma",
  "tnm_stage": {
    "T": null,
    "N": null,
    "M": null,
    "overall": "Stage IIIA",
    "evidence_span": "Stage IIIA\\nHer2-positive invasive ductal carcinoma of the left breast"
  }
}""",
    ),
    # ── Example 4: Note with ONLY a history mention — no staging extractable ──
    (
        """\
EMERGENCY DEPARTMENT NOTE — 2025-03-02

CHIEF COMPLAINT: Shortness of breath.

PAST MEDICAL HISTORY:
- History of lung cancer (treated, presumed cured, last oncology visit 2021)
- Hypertension
- Type 2 Diabetes Mellitus

ASSESSMENT:
Acute pulmonary embolism. No evidence of recurrent malignancy on today's CT.
""",
        """\
{
  "primary_site": null,
  "histology": null,
  "tnm_stage": {
    "T": null,
    "N": null,
    "M": null,
    "overall": null,
    "evidence_span": null
  }
}""",
    ),
    # ── Example 5: Colorectal — surgical pathology with explicit staging ───────
    (
        """\
SURGICAL PATHOLOGY REPORT

PROCEDURE: Right hemicolectomy with en bloc resection.
SPECIMEN: Ascending colon, 22 cm segment.

MICROSCOPIC DESCRIPTION:
Moderately differentiated adenocarcinoma of the ascending colon.
The tumor invades through the muscularis propria into pericolorectal
tissues (T3). Seventeen lymph nodes examined: 4 positive for
metastatic adenocarcinoma (N2a). Liver biopsy: negative for carcinoma (M0).

FINAL PATHOLOGIC STAGE: pT3 N2a M0 — Stage IIIC Colon Adenocarcinoma
(AJCC 8th Edition)
""",
        """\
{
  "primary_site": "ascending colon",
  "histology": "Moderately differentiated adenocarcinoma",
  "tnm_stage": {
    "T": "pT3",
    "N": "N2a",
    "M": "M0",
    "overall": "Stage IIIC Colon Adenocarcinoma",
    "evidence_span": "FINAL PATHOLOGIC STAGE: pT3 N2a M0 — Stage IIIC Colon Adenocarcinoma"
  }
}""",
    ),
]


def build_staging_prompt(
    document_type: Optional[str] = None,
    rag_context: Optional[str] = None,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Returns (system_prompt, few_shot_examples) for the staging LLM call.

    Args:
        document_type: From oncology_router — influences which examples are
                       prepended first (most-relevant first).
        rag_context:   Retrieved NCCN/NCI staging rules to inject into system
                       prompt for grounding (Pillar 2).

    Returns:
        (system_prompt_str, [(user_msg, assistant_msg), ...])
    """
    system = STAGING_SYSTEM_PROMPT

    if rag_context:
        system = (
            f"AUTHORITATIVE STAGING REFERENCE (from NCI guidelines):\n"
            f"{rag_context}\n\n"
            f"Use this reference to validate your extractions.\n\n"
            + system
        )

    # Re-order examples to put the most-relevant document type first
    examples = list(STAGING_FEW_SHOT_EXAMPLES)
    if document_type == "radiology":
        # Move the radiology example (index 1) to the front
        examples.insert(0, examples.pop(1))
    elif document_type == "progress_note":
        examples.insert(0, examples.pop(2))

    return system, examples
