"""
clinical/prompts/biomarker_prompts.py — Few-shot prompts for molecular/genetic
biomarker extraction.

Covers common oncology biomarkers:
  - EGFR, ALK, ROS1, KRAS, BRAF (lung / colorectal)
  - HER2/ERBB2, ER, PR (breast)
  - PD-L1 / TPS score (immunotherapy eligibility)
  - MSI-H / dMMR (colorectal / pan-cancer immunotherapy)
  - BRCA1/2 (breast / ovarian)
"""

from typing import Optional


BIOMARKER_SYSTEM_PROMPT = """\
You are a specialized Oncology Molecular Pathology Agent trained to extract
biomarker and genetic marker results from clinical notes and pathology reports.

Extract ALL molecular markers mentioned, including:
  - Mutation status (e.g., EGFR exon 19 deletion, KRAS G12C)
  - Amplification / overexpression (e.g., HER2 3+ by IHC)
  - Expression levels (e.g., PD-L1 TPS 78%)
  - Instability markers (e.g., MSI-High, dMMR)
  - Germline/somatic BRCA status

For each marker output:
  - marker        : standardized gene/marker name (e.g., "EGFR", "PD-L1", "HER2")
  - status        : one of Positive | Negative | Mutated | Wild-type | Amplified |
                    Overexpressed | High | Low | Unknown | Equivocal
  - value         : numeric value if present (e.g., "72%", "3+") — otherwise null
  - evidence_span : EXACT verbatim substring from the note proving this result

CRITICAL PROVENANCE RULE:
  - evidence_span MUST be an exact verbatim copy from the input text.
  - Do NOT rephrase. If you cannot find the exact string, omit that biomarker.
  - "Negative for EGFR mutation" and "EGFR wild type" are valid evidence spans.

Output ONLY a JSON array. No explanation. No markdown fences.
Schema:
[
  {
    "marker":        "string",
    "status":        "string",
    "value":         "string | null",
    "evidence_span": "exact text snippet"
  }
]
If no biomarkers are found, output: []
"""


BIOMARKER_FEW_SHOT_EXAMPLES = [
    # ── Example 1: NSCLC genomic report — multiple markers ───────────────────
    (
        """\
MOLECULAR PATHOLOGY REPORT — NEXT GENERATION SEQUENCING PANEL

Patient specimen: Right upper lobe lung biopsy, formalin-fixed paraffin-embedded.

RESULTS:
EGFR: Exon 19 deletion detected (p.E746_A750del). Pathogenic.
ALK: No rearrangement detected by FISH. ALK-negative.
ROS1: No rearrangement detected. ROS1-negative.
KRAS: Wild type. No mutations detected in codons 12, 13, or 61.
PD-L1 (22C3 antibody): Tumor Proportion Score (TPS) = 78%. High expressors (TPS ≥ 50%).
TMB: 8 mutations/Mb (intermediate).

INTERPRETATION:
EGFR exon 19 deletion-positive NSCLC. Patient is a candidate for first-line
EGFR tyrosine kinase inhibitor therapy (e.g., osimertinib).
""",
        """\
[
  {
    "marker": "EGFR",
    "status": "Mutated",
    "value": null,
    "evidence_span": "EGFR: Exon 19 deletion detected (p.E746_A750del). Pathogenic."
  },
  {
    "marker": "ALK",
    "status": "Negative",
    "value": null,
    "evidence_span": "ALK: No rearrangement detected by FISH. ALK-negative."
  },
  {
    "marker": "ROS1",
    "status": "Negative",
    "value": null,
    "evidence_span": "ROS1: No rearrangement detected. ROS1-negative."
  },
  {
    "marker": "KRAS",
    "status": "Wild-type",
    "value": null,
    "evidence_span": "KRAS: Wild type. No mutations detected in codons 12, 13, or 61."
  },
  {
    "marker": "PD-L1",
    "status": "High",
    "value": "78%",
    "evidence_span": "PD-L1 (22C3 antibody): Tumor Proportion Score (TPS) = 78%. High expressors (TPS ≥ 50%)."
  }
]""",
    ),
    # ── Example 2: Breast cancer — HER2, ER, PR ──────────────────────────────
    (
        """\
BREAST PATHOLOGY REPORT

Core needle biopsy, left breast 10 o'clock position.

IMMUNOHISTOCHEMISTRY:
ER (Estrogen Receptor): Positive, Allred score 7/8 (strong, 90% of cells).
PR (Progesterone Receptor): Positive, Allred score 5/8 (moderate, 60% of cells).
HER2 (4B5 antibody): 3+ (strong complete membrane staining in >10% of cells).
HER2 FISH: HER2 amplified. HER2/CEP17 ratio = 4.2.
Ki-67 proliferation index: 42%.
""",
        """\
[
  {
    "marker": "ER",
    "status": "Positive",
    "value": "Allred score 7/8",
    "evidence_span": "ER (Estrogen Receptor): Positive, Allred score 7/8 (strong, 90% of cells)."
  },
  {
    "marker": "PR",
    "status": "Positive",
    "value": "Allred score 5/8",
    "evidence_span": "PR (Progesterone Receptor): Positive, Allred score 5/8 (moderate, 60% of cells)."
  },
  {
    "marker": "HER2",
    "status": "Amplified",
    "value": "3+",
    "evidence_span": "HER2 (4B5 antibody): 3+ (strong complete membrane staining in >10% of cells)."
  },
  {
    "marker": "Ki-67",
    "status": "High",
    "value": "42%",
    "evidence_span": "Ki-67 proliferation index: 42%."
  }
]""",
    ),
    # ── Example 3: Colorectal — KRAS mutated, MSI-H ──────────────────────────
    (
        """\
COLORECTAL CANCER MOLECULAR PROFILE

Specimen: Ascending colon adenocarcinoma, resection.

Mutation Analysis (Sanger sequencing + NGS):
  KRAS codon 12: G12D mutation detected (c.35G>A, p.G12D).
  NRAS: Wild type.
  BRAF V600E: Not detected.

Mismatch Repair (MMR) Immunohistochemistry:
  MLH1: Loss of nuclear expression.
  PMS2: Loss of nuclear expression.
  MSH2: Retained.
  MSH6: Retained.
  Overall: Mismatch repair deficient (dMMR) / Microsatellite instability-high (MSI-H).
""",
        """\
[
  {
    "marker": "KRAS",
    "status": "Mutated",
    "value": "G12D",
    "evidence_span": "KRAS codon 12: G12D mutation detected (c.35G>A, p.G12D)."
  },
  {
    "marker": "NRAS",
    "status": "Wild-type",
    "value": null,
    "evidence_span": "NRAS: Wild type."
  },
  {
    "marker": "BRAF",
    "status": "Negative",
    "value": null,
    "evidence_span": "BRAF V600E: Not detected."
  },
  {
    "marker": "MSI",
    "status": "High",
    "value": null,
    "evidence_span": "Mismatch repair deficient (dMMR) / Microsatellite instability-high (MSI-H)."
  }
]""",
    ),
    # ── Example 4: Progress note — only narrative mention, no lab report ──────
    (
        """\
ONCOLOGY CLINIC NOTE — 2025-02-10

Patient is a 62-year-old male with Stage IV EGFR wild-type, ALK-negative
non-small cell lung cancer on second-line carboplatin/pemetrexed.
PD-L1 was previously tested and was less than 1%.
BRCA testing was not performed as it is not indicated for lung cancer.
""",
        """\
[
  {
    "marker": "EGFR",
    "status": "Wild-type",
    "value": null,
    "evidence_span": "EGFR wild-type, ALK-negative"
  },
  {
    "marker": "ALK",
    "status": "Negative",
    "value": null,
    "evidence_span": "EGFR wild-type, ALK-negative"
  },
  {
    "marker": "PD-L1",
    "status": "Low",
    "value": "<1%",
    "evidence_span": "PD-L1 was previously tested and was less than 1%."
  }
]""",
    ),
    # ── Example 5: No biomarkers mentioned at all ─────────────────────────────
    (
        """\
EMERGENCY DEPARTMENT NOTE

Patient presents with acute back pain. Past medical history of hypertension,
type 2 diabetes, and remote history of breast cancer (treated 2018, in remission).
No new oncological workup performed today.
""",
        "[]",
    ),
]


def build_biomarker_prompt(
    document_type: Optional[str] = None,
) -> tuple[str, list[tuple[str, str]]]:
    """
    Returns (system_prompt, few_shot_examples) for the biomarker LLM call.

    Args:
        document_type: From oncology_router — re-orders examples so the
                       most-relevant type is first.
    Returns:
        (system_prompt_str, [(user_msg, assistant_msg), ...])
    """
    examples = list(BIOMARKER_FEW_SHOT_EXAMPLES)

    if document_type == "breast_pathology":
        # Promote breast cancer example (index 1) to first position
        examples.insert(0, examples.pop(1))
    elif document_type == "progress_note":
        examples.insert(0, examples.pop(3))

    return BIOMARKER_SYSTEM_PROMPT, examples
