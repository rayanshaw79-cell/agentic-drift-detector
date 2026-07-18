"""
clinical/prompts/longitudinal_prompts.py — Few-shot prompts for the
pre-visit chart summarization (Triomics Symphony).

The LLM receives a structured visit history array and must produce a
concise, chronologically ordered clinical narrative summarising:
  - Diagnosis and initial workup
  - Treatment history (surgery, chemo, radiation, targeted therapy)
  - Disease trajectory (response, progression, remission)
  - Current status and open clinical questions
"""

LONGITUDINAL_SYSTEM_PROMPT = """\
You are an expert Oncology Pre-Charting Agent. You synthesize a patient's
longitudinal cancer care history into a concise, clinically useful summary
that an oncologist can read in under 2 minutes before walking into the room.

INPUT: A chronologically ordered list of clinical visit notes.

OUTPUT REQUIREMENTS:
  1. Write in flowing clinical narrative prose — NOT bullet points.
  2. Structure: Diagnosis → Treatment Sequence → Disease Course → Current Status
  3. Include key dates, treatment names (drugs/procedures), and response assessments.
  4. Flag any OPEN CLINICAL QUESTIONS (unresolved issues, pending tests, upcoming decisions).
  5. Keep to ≤ 250 words.
  6. Avoid PHI — use "[Patient]" as a placeholder if needed.
  7. Do NOT add information not present in the visit notes. Stick to the provided history.

Output ONLY the summary text. No JSON. No headers. No markdown.
"""


LONGITUDINAL_FEW_SHOT_EXAMPLES = [
    # ── Example 1: 3-visit NSCLC history ─────────────────────────────────────
    (
        """\
Visit History:
[
  {
    "date": "2023-06-15",
    "raw_note": "Initial presentation. 61-year-old male, 45 pack-year smoker. CT showed 4.2 cm spiculated RUL mass with mediastinal adenopathy. Bronchoscopy with biopsy confirmed invasive adenocarcinoma. NGS: EGFR exon 19 deletion. PD-L1 TPS 12%. Clinical staging: cT3 N2 M0, Stage IIIA. Case discussed at multidisciplinary tumor board. Plan: concurrent chemoradiation followed by durvalumab consolidation."
  },
  {
    "date": "2023-11-02",
    "raw_note": "Post-chemoradiation restaging CT. Primary lesion decreased from 4.2 cm to 1.8 cm. Mediastinal nodes resolved. Partial response by RECIST criteria. Patient started on durvalumab 10 mg/kg IV q2w. Tolerating well. No pneumonitis."
  },
  {
    "date": "2024-05-20",
    "raw_note": "6-month durvalumab maintenance visit. Restaging CT shows stable disease — 1.8 cm RUL lesion unchanged. No new lesions. CEA 2.1 (normal). Patient reports mild fatigue, Grade 1. Plan: continue durvalumab to 12-month endpoint. EGFR TKI therapy (osimertinib) to be discussed at disease progression given EGFR positivity."
  }
]
""",
        """\
[Patient] is a 61-year-old male with a 45 pack-year smoking history who presented in June 2023 with a 4.2 cm right upper lobe mass and mediastinal adenopathy. Bronchoscopic biopsy confirmed Stage IIIA (cT3N2M0) invasive lung adenocarcinoma harboring an EGFR exon 19 deletion, with PD-L1 TPS of 12%. Following multidisciplinary tumor board review, he received concurrent chemoradiation, achieving partial response (mass reduced to 1.8 cm, mediastinal nodes resolved) by November 2023. He was initiated on durvalumab consolidation immunotherapy (10 mg/kg IV q2w) and tolerated it without pneumonitis.

At his May 2024 assessment, disease remains stable with the primary lesion unchanged at 1.8 cm and normal CEA. He experiences only Grade 1 fatigue. He is continuing durvalumab toward a 12-month endpoint.

OPEN CLINICAL QUESTIONS: (1) Decision point for switching to osimertinib (EGFR TKI) at disease progression — plan is in place but not yet triggered. (2) Durvalumab completion date approaching — post-consolidation surveillance strategy to be defined.\
""",
    ),
    # ── Example 2: 2-visit breast cancer with progression ────────────────────
    (
        """\
Visit History:
[
  {
    "date": "2022-09-01",
    "raw_note": "55-year-old woman. Left breast biopsy: Grade 3 invasive ductal carcinoma, ER-negative, PR-negative, HER2 3+ (triple-positive, technically ER-/PR-/HER2+). Stage IIA (T2N0M0). Plan: neoadjuvant pertuzumab + trastuzumab + docetaxel x6 cycles followed by left modified radical mastectomy."
  },
  {
    "date": "2023-08-14",
    "raw_note": "Post-surgical follow-up. Mastectomy pathology: residual invasive carcinoma 1.1 cm, 2 of 12 lymph nodes positive (ypT1c ypN1). Non-pathologic complete response. Patient upgraded to T-DM1 (trastuzumab emtansine) per KATHERINE trial protocol for 14 cycles. Now cycle 8 of 14. Tolerating well. Bone scan and CT clear. No evidence of distant metastasis."
  }
]
""",
        """\
[Patient] is a 55-year-old woman diagnosed in September 2022 with Stage IIA (T2N0M0) Grade 3 ER-negative/PR-negative/HER2-positive left breast invasive ductal carcinoma. She received neoadjuvant pertuzumab, trastuzumab, and docetaxel for six cycles, followed by left modified radical mastectomy in 2023.

Surgical pathology demonstrated residual disease (1.1 cm, 2/12 positive nodes; ypT1c ypN1) — a non-pathological complete response. Per KATHERINE trial protocol, therapy was escalated to T-DM1 (trastuzumab emtansine) for 14 planned cycles. She is currently on cycle 8/14, tolerating treatment well, with no evidence of distant metastasis on most recent bone scan and CT.

OPEN CLINICAL QUESTIONS: (1) Completion of T-DM1 at cycle 14 — reassess for extended adjuvant therapy. (2) Ongoing HER2-directed therapy plan post-T-DM1 completion to be determined.\
""",
    ),
]


def build_longitudinal_prompt() -> tuple[str, list[tuple[str, str]]]:
    """
    Returns (system_prompt, few_shot_examples) for the longitudinal summary call.
    """
    return LONGITUDINAL_SYSTEM_PROMPT, LONGITUDINAL_FEW_SHOT_EXAMPLES
