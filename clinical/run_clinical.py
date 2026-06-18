"""
clinical/run_clinical.py — CLI entry point for the Clinical Coding Agent.

Usage:
    python -m clinical.run_clinical
    python -m clinical.run_clinical --note "Patient has T2DM and HTN"
    python -m clinical.run_clinical --simulate-batch 50
    python -m clinical.run_clinical --no-alerts
"""

import argparse
import logging
import os
import sys
import uuid

from dotenv import load_dotenv

from workflows.clinical_coding import clinical_coding_workflow
from schemas.clinical_state import ClinicalState
from drift.drift_detector import analyze_workflow
from telemetry.store import init_db
from telemetry.queue import enqueue
from config.tenant import get_current_tenant
from alerts.alert_manager import trigger_alert

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Sample notes for batch simulation ─────────────────────────────────────────
_SAMPLE_NOTES = [
    "Patient presents with essential hypertension (HTN) and is started on 5mg amlodipine.",
    "45-year-old with Type 2 diabetes mellitus (T2DM). HbA1c 8.2%. Continue Metformin.",
    "Chest pain on exertion. Rule out angina pectoris. ECG ordered.",
    "Chronic kidney disease stage 3. Monitor eGFR quarterly.",
    "Patient with atrial fibrillation on warfarin. INR within target range.",
    "New diagnosis of moderate persistent asthma. Initiate inhaled corticosteroids.",
    "Follow-up for major depressive disorder. Continue sertraline 100mg.",
    "Acute UTI — empirical trimethoprim course commenced.",
    "Screening visit — no active conditions identified. Routine bloods normal.",
    "Obesity (BMI 38). Diet and lifestyle counselling provided.",
]


def create_initial_state(raw_note: str) -> ClinicalState:
    return {
        "record_id":             str(uuid.uuid4())[:8],
        "raw_note":              raw_note,
        "extracted_diagnoses":   None,
        "ner_votes":             None,
        "icd10_codes":           None,
        "clinical_record":       None,
        "coding_status":         None,
        "claims_ready":          None,
        "meat_results":          None,
        "overall_confidence":    None,
        "current_step":          "",
        "step_count":            0,
        "retry_count":           0,
        "path_taken":            [],
        "execution_time_ms":     0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clinical/run_clinical.py",
        description="Agentic Drift Detector — Clinical Coding Agent CLI.",
    )
    parser.add_argument(
        "--note", metavar="TEXT", type=str, default=None,
        help="Run a single coding job on the given clinical note text.",
    )
    parser.add_argument(
        "--simulate-batch", metavar="N", type=int, default=1,
        help="Run N simulated executions across sample clinical notes (default: 1).",
    )
    parser.add_argument(
        "--no-alerts", action="store_true",
        help="Suppress Slack/Discord webhook alerts.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()
    init_db()
    tenant_id = get_current_tenant()

    simulate_count = args.simulate_batch
    single_note    = args.note

    if simulate_count > 1:
        log.info("Starting clinical batch simulation: %d runs (tenant=%s) …",
                 simulate_count, tenant_id)

    import itertools
    note_cycle = itertools.cycle(_SAMPLE_NOTES)

    final_state: dict | None = None
    analysis:    dict | None = None

    for i in range(simulate_count):
        if single_note:
            note = single_note
        else:
            note = next(note_cycle)

        initial_state = create_initial_state(note)
        final_state   = clinical_coding_workflow(initial_state)
        analysis      = analyze_workflow(final_state, workflow_type="clinical_coding")

        queued = enqueue(final_state, analysis, tenant_id)

        if simulate_count > 1:
            mode = "queued" if queued else "saved"
            log.info(
                "Run %d/%d — drift_score=%d risk=%s status=%s [%s]",
                i + 1, simulate_count,
                analysis.get("drift_score", 0),
                analysis.get("risk_level", "?"),
                final_state.get("coding_status", "?"),
                mode,
            )

    if simulate_count == 1 and not args.no_alerts and analysis:
        trigger_alert(analysis, {
            "incident_id": final_state.get("record_id", "—"),
            "decision":    final_state.get("coding_status", "—"),
            "retry_count": final_state.get("retry_count", 0),
            "severity":    "clinical_coding",
            "path_taken":  final_state.get("path_taken", []),
        })
    elif simulate_count > 1:
        log.info("Batch complete. Launch: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
