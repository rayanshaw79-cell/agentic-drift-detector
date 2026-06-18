"""
clinical/steps/clinical_output_step.py — Final output / notification node.

Prints a rich summary and writes telemetry.  Does NOT touch the medical
records database — the clinical_record dict is the output artefact.
"""

import json
import logging
import time

from schemas.clinical_state import ClinicalState

log = logging.getLogger(__name__)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    console = Console()
    HAS_RICH = True
except ImportError:
    console = None
    HAS_RICH = False


def clinical_output_step(state: ClinicalState) -> dict:
    """
    LangGraph node — Clinical Output / Notification.

    Reads:  state (all fields)
    Writes: step metadata only (all business fields already set)
    """
    start = time.perf_counter()

    record    = state.get("clinical_record") or {}
    status    = state.get("coding_status", "unknown")
    confidence = state.get("overall_confidence", 0.0)
    codes     = state.get("icd10_codes") or []

    if HAS_RICH:
        status_colour = "green" if status == "complete" else "red"
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                      border_style="dim", expand=False)
        table.add_column("Field",  style="dim", min_width=22)
        table.add_column("Value",  style="bold white")

        table.add_row("Record ID",   record.get("record_id", "—"))
        table.add_row("Status",      f"[bold {status_colour}]{status}[/]")
        table.add_row("Confidence",  f"{confidence:.2f}")
        table.add_row("Codes Found", str(len(codes)))

        for i, c in enumerate(codes, 1):
            table.add_row(
                f"  ICD-10 [{i}]",
                f"[bold]{c.get('code','?')}[/] — {c.get('description','?')} "
                f"(conf: {c.get('confidence', 0):.2f})",
            )

        if state.get("path_taken"):
            table.add_row("Path", " → ".join(state["path_taken"]))

        console.print()
        console.print(Panel(table, title="[bold cyan]Clinical Coding Result[/]",
                            border_style="cyan"))
        console.print()
    else:
        log.info("[CLINICAL OUTPUT] Status: %s | Confidence: %.2f | Codes: %d",
                 status, confidence, len(codes))
        log.info("[CLINICAL OUTPUT] Record: %s", json.dumps(record, indent=2))

    latency = int((time.perf_counter() - start) * 1000) + 10

    return {
        "current_step":    "clinical_output",
        "step_count":      1,
        "path_taken":      ["clinical_output"],
        "execution_time_ms": latency,
    }
