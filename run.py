"""
Agentic Drift Detector — CLI entry point.

Usage:
    python run.py                        # Run one execution
    python run.py --simulate-batch 60    # Build a healthy baseline
    python run.py --bias                 # Inject artificial bias for testing
    python run.py --clear                # Wipe the telemetry database
    python run.py --no-alerts            # Suppress webhook alerts (CI mode)
"""

import argparse
import logging
import os
import sqlite3
import sys
import uuid
from dotenv import load_dotenv

from workflows.incident_triage import incident_triage_workflow
from schemas.incident_state import IncidentState
from drift.drift_detector import analyze_workflow
from telemetry.store import init_db, save_execution_state, DB_PATH
from alerts.alert_manager import trigger_alert

# ─── Logging Setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Optional Rich ────────────────────────────────────────────────────────────
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


# ─── Helpers ─────────────────────────────────────────────────────────────────

def create_initial_state() -> IncidentState:
    return {
        "incident_id": str(uuid.uuid4())[:8],
        "incident_text": "Auth service latency spike",
        "severity": None,
        "investigation_summary": None,
        "decision": None,
        "confidence": None,
        "current_step": "",
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    }


def print_rich_summary(final_state: dict, analysis: dict) -> None:
    """Render a styled summary table using rich (if available)."""
    if not HAS_RICH:
        log.info("[FINAL STATE] %s", final_state)
        log.info("[DRIFT ANALYSIS] %s", analysis)
        return

    risk = analysis.get("risk_level", "healthy")
    score = analysis.get("drift_score", 0)
    risk_color = {"healthy": "green", "drift_detected": "yellow", "high_risk": "red"}.get(risk, "white")

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan",
                  border_style="dim", expand=False)
    table.add_column("Field", style="dim", min_width=22)
    table.add_column("Value", style="bold white")

    table.add_row("Incident ID",   final_state.get("incident_id", "—"))
    table.add_row("Severity",      final_state.get("severity", "—"))
    table.add_row("Decision",      final_state.get("decision", "—"))
    table.add_row("Confidence",    f"{final_state.get('confidence', 0):.2f}")
    table.add_row("Retries",       str(final_state.get("retry_count", 0)))
    table.add_row("Latency",       f"{final_state.get('execution_time_ms', 0)} ms")
    table.add_row("Path",          " → ".join(final_state.get("path_taken", [])))
    table.add_row("Drift Score",   f"[bold {risk_color}]{score}[/]")
    table.add_row("Risk Level",    f"[bold {risk_color}]{risk}[/]")

    healing = "intervention" in final_state.get("path_taken", [])
    if healing:
        table.add_row("Healing", "[bold yellow]⚡ Intervention triggered[/]")

    console.print()
    console.print(Panel(table, title="[bold cyan]Execution Summary[/]", border_style="cyan"))
    console.print()


def clear_database() -> None:
    """Wipe all telemetry records from the database."""
    if not os.path.exists(DB_PATH):
        log.warning("No database found at %s — nothing to clear.", DB_PATH)
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM executions")
        conn.execute("VACUUM")
    log.info("✅ Telemetry database cleared successfully.")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Agentic Drift Detector — run or simulate the incident triage workflow.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--simulate-batch", metavar="N", type=int, default=1,
        help="Run N simulated executions to build a baseline (default: 1).",
    )
    parser.add_argument(
        "--bias", action="store_true",
        help="Inject artificial classification & escalation bias for drift testing.",
    )
    parser.add_argument(
        "--no-alerts", action="store_true",
        help="Suppress Slack / Discord webhook alerts (useful in CI pipelines).",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Wipe the telemetry database and exit.",
    )
    return parser


def main() -> None:
    load_dotenv()
    args = build_parser().parse_args()

    # ── Clear mode ───────────────────────────────────────────────────────────
    if args.clear:
        clear_database()
        sys.exit(0)

    init_db()

    # ── Bias mode ────────────────────────────────────────────────────────────
    os.environ["SIMULATE_BIAS"] = "true" if args.bias else "false"
    if args.bias:
        log.warning("SIMULATE_BIAS enabled — artificial classification/escalation bias is active.")

    simulate_count = args.simulate_batch
    if simulate_count > 1:
        log.info("Starting batch simulation: %d executions …", simulate_count)

    # ── Run loop ─────────────────────────────────────────────────────────────
    final_state: dict | None = None
    analysis: dict | None = None

    for i in range(simulate_count):
        initial_state = create_initial_state()
        final_state = incident_triage_workflow(initial_state)
        analysis = analyze_workflow(final_state)
        save_execution_state(final_state, analysis)

        if simulate_count > 1:
            log.info("Simulation %d/%d complete — drift_score=%d risk=%s",
                     i + 1, simulate_count,
                     analysis.get("drift_score", 0),
                     analysis.get("risk_level", "?"))

    # ── Single-run rich summary ───────────────────────────────────────────────
    if simulate_count == 1 and final_state is not None and analysis is not None:
        print_rich_summary(final_state, analysis)

        if not args.no_alerts:
            trigger_alert(analysis, final_state)
        else:
            log.info("Alerts suppressed (--no-alerts).")
    elif simulate_count > 1:
        log.info("Batch complete. Launch the dashboard: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
