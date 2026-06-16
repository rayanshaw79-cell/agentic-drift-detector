"""
run.py — Agentic Drift Detector CLI entry point.

Usage:
    python run.py                        # Run one execution
    python run.py --simulate-batch 60    # Build a healthy baseline
    python run.py --bias                 # Inject artificial bias for testing
    python run.py --clear                # Wipe the telemetry database & queue
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
from telemetry.store import init_db, DB_PATH
from telemetry.queue import enqueue, flush_queue
from alerts.alert_manager import trigger_alert
from config.tenant import get_current_tenant

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Optional Rich ─────────────────────────────────────────────────────────────
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def create_initial_state() -> IncidentState:
    return {
        "incident_id":           str(uuid.uuid4())[:8],
        "incident_text":         "Auth service latency spike",
        "severity":              None,
        "investigation_summary": None,
        "decision":              None,
        "confidence":            None,
        "current_step":          "",
        "step_count":            0,
        "retry_count":           0,
        "path_taken":            [],
        "execution_time_ms":     0,
    }


def print_rich_summary(final_state: dict, analysis: dict, tenant_id: str) -> None:
    if not HAS_RICH:
        log.info("[FINAL STATE] %s", final_state)
        log.info("[DRIFT ANALYSIS] %s", analysis)
        return

    risk = analysis.get("risk_level", "healthy")
    score = analysis.get("drift_score", 0)
    risk_color = {"healthy": "green", "drift_detected": "yellow", "high_risk": "red"}.get(
        risk, "white"
    )

    table = Table(
        box=box.ROUNDED, show_header=True, header_style="bold cyan",
        border_style="dim", expand=False,
    )
    table.add_column("Field", style="dim", min_width=22)
    table.add_column("Value", style="bold white")

    table.add_row("Tenant",      f"[dim]{tenant_id}[/]")
    table.add_row("Incident ID", final_state.get("incident_id", "—"))
    table.add_row("Severity",    final_state.get("severity", "—"))
    table.add_row("Decision",    final_state.get("decision", "—"))
    table.add_row("Confidence",  f"{final_state.get('confidence', 0):.2f}")
    table.add_row("Retries",     str(final_state.get("retry_count", 0)))
    table.add_row("Latency",     f"{final_state.get('execution_time_ms', 0)} ms")
    table.add_row("Path",        " → ".join(final_state.get("path_taken", [])))
    table.add_row("Drift Score", f"[bold {risk_color}]{score}[/]")
    table.add_row("Risk Level",  f"[bold {risk_color}]{risk}[/]")

    if "intervention" in final_state.get("path_taken", []):
        table.add_row("Healing", "[bold yellow]⚡ Intervention triggered[/]")

    console.print()
    console.print(Panel(table, title="[bold cyan]Execution Summary[/]", border_style="cyan"))
    console.print()


def clear_database() -> None:
    """Wipe all telemetry records and flush the Redis queue."""
    # Flush Redis queue first
    flushed = flush_queue()
    if flushed:
        log.info("Flushed %d queued events from Redis.", flushed)

    # Clear active backend
    if os.getenv("DATABASE_URL"):
        _clear_postgres()
    else:
        _clear_sqlite()


def _clear_sqlite() -> None:
    if not os.path.exists(DB_PATH):
        log.warning("No SQLite database found at %s — nothing to clear.", DB_PATH)
        return
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM executions")
        conn.execute("VACUUM")
    log.info("✅ SQLite telemetry database cleared.")


def _clear_postgres() -> None:
    try:
        import psycopg2
        tenant_id = get_current_tenant()
        with psycopg2.connect(os.getenv("DATABASE_URL")) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM executions WHERE tenant_id = %s", (tenant_id,))
                deleted = cur.rowcount
        log.info("✅ Cleared %d PostgreSQL records for tenant '%s'.", deleted, tenant_id)
    except Exception as exc:
        log.error("Failed to clear PostgreSQL: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────

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
        help="Wipe the telemetry database and Redis queue, then exit.",
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
    tenant_id = get_current_tenant()

    # ── Bias mode ────────────────────────────────────────────────────────────
    os.environ["SIMULATE_BIAS"] = "true" if args.bias else "false"
    if args.bias:
        log.warning("SIMULATE_BIAS enabled — artificial bias is active (tenant=%s).", tenant_id)

    simulate_count = args.simulate_batch
    if simulate_count > 1:
        log.info(
            "Starting batch simulation: %d executions (tenant=%s) …",
            simulate_count, tenant_id,
        )

    # ── Run loop ─────────────────────────────────────────────────────────────
    final_state: dict | None = None
    analysis: dict | None = None

    for i in range(simulate_count):
        initial_state = create_initial_state()
        final_state = incident_triage_workflow(initial_state)
        analysis = analyze_workflow(final_state)

        # Enqueue telemetry (async via Redis, or sync fallback)
        queued = enqueue(final_state, analysis, tenant_id)

        if simulate_count > 1:
            mode = "queued" if queued else "saved"
            log.info(
                "Simulation %d/%d — drift_score=%d risk=%s [%s]",
                i + 1, simulate_count,
                analysis.get("drift_score", 0),
                analysis.get("risk_level", "?"),
                mode,
            )

    # ── Single-run rich summary ───────────────────────────────────────────────
    if simulate_count == 1 and final_state is not None and analysis is not None:
        print_rich_summary(final_state, analysis, tenant_id)

        if not args.no_alerts:
            trigger_alert(analysis, final_state)
        else:
            log.info("Alerts suppressed (--no-alerts).")
    elif simulate_count > 1:
        log.info(
            "Batch complete. "
            "Launch the dashboard: streamlit run dashboard.py"
        )


if __name__ == "__main__":
    main()
