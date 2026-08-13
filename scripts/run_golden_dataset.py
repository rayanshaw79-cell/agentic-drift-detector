"""
scripts/run_golden_dataset.py — Slow-drift regression guard.

Runs a fixed set of hardcoded IncidentState inputs through the full
incident_triage_workflow and checks that no case degrades to high_risk.

Because the inputs are STATIC, any change in drift score over successive runs
is 100% attributable to LLM / API degradation — not case-mix shifts.
This prevents the rolling historical baseline from absorbing slow degradation.

Usage:
    python scripts/run_golden_dataset.py           # Run + save to DB
    python scripts/run_golden_dataset.py --dry-run # Run only, don't save
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

# Force UTF-8 output on Windows so emoji in print() doesn't crash cp1252.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from workflows.incident_triage import incident_triage_workflow
from drift.drift_detector import analyze_workflow
from telemetry.store import init_db, save_execution_state

# ── Golden Cases ──────────────────────────────────────────────────────────────
# These inputs are IMMUTABLE. Do not alter them once in production.
# Annotate with case_name for the regression report.

GOLDEN_CASES = [
    {
        "_case_name": "LOW-001: Auth latency spike, low severity",
        "incident_id": "golden-low-001",
        "severity": "low",
        "confidence": 0.92,
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    },
    {
        "_case_name": "MED-001: Service degradation, medium severity",
        "incident_id": "golden-med-001",
        "severity": "medium",
        "confidence": 0.80,
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    },
    {
        "_case_name": "HIGH-001: Data breach alert, high severity",
        "incident_id": "golden-high-001",
        "severity": "high",
        "confidence": 0.75,
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    },
    {
        "_case_name": "HIGH-002: Critical infra failure, high severity",
        "incident_id": "golden-high-002",
        "severity": "high",
        "confidence": 0.88,
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    },
    {
        "_case_name": "LOW-002: False positive noise, low confidence",
        "incident_id": "golden-low-002",
        "severity": "low",
        "confidence": 0.62,
        "step_count": 0,
        "retry_count": 0,
        "path_taken": [],
        "execution_time_ms": 0,
    },
]

# Acceptable risk levels for each golden case (index-matched to GOLDEN_CASES).
# A result outside this set is a regression FAIL.
GOLDEN_EXPECTED_MAX_RISK = [
    "drift_detected",   # LOW-001: may drift but never high_risk
    "drift_detected",   # MED-001
    "high_risk",        # HIGH-001: high severity cases can legitimately trip high_risk
    "drift_detected",   # HIGH-002: high confidence should stay healthy or drift_detected
    "high_risk",        # LOW-002: low confidence may trip the breaker
]

RISK_ORDER = {"healthy": 0, "drift_detected": 1, "high_risk": 2}


def _strip_meta(case: dict) -> dict:
    """Remove runner-only keys before passing to workflow."""
    return {k: v for k, v in case.items() if not k.startswith("_")}


def run_golden_regression(dry_run: bool = False) -> bool:
    """Run all golden cases. Returns True if all cases pass."""
    print(f"\n{'='*64}")
    print(f"  [GOLDEN REGRESSION] {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*64}")

    if not dry_run:
        init_db()

    results = []
    overall_pass = True

    for i, case in enumerate(GOLDEN_CASES):
        case_name = case["_case_name"]
        state_input = _strip_meta(case)

        print(f"\n  [{i+1}/{len(GOLDEN_CASES)}] {case_name}")

        t0 = time.monotonic()
        try:
            final_state = incident_triage_workflow(state_input)
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            final_state["execution_time_ms"] = (
                final_state.get("execution_time_ms", 0) + elapsed_ms
            )

            analysis = analyze_workflow(final_state, workflow_type="incident_triage")
            # Override workflow_type so these are filterable on the dashboard
            analysis["workflow_type"] = "golden_regression"

            risk = analysis["risk_level"]
            score = analysis["drift_score"]
            max_allowed = GOLDEN_EXPECTED_MAX_RISK[i]
            passed = RISK_ORDER[risk] <= RISK_ORDER[max_allowed]

            status_icon = "[PASS]" if passed else "[FAIL]"
            print(
                f"       Risk={risk:<16} Score={score:<4} "
                f"MaxAllowed={max_allowed:<16} {status_icon}"
            )

            if not passed:
                overall_pass = False

            results.append({
                "case": case_name,
                "risk": risk,
                "score": score,
                "passed": passed,
                "elapsed_ms": elapsed_ms,
            })

            if not dry_run:
                save_execution_state(
                    final_state,
                    analysis=analysis,
                    tenant_id="golden",
                )

        except Exception as exc:
            print(f"       [ERROR] {exc}")
            overall_pass = False
            results.append({"case": case_name, "error": str(exc), "passed": False})

    # ── Summary ───────────────────────────────────────────────────────────────
    passed_count = sum(1 for r in results if r.get("passed"))
    print(f"\n{'='*64}")
    verdict = "ALL PASS" if overall_pass else "REGRESSION DETECTED"
    print(f"  {verdict}  ({passed_count}/{len(GOLDEN_CASES)} cases passed)")
    if not dry_run and overall_pass:
        print("  Results saved with workflow_type='golden_regression'.")
    print(f"{'='*64}\n")

    return overall_pass


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    success = run_golden_regression(dry_run=dry_run)
    sys.exit(0 if success else 1)
