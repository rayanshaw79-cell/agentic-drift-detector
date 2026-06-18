"""
telemetry/sqlite_store.py — SQLite backend for single-tenant / local dev.

This is the original store implementation, preserved as an explicit backend.
Used automatically when DATABASE_URL is not set.
"""

import json
import logging
import os
import sqlite3

from schemas.incident_state import IncidentState

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "telemetry.db")

# ── SQL constants ─────────────────────────────────────────────────────────────

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS executions (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id       TEXT,
        severity          TEXT,
        decision          TEXT,
        confidence        REAL,
        step_count        INTEGER,
        retry_count       INTEGER,
        path_taken        TEXT,
        execution_time_ms INTEGER,
        drift_score       INTEGER DEFAULT 0,
        risk_level        TEXT    DEFAULT 'healthy',
        workflow_type     TEXT    DEFAULT 'incident_triage',
        overall_confidence REAL   DEFAULT NULL,
        created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

_MIGRATIONS = [
    ("drift_score",       "INTEGER DEFAULT 0"),
    ("risk_level",        "TEXT DEFAULT 'healthy'"),
    ("created_at",        "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ("workflow_type",     "TEXT DEFAULT 'incident_triage'"),
    ("overall_confidence", "REAL DEFAULT NULL"),
]

_INSERT = """
    INSERT INTO executions (
        incident_id, severity, decision, confidence,
        step_count, retry_count, path_taken, execution_time_ms,
        drift_score, risk_level, workflow_type, overall_confidence
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_BASELINE_QUERY = """
    SELECT
        AVG(step_count)  AS avg_steps,
        AVG(retry_count) AS avg_retries,
        AVG(execution_time_ms) AS avg_latency,
        AVG(CASE WHEN decision = 'escalate' THEN 1.0 ELSE 0.0 END) AS escalation_rate,
        AVG(CASE WHEN severity = 'high'     THEN 1.0 ELSE 0.0 END) AS high_severity_rate,
        AVG(CASE WHEN severity = 'low'
                 THEN CASE WHEN decision = 'escalate' THEN 1.0 ELSE 0.0 END
                 ELSE NULL END) AS low_severity_escalation_rate,
        AVG(overall_confidence) AS avg_coding_confidence
    FROM (
        SELECT step_count, retry_count, execution_time_ms, decision, severity,
               overall_confidence
        FROM executions
        ORDER BY id DESC
        LIMIT ?
    )
"""

_DEFAULT_BASELINE = {
    "avg_steps":                   4.0,
    "avg_retries":                 0.0,
    "avg_latency":                 100.0,
    "escalation_rate":             0.2,
    "high_severity_rate":          0.2,
    "low_severity_escalation_rate": 0.05,
    "avg_coding_confidence":       0.75,
    "avg_unresolved_rate":         0.05,
}


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the executions table and apply pending column migrations."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_TABLE)
        for col, definition in _MIGRATIONS:
            try:
                conn.execute(f"ALTER TABLE executions ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.commit()
    log.debug("SQLite database initialised at %s", DB_PATH)


def save_execution_state(
    state,
    analysis: dict | None = None,
    *,
    tenant_id: str | None = None,  # accepted but ignored in SQLite mode
) -> None:
    """Persist a completed workflow execution to SQLite."""
    # Support both incident_triage states (incident_id) and clinical states (record_id)
    record_key = state.get("incident_id") or state.get("record_id")
    values = (
        record_key,
        state.get("severity"),
        state.get("decision") or state.get("coding_status"),
        state.get("confidence") or state.get("overall_confidence"),
        state.get("step_count", 0),
        state.get("retry_count", 0),
        json.dumps(state.get("path_taken", [])),
        state.get("execution_time_ms", 0),
        analysis.get("drift_score", 0) if analysis else 0,
        analysis.get("risk_level", "healthy") if analysis else "healthy",
        analysis.get("workflow_type", "incident_triage") if analysis else "incident_triage",
        state.get("overall_confidence"),
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_INSERT, values)
        conn.commit()
    log.debug("SQLite: saved execution %s", record_key)


def get_historical_metrics(
    limit: int = 100,
    *,
    tenant_id: str | None = None,  # ignored in SQLite mode
) -> dict:
    """Return population-level baseline metrics from the last *limit* executions."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(_BASELINE_QUERY, (limit,)).fetchone()

    if not row or row["avg_steps"] is None:
        log.debug("SQLite: empty database — using default baseline.")
        return dict(_DEFAULT_BASELINE)

    return {
        "avg_steps":                    row["avg_steps"],
        "avg_retries":                  row["avg_retries"],
        "avg_latency":                  row["avg_latency"],
        "escalation_rate":              row["escalation_rate"]              or 0.2,
        "high_severity_rate":           row["high_severity_rate"]           or 0.2,
        "low_severity_escalation_rate": row["low_severity_escalation_rate"] or 0.05,
        "avg_coding_confidence":        row["avg_coding_confidence"]        or 0.75,
        "avg_unresolved_rate":          0.05,  # computed separately if needed
    }
