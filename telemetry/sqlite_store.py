"""
telemetry/sqlite_store.py — SQLite backend for single-tenant / local dev.

This is the original store implementation, preserved as an explicit backend.
Used automatically when DATABASE_URL is not set.
"""

import json
import logging
import os
import sqlite3


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
        privacy_leak_risk REAL    DEFAULT 0.0,
        created_at        DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

_CREATE_HITL_TABLE = """
    CREATE TABLE IF NOT EXISTS human_interventions (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id         TEXT NOT NULL,
        action              TEXT NOT NULL,
        reviewed_by         TEXT DEFAULT 'clinician',
        notes               TEXT DEFAULT '',
        original_codes_json TEXT,
        final_codes_json    TEXT,
        created_at          DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

_MIGRATIONS = [
    ("drift_score",          "INTEGER DEFAULT 0"),
    ("risk_level",           "TEXT DEFAULT 'healthy'"),
    ("created_at",           "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ("workflow_type",        "TEXT DEFAULT 'incident_triage'"),
    ("overall_confidence",    "REAL DEFAULT NULL"),
    ("unresolved_count",      "INTEGER DEFAULT 0"),
    ("total_entities",        "INTEGER DEFAULT 0"),
    ("ml_explanation",        "TEXT DEFAULT NULL"),
    ("privacy_leak_risk",     "REAL DEFAULT 0.0"),
    ("sdoh_risk_label",       "TEXT DEFAULT NULL"),
    ("sdoh_risk_score",       "REAL DEFAULT NULL"),
    ("sdoh_shap_factors",     "TEXT DEFAULT NULL"),
    ("human_review_action",  "TEXT DEFAULT NULL"),
    ("human_notes",          "TEXT DEFAULT NULL"),
    ("reviewed_by",          "TEXT DEFAULT NULL"),
    ("icd10_codes_json",     "TEXT DEFAULT NULL"),
    ("patient_id",           "TEXT DEFAULT NULL"),
    ("lifestyle_risk_score", "REAL DEFAULT 0.0"),
    ("lifestyle_factors",    "TEXT DEFAULT '[]'"),
    ("complexity_class",     "TEXT DEFAULT 'simple'"),
]

_INSERT = """
    INSERT INTO executions (
        incident_id, severity, decision, confidence,
        step_count, retry_count, path_taken, execution_time_ms,
        drift_score, risk_level, workflow_type, overall_confidence,
        unresolved_count, total_entities, ml_explanation, privacy_leak_risk,
        sdoh_risk_label, sdoh_risk_score, sdoh_shap_factors,
        patient_id, lifestyle_risk_score, lifestyle_factors, complexity_class
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Unfiltered baseline — used when no complexity_class is specified (global view).
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
        AVG(overall_confidence) AS avg_coding_confidence,
        SUM(unresolved_count)*1.0 / NULLIF(SUM(total_entities), 0) AS avg_unresolved_rate,
        AVG(privacy_leak_risk) AS avg_privacy_leak_risk
    FROM (
        SELECT step_count, retry_count, execution_time_ms, decision, severity,
               overall_confidence, unresolved_count, total_entities, privacy_leak_risk
        FROM executions
        ORDER BY id DESC
        LIMIT ?
    )
"""

# Segmented baseline — filters by complexity_class before aggregating.
_BASELINE_QUERY_SEGMENTED = """
    SELECT
        AVG(step_count)  AS avg_steps,
        AVG(retry_count) AS avg_retries,
        AVG(execution_time_ms) AS avg_latency,
        AVG(CASE WHEN decision = 'escalate' THEN 1.0 ELSE 0.0 END) AS escalation_rate,
        AVG(CASE WHEN severity = 'high'     THEN 1.0 ELSE 0.0 END) AS high_severity_rate,
        AVG(CASE WHEN severity = 'low'
                 THEN CASE WHEN decision = 'escalate' THEN 1.0 ELSE 0.0 END
                 ELSE NULL END) AS low_severity_escalation_rate,
        AVG(overall_confidence) AS avg_coding_confidence,
        SUM(unresolved_count)*1.0 / NULLIF(SUM(total_entities), 0) AS avg_unresolved_rate,
        AVG(privacy_leak_risk) AS avg_privacy_leak_risk
    FROM (
        SELECT step_count, retry_count, execution_time_ms, decision, severity,
               overall_confidence, unresolved_count, total_entities, privacy_leak_risk
        FROM executions
        WHERE complexity_class = ?
        ORDER BY id DESC
        LIMIT ?
    )
"""

# Per-class default baselines reflect the realistic operating envelope of each tier.
# High-complexity cases legitimately have more steps, retries, and latency.
_DEFAULT_BASELINE_BY_CLASS: dict[str, dict] = {
    "simple": {
        "avg_steps":                    4.0,
        "avg_retries":                  0.0,
        "avg_latency":                  100.0,
        "escalation_rate":              0.2,
        "high_severity_rate":           0.2,
        "low_severity_escalation_rate": 0.05,
        "avg_coding_confidence":        0.75,
        "avg_unresolved_rate":          0.05,
        "avg_privacy_leak_risk":        0.0,
    },
    "moderate": {
        "avg_steps":                    6.0,
        "avg_retries":                  0.5,
        "avg_latency":                  300.0,
        "escalation_rate":              0.15,
        "high_severity_rate":           0.25,
        "low_severity_escalation_rate": 0.05,
        "avg_coding_confidence":        0.72,
        "avg_unresolved_rate":          0.08,
        "avg_privacy_leak_risk":        0.0,
    },
    "high_complexity": {
        "avg_steps":                    9.0,
        "avg_retries":                  1.0,
        "avg_latency":                  600.0,
        "escalation_rate":              0.1,
        "high_severity_rate":           0.4,
        "low_severity_escalation_rate": 0.05,
        "avg_coding_confidence":        0.68,
        "avg_unresolved_rate":          0.12,
        "avg_privacy_leak_risk":        0.0,
    },
    "preventive_screening": {
        "avg_steps":                    5.0,
        "avg_retries":                  0.2,
        "avg_latency":                  200.0,
        "escalation_rate":              0.3,
        "high_severity_rate":           0.3,
        "low_severity_escalation_rate": 0.1,
        "avg_coding_confidence":        0.70,
        "avg_unresolved_rate":          0.10,
        "avg_privacy_leak_risk":        0.0,
    },
}

# Backwards-compatible global default (used when complexity_class is None)
_DEFAULT_BASELINE = _DEFAULT_BASELINE_BY_CLASS["simple"]


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the executions and human_interventions tables and apply pending column migrations."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_HITL_TABLE)
        for col, definition in _MIGRATIONS:
            try:
                conn.execute(f"ALTER TABLE executions ADD COLUMN {col} {definition}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        conn.commit()
    log.debug("SQLite database initialised at %s", DB_PATH)


def _derive_complexity_class(state: dict, analysis: dict | None) -> str:
    """Derive a complexity_class tag from the execution state.

    Classes:
        simple               — incident_triage runs
        moderate             — clinical_coding with no biomarkers
        high_complexity      — clinical_coding with biomarkers present
        preventive_screening — preventive_screening workflow
    """
    workflow = (analysis or {}).get("workflow_type") or state.get("workflow_type", "incident_triage")
    if workflow == "preventive_screening":
        return "preventive_screening"
    if workflow == "clinical_coding":
        biomarkers = state.get("biomarkers")
        if biomarkers:  # non-empty list → high complexity
            return "high_complexity"
        return "moderate"
    return "simple"


def save_execution_state(
    state,
    analysis: dict | None = None,
    *,
    tenant_id: str | None = None,  # accepted but ignored in SQLite mode
) -> None:
    """Persist a completed workflow execution to SQLite."""
    # Support both incident_triage states (incident_id) and clinical states (record_id)
    record_key = state.get("incident_id") or state.get("record_id")
    icd10_codes = state.get("icd10_codes", [])
    unresolved_count = sum(1 for c in icd10_codes if c.get("code") == "UNRESOLVED") if icd10_codes else 0
    total_entities = len(icd10_codes) if icd10_codes else 0
    complexity_class = _derive_complexity_class(state, analysis)

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
        unresolved_count,
        total_entities,
        analysis.get("ml_explanation") if analysis else None,
        state.get("privacy_leak_risk", 0.0),
        state.get("sdoh_risk_label"),
        state.get("sdoh_risk_score"),
        json.dumps(state.get("sdoh_shap_factors", [])),
        state.get("patient_id"),
        state.get("lifestyle_risk_score", 0.0),
        json.dumps(state.get("lifestyle_factors", [])),
        complexity_class,
    )
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_INSERT, values)
        conn.commit()
    log.debug("SQLite: saved execution %s", record_key)


def get_historical_metrics(
    limit: int = 100,
    *,
    tenant_id: str | None = None,  # ignored in SQLite mode
    complexity_class: str | None = None,
) -> dict:
    """Return population-level baseline metrics from the last *limit* executions.

    If complexity_class is provided, the query is scoped to that class only,
    ensuring like-for-like baseline comparisons (e.g., high_complexity cases
    are never compared against a baseline diluted by simple triage runs).
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        if complexity_class:
            row = conn.execute(_BASELINE_QUERY_SEGMENTED, (complexity_class, limit)).fetchone()
        else:
            row = conn.execute(_BASELINE_QUERY, (limit,)).fetchone()

    default = dict(
        _DEFAULT_BASELINE_BY_CLASS.get(complexity_class or "simple", _DEFAULT_BASELINE)
    )

    if not row or row["avg_steps"] is None:
        log.debug(
            "SQLite: empty database for class=%s — using default baseline.",
            complexity_class or "global",
        )
        return default

    return {
        "avg_steps":                    row["avg_steps"],
        "avg_retries":                  row["avg_retries"],
        "avg_latency":                  row["avg_latency"],
        "escalation_rate":              row["escalation_rate"]              or default["escalation_rate"],
        "high_severity_rate":           row["high_severity_rate"]           or default["high_severity_rate"],
        "low_severity_escalation_rate": row["low_severity_escalation_rate"] or default["low_severity_escalation_rate"],
        "avg_coding_confidence":        row["avg_coding_confidence"]        or default["avg_coding_confidence"],
        "avg_unresolved_rate":          row["avg_unresolved_rate"]          or default["avg_unresolved_rate"],
        "avg_privacy_leak_risk":        row["avg_privacy_leak_risk"]        or 0.0,
    }


# ── Human-in-the-Loop (HITL) Helper API ────────────────────────────────────────

def save_human_intervention(
    incident_id: str,
    action: str,
    reviewed_by: str = "clinician",
    notes: str = "",
    original_codes: list | None = None,
    final_codes: list | None = None
) -> int:
    """Save an audit record of a human intervention decision."""
    query = """
        INSERT INTO human_interventions (
            incident_id, action, reviewed_by, notes, original_codes_json, final_codes_json
        ) VALUES (?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(query, (
            incident_id,
            action,
            reviewed_by,
            notes,
            json.dumps(original_codes or []),
            json.dumps(final_codes or [])
        ))
        conn.commit()
        return cursor.lastrowid or 0


def update_execution_human_status(
    record_id: str,
    new_status: str,
    human_action: str,
    notes: str,
    reviewed_by: str,
    final_codes: list | None = None
) -> None:
    """Update execution record after human approval or editing."""
    query = """
        UPDATE executions
        SET decision = ?,
            human_review_action = ?,
            human_notes = ?,
            reviewed_by = ?
        WHERE incident_id = ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(query, (new_status, human_action, notes, reviewed_by, record_id))
        conn.commit()


def get_pending_reviews(limit: int = 50) -> list[dict]:
    """Retrieve all execution records waiting for clinical human review."""
    init_db()
    query = """
        SELECT incident_id, decision, confidence, overall_confidence, step_count, retry_count,
               sdoh_risk_label, sdoh_risk_score, created_at, workflow_type
        FROM executions
        WHERE decision = 'requires_clinical_review'
          AND (human_review_action IS NULL OR human_review_action = 'pending')
        ORDER BY id DESC
        LIMIT ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_review_history(limit: int = 50) -> list[dict]:
    """Retrieve history of completed human review interventions."""
    init_db()
    query = """
        SELECT id, incident_id, action, reviewed_by, notes, original_codes_json, final_codes_json, created_at
        FROM human_interventions
        ORDER BY id DESC
        LIMIT ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (limit,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d["original_codes"] = json.loads(d["original_codes_json"]) if d["original_codes_json"] else []
                d["final_codes"] = json.loads(d["final_codes_json"]) if d["final_codes_json"] else []
            except Exception:
                d["original_codes"] = []
                d["final_codes"] = []
            result.append(d)
        return result


def get_breaker_events(limit: int = 100) -> list[dict]:
    """Retrieve all circuit breaker trigger events from the audit log.

    These are written by intervention_step every time the LangGraph circuit
    breaker fires. Intended for the clinical auditor dashboard view.
    """
    init_db()
    query = """
        SELECT id, incident_id, reviewed_by, notes, created_at
        FROM human_interventions
        WHERE action = 'circuit_breaker_triggered'
        ORDER BY id DESC
        LIMIT ?
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, (limit,)).fetchall()

    result = []
    for r in rows:
        d = dict(r)
        try:
            d["trigger_details"] = json.loads(d["notes"]) if d["notes"] else {}
        except Exception:
            d["trigger_details"] = {}
        result.append(d)
    return result
