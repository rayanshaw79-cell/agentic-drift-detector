"""
telemetry/postgres_store.py — PostgreSQL + TimescaleDB backend.

Used automatically when DATABASE_URL is set.

Features:
  - ThreadedConnectionPool (1–10 connections)
  - TimescaleDB hypertable with graceful fallback to plain PostgreSQL
  - Per-tenant queries scoped by tenant_id
  - JSONB storage for path_taken
"""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

log = logging.getLogger(__name__)

try:
    import psycopg2
    from psycopg2 import pool as pg_pool
    from psycopg2.extras import Json, RealDictCursor
except ImportError:
    raise ImportError(
        "DATABASE_URL is set but psycopg2 is not installed.\n"
        "Run: pip install psycopg2-binary"
    )

# ── Connection pool ───────────────────────────────────────────────────────────

_pool: pg_pool.ThreadedConnectionPool | None = None


def _get_pool() -> pg_pool.ThreadedConnectionPool:
    global _pool
    if _pool is None:
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            raise RuntimeError("DATABASE_URL is not set.")
        _pool = pg_pool.ThreadedConnectionPool(minconn=1, maxconn=10, dsn=dsn)
        log.debug("PostgreSQL connection pool created (maxconn=10).")
    return _pool


@contextmanager
def _get_conn() -> Generator:
    """Borrow a connection from the pool; auto-rollback on error."""
    p = _get_pool()
    conn = p.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


# ── SQL constants ─────────────────────────────────────────────────────────────

_INSERT = """
    INSERT INTO executions (
        tenant_id, incident_id, severity, decision, confidence,
        step_count, retry_count, path_taken, execution_time_ms,
        drift_score, risk_level
    ) VALUES (
        %(tenant_id)s, %(incident_id)s, %(severity)s, %(decision)s, %(confidence)s,
        %(step_count)s, %(retry_count)s, %(path_taken)s, %(execution_time_ms)s,
        %(drift_score)s, %(risk_level)s
    )
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
                 ELSE NULL END) AS low_severity_escalation_rate
    FROM (
        SELECT step_count, retry_count, execution_time_ms, decision, severity
        FROM executions
        WHERE tenant_id = %(tenant_id)s
        ORDER BY created_at DESC
        LIMIT %(limit)s
    ) sub
"""

_DEFAULT_BASELINE = {
    "avg_steps": 4.0,
    "avg_retries": 0.0,
    "avg_latency": 100.0,
    "escalation_rate": 0.2,
    "high_severity_rate": 0.2,
    "low_severity_escalation_rate": 0.05,
}


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Apply the initial PostgreSQL schema migration.

    - Creates the tenants + executions tables.
    - Attempts TimescaleDB hypertable creation (graceful fallback to plain PG).
    - Safe to call repeatedly (idempotent).
    """
    migration_path = Path(__file__).parent.parent / "migrations" / "001_initial_postgres.sql"
    sql = migration_path.read_text(encoding="utf-8")

    with _get_conn() as conn:
        with conn.cursor() as cur:
            # Run each statement individually to isolate TimescaleDB failures
            for statement in _split_sql(sql):
                try:
                    cur.execute(statement)
                    conn.commit()
                except psycopg2.Error as exc:
                    conn.rollback()
                    if "timescaledb" in str(exc).lower() or "create_hypertable" in statement.lower():
                        log.warning(
                            "TimescaleDB not available (%s). "
                            "Continuing with plain PostgreSQL — time-series features disabled.",
                            exc.pgcode,
                        )
                    else:
                        raise

    log.info("PostgreSQL schema initialised.")


def save_execution_state(
    state: dict,
    analysis: dict | None = None,
    *,
    tenant_id: str = "default",
) -> None:
    """Insert one execution record scoped to *tenant_id*."""
    from config.tenant import ensure_tenant_exists

    path_taken = state.get("path_taken", [])

    params = {
        "tenant_id":         tenant_id,
        "incident_id":       state.get("incident_id"),
        "severity":          state.get("severity"),
        "decision":          state.get("decision"),
        "confidence":        state.get("confidence"),
        "step_count":        state.get("step_count", 0),
        "retry_count":       state.get("retry_count", 0),
        "path_taken":        Json(path_taken if isinstance(path_taken, list)
                                 else json.loads(path_taken)),
        "execution_time_ms": state.get("execution_time_ms", 0),
        "drift_score":       analysis.get("drift_score", 0) if analysis else 0,
        "risk_level":        analysis.get("risk_level", "healthy") if analysis else "healthy",
    }

    with _get_conn() as conn:
        ensure_tenant_exists(conn, tenant_id)
        with conn.cursor() as cur:
            cur.execute(_INSERT, params)
        conn.commit()

    log.debug("PostgreSQL: saved execution %s (tenant=%s)", state.get("incident_id"), tenant_id)


def get_historical_metrics(
    limit: int = 100,
    *,
    tenant_id: str = "default",
) -> dict:
    """
    Return population-level baseline metrics for *tenant_id*
    from the last *limit* executions.
    """
    with _get_conn() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(_BASELINE_QUERY, {"tenant_id": tenant_id, "limit": limit})
            row = cur.fetchone()

    if not row or row["avg_steps"] is None:
        log.debug("PostgreSQL: no data for tenant '%s' — using default baseline.", tenant_id)
        return dict(_DEFAULT_BASELINE)

    return {
        "avg_steps":                    float(row["avg_steps"]),
        "avg_retries":                  float(row["avg_retries"]),
        "avg_latency":                  float(row["avg_latency"]),
        "escalation_rate":              float(row["escalation_rate"]              or 0.2),
        "high_severity_rate":           float(row["high_severity_rate"]           or 0.2),
        "low_severity_escalation_rate": float(row["low_severity_escalation_rate"] or 0.05),
    }


def get_tenants() -> list[str]:
    """Return all registered tenant IDs, ordered by creation time."""
    with _get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM tenants ORDER BY created_at")
            return [row[0] for row in cur.fetchall()]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _split_sql(sql: str) -> list[str]:
    """Split a SQL file into individual statements, skipping empty ones."""
    return [s.strip() for s in sql.split(";") if s.strip()]
