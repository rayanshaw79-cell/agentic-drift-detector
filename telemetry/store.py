"""
telemetry/store.py — Backend router.

Delegates all calls to the correct backend based on environment:
  - DATABASE_URL set   → PostgreSQL + TimescaleDB (telemetry/postgres_store.py)
  - DATABASE_URL unset → SQLite fallback            (telemetry/sqlite_store.py)

Public API is identical regardless of backend:
    init_db()
    save_execution_state(state, analysis, *, tenant_id=None)
    get_historical_metrics(limit, *, tenant_id=None)

This module is imported by run.py, drift_detector.py, and the test suite.
The test conftest patches DB_PATH here AND in sqlite_store to ensure isolation.
"""

import logging
import os

# Re-export DB_PATH from SQLite store for backward compatibility.
# Only meaningful when DATABASE_URL is not set.
from telemetry.sqlite_store import DB_PATH  # noqa: F401

log = logging.getLogger(__name__)

_USE_POSTGRES = bool(os.getenv("DATABASE_URL"))


def _backend():
    """Return the active backend module (lazy import to avoid import-time side effects)."""
    if _USE_POSTGRES:
        try:
            import telemetry.postgres_store as _pg
            return _pg
        except ImportError as exc:
            raise RuntimeError(
                f"DATABASE_URL is set but the PostgreSQL driver is missing: {exc}\n"
                "Fix: pip install psycopg2-binary"
            ) from exc
    import telemetry.sqlite_store as _sq
    return _sq


# ── Public API (thin delegation) ──────────────────────────────────────────────

def init_db() -> None:
    """Initialise the active database backend (create tables, run migrations)."""
    _backend().init_db()


def save_execution_state(
    state,
    analysis: dict | None = None,
    *,
    tenant_id: str | None = None,
) -> None:
    """
    Persist one execution record.

    tenant_id defaults to the TENANT_ID env var (resolved in postgres_store)
    and is silently ignored by the SQLite backend.
    """
    kwargs = {}
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    _backend().save_execution_state(state, analysis, **kwargs)


def get_historical_metrics(
    limit: int = 100,
    *,
    tenant_id: str | None = None,
    complexity_class: str | None = None,
) -> dict:
    """
    Return population-level baseline metrics.

    In PostgreSQL mode the query is scoped to tenant_id (defaults to
    TENANT_ID env var). In SQLite mode tenant_id is ignored.

    If complexity_class is provided ('simple', 'moderate', 'high_complexity',
    'preventive_screening'), the baseline is scoped to that class only,
    preventing case-mix shifts from masking real drift.
    """
    kwargs = {}
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    if complexity_class is not None:
        kwargs["complexity_class"] = complexity_class
    return _backend().get_historical_metrics(limit, **kwargs)

def get_recent_states(
    limit: int = 2000,
    *,
    tenant_id: str | None = None,
) -> list[dict]:
    """Retrieve recent raw states for ML retraining."""
    backend = _backend()
    kwargs = {}
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    if hasattr(backend, "get_recent_states"):
        return backend.get_recent_states(limit, **kwargs)
    return []


def save_human_intervention(
    incident_id: str,
    action: str,
    reviewed_by: str = "clinician",
    notes: str = "",
    original_codes: list | None = None,
    final_codes: list | None = None
) -> int:
    """Save an audit record of a human intervention decision."""
    backend = _backend()
    if hasattr(backend, "save_human_intervention"):
        return backend.save_human_intervention(
            incident_id=incident_id,
            action=action,
            reviewed_by=reviewed_by,
            notes=notes,
            original_codes=original_codes,
            final_codes=final_codes
        )
    return 0


def update_execution_human_status(
    record_id: str,
    new_status: str,
    human_action: str,
    notes: str,
    reviewed_by: str,
    final_codes: list | None = None
) -> None:
    """Update execution record after human approval or editing."""
    backend = _backend()
    if hasattr(backend, "update_execution_human_status"):
        backend.update_execution_human_status(
            record_id=record_id,
            new_status=new_status,
            human_action=human_action,
            notes=notes,
            reviewed_by=reviewed_by,
            final_codes=final_codes
        )


def get_pending_reviews(limit: int = 50) -> list[dict]:
    """Retrieve all execution records waiting for clinical human review."""
    backend = _backend()
    if hasattr(backend, "get_pending_reviews"):
        return backend.get_pending_reviews(limit=limit)
    return []


def get_review_history(limit: int = 50) -> list[dict]:
    """Retrieve history of completed human review interventions."""
    backend = _backend()
    if hasattr(backend, "get_review_history"):
        return backend.get_review_history(limit=limit)
    return []


def get_breaker_events(limit: int = 100) -> list[dict]:
    """Retrieve all circuit breaker trigger events from the audit log.

    Returns structured records written by intervention_step each time the
    LangGraph circuit breaker fires. Used by the dashboard audit log view.
    """
    backend = _backend()
    if hasattr(backend, "get_breaker_events"):
        return backend.get_breaker_events(limit=limit)
    return []
