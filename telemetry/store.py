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
) -> dict:
    """
    Return population-level baseline metrics.

    In PostgreSQL mode the query is scoped to tenant_id (defaults to
    TENANT_ID env var). In SQLite mode tenant_id is ignored.
    """
    if _USE_POSTGRES and tenant_id is None:
        from config.tenant import get_current_tenant
        tenant_id = get_current_tenant()

    kwargs = {}
    if tenant_id is not None:
        kwargs["tenant_id"] = tenant_id
    return _backend().get_historical_metrics(limit, **kwargs)
