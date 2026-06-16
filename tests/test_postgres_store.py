"""
tests/test_postgres_store.py — PostgreSQL + TimescaleDB backend tests.

These tests require a live PostgreSQL instance.
They are automatically SKIPPED when DATABASE_URL is not set.

To run locally:
    docker compose up -d
    pytest tests/test_postgres_store.py -v
"""

import os
import pytest

# ── Skip guard ────────────────────────────────────────────────────────────────
pg_required = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set — skipping PostgreSQL tests (run with Docker: docker compose up -d)",
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pg_store():
    """Return the postgres_store module with a fresh schema."""
    import telemetry.postgres_store as store
    store.init_db()
    return store


@pytest.fixture
def test_tenant():
    """A unique tenant ID per test to ensure isolation."""
    import uuid
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def healthy_state():
    return {
        "incident_id":       "pg-test-001",
        "severity":          "low",
        "decision":          "auto_resolve",
        "confidence":        0.9,
        "step_count":        4,
        "retry_count":       0,
        "path_taken":        ["triage", "investigation", "decision", "notification"],
        "execution_time_ms": 250,
    }


@pytest.fixture
def healthy_analysis():
    return {"drift_score": 5, "risk_level": "healthy"}


# ── Tests ─────────────────────────────────────────────────────────────────────

@pg_required
def test_init_db_is_idempotent(pg_store):
    """Calling init_db() multiple times must not raise errors."""
    pg_store.init_db()
    pg_store.init_db()


@pg_required
def test_save_and_retrieve_execution(pg_store, test_tenant, healthy_state, healthy_analysis):
    """A saved execution must appear in the baseline query for the same tenant."""
    pg_store.save_execution_state(healthy_state, healthy_analysis, tenant_id=test_tenant)

    metrics = pg_store.get_historical_metrics(limit=10, tenant_id=test_tenant)

    assert isinstance(metrics, dict)
    assert metrics["avg_steps"] == 4.0
    assert metrics["avg_retries"] == 0.0


@pg_required
def test_tenant_isolation(pg_store, healthy_state, healthy_analysis):
    """Executions saved under tenant A must not appear in tenant B's baseline."""
    import uuid
    tenant_a = f"isolated-a-{uuid.uuid4().hex[:6]}"
    tenant_b = f"isolated-b-{uuid.uuid4().hex[:6]}"

    # Save a high-retry event under tenant_a only
    biased_state = dict(healthy_state, retry_count=5, incident_id="bias-01")
    pg_store.save_execution_state(biased_state, healthy_analysis, tenant_id=tenant_a)

    # tenant_b should fall back to defaults (no data)
    baseline_b = pg_store.get_historical_metrics(limit=100, tenant_id=tenant_b)
    assert baseline_b["avg_retries"] == 0.0, "Tenant B must not see Tenant A's data"


@pg_required
def test_empty_tenant_returns_defaults(pg_store):
    """A tenant with no data must return default baseline values without crashing."""
    import uuid
    empty_tenant = f"empty-{uuid.uuid4().hex[:8]}"
    baseline = pg_store.get_historical_metrics(limit=100, tenant_id=empty_tenant)

    assert baseline["avg_steps"] == 4.0
    assert baseline["avg_retries"] == 0.0
    assert baseline["avg_latency"] == 100.0


@pg_required
def test_get_tenants_includes_saved_tenant(pg_store, test_tenant, healthy_state, healthy_analysis):
    """get_tenants() must include a tenant after their first execution is saved."""
    pg_store.save_execution_state(healthy_state, healthy_analysis, tenant_id=test_tenant)
    tenants = pg_store.get_tenants()
    assert test_tenant in tenants


@pg_required
def test_save_high_risk_execution(pg_store, test_tenant):
    """A high-risk execution must be stored correctly."""
    state = {
        "incident_id": "high-risk-01", "severity": "high", "decision": "escalate",
        "confidence": 0.2, "step_count": 9, "retry_count": 3,
        "path_taken": ["triage", "investigation", "decision", "decision",
                       "decision", "intervention", "notification"],
        "execution_time_ms": 8000,
    }
    analysis = {"drift_score": 85, "risk_level": "high_risk"}
    # Should not raise
    pg_store.save_execution_state(state, analysis, tenant_id=test_tenant)

    metrics = pg_store.get_historical_metrics(limit=5, tenant_id=test_tenant)
    assert metrics["avg_retries"] == 3.0
