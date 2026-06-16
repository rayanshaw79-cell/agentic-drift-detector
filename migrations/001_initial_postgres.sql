-- ============================================================
-- Agentic Drift Detector — PostgreSQL + TimescaleDB Schema
-- Migration 001: Initial schema
-- Run automatically by init_db() or docker-entrypoint-initdb.d
-- ============================================================

-- TimescaleDB extension (no-op if already installed)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ── Tenant registry ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tenants (
    id         TEXT        PRIMARY KEY,
    name       TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Seed the default tenant (safe to re-run)
INSERT INTO tenants (id, name)
VALUES ('default', 'Default')
ON CONFLICT (id) DO NOTHING;

-- ── Execution telemetry (time-series) ────────────────────────
CREATE TABLE IF NOT EXISTS executions (
    id                BIGSERIAL,
    tenant_id         TEXT        NOT NULL DEFAULT 'default'
                                  REFERENCES tenants(id),
    incident_id       TEXT,
    severity          TEXT,
    decision          TEXT,
    confidence        REAL,
    step_count        INTEGER,
    retry_count       INTEGER,
    path_taken        JSONB       NOT NULL DEFAULT '[]',
    execution_time_ms INTEGER,
    drift_score       INTEGER     NOT NULL DEFAULT 0,
    risk_level        TEXT        NOT NULL DEFAULT 'healthy',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (id, created_at)
);

-- Convert to TimescaleDB hypertable (partitioned by day)
-- If TimescaleDB is unavailable, init_db() catches and logs the error.
SELECT create_hypertable(
    'executions',
    'created_at',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists       => TRUE
);

-- ── Indexes ──────────────────────────────────────────────────
-- Most queries filter by tenant and order by time
CREATE INDEX IF NOT EXISTS idx_executions_tenant_time
    ON executions (tenant_id, created_at DESC);

-- Alert feed queries filter by tenant + risk_level
CREATE INDEX IF NOT EXISTS idx_executions_risk
    ON executions (tenant_id, risk_level, created_at DESC);

-- Baseline queries group by decision + severity per tenant
CREATE INDEX IF NOT EXISTS idx_executions_decision
    ON executions (tenant_id, decision, severity, created_at DESC);
