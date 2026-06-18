-- migrations/002_add_workflow_type.sql
-- PostgreSQL migration: add workflow_type and overall_confidence columns
-- to the executions table to support the Clinical Coding Agent.
--
-- Run once against the production PostgreSQL database:
--   psql $DATABASE_URL -f migrations/002_add_workflow_type.sql

ALTER TABLE executions
    ADD COLUMN IF NOT EXISTS workflow_type      TEXT    DEFAULT 'incident_triage',
    ADD COLUMN IF NOT EXISTS overall_confidence DECIMAL DEFAULT NULL;

-- Backfill existing rows
UPDATE executions
SET workflow_type = 'incident_triage'
WHERE workflow_type IS NULL;

-- Optional index to filter by workflow type efficiently
CREATE INDEX IF NOT EXISTS idx_executions_workflow_type
    ON executions (workflow_type);
