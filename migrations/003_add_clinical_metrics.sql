-- Migration 003: Add clinical metrics to PostgreSQL executions table

ALTER TABLE executions ADD COLUMN IF NOT EXISTS overall_confidence REAL DEFAULT NULL;
ALTER TABLE executions ADD COLUMN IF NOT EXISTS unresolved_count INTEGER DEFAULT 0;
ALTER TABLE executions ADD COLUMN IF NOT EXISTS total_entities INTEGER DEFAULT 0;
