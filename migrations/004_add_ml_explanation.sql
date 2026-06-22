-- Migration 004: Add ml_explanation to PostgreSQL executions table

ALTER TABLE executions ADD COLUMN IF NOT EXISTS ml_explanation TEXT DEFAULT NULL;
