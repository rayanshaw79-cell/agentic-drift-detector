-- migrations/006_add_complexity_class.sql
-- Adds a complexity_class column to the executions table for segmented baseline analysis.
-- Classes: 'simple' | 'moderate' | 'high_complexity' | 'preventive_screening'

ALTER TABLE executions ADD COLUMN IF NOT EXISTS complexity_class TEXT DEFAULT 'simple';
CREATE INDEX IF NOT EXISTS idx_executions_complexity_class ON executions (complexity_class);
