-- Migration 003: Add privacy_leak_risk column
ALTER TABLE executions ADD COLUMN IF NOT EXISTS privacy_leak_risk REAL DEFAULT 0.0;
