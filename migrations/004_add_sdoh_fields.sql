-- 004_add_sdoh_fields.sql
-- Add SDOH population health fields to executions table

ALTER TABLE executions ADD COLUMN IF NOT EXISTS sdoh_risk_label TEXT DEFAULT NULL;
ALTER TABLE executions ADD COLUMN IF NOT EXISTS sdoh_risk_score REAL DEFAULT NULL;
ALTER TABLE executions ADD COLUMN IF NOT EXISTS sdoh_shap_factors JSONB DEFAULT NULL;
