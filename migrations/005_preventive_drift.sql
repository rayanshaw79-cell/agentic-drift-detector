-- Add Patient Health Drift fields to executions table
ALTER TABLE executions
ADD COLUMN IF NOT EXISTS patient_id VARCHAR(255),
ADD COLUMN IF NOT EXISTS lifestyle_risk_score FLOAT DEFAULT 0.0,
ADD COLUMN IF NOT EXISTS lifestyle_factors JSONB DEFAULT '[]'::jsonb;
