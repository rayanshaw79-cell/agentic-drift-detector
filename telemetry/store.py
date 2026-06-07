import sqlite3
import json
import os
from schemas.incident_state import IncidentState

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "telemetry.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id TEXT,
            severity TEXT,
            decision TEXT,
            confidence REAL,
            step_count INTEGER,
            retry_count INTEGER,
            path_taken TEXT,
            execution_time_ms INTEGER,
            drift_score INTEGER DEFAULT 0,
            risk_level TEXT DEFAULT 'healthy',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Migrate existing tables that lack the new columns (safe to run repeatedly)
    for col, definition in [
        ("drift_score", "INTEGER DEFAULT 0"),
        ("risk_level",  "TEXT DEFAULT 'healthy'"),
        ("created_at",  "DATETIME DEFAULT CURRENT_TIMESTAMP"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE executions ADD COLUMN {col} {definition}")
        except Exception:
            pass  # Column already exists
    conn.commit()
    conn.close()

def save_execution_state(state: IncidentState, analysis: dict = None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO executions (
            incident_id, severity, decision, confidence,
            step_count, retry_count, path_taken, execution_time_ms,
            drift_score, risk_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        state.get("incident_id"),
        state.get("severity"),
        state.get("decision"),
        state.get("confidence"),
        state.get("step_count", 0),
        state.get("retry_count", 0),
        json.dumps(state.get("path_taken", [])),
        state.get("execution_time_ms", 0),
        analysis.get("drift_score", 0) if analysis else 0,
        analysis.get("risk_level", "healthy") if analysis else "healthy",
    ))
    conn.commit()
    conn.close()

def get_historical_metrics(limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT 
            AVG(step_count) as avg_steps,
            AVG(retry_count) as avg_retries,
            AVG(execution_time_ms) as avg_latency,
            AVG(CASE WHEN decision = 'escalate' THEN 1.0 ELSE 0.0 END) as escalation_rate,
            AVG(CASE WHEN severity = 'high' THEN 1.0 ELSE 0.0 END) as high_severity_rate,
            AVG(CASE WHEN severity = 'low' THEN 
                    CASE WHEN decision = 'escalate' THEN 1.0 ELSE 0.0 END 
                ELSE NULL END) as low_severity_escalation_rate
        FROM (
            SELECT step_count, retry_count, execution_time_ms, decision, severity
            FROM executions
            ORDER BY id DESC
            LIMIT ?
        )
    ''', (limit,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row or row["avg_steps"] is None:
        return {
            "avg_steps": 4.0,
            "avg_retries": 0.0,
            "avg_latency": 100.0,
            "escalation_rate": 0.2,
            "high_severity_rate": 0.2,
            "low_severity_escalation_rate": 0.05
        }
        
    return {
        "avg_steps": row["avg_steps"],
        "avg_retries": row["avg_retries"],
        "avg_latency": row["avg_latency"],
        "escalation_rate": row["escalation_rate"] if row["escalation_rate"] is not None else 0.2,
        "high_severity_rate": row["high_severity_rate"] if row["high_severity_rate"] is not None else 0.2,
        "low_severity_escalation_rate": row["low_severity_escalation_rate"] if row["low_severity_escalation_rate"] is not None else 0.05
    }
