"""
clinical/sdoh/patient_store.py — Longitudinal Patient Visit Database.

Provides a dedicated SQLite store for patient SDOH visit records,
separate from the main telemetry.db.

Database file: sdoh_patients.db (in the project root)
"""

import logging
import os
import sqlite3

log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "sdoh_patients.db")

_CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS patient_visits (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id       TEXT    NOT NULL,
        visit_number     INTEGER NOT NULL,
        visit_date       TEXT,
        zip_code         TEXT,
        age              INTEGER,
        gender           TEXT,
        race             TEXT,
        smoking_flag     INTEGER DEFAULT 0,
        alcohol_flag     INTEGER DEFAULT 0,
        exercise_score   REAL    DEFAULT 0.5,
        food_risk_score  REAL    DEFAULT 0.0,
        env_aqi          REAL    DEFAULT 80.0,
        env_poverty_rate REAL    DEFAULT 0.15,
        hcc_score        REAL    DEFAULT 0.0,
        icd10_codes      TEXT,          -- pipe-delimited e.g. "E11.9|N18.3"
        icd10_code_count INTEGER DEFAULT 0,
        chain_stage      INTEGER DEFAULT 0,
        sdoh_risk_score  REAL    DEFAULT 0.0,
        sdoh_risk_label  TEXT    DEFAULT 'low',
        created_at       DATETIME DEFAULT CURRENT_TIMESTAMP
    )
"""

_INSERT = """
    INSERT INTO patient_visits (
        patient_id, visit_number, visit_date, zip_code,
        age, gender, race,
        smoking_flag, alcohol_flag, exercise_score,
        food_risk_score, env_aqi, env_poverty_rate,
        hcc_score, icd10_codes, icd10_code_count, chain_stage,
        sdoh_risk_score, sdoh_risk_label
    ) VALUES (
        ?, ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?, ?,
        ?, ?
    )
"""

_SELECT_PATIENT = """
    SELECT * FROM patient_visits
    WHERE patient_id = ?
    ORDER BY visit_number ASC
"""

_SELECT_ALL_IDS = """
    SELECT DISTINCT patient_id FROM patient_visits ORDER BY patient_id
"""

_SELECT_ALL_VISITS = """
    SELECT * FROM patient_visits ORDER BY patient_id, visit_number
"""


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the patient_visits table if it does not exist."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_CREATE_TABLE)
        conn.commit()
    log.debug("SDOH patient store initialised at %s", DB_PATH)


def save_visit(record: dict) -> None:
    """Persist a single patient visit record."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(_INSERT, (
            record["patient_id"],
            record["visit_number"],
            record.get("visit_date"),
            record.get("zip_code"),
            record.get("age"),
            record.get("gender"),
            record.get("race"),
            int(record.get("smoking_flag", 0)),
            int(record.get("alcohol_flag", 0)),
            record.get("exercise_score", 0.5),
            record.get("food_risk_score", 0.0),
            record.get("env_aqi", 80.0),
            record.get("env_poverty_rate", 0.15),
            record.get("hcc_score", 0.0),
            record.get("icd10_codes", ""),
            record.get("icd10_code_count", 0),
            record.get("chain_stage", 0),
            record.get("sdoh_risk_score", 0.0),
            record.get("sdoh_risk_label", "low"),
        ))
        conn.commit()


def bulk_save(records: list[dict]) -> int:
    """Persist multiple patient visit records. Returns count saved."""
    init_db()
    saved = 0
    for r in records:
        try:
            save_visit(r)
            saved += 1
        except Exception as exc:
            log.warning("Failed to save visit for %s: %s", r.get("patient_id"), exc)
    log.info("SDOH store: saved %d / %d records", saved, len(records))
    return saved


def get_patient_history(patient_id: str) -> list[dict]:
    """Return all visits for a given patient, ordered chronologically."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(_SELECT_PATIENT, (patient_id,)).fetchall()
    return [dict(r) for r in rows]


def get_all_patient_ids() -> list[str]:
    """Return a list of all distinct patient IDs in the store."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(_SELECT_ALL_IDS).fetchall()
    return [r[0] for r in rows]


def get_all_visits() -> list[dict]:
    """Return all visit records (used for dashboard population view)."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(_SELECT_ALL_VISITS).fetchall()
    return [dict(r) for r in rows]
