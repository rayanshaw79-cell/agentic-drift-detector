import sqlite3

DB_PATH = 'telemetry.db'

print('Applying SQLite table rebuild migration (Part 3)...')
try:
    with sqlite3.connect(DB_PATH) as conn:
        # Just in case executions wasn't created because it failed after
        conn.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_id       TEXT,
                severity          TEXT,
                decision          TEXT,
                confidence        REAL,
                step_count        INTEGER,
                retry_count       INTEGER,
                path_taken        TEXT,
                execution_time_ms INTEGER,
                drift_score       INTEGER DEFAULT 0,
                risk_level        TEXT    DEFAULT 'healthy',
                workflow_type     TEXT    DEFAULT 'incident_triage',
                overall_confidence REAL   DEFAULT NULL,
                created_at        DATETIME DEFAULT CURRENT_TIMESTAMP,
                unresolved_count  INTEGER DEFAULT 0,
                total_entities    INTEGER DEFAULT 0,
                ml_explanation    TEXT    DEFAULT NULL
            )
        ''')
        
        cur = conn.execute('PRAGMA table_info(_executions_old)')
        old_cols = [row[1] for row in cur.fetchall()]
        
        if old_cols:
            select_cols = []
            insert_cols = []
            for col in old_cols:
                if col == 'timestamp':
                    select_cols.append('timestamp')
                    insert_cols.append('created_at')
                else:
                    select_cols.append(col)
                    insert_cols.append(col)
            
            sel_str = ', '.join(select_cols)
            ins_str = ', '.join(insert_cols)
            
            print("Mapping old columns to new schema...")
            conn.execute(f'INSERT INTO executions ({ins_str}) SELECT {sel_str} FROM _executions_old')
            conn.execute('DROP TABLE _executions_old')
            conn.commit()
            print('Migration complete! Data recovered.')
        else:
            print('No _executions_old table found, just created fresh schema.')
except Exception as e:
    print("Migration failed:", e)
