"""
telemetry/worker.py — Background telemetry drain worker.

Drains the Redis queue and writes each event to PostgreSQL.

Usage:
    python -m telemetry.worker

Run this as a persistent process alongside run.py / the Streamlit dashboard.
In Docker, the 'worker' service handles this automatically.

Signals:
    SIGINT / SIGTERM → finish the current item, then shut down cleanly.
"""

import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("drift.worker")

# ── Validation ────────────────────────────────────────────────────────────────

def _check_env() -> None:
    missing = []
    if not os.getenv("DATABASE_URL"):
        missing.append("DATABASE_URL")
    if not os.getenv("REDIS_URL"):
        missing.append("REDIS_URL")
    if missing:
        log.error(
            "Worker requires %s to be set. "
            "Copy .env.example to .env and fill in the values.",
            " and ".join(missing),
        )
        sys.exit(1)


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    _check_env()

    from telemetry.postgres_store import init_db, save_execution_state
    from telemetry.queue import dequeue, queue_depth

    init_db()

    depth = queue_depth()
    log.info(
        "Worker started. Queue depth: %d items. Waiting for events…",
        depth,
    )

    processed = 0
    errors = 0
    running = True

    def _handle_shutdown(sig, frame):
        nonlocal running
        log.info("Shutdown signal received — finishing current item then exiting.")
        running = False

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    while running:
        item = dequeue(timeout=1)
        if item is None:
            continue

        try:
            save_execution_state(
                item["state"],
                item["analysis"],
                tenant_id=item.get("tenant_id", "default"),
            )
            processed += 1

            if processed % 100 == 0:
                log.info("Worker milestone: %d records processed (%d errors).", processed, errors)

        except Exception as exc:
            errors += 1
            log.error(
                "Failed to persist item (incident=%s tenant=%s): %s",
                item.get("state", {}).get("incident_id"),
                item.get("tenant_id"),
                exc,
                exc_info=True,
            )

    log.info(
        "Worker shut down cleanly. Processed: %d  Errors: %d",
        processed,
        errors,
    )


if __name__ == "__main__":
    run()
