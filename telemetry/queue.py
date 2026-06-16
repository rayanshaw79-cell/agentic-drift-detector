"""
telemetry/queue.py — Redis-backed async telemetry queue.

The agent calls enqueue() after each run — it returns instantly.
A separate worker process (telemetry/worker.py) drains the queue and
writes to PostgreSQL, keeping zero latency on the agent's hot path.

Fallback behaviour:
  - If REDIS_URL is not set → writes synchronously to the active store backend.
  - If Redis is unreachable → logs a warning, falls back to sync write.
  - This means the system degrades gracefully: no crashes, no data loss.
"""

import json
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL")
_QUEUE_KEY = "drift:telemetry:queue"

# Module-level Redis client (lazy-initialised, cached)
_redis_client = None
_redis_available: Optional[bool] = None  # None = not yet tested


def _get_redis():
    """
    Return a Redis client, or None if Redis is unavailable.
    Result is cached after the first successful connection.
    """
    global _redis_client, _redis_available

    if _redis_available is False:
        return None
    if _redis_client is not None:
        return _redis_client

    if not REDIS_URL:
        log.debug("REDIS_URL not set — queue running in synchronous fallback mode.")
        _redis_available = False
        return None

    try:
        import redis as redis_lib
        client = redis_lib.from_url(REDIS_URL, decode_responses=True, socket_timeout=2)
        client.ping()
        _redis_client = client
        _redis_available = True
        log.info("Redis queue connected at %s", REDIS_URL)
        return _redis_client
    except Exception as exc:
        log.warning(
            "Redis unavailable (%s) — falling back to synchronous telemetry writes.", exc
        )
        _redis_available = False
        return None


def enqueue(state: dict, analysis: dict, tenant_id: str) -> bool:
    """
    Push one telemetry event onto the Redis queue.

    Returns True  → item was queued in Redis (async path).
    Returns False → Redis unavailable; item was written synchronously.
    """
    r = _get_redis()

    if r is None:
        # Synchronous fallback — write directly to the active store backend
        _sync_write(state, analysis, tenant_id)
        return False

    payload = json.dumps({
        "state":     _serialise_state(state),
        "analysis":  analysis,
        "tenant_id": tenant_id,
    })
    r.lpush(_QUEUE_KEY, payload)
    log.debug("Enqueued execution %s (tenant=%s)", state.get("incident_id"), tenant_id)
    return True


def dequeue(timeout: int = 1) -> Optional[dict]:
    """
    Block-pop one item from the queue (used by the worker).

    Returns None on timeout or if Redis is unavailable.
    """
    r = _get_redis()
    if r is None:
        return None

    result = r.brpop(_QUEUE_KEY, timeout=timeout)
    if result is None:
        return None

    _, raw = result
    return json.loads(raw)


def flush_queue() -> int:
    """
    Delete all pending items in the queue (used by --clear).

    Returns the number of items removed.
    """
    r = _get_redis()
    if r is None:
        return 0
    count = r.llen(_QUEUE_KEY)
    r.delete(_QUEUE_KEY)
    log.info("Flushed %d items from Redis queue.", count)
    return count


def queue_depth() -> int:
    """Return the number of items currently waiting in the queue."""
    r = _get_redis()
    if r is None:
        return 0
    return r.llen(_QUEUE_KEY)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sync_write(state: dict, analysis: dict, tenant_id: str) -> None:
    """Write directly to the active store backend (no Redis)."""
    from telemetry.store import save_execution_state
    save_execution_state(state, analysis, tenant_id=tenant_id)


def _serialise_state(state: dict) -> dict:
    """Ensure LangGraph state is JSON-serialisable (lists, not Annotated types)."""
    return {k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, str)
                else v)
            for k, v in state.items()}
