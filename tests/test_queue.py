"""
tests/test_queue.py — Redis queue tests.

These tests require a live Redis instance.
They are automatically SKIPPED when REDIS_URL is not set.

To run locally:
    docker compose up -d
    pytest tests/test_queue.py -v
"""

import os
import pytest

# ── Skip guard ────────────────────────────────────────────────────────────────
redis_required = pytest.mark.skipif(
    not os.getenv("REDIS_URL"),
    reason="REDIS_URL not set — skipping Redis tests (run with Docker: docker compose up -d)",
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def flush_test_queue():
    """Flush the queue before and after each test to prevent cross-test pollution."""
    from telemetry.queue import flush_queue
    flush_queue()
    yield
    flush_queue()


@pytest.fixture
def sample_payload():
    return {
        "state": {
            "incident_id": "q-test-001",
            "severity": "low",
            "decision": "auto_resolve",
            "confidence": 0.9,
            "step_count": 4,
            "retry_count": 0,
            "path_taken": ["triage", "investigation", "decision", "notification"],
            "execution_time_ms": 200,
        },
        "analysis": {"drift_score": 3, "risk_level": "healthy"},
    }


# ── Tests ─────────────────────────────────────────────────────────────────────

@redis_required
def test_enqueue_returns_true_when_redis_available(sample_payload):
    """enqueue() must return True when Redis is reachable."""
    from telemetry.queue import enqueue
    result = enqueue(sample_payload["state"], sample_payload["analysis"], "test-tenant")
    assert result is True


@redis_required
def test_queue_depth_increases_after_enqueue(sample_payload):
    """queue_depth() must reflect the number of enqueued items."""
    from telemetry.queue import enqueue, queue_depth

    assert queue_depth() == 0
    enqueue(sample_payload["state"], sample_payload["analysis"], "test-tenant")
    assert queue_depth() == 1
    enqueue(sample_payload["state"], sample_payload["analysis"], "test-tenant")
    assert queue_depth() == 2


@redis_required
def test_dequeue_returns_enqueued_item(sample_payload):
    """dequeue() must return the exact item that was enqueued."""
    from telemetry.queue import enqueue, dequeue

    enqueue(sample_payload["state"], sample_payload["analysis"], "tenant-x")
    item = dequeue(timeout=2)

    assert item is not None
    assert item["tenant_id"] == "tenant-x"
    assert item["analysis"]["drift_score"] == 3
    assert item["state"]["incident_id"] == "q-test-001"


@redis_required
def test_dequeue_returns_none_on_empty_queue():
    """dequeue() must return None when the queue is empty (no blocking hang)."""
    from telemetry.queue import dequeue
    item = dequeue(timeout=1)
    assert item is None


@redis_required
def test_flush_clears_all_items(sample_payload):
    """flush_queue() must remove all pending items."""
    from telemetry.queue import enqueue, flush_queue, queue_depth

    for _ in range(5):
        enqueue(sample_payload["state"], sample_payload["analysis"], "tenant-flush")

    assert queue_depth() == 5
    removed = flush_queue()
    assert removed == 5
    assert queue_depth() == 0


@redis_required
def test_fifo_ordering(sample_payload):
    """Items must be dequeued in FIFO order (first in, first out)."""
    from telemetry.queue import enqueue, dequeue

    for i in range(3):
        state = dict(sample_payload["state"], incident_id=f"order-{i}")
        enqueue(state, sample_payload["analysis"], "tenant-fifo")

    items = [dequeue(timeout=1) for _ in range(3)]
    ids = [item["state"]["incident_id"] for item in items]
    assert ids == ["order-0", "order-1", "order-2"]
