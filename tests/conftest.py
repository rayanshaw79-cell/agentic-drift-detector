"""
conftest.py — shared pytest fixtures for the Agentic Drift Detector test suite.

Key fixture
-----------
``temp_db``
    Automatically replaces the real telemetry.db path with a temporary
    on-disk path for the duration of each test, ensuring tests are fully
    isolated and do not pollute or depend on production data.

    Patches BOTH:
      - telemetry.store.DB_PATH       (the router's re-exported name)
      - telemetry.sqlite_store.DB_PATH (the backend that actually uses it)
"""

import pytest
import telemetry.store as store_module
import telemetry.sqlite_store as sqlite_store_module
from telemetry.store import init_db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Redirect all SQLite DB operations to an isolated temporary database.

    This fixture runs automatically for every test in the suite.
    The temporary database is discarded after each test.
    PostgreSQL tests are gated by their own skip conditions and do not
    depend on this fixture for isolation.
    """
    test_db_path = str(tmp_path / "test_telemetry.db")

    # Patch the re-exported name in the router
    monkeypatch.setattr(store_module, "DB_PATH", test_db_path)

    # Patch the actual backend (this is what sqlite_store.init_db() uses)
    monkeypatch.setattr(sqlite_store_module, "DB_PATH", test_db_path)

    init_db()
    yield test_db_path
