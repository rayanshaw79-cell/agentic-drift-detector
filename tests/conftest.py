"""
conftest.py — shared pytest fixtures for the Agentic Drift Detector test suite.

Key fixture
-----------
``temp_db``
    Automatically replaces the real telemetry.db path with a temporary
    in-memory/on-disk path for the duration of each test, ensuring tests are
    fully isolated and do not pollute or depend on production data.
"""

import pytest
import telemetry.store as store_module
from telemetry.store import init_db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Redirect all DB operations to an isolated temporary database.

    This fixture runs automatically for every test in the suite.
    The temporary database is discarded after each test.
    """
    test_db_path = str(tmp_path / "test_telemetry.db")
    monkeypatch.setattr(store_module, "DB_PATH", test_db_path)
    init_db()
    yield test_db_path
