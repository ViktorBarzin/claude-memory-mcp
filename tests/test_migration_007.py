"""PG-only tests for Alembic migration ``007_gin_search_fastupdate_off``.

Same substrate rules as ``test_migration_005.py``: a guarded
``MEMORY_TEST_DATABASE_URL`` or a disposable Docker container, and a skip (never a
failure) when neither is available.
"""

from __future__ import annotations

import pytest

from tests.test_migration_005 import _alembic_to, _query, psycopg2
from tests.test_migration_006 import pg_url  # noqa: F401 - shared module fixture


@pytest.fixture()
def db_at_006(pg_url: str) -> str:  # noqa: F811
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
    finally:
        conn.close()
    _alembic_to("006", pg_url)
    return pg_url


def _reloptions(db_url: str) -> list[str]:
    rows = _query(db_url, "SELECT reloptions FROM pg_class WHERE relname = 'idx_memories_search'")
    return list(rows[0][0] or [])


def test_upgrade_turns_off_the_gin_pending_list(db_at_006: str) -> None:
    assert "fastupdate=off" not in _reloptions(db_at_006)
    _alembic_to("007", db_at_006)
    assert "fastupdate=off" in _reloptions(db_at_006)


def test_upgrade_is_idempotent(db_at_006: str) -> None:
    _alembic_to("007", db_at_006)
    _alembic_to("stamp:006", db_at_006)
    _alembic_to("007", db_at_006)
    assert "fastupdate=off" in _reloptions(db_at_006)


def test_downgrade_restores_the_default(db_at_006: str) -> None:
    _alembic_to("007", db_at_006)
    _alembic_to("down:006", db_at_006)
    assert "fastupdate=off" not in _reloptions(db_at_006)
