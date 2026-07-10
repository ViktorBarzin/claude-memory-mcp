"""PG-only tests for Alembic migration ``006_add_memory_links``.

Same substrate rules as ``test_migration_005.py`` (whose container/alembic
helpers this module reuses): migrations are PostgreSQL-only, so the module
needs a real Postgres — a guarded ``MEMORY_TEST_DATABASE_URL`` or a disposable
Docker container — and **skips** (never fails) when neither is available.

What is asserted (the ADR-0007 link-substrate acceptance set):

* ``memory_links`` lands with the exact shape the recall/link endpoints rely
  on: ``id`` PK, ``user_id`` NOT NULL, ``src_id``/``dst_id`` FKs to
  ``memories(id)`` with ON DELETE CASCADE, ``link_type`` constrained to the
  **closed enum of four** (``part-of`` / ``supersedes`` / ``see-also`` /
  ``resolved-by``), ``created_at`` defaulting to now.
* the per-user edge is unique: ``UNIQUE(user_id, src_id, dst_id, link_type)``
  rejects a duplicate edge but allows the same pair under a different type.
* both lookup indexes exist (``(user_id, src_id)`` and ``(user_id, dst_id)``)
  — the recall post-processing batch query walks edges in both directions.
* ``upgrade`` is additive + **idempotent** (stamp-back re-run is a no-op) and
  ``downgrade`` drops ONLY ``memory_links``, leaving the 005 schema intact.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from tests.test_migration_005 import (
    PGVECTOR_IMAGE,
    _alembic_to,
    _column_type,
    _docker_available,
    _index_exists,
    _query,
    _start_pg_container,
    _table_exists,
    _wait_until_ready,
    psycopg2,
)

LINK_TYPES = ("part-of", "supersedes", "see-also", "resolved-by")


@pytest.fixture(scope="module")
def pg_url() -> Iterator[str]:
    """A live Postgres URL (guarded env or disposable container), or skip."""
    import os

    guarded = os.environ.get("MEMORY_TEST_DATABASE_URL")
    if guarded:
        _wait_until_ready(guarded)
        yield guarded
        return

    if not _docker_available():
        pytest.skip(
            "no Postgres available: set MEMORY_TEST_DATABASE_URL or run a Docker daemon "
            f"(migration tests need {PGVECTOR_IMAGE})"
        )

    dsn, name = _start_pg_container(PGVECTOR_IMAGE)
    try:
        yield dsn
    finally:
        import subprocess

        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


@pytest.fixture()
def base_db(pg_url: str) -> str:
    """A database migrated up to **005** (the pre-006 baseline) for each test."""
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
    finally:
        conn.close()
    _alembic_to("005", pg_url)
    return pg_url


def _execute(db_url: str, sql: str, params: tuple[object, ...] = ()) -> None:
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
    finally:
        conn.close()


def _insert_memory(db_url: str, user_id: str = "u1", content: str = "mem") -> int:
    # NOT _query: that helper never commits (fine for SELECTs; an INSERT would roll back).
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memories (user_id, content) VALUES (%s, %s) RETURNING id",
                (user_id, content),
            )
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def _insert_link(
    db_url: str, user_id: str, src_id: int, dst_id: int, link_type: str
) -> None:
    _execute(
        db_url,
        "INSERT INTO memory_links (user_id, src_id, dst_id, link_type) VALUES (%s, %s, %s, %s)",
        (user_id, src_id, dst_id, link_type),
    )


def test_upgrade_creates_memory_links_with_expected_shape(base_db: str) -> None:
    """``memory_links`` lands with the columns, defaults and indexes ADR-0007 needs."""
    _alembic_to("006", base_db)

    assert _table_exists(base_db, "memory_links")
    assert _column_type(base_db, "memory_links", "id") is not None
    assert _column_type(base_db, "memory_links", "user_id") == "text"
    assert _column_type(base_db, "memory_links", "src_id") == "int4"
    assert _column_type(base_db, "memory_links", "dst_id") == "int4"
    assert _column_type(base_db, "memory_links", "link_type") == "text"
    assert _column_type(base_db, "memory_links", "created_at") == "timestamptz"

    # NOT NULLs
    for col in ("user_id", "src_id", "dst_id", "link_type"):
        rows = _query(
            base_db,
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name = 'memory_links' AND column_name = %s",
            (col,),
        )
        assert rows[0][0] == "NO", f"{col} must be NOT NULL"

    # both direction indexes
    assert _index_exists(base_db, "idx_memory_links_user_src")
    assert _index_exists(base_db, "idx_memory_links_user_dst")

    # created_at defaults to now(): an insert without it gets a timestamp.
    a = _insert_memory(base_db)
    b = _insert_memory(base_db)
    _insert_link(base_db, "u1", a, b, "see-also")
    rows = _query(base_db, "SELECT created_at FROM memory_links WHERE src_id = %s", (a,))
    assert rows and rows[0][0] is not None


def test_link_type_check_constraint_enforces_closed_enum(base_db: str) -> None:
    """The four ADR-0007 types insert; anything else violates the CHECK."""
    _alembic_to("006", base_db)
    a = _insert_memory(base_db)
    b = _insert_memory(base_db)

    for lt in LINK_TYPES:
        _insert_link(base_db, "u1", a, b, lt)  # all four accepted

    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert_link(base_db, "u1", a, b, "related-to")


def test_unique_edge_per_user(base_db: str) -> None:
    """UNIQUE(user_id, src_id, dst_id, link_type): duplicate edge rejected; the
    same pair under a different type — or the same edge by a different user — is fine."""
    _alembic_to("006", base_db)
    a = _insert_memory(base_db)
    b = _insert_memory(base_db)

    _insert_link(base_db, "u1", a, b, "see-also")
    _insert_link(base_db, "u1", a, b, "part-of")  # different type: OK
    _insert_link(base_db, "u2", a, b, "see-also")  # different user: OK

    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert_link(base_db, "u1", a, b, "see-also")


def test_fk_cascade_deletes_links_with_memory(base_db: str) -> None:
    """Deleting a memory hard-deletes its edges in BOTH directions (ON DELETE CASCADE)."""
    _alembic_to("006", base_db)
    a = _insert_memory(base_db)
    b = _insert_memory(base_db)
    c = _insert_memory(base_db)
    _insert_link(base_db, "u1", a, b, "supersedes")  # b's incoming
    _insert_link(base_db, "u1", b, c, "resolved-by")  # b's outgoing

    _execute(base_db, "DELETE FROM memories WHERE id = %s", (b,))

    rows = _query(base_db, "SELECT COUNT(*) FROM memory_links WHERE src_id = %s OR dst_id = %s", (b, b))
    assert int(rows[0][0]) == 0  # type: ignore[arg-type]


def test_upgrade_is_idempotent(base_db: str) -> None:
    """Stamp-back re-run of the 006 body over existing objects is a no-op."""
    _alembic_to("006", base_db)
    _alembic_to("stamp:005", base_db)  # rewind the pointer only; the table stays
    _alembic_to("006", base_db)  # re-run the body over the existing table → no-op

    assert _table_exists(base_db, "memory_links")
    assert _index_exists(base_db, "idx_memory_links_user_src")
    assert _index_exists(base_db, "idx_memory_links_user_dst")


def test_downgrade_drops_only_memory_links(base_db: str) -> None:
    """``downgrade`` removes memory_links and nothing else from the 005 schema."""
    _alembic_to("006", base_db)
    assert _table_exists(base_db, "memory_links")

    _alembic_to("down:005", base_db)

    assert not _table_exists(base_db, "memory_links")
    # 005-and-earlier schema intact
    for table in ("concepts", "concept_edges", "memory_concepts", "memories", "memory_shares"):
        assert _table_exists(base_db, table), f"{table} must survive the 006 downgrade"
    assert _column_type(base_db, "memories", "search_vector") == "tsvector"
    assert _index_exists(base_db, "idx_memories_search")
