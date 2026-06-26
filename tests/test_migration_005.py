"""PG-only tests for Alembic migration ``005_add_embeddings_and_graph``.

Migrations are **PostgreSQL-only** — the schema uses ``information_schema``,
``SERIAL``, a generated ``tsvector`` column, a GIN index and pgvector's
``halfvec`` type, none of which run on SQLite. There is **no** pre-existing
migration-test pattern in this repo to mirror (``tests/`` has none), so this
module is net-new and deliberately spins up a *real* Postgres.

Postgres source resolution (first that works wins):

1. ``MEMORY_TEST_DATABASE_URL`` — a guarded URL to a throwaway Postgres that
   already has the ``vector`` extension *available* (``CREATE EXTENSION`` must
   succeed). Used verbatim; the test owns the schema and rolls it back.
2. A disposable ``pgvector/pgvector:pg16`` container driven through the
   ``docker`` CLI (no ``testcontainers`` dependency). Matches the production
   PG16 + pgvector substrate.

If neither is available the whole module **skips** (never fails) — CI without a
Docker daemon and a guard URL simply does not exercise these. The image is
pulled on demand the first time; subsequent runs reuse the local layer cache.

What is asserted (the S7 acceptance set):

* ``upgrade`` is **idempotent** — running it twice is a no-op (guards via
  ``IF NOT EXISTS`` / ``_table_exists`` / ``_column_exists``).
* the ``embedding`` column is ``halfvec(1024)`` and **NULL-able**.
* the HNSW index exists, uses ``halfvec_cosine_ops`` and was built with the
  ADR-0006 build params, and is created **CONCURRENTLY inside an
  ``autocommit_block``** (asserted by the migration succeeding at all — a bare
  ``CREATE INDEX CONCURRENTLY`` inside ``env.py``'s wrapping transaction would
  raise ``25001``).
* the three graph tables (``concepts`` / ``concept_edges`` /
  ``memory_concepts``) are created.
* the lexical schema is **UNTOUCHED**: the generated ``search_vector`` column
  and the ``idx_memories_search`` GIN index from migration 001 survive every
  upgrade/downgrade cycle.
* ``downgrade`` drops **only** the additions and leaves the migration-004
  schema exactly as it was.
"""

from __future__ import annotations

import os
import secrets
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# psycopg2 ships in the ``api`` optional extra (CI runs ``uv sync --all-extras``).
# If it is somehow absent the module skips rather than erroring at collection.
psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 (api extra) required for PG migration tests")

REPO_ROOT = Path(__file__).resolve().parents[1]
PGVECTOR_IMAGE = "pgvector/pgvector:pg16"
#: stock PG16 with NO pgvector — the pre-infra substrate for the gating test.
PLAIN_PG_IMAGE = "postgres:16"
_CONTAINER_READY_TIMEOUT_S = 60.0


def _docker_available() -> bool:
    """True iff a usable ``docker`` CLI + running daemon are present."""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _wait_until_ready(dsn: str) -> None:
    """Poll ``dsn`` until a connection succeeds or the timeout elapses."""
    deadline = time.monotonic() + _CONTAINER_READY_TIMEOUT_S
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            conn = psycopg2.connect(dsn)
        except psycopg2.OperationalError as exc:  # not up yet
            last_err = exc
            time.sleep(0.5)
            continue
        conn.close()
        return
    raise RuntimeError(f"Postgres at {dsn} not ready within {_CONTAINER_READY_TIMEOUT_S}s: {last_err}")


def _start_pg_container(image: str) -> tuple[str, str]:
    """Start a disposable Postgres container from ``image``.

    Returns ``(database_url, container_name)``. The caller is responsible for
    ``docker rm -f`` on the returned name. Pulls the image on first use.
    """
    # Map the container's 5432 to an ephemeral host port (``-P``-style explicit
    # publish to 0 lets Docker choose, avoiding host-port collisions when tests
    # run in parallel CI lanes).
    name = f"cm-pg-test-{secrets.token_hex(6)}"
    password = "postgres"  # noqa: S105 — throwaway local container, never networked
    run = subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--rm",
            "--name",
            name,
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "-e",
            "POSTGRES_DB=claude_memory",
            "-p",
            "127.0.0.1:0:5432",
            image,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(f"failed to start {image}: {run.stderr.strip()}")

    port_proc = subprocess.run(
        ["docker", "port", name, "5432/tcp"],
        capture_output=True,
        text=True,
        check=True,
    )
    # e.g. "127.0.0.1:49153" — take the trailing port.
    host_port = port_proc.stdout.strip().splitlines()[0].rsplit(":", 1)[1]
    dsn = f"postgresql://postgres:{password}@127.0.0.1:{host_port}/claude_memory"
    _wait_until_ready(dsn)
    return dsn, name


@pytest.fixture(scope="module")
def pg_url() -> Iterator[str]:
    """A live Postgres URL with pgvector *available*, or skip the module.

    Prefers a guarded ``MEMORY_TEST_DATABASE_URL`` so CI/operators can point at
    an existing throwaway PG; otherwise spins a disposable pgvector container.
    """
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
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


@pytest.fixture(scope="module")
def plain_pg_url() -> Iterator[str]:
    """A live Postgres with pgvector **NOT** available (stock ``postgres:16``).

    Models the pre-infra state: the operand image does not yet bundle pgvector,
    so ``CREATE EXTENSION vector`` would hard-fail. Used to prove migration 005's
    availability gate makes it a clean no-op for the embedding column + HNSW
    index while the graph tables still land. Docker-only (no guarded-URL form —
    a guard URL is assumed pgvector-enabled); skips without a daemon.
    """
    if not _docker_available():
        pytest.skip(f"no Docker daemon: the pgvector-absent gating test needs {PLAIN_PG_IMAGE}")
    dsn, name = _start_pg_container(PLAIN_PG_IMAGE)
    try:
        yield dsn
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _alembic_to(revision: str, db_url: str) -> None:
    """Drive Alembic against ``db_url`` via the command API, in a child process.

    ``revision`` selects the operation:

    * ``"<rev>"``        — ``command.upgrade`` to that revision (e.g. ``"004"``, ``"005"``).
    * ``"down:<rev>"``   — ``command.downgrade`` to that revision.
    * ``"stamp:<rev>"``  — ``command.stamp`` (move the version pointer only, no DDL).

    A child process gives each call a clean Alembic/SQLAlchemy state and the
    ``DATABASE_URL`` env the project's ``env.py`` reads. ``REPO_ROOT`` is the cwd
    so ``alembic.ini`` / ``script_location = migrations`` resolve.
    """
    env = dict(os.environ)
    env["DATABASE_URL"] = db_url
    if revision.startswith("down:"):
        op_call = f"command.downgrade(cfg, {revision.split(':', 1)[1]!r})"
    elif revision.startswith("stamp:"):
        op_call = f"command.stamp(cfg, {revision.split(':', 1)[1]!r})"
    else:
        op_call = f"command.upgrade(cfg, {revision!r})"
    code = "from alembic.config import Config; from alembic import command; cfg = Config('alembic.ini'); " + op_call
    # sys.executable (NOT a bare "python"): the subprocess must use the SAME
    # interpreter running pytest so it sees alembic from this venv. A bare "python"
    # resolves to whatever is first on PATH (system python, no alembic) when pytest
    # is invoked as ".venv/bin/python -m pytest" rather than via "uv run".
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(f"alembic {revision} failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")


def _query(db_url: str, sql: str, params: tuple[object, ...] = ()) -> list[tuple[object, ...]]:
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return list(cur.fetchall())
    finally:
        conn.close()


def _column_type(db_url: str, table: str, column: str) -> str | None:
    rows = _query(
        db_url,
        "SELECT data_type, udt_name, is_nullable FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        (table, column),
    )
    if not rows:
        return None
    # udt_name carries the concrete extension type (e.g. 'halfvec'); data_type
    # is 'USER-DEFINED' for it.
    return str(rows[0][1])


def _is_nullable(db_url: str, table: str, column: str) -> bool:
    rows = _query(
        db_url,
        "SELECT is_nullable FROM information_schema.columns WHERE table_name = %s AND column_name = %s",
        (table, column),
    )
    return bool(rows) and rows[0][0] == "YES"


def _table_exists(db_url: str, table: str) -> bool:
    rows = _query(
        db_url,
        "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        (table,),
    )
    return bool(rows[0][0])


def _index_exists(db_url: str, index: str) -> bool:
    rows = _query(
        db_url,
        "SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = %s)",
        (index,),
    )
    return bool(rows[0][0])


def _index_def(db_url: str, index: str) -> str:
    rows = _query(db_url, "SELECT indexdef FROM pg_indexes WHERE indexname = %s", (index,))
    return str(rows[0][0]) if rows else ""


@pytest.fixture()
def base_db(pg_url: str) -> str:
    """A database migrated up to **004** (the pre-005 baseline) for each test.

    Each test gets a freshly-reset schema so idempotency / downgrade assertions
    do not leak across tests. We drop the public schema wholesale (fast, total)
    then run 001..004.
    """
    conn = psycopg2.connect(pg_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
    finally:
        conn.close()
    _alembic_to("004", pg_url)
    return pg_url


def test_upgrade_adds_embedding_halfvec_nullable(base_db: str) -> None:
    """``embedding`` lands as a NULL-able ``halfvec(1024)`` column."""
    _alembic_to("005", base_db)

    assert _column_type(base_db, "memories", "embedding") == "halfvec"
    assert _is_nullable(base_db, "memories", "embedding") is True
    # column width is halfvec(1024): atttypmod encodes the dimension.
    dim = _query(
        base_db,
        "SELECT atttypmod FROM pg_attribute "
        "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'",
    )
    assert dim and int(dim[0][0]) == 1024


def test_upgrade_creates_hnsw_index_with_cosine_ops(base_db: str) -> None:
    """The HNSW index exists, uses ``halfvec_cosine_ops`` and the ADR-0006 params.

    That the upgrade even *succeeds* proves ``CREATE INDEX CONCURRENTLY`` ran
    inside an ``autocommit_block`` — env.py wraps the migration in a transaction,
    and a bare CONCURRENTLY there raises ``25001 active sql transaction``.
    """
    _alembic_to("005", base_db)

    assert _index_exists(base_db, "idx_memories_embedding_hnsw")
    idxdef = _index_def(base_db, "idx_memories_embedding_hnsw").lower()
    assert "using hnsw" in idxdef
    assert "halfvec_cosine_ops" in idxdef
    assert "m='16'" in idxdef or "m=16" in idxdef
    assert "ef_construction='64'" in idxdef or "ef_construction=64" in idxdef


def test_upgrade_creates_graph_tables(base_db: str) -> None:
    """The three additive graph tables are created."""
    _alembic_to("005", base_db)

    for table in ("concepts", "concept_edges", "memory_concepts"):
        assert _table_exists(base_db, table), f"{table} missing after upgrade"

    # spot-check the key columns mirror the offline dataclasses (graph_build.py)
    assert _column_type(base_db, "concepts", "canonical_name") is not None
    assert _column_type(base_db, "concepts", "embedding") == "halfvec"
    assert _column_type(base_db, "concept_edges", "relation") is not None
    assert _column_type(base_db, "concept_edges", "evidence_memory_ids") is not None
    assert _column_type(base_db, "memory_concepts", "concept_id") is not None


def test_upgrade_is_idempotent(base_db: str) -> None:
    """Re-running the upgrade body when its objects already exist is a no-op.

    A plain second ``command.upgrade(005)`` would short-circuit at the version
    table and never touch the body. To exercise the body's IF-NOT-EXISTS /
    ``_table_exists`` / ``_column_exists`` guards against an
    already-fully-migrated database, we ``stamp`` the version pointer back to
    004 (DDL untouched — the 005 column/index/tables remain) and upgrade to 005
    a second time. The body must run clean over the existing objects.
    """
    _alembic_to("005", base_db)
    _alembic_to("stamp:004", base_db)  # rewind the pointer only; 005 objects stay
    _alembic_to("005", base_db)  # re-run the body over existing objects → no-op

    # everything still present (and not duplicated — UniqueConstraints would have
    # raised on a re-create that slipped past the guards).
    assert _column_type(base_db, "memories", "embedding") == "halfvec"
    assert _index_exists(base_db, "idx_memories_embedding_hnsw")
    for table in ("concepts", "concept_edges", "memory_concepts"):
        assert _table_exists(base_db, table)


def test_lexical_schema_untouched_by_upgrade(base_db: str) -> None:
    """The generated ``search_vector`` column + GIN index from 001 survive."""
    # baseline (pre-005)
    assert _column_type(base_db, "memories", "search_vector") == "tsvector"
    assert _index_exists(base_db, "idx_memories_search")
    gin_before = _index_def(base_db, "idx_memories_search")

    _alembic_to("005", base_db)

    assert _column_type(base_db, "memories", "search_vector") == "tsvector"
    assert _index_exists(base_db, "idx_memories_search")
    assert _index_def(base_db, "idx_memories_search") == gin_before  # byte-identical


def test_downgrade_drops_only_additions(base_db: str) -> None:
    """``downgrade`` removes the 005 additions and restores the 004 schema."""
    _alembic_to("005", base_db)
    assert _column_type(base_db, "memories", "embedding") == "halfvec"

    _alembic_to("down:004", base_db)

    # additions gone
    assert _column_type(base_db, "memories", "embedding") is None
    assert not _index_exists(base_db, "idx_memories_embedding_hnsw")
    for table in ("concepts", "concept_edges", "memory_concepts"):
        assert not _table_exists(base_db, table), f"{table} should be dropped by downgrade"

    # 004-and-earlier schema intact: lexical + sharing + sensitivity columns
    assert _column_type(base_db, "memories", "search_vector") == "tsvector"
    assert _index_exists(base_db, "idx_memories_search")
    assert _column_type(base_db, "memories", "is_sensitive") is not None
    assert _table_exists(base_db, "memory_shares")


def test_upgrade_gated_when_pgvector_unavailable(plain_pg_url: str) -> None:
    """Before infra enables pgvector, 005 no-ops the vector steps but still lands
    the graph tables — proving the availability gate makes it pre-infra-safe.

    On stock ``postgres:16`` ``CREATE EXTENSION vector`` would hard-fail
    ("extension is not available"); the migration must NOT attempt it. The
    embedding column, the HNSW index, and ``concepts.embedding`` are skipped; the
    (vector-free) graph tables are created so the graph leg's schema is ready.
    """
    # fresh 004 baseline on the no-pgvector server
    conn = psycopg2.connect(plain_pg_url)
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE")
            cur.execute("CREATE SCHEMA public")
    finally:
        conn.close()
    _alembic_to("004", plain_pg_url)

    # sanity: pgvector genuinely unavailable here
    avail = _query(
        plain_pg_url,
        "SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')",
    )
    assert avail[0][0] is False, "fixture invariant: postgres:16 must NOT have pgvector available"

    _alembic_to("005", plain_pg_url)  # must succeed despite no pgvector

    # vector steps skipped
    assert _column_type(plain_pg_url, "memories", "embedding") is None
    assert not _index_exists(plain_pg_url, "idx_memories_embedding_hnsw")
    assert _column_type(plain_pg_url, "concepts", "embedding") is None

    # graph tables (vector-free parts) still land
    for table in ("concepts", "concept_edges", "memory_concepts"):
        assert _table_exists(plain_pg_url, table), f"{table} should still be created without pgvector"

    # lexical schema untouched
    assert _column_type(plain_pg_url, "memories", "search_vector") == "tsvector"
    assert _index_exists(plain_pg_url, "idx_memories_search")

    # and downgrade is clean even though the vector objects were never created
    _alembic_to("down:004", plain_pg_url)
    for table in ("concepts", "concept_edges", "memory_concepts"):
        assert not _table_exists(plain_pg_url, table)
