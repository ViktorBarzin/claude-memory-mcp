import asyncio
import logging
import os

import asyncpg  # type: ignore[import-untyped]

from claude_memory.api import metrics

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

#: Seconds between pg_prewarm runs over the recall relations; 0 turns the loop off.
PREWARM_INTERVAL_ENV = "MEMORY_PREWARM_INTERVAL_S"
DEFAULT_PREWARM_INTERVAL_S = 60.0

#: Every relation one recall reads: the memories heap, its TOAST table and that table's
#: index (the OR-broaden detoasts every row's search_vector), the GIN index (AND-match),
#: the HNSW index (dense leg), and memory_links with its indexes (link semantics). On the
#: shared pg-cluster these fall out of shared_buffers between recalls and come back one
#: random read at a time from a disk that measured 27-170 ms per read, which is where
#: multi-second recalls came from. pg_prewarm reads whatever is missing and bumps the
#: usage count of what is resident, so it keeps ~130 MB hot for ~12 ms of work when
#: nothing was evicted. to_regclass skips a relation that does not exist (no pgvector,
#: an older schema).
PREWARM_SQL = """
SELECT coalesce(sum(pg_prewarm(rel)), 0)::bigint
FROM unnest(ARRAY[
    to_regclass('memories'),
    (SELECT nullif(reltoastrelid, 0)::regclass FROM pg_class WHERE oid = to_regclass('memories')),
    (SELECT i.indexrelid::regclass FROM pg_index i JOIN pg_class c ON c.reltoastrelid = i.indrelid
      WHERE c.oid = to_regclass('memories')),
    to_regclass('idx_memories_search'),
    to_regclass('idx_memories_embedding_hnsw'),
    to_regclass('memory_links')
] || ARRAY(SELECT indexrelid::regclass FROM pg_index WHERE indrelid = to_regclass('memory_links')))
    AS rel
WHERE rel IS NOT NULL
"""

pool: asyncpg.Pool | None = None


def run_migrations() -> None:
    """Run Alembic migrations to latest revision."""
    try:
        from alembic import command
        from alembic.config import Config

        alembic_cfg = Config()
        migrations_dir = os.environ.get("ALEMBIC_MIGRATIONS_DIR", "")
        if not migrations_dir:
            for candidate in [
                os.path.join(os.path.dirname(__file__), "..", "..", "..", "migrations"),
                os.path.join(os.getcwd(), "migrations"),
                "/app/migrations",
            ]:
                if os.path.isdir(candidate):
                    migrations_dir = candidate
                    break

        if not migrations_dir or not os.path.isdir(migrations_dir):
            logger.warning("Alembic migrations directory not found, skipping migrations")
            return

        alembic_cfg.set_main_option("script_location", migrations_dir)
        alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed successfully")
    except Exception as e:
        logger.warning("Failed to run Alembic migrations: %s", e)


async def init_pool() -> asyncpg.Pool:
    """Initialize connection pool with retries for database availability."""
    global pool
    run_migrations()

    max_retries = 5
    for attempt in range(max_retries):
        try:
            # max_inactive_connection_lifetime=0 keeps idle connections open. asyncpg's
            # default closes them after 300 s, and recalls arrive minutes apart, so a recall
            # often landed on a new backend with empty catalog caches whose first plan
            # re-read evicted pg_class/pg_statistic pages (0.8-2 s measured vs 3 ms warm).
            pool = await asyncpg.create_pool(
                DATABASE_URL, min_size=2, max_size=10, max_inactive_connection_lifetime=0
            )
            logger.info("Database pool initialized successfully")
            return pool
        except (asyncpg.CannotConnectNowError, OSError, ConnectionRefusedError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Database not ready (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, max_retries, e, wait)
                await asyncio.sleep(wait)
            else:
                raise


def prewarm_interval_s() -> float:
    """The keep-warm interval from the environment: default 60 s, 0 (or negative) = off."""
    raw = os.environ.get(PREWARM_INTERVAL_ENV, "").strip()
    if not raw:
        return DEFAULT_PREWARM_INTERVAL_S
    try:
        return max(float(raw), 0.0)
    except ValueError:
        logger.warning("%s=%r is not a number; using %ss", PREWARM_INTERVAL_ENV, raw, DEFAULT_PREWARM_INTERVAL_S)
        return DEFAULT_PREWARM_INTERVAL_S


async def keep_recall_relations_warm(db_pool: asyncpg.Pool, interval_s: float) -> None:
    """Re-read the recall relations with pg_prewarm every ``interval_s``, off the request path.

    Runs until cancelled. A failed run is logged and retried next interval; a database
    without the pg_prewarm extension stops the loop, since retrying cannot fix that.
    memory_prewarm_seconds shows how much had been evicted: ~10 ms means nothing was,
    seconds mean the run read pages back from disk that a recall would otherwise have.
    """
    while True:
        try:
            with metrics.PREWARM_LATENCY.time():
                async with db_pool.acquire() as conn:
                    await conn.fetchval(PREWARM_SQL)
        except asyncpg.UndefinedFunctionError:
            logger.warning("pg_prewarm is not installed; recall relations will not be kept warm")
            return
        except Exception as exc:  # noqa: BLE001 - keep-warm is best-effort, never fatal
            logger.warning("prewarming the recall relations failed: %s", exc)
        await asyncio.sleep(interval_s)


async def close_pool() -> None:
    global pool
    if pool:
        await pool.close()
        pool = None


async def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
