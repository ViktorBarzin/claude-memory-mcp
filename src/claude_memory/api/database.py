import asyncio
import logging
import os

import asyncpg

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get("DATABASE_URL", "")

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
            pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
            logger.info("Database pool initialized successfully")
            return pool
        except (asyncpg.CannotConnectNowError, OSError, ConnectionRefusedError) as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("Database not ready (attempt %d/%d): %s. Retrying in %ds...", attempt + 1, max_retries, e, wait)
                await asyncio.sleep(wait)
            else:
                raise


async def close_pool() -> None:
    global pool
    if pool:
        await pool.close()
        pool = None


async def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
