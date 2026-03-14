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
        # Find migrations directory relative to this file or project root
        migrations_dir = os.environ.get("ALEMBIC_MIGRATIONS_DIR", "")
        if not migrations_dir:
            # Check common locations
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
    global pool
    run_migrations()
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    return pool


async def close_pool() -> None:
    global pool
    if pool:
        await pool.close()
        pool = None


async def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
