import os

import asyncpg

DATABASE_URL = os.environ.get("DATABASE_URL", "")

pool: asyncpg.Pool | None = None


async def init_pool() -> asyncpg.Pool:
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id SERIAL PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL DEFAULT 'default',
                content TEXT NOT NULL,
                category VARCHAR(50) DEFAULT 'facts',
                tags TEXT DEFAULT '',
                expanded_keywords TEXT DEFAULT '',
                importance REAL DEFAULT 0.5,
                is_sensitive BOOLEAN DEFAULT FALSE,
                vault_path TEXT DEFAULT NULL,
                encrypted_content BYTEA DEFAULT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                search_vector tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(expanded_keywords, '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(tags, '')), 'C') ||
                    setweight(to_tsvector('english', coalesce(category, '')), 'D')
                ) STORED
            )
        """)
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING GIN(search_vector)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)"
        )
    return pool


async def close_pool():
    global pool
    if pool:
        await pool.close()
        pool = None


async def get_pool() -> asyncpg.Pool:
    if pool is None:
        raise RuntimeError("Database pool not initialized")
    return pool
