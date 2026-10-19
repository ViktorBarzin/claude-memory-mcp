"""Tests for the recall-relation keep-warm loop in ``api/database.py``.

Recall reads the memories heap, its TOAST, the GIN index and the HNSW index. On the
shared Postgres those pages fall out of shared_buffers between recalls and come back
one random read at a time from a slow disk, which is the multi-second recall tail.
``keep_recall_relations_warm`` re-reads them with pg_prewarm on a timer, off the
request path.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg  # type: ignore[import-untyped]
import pytest

from claude_memory.api import database


def _pool_with(conn: AsyncMock) -> MagicMock:
    pool = MagicMock()
    acm = MagicMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acm
    return pool


def _sample(name: str) -> float:
    from prometheus_client import REGISTRY

    return REGISTRY.get_sample_value(name) or 0.0


@pytest.mark.asyncio
async def test_prewarm_reads_every_recall_relation_then_waits_the_interval():
    conn = AsyncMock()
    conn.fetchval.return_value = 16_000
    sleep = AsyncMock(side_effect=asyncio.CancelledError)
    before = _sample("memory_prewarm_seconds_count")

    with patch.object(database.asyncio, "sleep", sleep), pytest.raises(asyncio.CancelledError):
        await database.keep_recall_relations_warm(_pool_with(conn), interval_s=120)

    sql = conn.fetchval.call_args.args[0]
    assert "pg_prewarm" in sql
    for relation in ("'memories'", "reltoastrelid", "idx_memories_search", "idx_memories_embedding_hnsw"):
        assert relation in sql, relation
    sleep.assert_awaited_once_with(120)
    assert _sample("memory_prewarm_seconds_count") == before + 1


@pytest.mark.asyncio
async def test_prewarm_stops_when_the_extension_is_missing():
    conn = AsyncMock()
    conn.fetchval.side_effect = asyncpg.UndefinedFunctionError("function pg_prewarm does not exist")
    sleep = AsyncMock()

    with patch.object(database.asyncio, "sleep", sleep):
        await database.keep_recall_relations_warm(_pool_with(conn), interval_s=120)

    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_prewarm_survives_a_failed_run():
    conn = AsyncMock()
    conn.fetchval.side_effect = OSError("connection reset")
    sleep = AsyncMock(side_effect=asyncio.CancelledError)

    with patch.object(database.asyncio, "sleep", sleep), pytest.raises(asyncio.CancelledError):
        await database.keep_recall_relations_warm(_pool_with(conn), interval_s=30)

    sleep.assert_awaited_once_with(30)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, 60.0), ("", 60.0), ("0", 0.0), ("45", 45.0), ("-5", 0.0), ("soon", 60.0)],
)
def test_prewarm_interval_from_env(raw, expected):
    env = {} if raw is None else {"MEMORY_PREWARM_INTERVAL_S": raw}
    with patch.dict(os.environ, env, clear=False):
        if raw is None:
            os.environ.pop("MEMORY_PREWARM_INTERVAL_S", None)
        assert database.prewarm_interval_s() == expected
