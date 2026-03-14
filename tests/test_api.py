"""Tests for the Claude Memory API endpoints."""

import importlib
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from claude_memory.api.auth import AuthUser


# Helpers to build mock asyncpg rows (they behave like dicts with attribute access)
class MockRow(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _make_memory_row(**overrides):
    now = datetime.now(timezone.utc)
    defaults = {
        "id": 1,
        "user_id": "testuser",
        "content": "test content",
        "category": "facts",
        "tags": "",
        "expanded_keywords": "",
        "importance": 0.5,
        "is_sensitive": False,
        "vault_path": None,
        "encrypted_content": None,
        "rank": 0.5,
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
    }
    defaults.update(overrides)
    return MockRow(defaults)


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool with connection context manager."""
    pool = MagicMock()
    conn = AsyncMock()

    # pool.acquire() returns an async context manager yielding conn
    acm = MagicMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acm

    return pool, conn


@pytest.fixture
def test_user():
    return AuthUser(user_id="testuser")


@pytest.fixture
def client(mock_pool, test_user):
    """Create an AsyncClient with mocked dependencies."""
    pool, conn = mock_pool

    # Reload modules with test API key
    with patch.dict(os.environ, {"API_KEY": "test-key", "API_KEYS": "", "DATABASE_URL": "postgresql://test"}):
        import claude_memory.api.auth as auth_mod
        import claude_memory.api.database as db_mod
        import claude_memory.api.app as app_mod

        importlib.reload(auth_mod)
        importlib.reload(db_mod)
        importlib.reload(app_mod)

        # Override database pool
        db_mod.pool = pool

        # Override auth to return our test user
        async def mock_get_user(authorization: str = ""):
            return test_user

        app_mod.app.dependency_overrides[auth_mod.get_current_user] = mock_get_user

        transport = ASGITransport(app=app_mod.app)
        return AsyncClient(transport=transport, base_url="http://test"), conn, app_mod


@pytest.mark.asyncio
async def test_health_endpoint_no_auth(client):
    ac, conn, app_mod = client
    async with ac:
        resp = await ac.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_store_memory_creates_record_with_user_id(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=42, category="facts", importance=0.7)

    async with ac:
        resp = await ac.post(
            "/api/memories",
            json={"content": "Python is great", "category": "facts", "importance": 0.7},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 42
    assert data["category"] == "facts"
    assert data["importance"] == 0.7

    # Verify INSERT was called with user_id
    call_args = conn.fetchrow.call_args
    assert call_args[0][1] == "testuser"  # user_id is the second positional arg


@pytest.mark.asyncio
async def test_recall_returns_only_user_memories(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=1, content="user memory", is_sensitive=False),
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "test query"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    results = data["memories"]
    assert len(results) == 1
    assert results[0]["content"] == "user memory"

    # Verify query includes user_id filter
    call_args = conn.fetch.call_args
    assert call_args[0][1] == "testuser"


@pytest.mark.asyncio
async def test_recall_redacts_sensitive_memories(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=5, content="[REDACTED]", is_sensitive=True),
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "secrets"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    results = data["memories"]
    assert "[SENSITIVE" in results[0]["content"]
    assert "secret_get(id=5)" in results[0]["content"]


@pytest.mark.asyncio
async def test_list_returns_only_user_memories(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=1, content="mem1"),
        _make_memory_row(id=2, content="mem2"),
    ]

    async with ac:
        resp = await ac.get(
            "/api/memories",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    results = data["memories"]
    assert len(results) == 2

    # Verify user_id filter
    call_args = conn.fetch.call_args
    assert call_args[0][1] == "testuser"


@pytest.mark.asyncio
async def test_delete_only_user_memories(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=10, vault_path=None, preview="test content")
    conn.execute.return_value = None

    async with ac:
        resp = await ac.delete(
            "/api/memories/10",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["deleted"] == 10
    assert "preview" in data

    # Verify both SELECT and DELETE include user_id
    fetchrow_args = conn.fetchrow.call_args
    assert fetchrow_args[0][1] == 10  # memory_id
    assert fetchrow_args[0][2] == "testuser"  # user_id


@pytest.mark.asyncio
async def test_delete_nonexistent_memory_returns_404(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None

    async with ac:
        resp = await ac.delete(
            "/api/memories/999",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_secret_endpoint_returns_plaintext(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(
        id=7, content="my secret value", is_sensitive=False,
        vault_path=None, encrypted_content=None,
    )

    async with ac:
        resp = await ac.post(
            "/api/memories/7/secret",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 7
    assert data["content"] == "my secret value"
    assert data["source"] == "plaintext"


@pytest.mark.asyncio
async def test_secret_endpoint_returns_vault_content(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(
        id=8, content="[REDACTED]", is_sensitive=True,
        vault_path="claude-memory/testuser/mem-8", encrypted_content=None,
    )

    with patch("claude_memory.api.app.get_secret", return_value="actual-secret-from-vault"):
        async with ac:
            resp = await ac.post(
                "/api/memories/8/secret",
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["content"] == "actual-secret-from-vault"
    assert data["source"] == "vault"


@pytest.mark.asyncio
async def test_secret_endpoint_nonexistent_returns_404(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None

    async with ac:
        resp = await ac.post(
            "/api/memories/999/secret",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_import_memories(client):
    ac, conn, app_mod = client
    conn.fetchrow.side_effect = [
        _make_memory_row(id=100, category="facts", importance=0.5),
        _make_memory_row(id=101, category="preferences", importance=0.8),
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/import",
            json=[
                {"content": "fact one", "category": "facts"},
                {"content": "pref one", "category": "preferences", "importance": 0.8},
            ],
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["id"] == 100
    assert data[1]["id"] == 101


# ─── Sync endpoint tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sync_full_dump_without_since(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=1, content="mem1", deleted_at=None),
        _make_memory_row(id=2, content="mem2", deleted_at=None),
    ]

    async with ac:
        resp = await ac.get(
            "/api/memories/sync",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 2
    assert "server_time" in data
    assert data["memories"][0]["id"] == 1
    assert data["memories"][1]["id"] == 2

    # Without since param, should query non-deleted only
    call_args = conn.fetch.call_args
    query = call_args[0][0]
    assert "deleted_at IS NULL" in query


@pytest.mark.asyncio
async def test_sync_incremental_with_since(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=3, content="updated mem", deleted_at=None),
    ]

    async with ac:
        resp = await ac.get(
            "/api/memories/sync?since=2026-03-14T10:00:00Z",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 1

    # With since param, should include updated_at filter (includes soft-deleted)
    call_args = conn.fetch.call_args
    query = call_args[0][0]
    assert "updated_at >" in query
    assert "deleted_at IS NULL" not in query


@pytest.mark.asyncio
async def test_sync_includes_soft_deleted_with_since(client):
    ac, conn, app_mod = client
    now = datetime.now(timezone.utc)
    conn.fetch.return_value = [
        _make_memory_row(id=5, content="deleted mem", deleted_at=now),
    ]

    async with ac:
        resp = await ac.get(
            "/api/memories/sync?since=2026-03-14T10:00:00Z",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 1
    assert data["memories"][0]["deleted_at"] is not None


# ─── Soft delete tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_is_soft_delete(client):
    """Delete should SET deleted_at, not DELETE the row."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=10, vault_path=None, preview="test content")
    conn.execute.return_value = None

    async with ac:
        resp = await ac.delete(
            "/api/memories/10",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200

    # Verify the execute call uses UPDATE SET deleted_at, not DELETE
    execute_args = conn.execute.call_args
    query = execute_args[0][0]
    assert "UPDATE" in query
    assert "deleted_at" in query
    assert "DELETE" not in query.upper().split("SET")[0]  # No DELETE before SET


@pytest.mark.asyncio
async def test_delete_excludes_already_deleted(client):
    """DELETE endpoint should not find already-deleted memories."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None  # Not found because deleted_at IS NULL filter

    async with ac:
        resp = await ac.delete(
            "/api/memories/10",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 404

    # Verify query includes deleted_at IS NULL
    call_args = conn.fetchrow.call_args
    query = call_args[0][0]
    assert "deleted_at IS NULL" in query
