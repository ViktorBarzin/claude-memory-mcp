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
    results = resp.json()
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
    results = resp.json()
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
    results = resp.json()
    assert len(results) == 2

    # Verify user_id filter
    call_args = conn.fetch.call_args
    assert call_args[0][1] == "testuser"


@pytest.mark.asyncio
async def test_delete_only_user_memories(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=10, vault_path=None)
    conn.execute.return_value = None

    async with ac:
        resp = await ac.delete(
            "/api/memories/10",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    assert resp.json() == {"deleted": 10}

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
