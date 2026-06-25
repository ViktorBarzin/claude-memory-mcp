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
        "owner": "testuser",
        "shared_by": None,
        "share_permission": None,
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
async def test_auth_check_endpoint(client):
    ac, conn, app_mod = client
    async with ac:
        resp = await ac.get(
            "/api/auth-check",
            headers={"Authorization": "Bearer test-key"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["user_id"] == "testuser"


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
async def test_recall_returns_all_memories(client):
    ac, conn, app_mod = client
    # recall now runs a single query (all memories are public)
    conn.fetch.return_value = [
        _make_memory_row(id=1, content="user memory", is_sensitive=False, owner="testuser", shared_by=None),
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
    assert results[0]["owner"] == "testuser"


@pytest.mark.asyncio
async def test_recall_default_limit_is_capped(client):
    """Default recall limit must be a small top-N (30), not the whole store.

    Regression for the 'recall returns the entire ~1460-memory store' bug:
    the default was 10000, so a default call returned everything.
    """
    ac, conn, app_mod = client
    conn.fetch.return_value = [_make_memory_row(id=1)]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "singleword"},  # 1 word -> no OR-broadening
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    # fetch(sql, user_id, query_text, limit[, category]) -> limit is the 4th arg
    first_args = conn.fetch.call_args_list[0].args
    assert first_args[3] == 30, f"default recall limit should be 30, got {first_args[3]}"


@pytest.mark.asyncio
async def test_recall_or_broadening_is_relevance_bounded(client):
    """When the precise AND-match is sparse, the OR-broadening fallback must
    order by relevance (ts_rank) and apply a minimum-rank floor — not pad up to
    `limit` ordered by the importance hybrid (which floods with high-importance
    but irrelevant memories).
    """
    ac, conn, app_mod = client
    conn.fetch.side_effect = [
        [_make_memory_row(id=1)],  # AND-match: 1 row (< limit -> triggers OR)
        [_make_memory_row(id=2)],  # OR-broadening result
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "two words"},  # >1 word -> OR-broadening fires
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    assert len(conn.fetch.call_args_list) == 2, "OR-broadening query should fire"
    or_sql = conn.fetch.call_args_list[1].args[0]
    assert "ts_rank(search_vector, query) DESC" in or_sql, "OR matches must be ordered by relevance"
    assert "ts_rank(search_vector, query) >" in or_sql, "OR matches must have a minimum-rank floor"


@pytest.mark.asyncio
async def test_recall_redacts_sensitive_memories(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=5, content="[REDACTED]", is_sensitive=True, owner="testuser", shared_by=None),
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
async def test_list_returns_all_memories(client):
    ac, conn, app_mod = client
    conn.fetch.return_value = [
        _make_memory_row(id=1, content="mem1", owner="testuser"),
        _make_memory_row(id=2, content="mem2", owner="otheruser"),
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
    assert results[0]["owner"] == "testuser"
    assert results[1]["owner"] == "otheruser"


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
async def test_delete_nonexistent_memory_is_idempotent(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None

    async with ac:
        resp = await ac.delete(
            "/api/memories/999",
            headers={"Authorization": "Bearer test-key"},
        )

    # Idempotent: returns 200 even if already deleted (prevents retry loops)
    assert resp.status_code == 200
    assert resp.json()["preview"] == "[already deleted]"


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
    """DELETE endpoint filters by deleted_at IS NULL and returns idempotent 200."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None  # Not found because deleted_at IS NULL filter

    async with ac:
        resp = await ac.delete(
            "/api/memories/10",
            headers={"Authorization": "Bearer test-key"},
        )

    # Idempotent: returns 200 even if already soft-deleted
    assert resp.status_code == 200
    assert resp.json()["preview"] == "[already deleted]"

    # Verify query includes deleted_at IS NULL
    call_args = conn.fetchrow.call_args
    query = call_args[0][0]
    assert "deleted_at IS NULL" in query


# ─── Sharing endpoint tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_share_memory_creates_record(client):
    """POST /api/memories/{id}/share creates a sharing record."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=10, user_id="testuser")
    conn.execute.return_value = None

    async with ac:
        resp = await ac.post(
            "/api/memories/10/share",
            json={"shared_with": "otheruser", "permission": "read"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["shared"] == 10
    assert data["with"] == "otheruser"
    assert data["permission"] == "read"


@pytest.mark.asyncio
async def test_share_memory_nonexistent_returns_404(client):
    """POST /api/memories/{id}/share returns 404 for non-existent memory."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None

    async with ac:
        resp = await ac.post(
            "/api/memories/999/share",
            json={"shared_with": "otheruser", "permission": "read"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unshare_memory(client):
    """DELETE /api/memories/{id}/share/{user} removes sharing."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=10, user_id="testuser")
    conn.execute.return_value = None

    async with ac:
        resp = await ac.delete(
            "/api/memories/10/share/otheruser",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_share_tag_creates_record(client):
    """POST /api/memories/share-tag creates a tag sharing record."""
    ac, conn, app_mod = client
    conn.execute.return_value = None

    async with ac:
        resp = await ac.post(
            "/api/memories/share-tag",
            json={"tag": "python", "shared_with": "otheruser", "permission": "read"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["shared_tag"] == "python"
    assert data["with"] == "otheruser"
    assert data["permission"] == "read"


@pytest.mark.asyncio
async def test_unshare_tag(client):
    """DELETE /api/memories/share-tag removes tag sharing."""
    ac, conn, app_mod = client
    conn.execute.return_value = None

    async with ac:
        resp = await ac.request(
            "DELETE",
            "/api/memories/share-tag",
            json={"tag": "python", "shared_with": "otheruser"},
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["unshared_tag"] == "python"


@pytest.mark.asyncio
async def test_shared_with_me_returns_shared_memories(client):
    """GET /api/memories/shared-with-me returns individually and tag-shared memories."""
    ac, conn, app_mod = client
    # Mock conn.fetch called twice: individual shares, then tag shares
    # Need to include permission field for the sharing queries
    conn.fetch.side_effect = [
        [_make_memory_row(id=1, content="shared memory", user_id="owner1", shared_by="owner1", permission="read")],  # individual
        [_make_memory_row(id=2, content="tag shared", user_id="owner2", shared_by="owner2", permission="write")],  # tag-shared
    ]

    async with ac:
        resp = await ac.get(
            "/api/memories/shared-with-me",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memories"]) == 2
    assert data["memories"][0]["id"] == 1
    assert data["memories"][1]["id"] == 2


@pytest.mark.asyncio
async def test_my_shares_returns_outgoing_shares(client):
    """GET /api/memories/my-shares returns outgoing memory and tag shares."""
    ac, conn, app_mod = client
    now = datetime.now(timezone.utc)
    # Mock conn.fetch called twice: memory_shares, then tag_shares
    conn.fetch.side_effect = [
        [MockRow({"memory_id": 1, "shared_with": "user1", "permission": "read", "preview": "memory preview", "created_at": now})],  # memory_shares
        [MockRow({"tag": "python", "shared_with": "user2", "permission": "write", "created_at": now})],  # tag_shares
    ]

    async with ac:
        resp = await ac.get(
            "/api/memories/my-shares",
            headers={"Authorization": "Bearer test-key"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert len(data["memory_shares"]) == 1
    assert len(data["tag_shares"]) == 1
    assert data["memory_shares"][0]["memory_id"] == 1
    assert data["tag_shares"][0]["tag"] == "python"


@pytest.mark.asyncio
async def test_recall_includes_all_users_memories(client):
    """POST /api/memories/recall returns all users' memories with owner field."""
    ac, conn, app_mod = client
    # Single query returns all memories (public by default)
    conn.fetch.return_value = [
        _make_memory_row(id=1, content="own memory", owner="testuser", shared_by=None),
        _make_memory_row(id=2, content="other memory", owner="owner1", shared_by="owner1"),
        _make_memory_row(id=3, content="another memory", owner="owner2", shared_by="owner2"),
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
    assert len(results) == 3
    assert results[0]["owner"] == "testuser"
    assert results[0]["shared_by"] is None
    assert results[1]["owner"] == "owner1"
    assert results[1]["shared_by"] == "owner1"


@pytest.mark.asyncio
async def test_update_shared_memory_with_write_permission(client):
    """PUT /api/memories/{id} succeeds when user has write permission."""
    ac, conn, app_mod = client

    # Mock check_memory_permission to return (True, "owner")
    async def mock_check_permission(conn, memory_id, user_id, perm):
        return (True, "owner")

    with patch("claude_memory.api.app.check_memory_permission", side_effect=mock_check_permission):
        conn.execute.return_value = None

        async with ac:
            resp = await ac.put(
                "/api/memories/10",
                json={"content": "updated content"},
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["updated"] == 10


@pytest.mark.asyncio
async def test_update_shared_memory_without_write_fails(client):
    """PUT /api/memories/{id} returns 403 when user lacks write permission."""
    ac, conn, app_mod = client

    # Mock check_memory_permission to return (False, "owner")
    async def mock_check_permission(conn, memory_id, user_id, perm):
        return (False, "owner")

    with patch("claude_memory.api.app.check_memory_permission", side_effect=mock_check_permission):
        async with ac:
            resp = await ac.put(
                "/api/memories/10",
                json={"content": "updated content"},
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 403


# ─── Shared fused-recall helper (S8) ─────────────────────────────────────────
#
# api/recall._fused_recall is the ONE retrieval helper both recall entry points
# (REST recall_memories + FastMCP memory_recall) call, so the lexical→hybrid logic
# cannot drift between them. The endpoints keep their own response-shaping loops
# (intentionally different); the helper returns the ordered rows.


def _mock_conn():
    """A bare AsyncMock conn with an async-context-manager pool wrapping it."""
    conn = AsyncMock()
    pool = MagicMock()
    acm = MagicMock()
    acm.__aenter__ = AsyncMock(return_value=conn)
    acm.__aexit__ = AsyncMock(return_value=False)
    pool.acquire.return_value = acm
    return pool, conn


@pytest.mark.asyncio
async def test_fused_recall_flags_off_runs_verbatim_ts_rank_sql():
    """challenger must_fix #3: with BOTH flags OFF, _fused_recall is a TRUE no-op —
    it runs the EXACT current ts_rank SQL (additive ts_rank*0.7+importance*0.3 blend,
    plainto_tsquery AND-match), NOT an RRF collapse, and returns the DB rows verbatim.
    """
    from claude_memory.api import recall as recall_mod

    pool, conn = _mock_conn()
    rows = [_make_memory_row(id=1, rank=0.9), _make_memory_row(id=2, rank=0.1)]
    conn.fetch.return_value = rows

    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "", "MEMORY_GRAPH_ENABLED": ""}):
        out = await recall_mod._fused_recall(
            conn, user_id="u", query_text="single", sort_by="relevance",
            category=None, limit=30, pool=pool,
        )

    # Returns the lexical rows unchanged (verbatim no-op).
    assert out == rows
    # Exactly the current lexical query ran; no dense/HNSW leg, no RRF.
    assert conn.fetch.call_count == 1
    sql = conn.fetch.call_args_list[0].args[0]
    assert "ts_rank(search_vector, query) * 0.7 + importance * 0.3" in sql
    assert "plainto_tsquery('english'" in sql
    assert "<=>" not in sql  # no dense leg


@pytest.mark.asyncio
async def test_fused_recall_flags_off_importance_sort_uses_current_blend():
    """sort_by='importance' off-flags must use the current 0.4/0.6 blend verbatim."""
    from claude_memory.api import recall as recall_mod

    pool, conn = _mock_conn()
    conn.fetch.return_value = [_make_memory_row(id=1)]

    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "", "MEMORY_GRAPH_ENABLED": ""}):
        await recall_mod._fused_recall(
            conn, user_id="u", query_text="single", sort_by="importance",
            category=None, limit=30, pool=pool,
        )

    sql = conn.fetch.call_args_list[0].args[0]
    assert "ts_rank(search_vector, query) * 0.4 + importance * 0.6" in sql


@pytest.mark.asyncio
async def test_fused_recall_flags_off_or_broaden_preserved():
    """The OR-broaden fallback (sparse AND-match) must be byte-identical when flags off."""
    from claude_memory.api import recall as recall_mod

    pool, conn = _mock_conn()
    conn.fetch.side_effect = [
        [_make_memory_row(id=1)],  # AND-match: 1 row (< limit -> triggers OR-broaden)
        [_make_memory_row(id=2)],  # OR-broaden result
    ]

    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "", "MEMORY_GRAPH_ENABLED": ""}):
        out = await recall_mod._fused_recall(
            conn, user_id="u", query_text="two words", sort_by="relevance",
            category=None, limit=30, pool=pool,
        )

    assert conn.fetch.call_count == 2, "OR-broaden must still fire when AND-match is sparse"
    or_sql = conn.fetch.call_args_list[1].args[0]
    assert "ts_rank(search_vector, query) DESC" in or_sql
    assert "ts_rank(search_vector, query) >" in or_sql
    assert {r["id"] for r in out} == {1, 2}


@pytest.mark.asyncio
async def test_fused_recall_embeddings_on_dense_leg_reranks_with_post_fusion_importance():
    """challenger must_fix: with embeddings ON, the dense leg joins the SHARED fused pool
    (weighted RRF), and importance is a POST-fusion MULTIPLIER (not a fused leg).

    Lexical ranks A>B; dense ranks B>A. With equal leg weights the RRF tie is broken by
    the importance multiplier: B has higher importance, so B sorts first.
    """
    from claude_memory.api import recall as recall_mod

    pool, conn = _mock_conn()
    row_a = _make_memory_row(id=1, content="A", importance=0.1)
    row_b = _make_memory_row(id=2, content="B", importance=0.9)
    # fetch is called twice: lexical leg, then dense leg.
    conn.fetch.side_effect = [
        [row_a, row_b],   # lexical: A (rank 1) > B (rank 2)
        [row_b, row_a],   # dense:   B (rank 1) > A (rank 2)
    ]

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 1024

    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "1", "MEMORY_GRAPH_ENABLED": ""}):
        out = await recall_mod._fused_recall(
            conn, user_id="u", query_text="q", sort_by="relevance",
            category=None, limit=30, pool=pool, embedder=embedder,
        )

    # A dense leg ran (HNSW cosine <=>) and the query was embedded once.
    embedder.embed_query.assert_called_once()
    dense_sql = conn.fetch.call_args_list[1].args[0]
    assert "<=>" in dense_sql
    assert "embedding IS NOT NULL" in dense_sql
    # Symmetric RRF ranks tie A and B; the post-fusion importance multiplier
    # (0.7 + 0.3*importance) breaks it toward B (0.9 > 0.1).
    assert [r["id"] for r in out] == [2, 1]


@pytest.mark.asyncio
async def test_fused_recall_dense_leg_excludes_sensitive_rows():
    """challenger / ADR-0003: sensitive rows are NEVER in the dense leg — the dense CTE
    filters ``embedding IS NOT NULL`` and sensitive rows have a NULL embedding, so they
    can only ever reach results via the lexical leg.
    """
    from claude_memory.api import recall as recall_mod

    pool, conn = _mock_conn()
    sensitive = _make_memory_row(id=9, content="[REDACTED]", is_sensitive=True, importance=0.5)
    conn.fetch.side_effect = [
        [sensitive],  # lexical leg can surface a sensitive row
        [],           # dense leg returns nothing (sensitive rows have NULL embedding)
    ]

    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 1024

    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "1", "MEMORY_GRAPH_ENABLED": ""}):
        out = await recall_mod._fused_recall(
            conn, user_id="u", query_text="secret", sort_by="relevance",
            category=None, limit=30, pool=pool, embedder=embedder,
        )

    dense_sql = conn.fetch.call_args_list[1].args[0]
    assert "embedding IS NOT NULL" in dense_sql
    # The sensitive row still appears (via lexical), but only because the dense leg
    # never returned it.
    assert [r["id"] for r in out] == [9]


@pytest.mark.asyncio
async def test_schedule_embedding_flag_off_is_noop():
    """schedule_embedding returns None and schedules nothing when the flag is off."""
    from claude_memory.api import recall as recall_mod

    pool, _ = _mock_conn()
    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": ""}):
        task = recall_mod.schedule_embedding(pool, 1, "content", is_sensitive=False)
    assert task is None


@pytest.mark.asyncio
async def test_schedule_embedding_skips_sensitive_rows():
    """challenger / ADR-0003: sensitive rows are NEVER embedded, even with the flag on."""
    from claude_memory.api import recall as recall_mod

    pool, _ = _mock_conn()
    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "1"}):
        task = recall_mod.schedule_embedding(pool, 1, "secret", is_sensitive=True)
    assert task is None


@pytest.mark.asyncio
async def test_schedule_embedding_does_not_block_and_persists_off_hot_path():
    """(e) store triggers async embed WITHOUT blocking: schedule_embedding returns a
    pending task immediately (the caller never awaits it), and the task later embeds and
    UPDATEs memories.embedding using its OWN pool connection.
    """
    from claude_memory.api import recall as recall_mod

    pool, conn = _mock_conn()

    embedder = MagicMock()
    embedder.embed_document.return_value = [0.5] * 1024

    with patch.dict(os.environ, {"MEMORY_EMBEDDINGS_ENABLED": "1"}), \
         patch.object(recall_mod, "select_embedder", return_value=embedder):
        task = recall_mod.schedule_embedding(pool, 42, "embed me", is_sensitive=False)
        # Returned immediately as a not-yet-awaited task — the store response is not blocked.
        assert task is not None
        assert not task.done()
        await task  # drive the off-hot-path work to completion

    embedder.embed_document.assert_called_once_with("embed me", is_sensitive=False)
    # It UPDATEd the embedding column for the right row, via its own acquired connection.
    update_calls = [c for c in conn.execute.call_args_list if "embedding" in c.args[0].lower()]
    assert update_calls, "embed task must UPDATE memories.embedding"
    assert "UPDATE memories SET embedding" in update_calls[0].args[0]
    assert update_calls[0].args[2] == 42  # memory_id is the WHERE id param


# ─── Both REST entry points wire through the shared helpers (no drift, S8) ────


@pytest.mark.asyncio
async def test_rest_recall_delegates_to_shared_fused_recall(client):
    """(f) REST recall_memories must retrieve via the shared _fused_recall helper."""
    ac, conn, app_mod = client
    captured = {}

    async def fake_fused(conn_arg, **kwargs):
        captured.update(kwargs)
        return [_make_memory_row(id=1, content="from helper", owner="testuser", shared_by=None)]

    with patch("claude_memory.api.app._fused_recall", side_effect=fake_fused) as mock_fused:
        async with ac:
            resp = await ac.post(
                "/api/memories/recall",
                json={"context": "hello world", "limit": 7},
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200
    assert resp.json()["memories"][0]["content"] == "from helper"
    mock_fused.assert_called_once()
    # The endpoint passes through the user's query + limit to the shared helper.
    assert captured["query_text"] == "hello world"
    assert captured["limit"] == 7
    assert captured["user_id"] == "testuser"


@pytest.mark.asyncio
async def test_rest_store_schedules_embed_on_write(client):
    """(e/f) REST store_memory must call the shared schedule_embedding after the INSERT."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=55, category="facts", importance=0.5)

    with patch("claude_memory.api.app.schedule_embedding") as mock_sched:
        async with ac:
            resp = await ac.post(
                "/api/memories",
                json={"content": "remember this", "category": "facts"},
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200
    mock_sched.assert_called_once()
    args, kwargs = mock_sched.call_args
    # schedule_embedding(pool, memory_id, content, is_sensitive=...)
    assert args[1] == 55
    assert args[2] == "remember this"
    assert kwargs["is_sensitive"] is False


@pytest.mark.asyncio
async def test_rest_store_sensitive_memory_is_not_embedded(client):
    """(c) A sensitive store passes is_sensitive=True to schedule_embedding (which then
    refuses to embed) — the dense path never sees sensitive content."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=56, category="facts", importance=0.5)

    with patch("claude_memory.api.app.schedule_embedding") as mock_sched:
        async with ac:
            resp = await ac.post(
                "/api/memories",
                json={"content": "token sk-secret", "force_sensitive": True},
                headers={"Authorization": "Bearer test-key"},
            )

    assert resp.status_code == 200
    mock_sched.assert_called_once()
    assert mock_sched.call_args.kwargs["is_sensitive"] is True
