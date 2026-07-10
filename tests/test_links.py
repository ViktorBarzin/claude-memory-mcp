"""Tests for ADR-0007 typed Memory→Memory links: CRUD endpoints, the single-memory
GET (with links both directions), and the recall post-processing semantics
(supersedes-redirect, resolved-by auto-attach, links summary).

Follows the test_api.py pattern: the app is stood up with a mocked asyncpg pool
and an auth override; SQL behaviour is asserted through the mock connection.
"""

import importlib
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg  # type: ignore[import-untyped]
import pytest
from httpx import ASGITransport, AsyncClient

from claude_memory.api.auth import AuthUser


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
        "rank": 0.5,
        "created_at": now,
        "updated_at": now,
        "deleted_at": None,
        "owner": "testuser",
        "shared_by": None,
    }
    defaults.update(overrides)
    return MockRow(defaults)


def _make_link_row(src_id, dst_id, link_type, *, link_id=1, age_seconds=0):
    return MockRow(
        {
            "id": link_id,
            "src_id": src_id,
            "dst_id": dst_id,
            "link_type": link_type,
            "created_at": datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
        }
    )


@pytest.fixture
def mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
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
    pool, conn = mock_pool

    with patch.dict(os.environ, {"API_KEY": "test-key", "API_KEYS": "", "DATABASE_URL": "postgresql://test"}):
        import claude_memory.api.auth as auth_mod
        import claude_memory.api.database as db_mod
        import claude_memory.api.app as app_mod

        importlib.reload(auth_mod)
        importlib.reload(db_mod)
        importlib.reload(app_mod)

        db_mod.pool = pool

        async def mock_get_user(authorization: str = ""):
            return test_user

        app_mod.app.dependency_overrides[auth_mod.get_current_user] = mock_get_user

        transport = ASGITransport(app=app_mod.app)
        return AsyncClient(transport=transport, base_url="http://test"), conn, app_mod


AUTH = {"Authorization": "Bearer test-key"}


def _allow_read(*results):
    """A check_memory_permission side_effect: the Nth call gets the Nth (allowed,
    owner) tuple; extra calls repeat the last one."""

    async def _counting(conn, memory_id, user_id, perm):
        result = results[min(_counting.calls, len(results) - 1)]
        _counting.calls += 1
        return result

    _counting.calls = 0
    return _counting


# ─── POST /api/memories/{id}/links ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_link_inserts_edge_for_caller(client):
    ac, conn, app_mod = client
    now = datetime.now(timezone.utc)
    conn.fetchrow.return_value = MockRow({"id": 7, "created_at": now})

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((True, "testuser"))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "see-also"},
                headers=AUTH,
            )

    assert resp.status_code == 200
    data = resp.json()
    assert data["src_id"] == 1
    assert data["dst_id"] == 2
    assert data["link_type"] == "see-also"
    insert_args = conn.fetchrow.call_args.args
    assert "INSERT INTO memory_links" in insert_args[0]
    assert insert_args[1:] == ("testuser", 1, 2, "see-also")


@pytest.mark.asyncio
async def test_create_link_rejects_unknown_type(client):
    """link_type is the CLOSED enum of four — anything else is a 422."""
    ac, conn, app_mod = client

    async with ac:
        resp = await ac.post(
            "/api/memories/1/links",
            json={"target_id": 2, "link_type": "related-to"},
            headers=AUTH,
        )

    assert resp.status_code == 422
    conn.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_create_link_rejects_self_link(client):
    ac, conn, app_mod = client

    async with ac:
        resp = await ac.post(
            "/api/memories/5/links",
            json={"target_id": 5, "link_type": "see-also"},
            headers=AUTH,
        )

    assert resp.status_code == 422
    conn.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_create_link_source_not_found_is_404(client):
    ac, conn, app_mod = client

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((False, None))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "see-also"},
                headers=AUTH,
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_link_target_not_found_is_404(client):
    ac, conn, app_mod = client

    with patch(
        "claude_memory.api.app.check_memory_permission",
        side_effect=_allow_read((True, "testuser"), (False, None)),
    ):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "see-also"},
                headers=AUTH,
            )

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_create_link_unreadable_target_is_403(client):
    """Both ends must be readable by the caller (ownership or shared-read)."""
    ac, conn, app_mod = client

    with patch(
        "claude_memory.api.app.check_memory_permission",
        side_effect=_allow_read((True, "testuser"), (False, "otheruser")),
    ):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "see-also"},
                headers=AUTH,
            )

    assert resp.status_code == 403
    conn.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_create_duplicate_link_is_409(client):
    ac, conn, app_mod = client
    conn.fetchrow.side_effect = asyncpg.UniqueViolationError("duplicate key")

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((True, "testuser"))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "see-also"},
                headers=AUTH,
            )

    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_create_supersedes_direct_cycle_is_422(client):
    """Creating 1-supersedes->2 when 2 already supersedes 1 is a cycle."""
    ac, conn, app_mod = client
    conn.fetch.return_value = [MockRow({"dst_id": 1})]  # 2's outgoing supersedes reaches 1

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((True, "testuser"))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "supersedes"},
                headers=AUTH,
            )

    assert resp.status_code == 422
    assert "cycle" in resp.text
    conn.fetchrow.assert_not_called()  # no INSERT


@pytest.mark.asyncio
async def test_create_supersedes_transitive_cycle_is_422(client):
    """1->2 with an existing chain 2->3->1 is a (transitive) cycle."""
    ac, conn, app_mod = client
    conn.fetch.side_effect = [
        [MockRow({"dst_id": 3})],  # hop 1: 2 supersedes 3
        [MockRow({"dst_id": 1})],  # hop 2: 3 supersedes 1 → reaches the new src
    ]

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((True, "testuser"))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "supersedes"},
                headers=AUTH,
            )

    assert resp.status_code == 422
    conn.fetchrow.assert_not_called()


@pytest.mark.asyncio
async def test_create_supersedes_acyclic_chain_is_allowed(client):
    ac, conn, app_mod = client
    now = datetime.now(timezone.utc)
    conn.fetch.side_effect = [
        [MockRow({"dst_id": 3})],  # 2 supersedes 3
        [],                        # 3 supersedes nothing → chain ends
    ]
    conn.fetchrow.return_value = MockRow({"id": 9, "created_at": now})

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((True, "testuser"))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "supersedes"},
                headers=AUTH,
            )

    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_supersedes_cycle_walk_caps_at_depth_10(client):
    """The cycle walk is bounded: 10 hops maximum, then the link is allowed."""
    ac, conn, app_mod = client
    now = datetime.now(timezone.utc)
    # Each hop discovers one more node (100+n), never reaching src=1.
    conn.fetch.side_effect = [[MockRow({"dst_id": 100 + n})] for n in range(15)]
    conn.fetchrow.return_value = MockRow({"id": 9, "created_at": now})

    with patch("claude_memory.api.app.check_memory_permission", side_effect=_allow_read((True, "testuser"))):
        async with ac:
            resp = await ac.post(
                "/api/memories/1/links",
                json={"target_id": 2, "link_type": "supersedes"},
                headers=AUTH,
            )

    assert resp.status_code == 200
    assert conn.fetch.call_count == 10, "supersedes walk must stop at the depth cap"


# ─── DELETE /api/memories/{id}/links/{dst_id}/{link_type} ────────────────────


@pytest.mark.asyncio
async def test_delete_link_removes_the_edge(client):
    ac, conn, app_mod = client
    conn.execute.return_value = "DELETE 1"

    async with ac:
        resp = await ac.delete("/api/memories/1/links/2/see-also", headers=AUTH)

    assert resp.status_code == 200
    args = conn.execute.call_args.args
    assert "DELETE FROM memory_links" in args[0]
    assert args[1:] == ("testuser", 1, 2, "see-also")


@pytest.mark.asyncio
async def test_delete_missing_link_is_404(client):
    ac, conn, app_mod = client
    conn.execute.return_value = "DELETE 0"

    async with ac:
        resp = await ac.delete("/api/memories/1/links/2/see-also", headers=AUTH)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_delete_link_rejects_unknown_type(client):
    ac, conn, app_mod = client

    async with ac:
        resp = await ac.delete("/api/memories/1/links/2/related-to", headers=AUTH)

    assert resp.status_code == 422
    conn.execute.assert_not_called()


# ─── GET /api/memories/{id} ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_memory_returns_full_entry_with_links_both_directions(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=10, content="the full memory")
    conn.fetch.return_value = [
        _make_link_row(10, 2, "part-of", link_id=1),
        _make_link_row(3, 10, "supersedes", link_id=2),
        _make_link_row(10, 4, "resolved-by", link_id=3),
    ]

    async with ac:
        resp = await ac.get("/api/memories/10", headers=AUTH)

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == 10
    assert data["content"] == "the full memory"
    assert data["links_out"] == [
        {"id": 2, "type": "part-of"},
        {"id": 4, "type": "resolved-by"},
    ]
    assert data["links_in"] == [{"id": 3, "type": "supersedes"}]


@pytest.mark.asyncio
async def test_get_memory_not_found_is_404(client):
    ac, conn, app_mod = client
    conn.fetchrow.return_value = None

    async with ac:
        resp = await ac.get("/api/memories/999", headers=AUTH)

    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_memory_redacts_sensitive_content(client):
    """The single-memory GET respects the existing sensitive-redaction rules."""
    ac, conn, app_mod = client
    conn.fetchrow.return_value = _make_memory_row(id=11, content="[REDACTED]", is_sensitive=True)
    conn.fetch.return_value = []

    async with ac:
        resp = await ac.get("/api/memories/11", headers=AUTH)

    assert resp.status_code == 200
    data = resp.json()
    assert "[SENSITIVE" in data["content"]
    assert "secret_get(id=11)" in data["content"]
    assert data["links_out"] == []
    assert data["links_in"] == []
