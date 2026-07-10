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


# ─── Recall post-processing: apply_link_semantics (unit) ─────────────────────
#
# The conn.fetch side_effect lists below mirror the helper's fixed query order:
#   1. edges touching the ranked ids (ONE batched query — the common case's only one)
#   2. supersedes hop queries, one per chain hop, only while unresolved superseders
#      remain (depth-capped)
#   3. edges touching redirect heads that weren't ranked (for their links summary
#      + their resolved-by edges)
#   4. edges touching attach-only targets (for their links summary)
#   5. one row fetch for every id served that wasn't in the ranked set


def _conn():
    return AsyncMock()


async def _apply(conn, rows):
    from claude_memory.api import recall as recall_mod

    return await recall_mod.apply_link_semantics(conn, user_id="testuser", rows=rows)


@pytest.mark.asyncio
async def test_apply_no_rows_short_circuits_without_queries():
    conn = _conn()

    out = await _apply(conn, [])

    assert out == []
    conn.fetch.assert_not_called()


@pytest.mark.asyncio
async def test_apply_no_links_is_one_batched_query():
    """(c) the links lookup is ONE batched query for the whole result set, not N+1."""
    conn = _conn()
    rows = [_make_memory_row(id=1), _make_memory_row(id=2), _make_memory_row(id=3)]
    conn.fetch.return_value = []

    out = await _apply(conn, rows)

    assert conn.fetch.call_count == 1
    sql = conn.fetch.call_args.args[0]
    assert "memory_links" in sql
    assert "ANY" in sql  # batched over the id set
    assert [o["row"]["id"] for o in out] == [1, 2, 3]
    assert all(o["links"] == [] for o in out)
    assert all(o["redirected_from"] is None for o in out)


@pytest.mark.asyncio
async def test_apply_links_summary_covers_all_four_types_both_directions():
    conn = _conn()
    rows = [_make_memory_row(id=1)]
    conn.fetch.side_effect = [
        [
            _make_link_row(1, 2, "part-of", link_id=1, age_seconds=40),
            _make_link_row(1, 3, "see-also", link_id=2, age_seconds=30),
            _make_link_row(4, 1, "resolved-by", link_id=3, age_seconds=20),
            _make_link_row(1, 5, "supersedes", link_id=4, age_seconds=10),
        ],
        [],  # hop: superseder chain of... nothing; 1 has no INCOMING supersedes
    ]

    out = await _apply(conn, rows)

    assert out[0]["links"] == [
        {"type": "part-of", "dir": "out", "id": 2},
        {"type": "see-also", "dir": "out", "id": 3},
        {"type": "resolved-by", "dir": "in", "id": 4},
        {"type": "supersedes", "dir": "out", "id": 5},
    ]
    # outgoing supersedes does NOT redirect (only an INCOMING edge does).
    assert out[0]["row"]["id"] == 1
    assert out[0]["redirected_from"] is None
    assert conn.fetch.call_count == 1


@pytest.mark.asyncio
async def test_apply_supersedes_redirect_serves_successor_in_place():
    """(a) an INCOMING supersedes edge redirects: the head is served at the same
    rank, marked redirected_from."""
    conn = _conn()
    old = _make_memory_row(id=1, content="stale truth")
    other = _make_memory_row(id=5)
    successor_row = _make_memory_row(id=2, content="current truth", rank=0.0)
    edge = _make_link_row(2, 1, "supersedes", link_id=10)
    conn.fetch.side_effect = [
        [edge],            # edges touching {1, 5}
        [],                # hop: 2 has no incoming supersedes → chain head is 2
        [edge],            # edges touching the new head {2} (same edge, deduped)
        [successor_row],   # row fetch for {2}
    ]

    out = await _apply(conn, [old, other])

    assert [o["row"]["id"] for o in out] == [2, 5], "head replaces the old entry at the SAME rank"
    assert out[0]["redirected_from"] == 1
    assert out[0]["row"]["content"] == "current truth"
    assert out[0]["links"] == [{"type": "supersedes", "dir": "out", "id": 1}]
    assert out[1]["redirected_from"] is None


@pytest.mark.asyncio
async def test_apply_redirect_follows_chain_to_newest_head():
    """A supersedes CHAIN is followed to its head, picking the NEWEST superseder
    at each fan-in."""
    conn = _conn()
    old = _make_memory_row(id=1)
    e_old_superseder = _make_link_row(2, 1, "supersedes", link_id=1, age_seconds=100)
    e_new_superseder = _make_link_row(3, 1, "supersedes", link_id=2, age_seconds=50)
    e_head = _make_link_row(4, 3, "supersedes", link_id=3, age_seconds=10)
    head_row = _make_memory_row(id=4, rank=0.0)
    conn.fetch.side_effect = [
        [e_old_superseder, e_new_superseder],  # edges touching {1}
        [e_head],                              # hop 1: incoming supersedes of {2, 3}
        [],                                    # hop 2: incoming supersedes of {4}
        [e_head],                              # edges touching the head {4} (deduped)
        [head_row],                            # row fetch for {4}
    ]

    out = await _apply(conn, [old])

    # newest superseder of 1 is 3 (not 2); 3 is superseded by 4; head = 4.
    assert [o["row"]["id"] for o in out] == [4]
    assert out[0]["redirected_from"] == 1


@pytest.mark.asyncio
async def test_apply_redirect_dedupes_when_head_also_ranked():
    """If the head already ranked, the superseded entry folds into it (keep the
    best rank, no duplicate)."""
    conn = _conn()
    head = _make_memory_row(id=2)
    old = _make_memory_row(id=1)
    edge = _make_link_row(2, 1, "supersedes", link_id=1)
    conn.fetch.side_effect = [[edge]]  # both ends ranked → no hops, no fetches

    out = await _apply(conn, [head, old])

    assert [o["row"]["id"] for o in out] == [2]
    # the head ranked natively at the better position — not a substitution.
    assert out[0]["redirected_from"] is None
    assert conn.fetch.call_count == 1


@pytest.mark.asyncio
async def test_apply_redirect_dedupe_keeps_substituted_slot_when_it_ranks_better():
    conn = _conn()
    old = _make_memory_row(id=1)
    head = _make_memory_row(id=2)
    edge = _make_link_row(2, 1, "supersedes", link_id=1)
    conn.fetch.side_effect = [[edge]]

    out = await _apply(conn, [old, head])  # old outranks its own successor

    assert [o["row"]["id"] for o in out] == [2]
    assert out[0]["redirected_from"] == 1  # served via the substitution at rank 0


@pytest.mark.asyncio
async def test_apply_redirect_survives_pre_existing_cycle_in_data():
    """Legacy mutual supersedes edges must not loop the walk (visited set)."""
    conn = _conn()
    old = _make_memory_row(id=1)
    e21 = _make_link_row(2, 1, "supersedes", link_id=1, age_seconds=10)
    e12 = _make_link_row(1, 2, "supersedes", link_id=2, age_seconds=20)
    successor_row = _make_memory_row(id=2, rank=0.0)
    conn.fetch.side_effect = [
        [e21, e12],        # edges touching {1} (both edges touch 1)
        [e12],             # hop: incoming supersedes of {2} → the back-edge (deduped)
        [e21, e12],        # edges touching the head {2}
        [successor_row],   # row fetch for {2}
    ]

    out = await _apply(conn, [old])

    assert [o["row"]["id"] for o in out] == [2]
    assert out[0]["redirected_from"] == 1


@pytest.mark.asyncio
async def test_apply_redirect_walk_is_depth_capped():
    """A 12-hop supersedes chain stops at the 10-hop cap."""
    conn = _conn()
    ranked = _make_memory_row(id=0)
    chain_edges = [
        _make_link_row(n + 1, n, "supersedes", link_id=100 + n) for n in range(12)
    ]
    capped_row = _make_memory_row(id=10, rank=0.0)
    side_effect = [[chain_edges[0]]]  # edges touching {0}
    # hop k (k=1..10): incoming supersedes of {k} → edge (k+1 → k)
    side_effect += [[chain_edges[k]] for k in range(1, 11)]
    side_effect += [
        [chain_edges[9], chain_edges[10]],  # edges touching the head {10}
        [capped_row],                       # row fetch for {10}
    ]
    conn.fetch.side_effect = side_effect

    out = await _apply(conn, [ranked])

    assert [o["row"]["id"] for o in out] == [10], "walk must stop at the depth cap"
    assert out[0]["redirected_from"] == 0


@pytest.mark.asyncio
async def test_apply_resolved_by_attaches_target_as_extra_result():
    """(b) an outgoing resolved-by edge auto-attaches its target after the ranked
    results, marked attached_via — without consuming the caller's limit."""
    conn = _conn()
    symptom = _make_memory_row(id=1, content="the symptom")
    answer_row = _make_memory_row(id=9, content="the root cause", rank=0.0)
    edge = _make_link_row(1, 9, "resolved-by", link_id=1)
    conn.fetch.side_effect = [
        [edge],        # edges touching {1}
        [edge],        # edges touching the attach target {9} (deduped)
        [answer_row],  # row fetch for {9}
    ]

    out = await _apply(conn, [symptom])

    assert [o["row"]["id"] for o in out] == [1, 9]
    assert out[0].get("attached_via") is None
    assert out[1]["attached_via"] == {"type": "resolved-by", "source": 1}
    assert out[1]["links"] == [{"type": "resolved-by", "dir": "in", "id": 1}]


@pytest.mark.asyncio
async def test_apply_resolved_by_attachments_cap_at_three_per_response():
    conn = _conn()
    src = _make_memory_row(id=1)
    edges = [
        _make_link_row(1, 10 + n, "resolved-by", link_id=n + 1, age_seconds=100 - n)
        for n in range(5)
    ]
    attach_rows = [_make_memory_row(id=10 + n, rank=0.0) for n in range(3)]
    conn.fetch.side_effect = [
        edges,        # edges touching {1}
        [],           # edges touching the attach targets {10, 11, 12}
        attach_rows,  # row fetch for {10, 11, 12}
    ]

    out = await _apply(conn, [src])

    assert len(out) == 4  # 1 ranked + 3 attachments (cap), not 6
    attached = [o for o in out if o.get("attached_via")]
    assert [o["row"]["id"] for o in attached] == [10, 11, 12]  # oldest edges first


@pytest.mark.asyncio
async def test_apply_resolved_by_skips_targets_already_in_results():
    conn = _conn()
    symptom = _make_memory_row(id=1)
    answer = _make_memory_row(id=9)
    edge = _make_link_row(1, 9, "resolved-by", link_id=1)
    conn.fetch.side_effect = [[edge]]

    out = await _apply(conn, [symptom, answer])

    assert [o["row"]["id"] for o in out] == [1, 9]
    assert all(o.get("attached_via") is None for o in out)
    assert conn.fetch.call_count == 1


@pytest.mark.asyncio
async def test_apply_redirect_head_gets_its_own_resolved_by_attachment():
    """(b) applies to SURVIVING results: a redirect head's outgoing resolved-by
    edges attach too."""
    conn = _conn()
    old = _make_memory_row(id=1)
    sup_edge = _make_link_row(2, 1, "supersedes", link_id=1)
    res_edge = _make_link_row(2, 7, "resolved-by", link_id=2)
    head_row = _make_memory_row(id=2, rank=0.0)
    answer_row = _make_memory_row(id=7, rank=0.0)
    conn.fetch.side_effect = [
        [sup_edge],             # edges touching {1}
        [],                     # hop: incoming supersedes of {2}
        [sup_edge, res_edge],   # edges touching the head {2} → reveals its resolved-by
        [res_edge],             # edges touching the attach target {7}
        [head_row, answer_row],  # row fetch for {2, 7}
    ]

    out = await _apply(conn, [old])

    assert [o["row"]["id"] for o in out] == [2, 7]
    assert out[0]["redirected_from"] == 1
    assert out[1]["attached_via"] == {"type": "resolved-by", "source": 2}


# ─── Recall post-processing: REST endpoint wiring ────────────────────────────


@pytest.mark.asyncio
async def test_recall_endpoint_attaches_resolved_by_and_links(client):
    """POST /api/memories/recall serves link semantics: links on every result,
    resolved-by targets attached beyond the ranked set."""
    ac, conn, app_mod = client
    symptom = _make_memory_row(id=1, content="the symptom")
    answer_row = _make_memory_row(id=9, content="the root cause", rank=0.0)
    edge = _make_link_row(1, 9, "resolved-by", link_id=1)
    conn.fetch.side_effect = [
        [symptom],     # lexical AND-match (single word → no OR-broaden)
        [edge],        # edges touching {1}
        [edge],        # edges touching the attach target {9}
        [answer_row],  # row fetch for {9}
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "symptom"},
            headers=AUTH,
        )

    assert resp.status_code == 200
    memories = resp.json()["memories"]
    assert [m["id"] for m in memories] == [1, 9]
    assert memories[0]["links"] == [{"type": "resolved-by", "dir": "out", "id": 9}]
    assert "attached_via" not in memories[0]
    assert memories[1]["attached_via"] == {"type": "resolved-by", "source": 1}
    assert memories[1]["content"] == "the root cause"


@pytest.mark.asyncio
async def test_recall_endpoint_serves_supersedes_redirect(client):
    """POST /api/memories/recall replaces a superseded entry with its chain head,
    marked redirected_from, at the same rank."""
    ac, conn, app_mod = client
    old = _make_memory_row(id=1, content="stale")
    successor_row = _make_memory_row(id=2, content="current", rank=0.0)
    edge = _make_link_row(2, 1, "supersedes", link_id=1)
    conn.fetch.side_effect = [
        [old],             # lexical AND-match
        [edge],            # edges touching {1}
        [],                # hop: incoming supersedes of {2}
        [edge],            # edges touching the head {2}
        [successor_row],   # row fetch for {2}
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "stale"},
            headers=AUTH,
        )

    assert resp.status_code == 200
    memories = resp.json()["memories"]
    assert len(memories) == 1
    assert memories[0]["id"] == 2
    assert memories[0]["content"] == "current"
    assert memories[0]["redirected_from"] == 1
    assert memories[0]["links"] == [{"type": "supersedes", "dir": "out", "id": 1}]


@pytest.mark.asyncio
async def test_recall_endpoint_redacts_sensitive_attachments(client):
    """Attached results flow through the SAME sensitive-redaction as ranked ones."""
    ac, conn, app_mod = client
    symptom = _make_memory_row(id=1)
    secret_row = _make_memory_row(id=9, content="[REDACTED]", is_sensitive=True, rank=0.0)
    edge = _make_link_row(1, 9, "resolved-by", link_id=1)
    conn.fetch.side_effect = [
        [symptom],
        [edge],
        [edge],
        [secret_row],
    ]

    async with ac:
        resp = await ac.post(
            "/api/memories/recall",
            json={"context": "symptom"},
            headers=AUTH,
        )

    memories = resp.json()["memories"]
    assert "[SENSITIVE" in memories[1]["content"]
    assert "secret_get(id=9)" in memories[1]["content"]
