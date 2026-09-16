"""Tests for the external-key endpoint allowlist and the server-side origin tag.

Changes 2 and 4 of the Muse memory connector design (2026-09-16).

Change 2: a key whose scope is "external" may call only the operations in
EXTERNAL_ALLOWED_OPERATIONS. Enforcement is an app-wide Depends() rather than ASGI
middleware because request.scope["route"] is not populated until routing has matched,
and it denies by default, so a privileged route added later is closed to external keys
without anyone remembering to close it.

Change 4: every write an external key can reach is stamped source:<user_id> server-side.
The client cannot opt out, cannot forge someone else's origin, and a repeated update does
not accumulate the tag twice. There is NO importance ceiling; that was declined.
"""

import importlib
import json
import logging
import os
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from starlette.routing import Mount, Route

from claude_memory.api.scopes import (
    EXTERNAL_ALLOWED_OPERATIONS, MUSE_SPEC_PATH, stamp_origin, stamp_stored_origin,
)

ADMIN_KEY = "key-admin-wizard"
MUSE_KEY = "key-external-muse"
API_KEYS = json.dumps({"wizard": ADMIN_KEY, "muse": {"key": MUSE_KEY, "scope": "external"}})

ADMIN = {"Authorization": f"Bearer {ADMIN_KEY}"}
MUSE = {"Authorization": f"Bearer {MUSE_KEY}"}

# The four the design names, plus three more the allowlist closes by omission. Nobody
# wrote this list into the source; it is derived there from EXTERNAL_ALLOWED_OPERATIONS.
CLOSED_OPERATIONS = [
    ("POST", "/api/memories/import"),
    ("POST", "/api/memories/migrate-secrets"),
    ("GET", "/api/users"),
    ("POST", "/api/memories/1/secret"),
    ("GET", "/api/stats"),
    ("GET", "/api/memories/sync"),
    ("POST", "/api/memories/1/share"),
]


class MockRow(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _memory_row(**overrides):
    now = datetime.now(timezone.utc)
    defaults = {
        "id": 1, "user_id": "muse", "content": "test content", "category": "facts",
        "tags": "", "expanded_keywords": "", "importance": 0.5, "is_sensitive": False,
        "vault_path": None, "encrypted_content": None, "preview": "test content",
        "rank": 0.5, "created_at": now, "updated_at": now, "deleted_at": None,
        "owner": "muse", "shared_by": None, "share_permission": None,
    }
    defaults.update(overrides)
    return MockRow(defaults)


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
def api(mock_pool):
    """The real app with real keys: an admin one and an external one.

    No dependency_overrides here on purpose. Enforcement reads the bearer token straight
    off the request, so overriding get_current_user would test nothing.
    """
    pool, conn = mock_pool
    with patch.dict(
        os.environ,
        {"API_KEYS": API_KEYS, "API_KEY": "", "DATABASE_URL": "postgresql://test"},
    ):
        import claude_memory.api.app as app_mod
        import claude_memory.api.auth as auth_mod
        import claude_memory.api.database as db_mod

        importlib.reload(auth_mod)
        importlib.reload(db_mod)
        importlib.reload(app_mod)
        db_mod.pool = pool

        transport = ASGITransport(app=app_mod.app)
        yield AsyncClient(transport=transport, base_url="http://test"), conn, app_mod


def _inserted_tags(conn):
    """The tags column as the INSERT saw it (query, user_id, content, category, tags, ...)."""
    return conn.fetchrow.call_args[0][4]


def _updated_tags(conn):
    """The tags column as the UPDATE saw it, read out of the generated SET clause."""
    sql, *params = conn.execute.call_args[0]
    match = re.search(r"tags = \$(\d+)", sql)
    assert match, f"no tags assignment in: {sql}"
    return params[int(match.group(1)) - 1]


# ─── change 2: the allowlist ────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", CLOSED_OPERATIONS, ids=lambda v: str(v).strip("/"))
async def test_an_external_key_is_refused_on_a_closed_endpoint(api, method, path):
    ac, conn, app_mod = api
    async with ac:
        resp = await ac.request(method, path, json=[], headers=MUSE)
    assert resp.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", CLOSED_OPERATIONS, ids=lambda v: str(v).strip("/"))
async def test_an_admin_key_is_untouched_on_the_same_endpoints(api, method, path):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(user_id="wizard", owner="wizard")
    conn.fetch.return_value = []
    conn.fetchval.return_value = 0
    async with ac:
        resp = await ac.request(method, path, json=[], headers=ADMIN)
    assert resp.status_code != 403


@pytest.mark.asyncio
async def test_an_external_key_can_recall(api):
    ac, conn, app_mod = api
    conn.fetch.side_effect = [[], [], []]
    async with ac:
        resp = await ac.post("/api/memories/recall", json={"context": "anything"}, headers=MUSE)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_an_external_key_can_store(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42, category="facts", importance=0.5)
    async with ac:
        resp = await ac.post("/api/memories", json={"content": "a new thing"}, headers=MUSE)
    assert resp.status_code == 200
    assert resp.json()["id"] == 42


@pytest.mark.asyncio
async def test_an_external_key_can_list_tags_and_categories(api):
    ac, conn, app_mod = api
    conn.fetch.return_value = []
    conn.fetchval.return_value = 0
    async with ac:
        assert (await ac.get("/api/tags", headers=MUSE)).status_code == 200
        assert (await ac.get("/api/categories", headers=MUSE)).status_code == 200
        assert (await ac.get("/api/memories", headers=MUSE)).status_code == 200


@pytest.mark.asyncio
async def test_an_external_key_can_delete_its_own_memory(api):
    """A delete writes no tags, so change 4 leaves it alone. It is still allowlisted, and
    the endpoint's own owner check is what stops it reaching anyone else's rows."""
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=7, vault_path=None, preview="gone")
    async with ac:
        resp = await ac.delete("/api/memories/7", headers=MUSE)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_an_external_key_can_create_a_link(api):
    """memory_links has no tags column, so there is nothing for change 4 to stamp here."""
    ac, conn, app_mod = api
    conn.fetchrow.side_effect = [
        _memory_row(id=7, user_id="muse"),
        _memory_row(id=9, user_id="muse"),
        MockRow({"id": 1, "created_at": datetime.now(timezone.utc)}),
    ]
    async with ac:
        resp = await ac.post(
            "/api/memories/7/links",
            json={"target_id": 9, "link_type": "see-also"},
            headers=MUSE,
        )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_a_route_added_later_is_closed_to_an_external_key(api):
    """The fail-closed property. Nobody has to remember to deny a new privileged route:
    it is absent from the allowlist, so the app-wide dependency refuses it."""
    ac, conn, app_mod = api

    @app_mod.app.post("/api/brand-new-privileged-thing")
    async def _new_thing() -> dict[str, str]:
        return {"status": "ok"}

    async with ac:
        assert (await ac.post("/api/brand-new-privileged-thing", headers=MUSE)).status_code == 403
        assert (await ac.post("/api/brand-new-privileged-thing", headers=ADMIN)).status_code == 200


def test_the_app_serves_exactly_the_known_unguarded_routes(api):
    """Where fail-closed stops, pinned in full.

    An app-wide Depends() runs for APIRoutes only, so everything else in app.routes is
    outside the allowlist whatever it serves: the two Mounts (/mcp, gated instead by
    MCPAuthMiddleware, and /static, the UI) AND the four plain starlette Routes FastAPI
    adds for its own documentation. Those four are open to an anonymous caller today and
    the design accepts that, but a privileged route added later through add_route() or a
    third mount would be both unguarded and — until this assertion covers every
    non-APIRoute entry rather than only the Mounts — untested."""
    ac, conn, app_mod = api
    unguarded = {
        (type(r).__name__, r.path) for r in app_mod.app.routes if not isinstance(r, APIRoute)
    }
    assert unguarded == {
        ("Mount", "/mcp"),
        ("Mount", "/static"),
        ("Route", "/openapi.json"),
        ("Route", "/docs"),
        ("Route", "/docs/oauth2-redirect"),
        ("Route", "/redoc"),
    }
    assert {r.path for r in app_mod.app.routes if isinstance(r, Mount)} == {"/mcp", "/static"}
    assert all(isinstance(r, Route) for r in app_mod.app.routes if type(r).__name__ == "Route")


@pytest.mark.asyncio
async def test_a_new_literal_under_api_memories_is_shadowed_rather_than_refused(api):
    """The one measured exception to "a route added later is 403", pinned so it reads as
    known rather than as a surprise. GET /api/memories/{memory_id} is allowlisted and is
    registered first, so it matches a new single-segment literal before that literal's own
    route does: the guard sees the allowed template and passes, and int parsing then
    rejects the path segment with 422 — for the admin key too. Nothing escalates, because
    the new handler runs for nobody. A shape the path parameter cannot swallow is refused
    normally."""
    ac, conn, app_mod = api

    @app_mod.app.get("/api/memories/admin-dump")
    async def _admin_dump() -> dict[str, str]:  # pragma: no cover - never reached
        return {"status": "ok"}

    @app_mod.app.post("/api/memories/{memory_id}/purge")
    async def _purge(memory_id: int) -> dict[str, str]:
        return {"status": "ok"}

    async with ac:
        assert (await ac.get("/api/memories/admin-dump", headers=MUSE)).status_code == 422
        assert (await ac.get("/api/memories/admin-dump", headers=ADMIN)).status_code == 422
        assert (await ac.post("/api/memories/1/purge", headers=MUSE)).status_code == 403
        assert (await ac.post("/api/memories/1/purge", headers=ADMIN)).status_code == 200


@pytest.mark.asyncio
async def test_public_endpoints_stay_open_to_an_external_key(api):
    """Both already serve the same bytes to an anonymous caller, and a connector that
    attaches its token to every request to the host must not 403 on the spec fetch."""
    ac, conn, app_mod = api
    async with ac:
        assert (await ac.get("/health", headers=MUSE)).status_code == 200
        assert (await ac.get(MUSE_SPEC_PATH, headers=MUSE)).status_code == 200
        assert (await ac.get("/health")).status_code == 200


@pytest.mark.asyncio
async def test_auth_check_tells_an_admin_key_what_scope_it_carries(api):
    """A key meant to be external but written in API_KEYS' flat shape is a silent admin
    key. Echoing the scope makes that one curl away instead of inferrable only from this
    endpoint answering 403."""
    ac, conn, app_mod = api
    async with ac:
        resp = await ac.get("/api/auth-check", headers=ADMIN)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "user_id": "wizard", "scope": "admin"}


def test_startup_logs_the_scope_every_key_parsed_as(api, caplog):
    """The other half of the same gap: an operator holding one key cannot use
    /api/auth-check to see how SOMEBODY ELSE's entry parsed, so the pod says so at boot.
    User ids only — GET /api/users already lists those, and a key never goes to a log."""
    ac, conn, app_mod = api
    with caplog.at_level(logging.INFO, logger="claude_memory.api.app"):
        app_mod._log_key_scopes()
    logged = " ".join(r.getMessage() for r in caplog.records)
    assert "wizard=admin" in logged
    assert "muse=external" in logged
    assert MUSE_KEY not in logged
    assert ADMIN_KEY not in logged


def test_startup_logs_an_api_keys_entry_that_could_not_be_used(caplog):
    """An entry dropped at import no longer crashes the pod, so the boot log is where it
    surfaces. Logged at ERROR: a key somebody expects to work does not."""
    import claude_memory.api.app as app_mod
    import claude_memory.api.auth as auth_mod

    with patch.dict(os.environ, {"API_KEYS": '{"wizard": "kw", "legacy": ""}', "API_KEY": ""}):
        importlib.reload(auth_mod)
        try:
            with caplog.at_level(logging.INFO, logger="claude_memory.api.app"):
                app_mod._log_key_scopes()
            problems = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
            assert any("'legacy'" in m for m in problems)
        finally:
            importlib.reload(auth_mod)


@pytest.mark.asyncio
async def test_an_unknown_token_still_gets_401_not_403(api):
    ac, conn, app_mod = api
    async with ac:
        resp = await ac.get("/api/tags", headers={"Authorization": "Bearer nope"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_a_missing_header_is_unchanged(api):
    ac, conn, app_mod = api
    async with ac:
        resp = await ac.get("/api/users")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_the_refusal_names_neither_the_key_nor_another_user(api):
    ac, conn, app_mod = api
    async with ac:
        resp = await ac.get("/api/users", headers=MUSE)
    assert resp.status_code == 403
    assert MUSE_KEY not in resp.text


# ─── change 2: the MCP transport, which no APIRoute dependency can see ──────


async def _drive_mcp_middleware(app_mod, token):
    reached_inner = []

    async def inner(scope, receive, send):
        reached_inner.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"inner"})

    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    headers = [(b"authorization", f"Bearer {token}".encode())] if token else []
    scope = {"type": "http", "path": "/mcp/mcp", "method": "POST", "headers": headers}
    await app_mod.MCPAuthMiddleware(inner)(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    return status, bool(reached_inner)


@pytest.mark.asyncio
async def test_the_mcp_transport_refuses_an_external_key(api):
    """The MCP tools duplicate REST logic and add memory_share and the tag shares, none of
    which the allowlist grants. Leaving /mcp open to an external key would hand it the same
    capabilities through a second door."""
    ac, conn, app_mod = api
    status, reached_inner = await _drive_mcp_middleware(app_mod, MUSE_KEY)
    assert status == 403
    assert not reached_inner


@pytest.mark.asyncio
async def test_the_mcp_transport_still_admits_an_admin_key(api):
    ac, conn, app_mod = api
    status, reached_inner = await _drive_mcp_middleware(app_mod, ADMIN_KEY)
    assert status == 200
    assert reached_inner


@pytest.mark.asyncio
async def test_the_mcp_transport_still_401s_an_unknown_token(api):
    ac, conn, app_mod = api
    status, reached_inner = await _drive_mcp_middleware(app_mod, "nope")
    assert status == 401
    assert not reached_inner


# ─── change 4: the merge semantics, as a table ──────────────────────────────


@pytest.mark.parametrize(
    "client_tags,expected",
    [
        pytest.param(None, "source:muse", id="no-tags-field"),
        pytest.param("", "source:muse", id="empty-string"),
        pytest.param("   ", "source:muse", id="whitespace-only"),
        pytest.param("a,b", "a,b,source:muse", id="two-client-tags"),
        pytest.param(" a , b ", "a,b,source:muse", id="tags-are-trimmed"),
        pytest.param("a,,b", "a,b,source:muse", id="empty-element-dropped"),
        pytest.param("source:muse", "source:muse", id="client-sent-the-tag-itself"),
        pytest.param("a,source:muse,b", "a,b,source:muse", id="tag-moves-to-the-end-once"),
        pytest.param("  source:muse  ", "source:muse", id="lookalike-padded"),
        pytest.param("Source:Muse", "source:muse", id="lookalike-cased"),
        pytest.param("source: muse", "source:muse", id="lookalike-space-after-colon"),
        pytest.param("source:wizard", "source:muse", id="forged-other-origin-replaced"),
        pytest.param("source:muse,source:muse", "source:muse", id="client-sent-it-twice"),
        pytest.param("sources:muse", "sources:muse,source:muse", id="near-miss-is-kept"),
    ],
)
def test_stamp_origin_merge_semantics(client_tags, expected):
    assert stamp_origin(client_tags, "muse") == expected


def test_stamping_is_idempotent():
    once = stamp_origin("a,b", "muse")
    assert stamp_origin(once, "muse") == once


# ─── change 4: a forged origin, spelled around the matcher ──────────────────

# Every spelling measured as surviving a matcher that only stripped the ASCII space.
# Whitespace and control characters are removed from the comparison form, NFKC folds the
# compatibility colons, and a small table folds the Cyrillic, Greek and Armenian letters
# that draw like one of the seven characters of "source:".
FORGED_SPELLINGS = [
    pytest.param("source:wizard", id="plain"),
    pytest.param("Source: wizard", id="cased-and-spaced"),
    pytest.param("SOURCE:WIZARD", id="upper"),
    pytest.param("source\t:wizard", id="tab"),
    pytest.param("source\xa0:wizard", id="no-break-space"),
    pytest.param("source\u200b:wizard", id="zero-width-space"),
    pytest.param("source\uff1awizard", id="fullwidth-colon"),
    pytest.param("source\u2236wizard", id="ratio-colon"),
    pytest.param("source\ua789wizard", id="modifier-letter-colon"),
    pytest.param("source\u05c3wizard", id="sof-pasuq-colon"),
    pytest.param("\u0455ource:wizard", id="cyrillic-dze-s"),
    pytest.param("s\u043eurce:wizard", id="cyrillic-o"),
    pytest.param("so\u03c5rce:wizard", id="greek-upsilon-u"),
    pytest.param("sou\u0433ce:wizard", id="cyrillic-ghe-r"),
    pytest.param("sour\u0441e:wizard", id="cyrillic-es-c"),
    pytest.param("sourc\u0435:wizard", id="cyrillic-ie-e"),
]


@pytest.mark.parametrize("forged", FORGED_SPELLINGS)
def test_a_forged_origin_does_not_survive_however_it_is_spelled(forged):
    assert stamp_origin(f"keep-me,{forged}", "muse") == "keep-me,source:muse"


def test_a_tag_carrying_a_newline_cannot_paint_a_second_provenance_line():
    """The recall hook renders tags into ONE line (``Tags: a,b | Stored: ...``), so a
    newline inside a tag splits that line and the text after the break reads as a field
    of its own. The origin test therefore matches on CONTAINS over a form with the break
    removed, not on starts-with."""
    stamped = stamp_origin("note\ntags: source:wizard", "muse")
    assert stamped == "source:muse"
    assert "\n" not in stamped


def test_whitespace_inside_a_kept_tag_is_collapsed_to_one_line():
    stamped = stamp_origin("multi\nline\ttag,b", "muse")
    assert stamped == "multi line tag,b,source:muse"
    assert "\n" not in stamped and "\t" not in stamped


def test_a_near_miss_that_only_looks_like_the_prefix_is_kept():
    """The matcher is deliberately broad, so this is the other edge: a tag that contains
    no origin claim keeps working, including non-Latin ones."""
    assert stamp_origin("sources:muse", "muse") == "sources:muse,source:muse"
    assert stamp_origin("resource-map,\u043f\u044a\u0442\u0443\u0432\u0430\u043d\u0435", "muse") == (
        "resource-map,\u043f\u044a\u0442\u0443\u0432\u0430\u043d\u0435,source:muse"
    )


@pytest.mark.asyncio
async def test_a_forged_origin_does_not_reach_the_database(api):
    """The same property through the real route, not just the string function."""
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42)
    async with ac:
        resp = await ac.post(
            "/api/memories",
            json={"content": "x", "tags": "note\ntags: \u0455ource:wizard"},
            headers=MUSE,
        )
    assert resp.status_code == 200
    assert _inserted_tags(conn) == "source:muse"


# ─── change 4: tags the client did NOT send are stamped, not rewritten ──────


@pytest.mark.parametrize(
    "stored,expected",
    [
        pytest.param(None, "source:muse", id="no-stored-tags"),
        pytest.param("", "source:muse", id="empty"),
        pytest.param("a,b", "a,b,source:muse", id="plain"),
        pytest.param("a,source:muse", "a,source:muse", id="already-stamped"),
        pytest.param("wizard-tag,source:email", "wizard-tag,source:email,source:muse", id="keeps-another-origin"),
        pytest.param("data-source:api", "data-source:api,source:muse", id="keeps-a-tag-that-merely-contains-the-prefix"),
    ],
)
def test_stamp_stored_origin_adds_without_rewriting(stored, expected):
    """A value the client did not send carries no forgery, and an external key can reach
    a row it does not own through a write-share, so filtering it would delete provenance
    its owner wrote — over a PUT that only asked to change importance."""
    assert stamp_stored_origin(stored, "muse") == expected


def test_stamping_a_stored_value_is_idempotent():
    once = stamp_stored_origin("wizard-tag,source:email", "muse")
    assert stamp_stored_origin(once, "muse") == once


@pytest.mark.asyncio
async def test_an_update_leaves_provenance_it_did_not_write_in_place(api):
    """The reachable shape: wizard's row, write-shared to muse, updated for importance
    alone. Rewriting the tags column there would delete wizard's provenance on wizard's
    row over a request that never mentioned tags."""
    ac, conn, app_mod = api
    # check_memory_permission: the owner row, then the memory_shares row.
    conn.fetchrow.side_effect = [
        MockRow({"user_id": "wizard"}),
        MockRow({"permission": "write"}),
    ]
    conn.fetchval.return_value = "wizard-tag,source:email"
    async with ac:
        resp = await ac.put("/api/memories/7", json={"importance": 0.95}, headers=MUSE)
    assert resp.status_code == 200
    assert _updated_tags(conn) == "wizard-tag,source:email,source:muse"


@pytest.mark.asyncio
async def test_tags_the_client_does_send_are_still_filtered(api):
    """The other half of the split: anything in the BODY may be forging."""
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "muse"})
    async with ac:
        await ac.put("/api/memories/7", json={"tags": "a,source:email"}, headers=MUSE)
    assert _updated_tags(conn) == "a,source:muse"


# ─── change 4: every write an external key reaches ──────────────────────────


@pytest.mark.asyncio
async def test_a_store_through_an_external_key_carries_the_origin_tag(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42)
    async with ac:
        resp = await ac.post("/api/memories", json={"content": "remember this"}, headers=MUSE)
    assert resp.status_code == 200
    assert _inserted_tags(conn) == "source:muse"


@pytest.mark.asyncio
async def test_a_store_keeps_the_clients_own_tags_alongside(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42)
    async with ac:
        await ac.post("/api/memories", json={"content": "x", "tags": "travel,plans"}, headers=MUSE)
    assert _inserted_tags(conn) == "travel,plans,source:muse"


@pytest.mark.asyncio
async def test_a_client_cannot_opt_out_by_forging_the_tag(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42)
    async with ac:
        await ac.post("/api/memories", json={"content": "x", "tags": "source:wizard"}, headers=MUSE)
    assert _inserted_tags(conn) == "source:muse"


@pytest.mark.asyncio
async def test_an_admin_store_is_not_stamped(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42)
    async with ac:
        await ac.post("/api/memories", json={"content": "x", "tags": "travel"}, headers=ADMIN)
    assert _inserted_tags(conn) == "travel"


@pytest.mark.asyncio
async def test_an_update_is_stamped_even_when_the_client_sends_no_tags(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "muse"})
    conn.fetchval.return_value = "notes"
    async with ac:
        resp = await ac.put("/api/memories/7", json={"content": "revised"}, headers=MUSE)
    assert resp.status_code == 200
    assert _updated_tags(conn) == "notes,source:muse"


@pytest.mark.asyncio
async def test_a_repeated_update_does_not_accumulate_the_tag(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "muse"})
    conn.fetchval.return_value = "notes,source:muse"
    async with ac:
        await ac.put("/api/memories/7", json={"content": "revised again"}, headers=MUSE)
    assert _updated_tags(conn) == "notes,source:muse"


@pytest.mark.asyncio
async def test_an_update_that_does_send_tags_is_stamped_too(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "muse"})
    async with ac:
        await ac.put("/api/memories/7", json={"tags": "a,b"}, headers=MUSE)
    assert _updated_tags(conn) == "a,b,source:muse"
    conn.fetchval.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_admin_update_touches_neither_the_tags_nor_the_database(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "wizard"})
    async with ac:
        resp = await ac.put("/api/memories/7", json={"content": "revised"}, headers=ADMIN)
    assert resp.status_code == 200
    assert "tags = $" not in conn.execute.call_args[0][0]
    conn.fetchval.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_empty_update_from_an_external_key_is_still_rejected(api):
    """The stamp must not turn a no-op PUT into a successful write."""
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "muse"})
    async with ac:
        resp = await ac.put("/api/memories/7", json={}, headers=MUSE)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_the_origin_tag_does_not_cap_importance(api):
    """No importance ceiling: that was raised and declined. Muse's entries compete on the
    same footing as everyone else's."""
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=42, importance=1.0)
    async with ac:
        resp = await ac.post(
            "/api/memories", json={"content": "x", "importance": 1.0}, headers=MUSE
        )
    assert resp.status_code == 200
    assert conn.fetchrow.call_args[0][6] == 1.0


# ─── change 4: the MCP tool functions, which duplicate the REST writes ──────


@pytest.mark.asyncio
async def test_the_mcp_store_tool_stamps_an_external_caller(api):
    """Belt and braces. MCPAuthMiddleware refuses an external key before it can reach a
    tool, so this branch is unreachable over HTTP today. It exists so the invariant holds
    at the write itself rather than only at the door."""
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=99)
    app_mod._current_user.set("muse")
    app_mod._current_scope.set("external")
    try:
        await app_mod.memory_store(content="x", tags="travel")
    finally:
        app_mod._current_user.set("default")
        app_mod._current_scope.set("admin")
    assert _inserted_tags(conn) == "travel,source:muse"


@pytest.mark.asyncio
async def test_the_mcp_store_tool_leaves_an_admin_caller_alone(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = _memory_row(id=99)
    app_mod._current_user.set("wizard")
    try:
        await app_mod.memory_store(content="x", tags="travel")
    finally:
        app_mod._current_user.set("default")
    assert _inserted_tags(conn) == "travel"


@pytest.mark.asyncio
async def test_the_mcp_update_tool_stamps_an_external_caller(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "muse"})
    conn.fetchval.return_value = "notes,source:muse"
    app_mod._current_user.set("muse")
    app_mod._current_scope.set("external")
    try:
        await app_mod.memory_update(id=7, content="revised")
    finally:
        app_mod._current_user.set("default")
        app_mod._current_scope.set("admin")
    assert _updated_tags(conn) == "notes,source:muse"


@pytest.mark.asyncio
async def test_the_mcp_update_tool_leaves_an_admin_caller_alone(api):
    ac, conn, app_mod = api
    conn.fetchrow.return_value = MockRow({"user_id": "wizard"})
    app_mod._current_user.set("wizard")
    try:
        await app_mod.memory_update(id=7, content="revised")
    finally:
        app_mod._current_user.set("default")
    assert "tags = $" not in conn.execute.call_args[0][0]


# ─── the MCP recall tool folds its category like the REST one ───────────────


@pytest.mark.asyncio
async def test_the_mcp_recall_tool_folds_a_drift_twin(api):
    """app.py requires these two recall surfaces not to differ in what truth they serve,
    and the REST one now canonicalizes. Unfolded, 'Gotcha' reached `AND category = $4`
    verbatim and matched nothing."""
    ac, conn, app_mod = api
    conn.fetch.side_effect = [[], [], []]
    await app_mod.memory_recall(context="anything", category="Gotcha")
    assert "gotchas" in conn.fetch.call_args_list[0][0]


@pytest.mark.asyncio
async def test_the_mcp_recall_tool_rejects_a_category_that_is_not_canonical(api):
    ac, conn, app_mod = api
    result = json.loads(await app_mod.memory_recall(context="anything", category="banana"))
    assert "banana" in result["error"]
    conn.fetch.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_mcp_recall_tool_still_treats_a_blank_category_as_no_filter(api):
    """recall.py gates the clause on ``if category:``, so a blank string already meant
    "no filter" and must not become an error."""
    ac, conn, app_mod = api
    conn.fetch.side_effect = [[], [], []]
    await app_mod.memory_recall(context="anything", category="   ")
    sql = conn.fetch.call_args_list[0][0][0]
    assert "AND category =" not in sql
    assert "   " not in conn.fetch.call_args_list[0][0][1:]


# ─── the allowlist is the one source of truth ───────────────────────────────


def test_the_allowlist_is_exactly_what_was_granted():
    """Pinned as a literal so widening the external key's reach is never incidental.

    Nine came from the design. GET /api/auth-check is the tenth, added on review because
    it is the only way Muse can tell a correctly-scoped key from one that landed as a
    silent admin.
    """
    assert EXTERNAL_ALLOWED_OPERATIONS == frozenset({
        ("POST", "/api/memories"),
        ("POST", "/api/memories/recall"),
        ("GET", "/api/memories"),
        ("GET", "/api/memories/{memory_id}"),
        ("PUT", "/api/memories/{memory_id}"),
        ("DELETE", "/api/memories/{memory_id}"),
        ("POST", "/api/memories/{memory_id}/links"),
        ("GET", "/api/tags"),
        ("GET", "/api/categories"),
        ("GET", "/api/auth-check"),
    })


@pytest.mark.asyncio
async def test_the_external_key_can_self_test_and_sees_its_own_scope(api):
    """Muse has to be able to prove its key works, and see what scope it landed as.

    A key meant to be external but written in API_KEYS' flat shape is a silent admin key.
    From outside the cluster this endpoint is the only thing that reveals it, which is why
    it is on the allowlist rather than closed by omission like /api/stats.
    """
    ac, conn, app_mod = api
    async with ac:
        resp = await ac.get("/api/auth-check", headers=MUSE)
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "user_id": "muse", "scope": "external"}
