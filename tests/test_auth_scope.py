"""Tests for API-key scopes in API_KEYS.

Change 1 of the Muse memory connector design (2026-09-16): API_KEYS grows a richer
per-user shape carrying a scope, and the flat shape keeps working so existing keys survive
the rollout.

A malformed DOCUMENT still stops the process, as it did before scopes existed. A malformed
ENTRY is dropped, reported by user id and never by key, and the rest of the map loads:
the value is hand-edited in Vault and reaches the pod at the next restart, so refusing to
boot takes down the keys that were fine, hours or days after the edit. Dropping only ever
removes access, and it closes two shapes master booted with (an empty key, which
``Authorization: Bearer `` would have matched, and one key held by two users).
"""

import importlib
import json
import os
from unittest.mock import patch

import pytest
from fastapi import HTTPException

FLAT = '{"wizard": "key-wizard", "emo": "key-emo"}'
SCOPED = '{"wizard": {"key": "key-wizard", "scope": "admin"}, "muse": {"key": "key-muse", "scope": "external"}}'
MIXED = '{"wizard": "key-wizard", "muse": {"key": "key-muse", "scope": "external"}}'


def _reload_auth(env: dict):
    """Reload the auth module with exactly these API_KEY / API_KEYS values."""
    with patch.dict(os.environ, {}, clear=False):
        for name in ("API_KEY", "API_KEYS"):
            os.environ.pop(name, None)
        os.environ.update(env)

        import claude_memory.api.auth as auth_mod

        importlib.reload(auth_mod)
        return auth_mod


@pytest.fixture(autouse=True)
def restore_auth_module():
    """Leave the module as a fresh import with no keys, so later test files are unaffected."""
    yield
    _reload_auth({})


# --- the flat shape still works ---


@pytest.mark.asyncio
async def test_flat_shape_still_authenticates():
    auth = _reload_auth({"API_KEYS": FLAT})
    assert (await auth.get_current_user(authorization="Bearer key-wizard")).user_id == "wizard"
    assert (await auth.get_current_user(authorization="Bearer key-emo")).user_id == "emo"


@pytest.mark.asyncio
async def test_flat_shape_defaults_to_admin_scope():
    auth = _reload_auth({"API_KEYS": FLAT})
    user = await auth.get_current_user(authorization="Bearer key-wizard")
    assert user.scope == "admin"


def test_flat_shape_keeps_the_key_to_user_map():
    # app.py imports _key_to_user directly for GET /api/users and the MCP SSE token path.
    auth = _reload_auth({"API_KEYS": FLAT})
    assert auth._key_to_user == {"key-wizard": "wizard", "key-emo": "emo"}


# --- the scoped shape ---


@pytest.mark.asyncio
async def test_scoped_shape_carries_the_scope():
    auth = _reload_auth({"API_KEYS": SCOPED})
    admin = await auth.get_current_user(authorization="Bearer key-wizard")
    assert (admin.user_id, admin.scope) == ("wizard", "admin")

    external = await auth.get_current_user(authorization="Bearer key-muse")
    assert (external.user_id, external.scope) == ("muse", "external")


def test_scoped_shape_keeps_the_key_to_user_map():
    auth = _reload_auth({"API_KEYS": SCOPED})
    assert auth._key_to_user == {"key-wizard": "wizard", "key-muse": "muse"}


@pytest.mark.asyncio
async def test_both_shapes_can_be_mixed_during_the_rollout():
    auth = _reload_auth({"API_KEYS": MIXED})
    assert (await auth.get_current_user(authorization="Bearer key-wizard")).scope == "admin"
    assert (await auth.get_current_user(authorization="Bearer key-muse")).scope == "external"


@pytest.mark.asyncio
async def test_scoped_entry_without_a_scope_is_admin():
    auth = _reload_auth({"API_KEYS": '{"wizard": {"key": "key-wizard"}}'})
    user = await auth.get_current_user(authorization="Bearer key-wizard")
    assert (user.user_id, user.scope) == ("wizard", "admin")


@pytest.mark.asyncio
async def test_bearer_prefix_and_surrounding_space_are_stripped():
    auth = _reload_auth({"API_KEYS": SCOPED})
    assert (await auth.get_current_user(authorization="key-muse")).scope == "external"
    assert (await auth.get_current_user(authorization="Bearer  key-muse ")).scope == "external"


# --- the single-key fallback ---


@pytest.mark.asyncio
async def test_single_api_key_is_default_user_with_admin_scope():
    auth = _reload_auth({"API_KEY": "solo-key"})
    user = await auth.get_current_user(authorization="Bearer solo-key")
    assert (user.user_id, user.scope) == ("default", "admin")


@pytest.mark.asyncio
async def test_api_keys_wins_over_the_single_key_fallback():
    auth = _reload_auth({"API_KEYS": SCOPED, "API_KEY": "solo-key"})
    with pytest.raises(HTTPException):
        await auth.get_current_user(authorization="Bearer solo-key")


# --- rejection ---


@pytest.mark.asyncio
async def test_unknown_token_still_raises_401():
    auth = _reload_auth({"API_KEYS": SCOPED})
    with pytest.raises(HTTPException) as exc_info:
        await auth.get_current_user(authorization="Bearer not-a-key")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_a_user_id_is_not_a_token():
    auth = _reload_auth({"API_KEYS": SCOPED})
    with pytest.raises(HTTPException) as exc_info:
        await auth.get_current_user(authorization="Bearer muse")
    assert exc_info.value.status_code == 401


# --- a malformed DOCUMENT still fails loudly at import ---


@pytest.mark.parametrize(
    "api_keys",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param('["wizard"]', id="top-level-not-an-object"),
    ],
)
def test_a_malformed_document_raises(api_keys):
    """There is no partial result to salvage, and both shapes failed this way before
    scopes existed (master: json.loads, then .items() on a list)."""
    with pytest.raises(ValueError):
        _reload_auth({"API_KEYS": api_keys})


# --- a malformed ENTRY is dropped, and the rest of the map still loads ---

# Every shape that raised at import while this branch was in review. Each is delivered by
# a hand-edit in Vault, synced into an env var, and read only at the NEXT restart — hours
# or days later, with strategy Recreate stopping the old pod first — so raising here is
# downtime for every other key rather than a blocked rollout. Two of these booted on
# master ('{"legacy": ""}' mapping the empty token, and the duplicate key, last-wins);
# both are now dropped, which is stricter than master AND still boots.
BAD_ENTRIES = [
    pytest.param('{"wizard": "key-wizard", "legacy": ""}', "legacy", id="empty-key-flat"),
    pytest.param('{"wizard": "key-wizard", "legacy": 5}', "legacy", id="entry-neither-string-nor-object"),
    pytest.param('{"wizard": "key-wizard", "legacy": null}', "legacy", id="entry-null"),
    pytest.param('{"wizard": "key-wizard", "muse": {"scope": "external"}}', "muse", id="object-entry-without-key"),
    pytest.param('{"wizard": "key-wizard", "muse": {"key": 5}}', "muse", id="non-string-key"),
    pytest.param('{"wizard": "key-wizard", "muse": {"key": ""}}', "muse", id="empty-key-in-object"),
    pytest.param('{"wizard": "key-wizard", "muse": {"key": "k", "scope": "extrenal"}}', "muse", id="misspelled-scope"),
    pytest.param('{"wizard": "key-wizard", "muse": {"key": "k", "scope": ""}}', "muse", id="empty-scope"),
    pytest.param('{"wizard": "key-wizard", "muse": {"key": "k", "scope": null}}', "muse", id="null-scope"),
    pytest.param('{"wizard": "key-wizard", "muse": {"key": "k", "scoope": "external"}}', "muse", id="misspelled-field"),
    pytest.param('{"wizard": "key-wizard", "": "key-nobody"}', "", id="empty-user-id"),
    pytest.param('{"wizard": "key-wizard", "   ": "key-blank"}', "   ", id="blank-user-id"),
    pytest.param('{"wizard": "key-wizard", "mu,se": {"key": "k"}}', "mu,se", id="comma-in-user-id"),
    pytest.param('{"wizard": "key-wizard", "mu\\nse": {"key": "k"}}', "mu\nse", id="control-char-in-user-id"),
]


@pytest.mark.parametrize("api_keys,bad_user", BAD_ENTRIES)
def test_a_bad_entry_does_not_take_down_the_keys_that_are_fine(api_keys, bad_user):
    auth = _reload_auth({"API_KEYS": api_keys})
    assert auth._key_to_user == {"key-wizard": "wizard"}
    assert auth._key_to_scope == {"key-wizard": "admin"}


@pytest.mark.parametrize("api_keys,bad_user", BAD_ENTRIES)
def test_a_dropped_entry_is_reported_by_name(api_keys, bad_user):
    auth = _reload_auth({"API_KEYS": api_keys})
    assert len(auth.API_KEYS_PROBLEMS) == 1
    assert repr(bad_user) in auth.API_KEYS_PROBLEMS[0]


@pytest.mark.asyncio
async def test_a_dropped_entry_authenticates_nobody():
    auth = _reload_auth({"API_KEYS": '{"wizard": "key-wizard", "muse": {"key": "k", "scope": "extrenal"}}'})
    with pytest.raises(HTTPException) as exc_info:
        await auth.get_current_user(authorization="Bearer k")
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_an_empty_key_is_never_accepted_as_a_token():
    # The dangerous case behind the empty-key drop, and the one master BOOTED with:
    # `Authorization: Bearer ` strips to "", which would otherwise match and authenticate
    # as that user.
    auth = _reload_auth({"API_KEYS": '{"wizard": "key-wizard", "legacy": ""}'})
    with pytest.raises(HTTPException):
        await auth.get_current_user(authorization="Bearer ")


@pytest.mark.asyncio
async def test_one_key_given_to_two_users_authenticates_neither():
    """Master booted last-wins. Last-wins is an escalation when the two entries disagree
    on scope, and the owning user is ambiguous either way, so the key is dropped."""
    auth = _reload_auth({"API_KEYS": '{"wizard": "key-wizard", "muse": {"key": "shared", "scope": "external"}, "emo": "shared"}'})
    assert auth._key_to_user == {"key-wizard": "wizard"}
    with pytest.raises(HTTPException):
        await auth.get_current_user(authorization="Bearer shared")
    assert any("'muse'" in p and "'emo'" in p for p in auth.API_KEYS_PROBLEMS)


def test_a_dropped_entry_never_echoes_its_key():
    auth = _reload_auth({"API_KEYS": '{"muse": {"key": "s3cr3t-key", "scope": "extrenal"}}'})
    reported = " ".join(auth.API_KEYS_PROBLEMS)
    assert "muse" in reported
    assert "s3cr3t-key" not in reported


def test_a_clean_document_reports_no_problems():
    assert _reload_auth({"API_KEYS": SCOPED}).API_KEYS_PROBLEMS == []


# --- the user id has to be usable as the origin tag's payload (scopes.stamp_origin) ---


def test_a_user_id_with_a_comma_is_dropped():
    """source:<user_id> is ONE element of a comma-separated tags column. With a comma in
    the id the stamp splits in two, the trailing fragment is not recognised as an origin
    claim, and one copy accumulates per external update."""
    auth = _reload_auth({"API_KEYS": '{"mu,se": {"key": "k", "scope": "external"}}'})
    assert auth._key_to_user == {}
    assert "comma" in auth.API_KEYS_PROBLEMS[0]


def test_every_user_id_auth_accepts_gives_an_idempotent_stamp():
    from claude_memory.api.scopes import stamp_origin

    auth = _reload_auth({"API_KEYS": json.dumps({
        "wizard": "k1", "muse": {"key": "k2", "scope": "external"},
        "some user": "k3", "user.with-punct+1": "k4", "потребител": "k5",
    })})
    for user_id in auth._key_to_user.values():
        once = stamp_origin("a,b", user_id)
        assert stamp_origin(once, user_id) == once
        assert stamp_origin(stamp_origin(once, user_id), user_id) == once
        assert once.count(",") == 2


# --- which scopes the allowlist restricts ---


def test_admin_is_the_only_unrestricted_scope():
    """The route guard tests membership of UNRESTRICTED_SCOPES rather than the string
    "external", so a third scope added to VALID_SCOPES is restricted from the moment it
    exists instead of silently getting admin-equivalent route access."""
    import claude_memory.api.auth as auth_mod

    assert auth_mod.UNRESTRICTED_SCOPES == frozenset({"admin"})
    assert set(auth_mod.UNRESTRICTED_SCOPES) <= set(auth_mod.VALID_SCOPES)
    for scope in auth_mod.VALID_SCOPES:
        assert auth_mod.is_restricted_scope(scope) is (scope not in auth_mod.UNRESTRICTED_SCOPES)


# --- the startup summary ---


def test_scope_by_user_reports_what_each_entry_parsed_as():
    """The flat shape silently means admin, so a muse key pasted into it is a working
    admin key. This map is what makes that visible without probing."""
    auth = _reload_auth({"API_KEYS": MIXED})
    assert auth.scope_by_user() == {"wizard": "admin", "muse": "external"}


def test_scope_by_user_shows_a_flat_muse_entry_as_admin():
    auth = _reload_auth({"API_KEYS": '{"wizard": "key-wizard", "muse": "key-muse"}'})
    assert auth.scope_by_user() == {"wizard": "admin", "muse": "admin"}


def test_scope_by_user_covers_the_single_key_fallback():
    assert _reload_auth({"API_KEY": "solo-key"}).scope_by_user() == {"default": "admin"}
