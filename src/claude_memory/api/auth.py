import json
import logging
import os
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# A key's scope says which endpoints it may call. "admin" is every endpoint and is what a
# key with no declared scope gets, so keys written in the flat shape keep working through
# the rollout. "external" is for a key handed to a third-party assistant (Muse); the
# endpoint allowlist enforces the difference, this module only carries the label.
Scope = Literal["admin", "external"]

VALID_SCOPES: tuple[Scope, ...] = ("admin", "external")
DEFAULT_SCOPE: Scope = "admin"

#: Scopes the endpoint allowlist leaves alone. Written as the UNrestricted set rather than
#: as a test against "external", so a third scope added to VALID_SCOPES is restricted from
#: the moment it exists and has to be listed here deliberately to gain full route access.
UNRESTRICTED_SCOPES: frozenset[Scope] = frozenset({"admin"})


def is_restricted_scope(scope: Scope) -> bool:
    """Whether the endpoint allowlist and the origin stamp apply to a key with this scope."""
    return scope not in UNRESTRICTED_SCOPES


@dataclass
class AuthUser:
    user_id: str
    scope: Scope = DEFAULT_SCOPE


class _EntryProblem(ValueError):
    """One API_KEYS entry is unusable, so that entry is dropped and the rest still load."""


def _coerce_scope(value: object) -> Scope:
    for scope in VALID_SCOPES:
        if value == scope:
            return scope
    valid = ", ".join(repr(s) for s in VALID_SCOPES)
    raise _EntryProblem(f"unrecognised scope, expected one of: {valid}")


def _require_key(key: str) -> str:
    if not key:
        # "Authorization: Bearer " strips to "", which an empty key would authenticate.
        raise _EntryProblem("the key is empty")
    return key


def _validate_user_id(user_id: object) -> str:
    """The user id has to be usable as a row's owner AND as the origin tag's payload."""
    if not isinstance(user_id, str) or not user_id.strip():
        raise _EntryProblem("the user id is not a non-empty string")
    if "," in user_id:
        # scopes.stamp_origin writes source:<user_id> as ONE element of a comma-separated
        # tags column, so a comma splits the stamp in two: the trailing fragment is not
        # recognised as an origin claim and one copy accumulates per external update.
        raise _EntryProblem("the user id contains a comma, which the origin tag cannot carry")
    if any(unicodedata.category(ch)[0] == "C" for ch in user_id):
        raise _EntryProblem("the user id contains a control character")
    return user_id


def _parse_entry(entry: object) -> tuple[str, Scope]:
    """Read one API_KEYS entry, in either the flat or the scoped shape."""
    if isinstance(entry, str):
        return _require_key(entry), DEFAULT_SCOPE
    if isinstance(entry, dict):
        unknown = sorted(str(field) for field in entry if field not in ("key", "scope"))
        if unknown:
            raise _EntryProblem(f"unrecognised field(s): {', '.join(unknown)}")
        key = entry.get("key")
        if not isinstance(key, str):
            raise _EntryProblem("no string 'key'")
        return _require_key(key), _coerce_scope(entry.get("scope", DEFAULT_SCOPE))
    raise _EntryProblem("it is neither a key string nor an object with 'key' and an optional 'scope'")


def _parse_api_keys(raw: str) -> tuple[dict[str, str], dict[str, Scope], list[str]]:
    """Invert API_KEYS into key -> user_id and key -> scope, plus what could not be used.

    A malformed ENTRY is dropped and reported, and the rest of the map still loads.
    Dropping only ever removes access, while raising takes the whole service down at the
    next restart — which arrives hours or days after the hand-edit in Vault, carries the
    keys that were fine with it, and stops the old pod first (strategy Recreate). One
    unusable entry costing its own holder a 401 is the smaller failure, and the startup
    log says which entry and why.

    A malformed DOCUMENT still raises: there is no partial result to salvage, and a value
    that is not a JSON object failed exactly this way before scopes existed.
    """
    try:
        parsed: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"API_KEYS is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("API_KEYS must be a JSON object mapping user_id to a key or a scoped entry")

    problems: list[str] = []
    holders: dict[str, list[tuple[str, Scope]]] = defaultdict(list)
    entries: dict[Any, Any] = parsed
    for raw_user_id, entry in entries.items():
        try:
            user_id = _validate_user_id(raw_user_id)
            key, scope = _parse_entry(entry)
        except _EntryProblem as exc:
            # The offending VALUE is never echoed: a key pasted into the wrong field would
            # otherwise land in the container log.
            problems.append(f"API_KEYS entry for {str(raw_user_id)!r} dropped, {exc}")
            continue
        holders[key].append((user_id, scope))

    key_to_user: dict[str, str] = {}
    key_to_scope: dict[str, Scope] = {}
    for key, claimants in holders.items():
        if len(claimants) > 1:
            names = ", ".join(repr(user_id) for user_id, _ in claimants)
            problems.append(
                f"API_KEYS gives one key to {names}; that key is dropped, because which user "
                "it authenticates and which scope it carries are both ambiguous"
            )
            continue
        key_to_user[key], key_to_scope[key] = claimants[0]
    return key_to_user, key_to_scope, problems


# Multi-user mode, two shapes, mixable while a deployment migrates from one to the other:
#   API_KEYS='{"viktor": "key1", "user2": "key2"}'               (flat, scope admin)
#   API_KEYS='{"muse": {"key": "key3", "scope": "external"}}'    (scoped)
# Single-user mode: API_KEY="some-key" (backward compatible, user_id="default")
_api_keys_json = os.environ.get("API_KEYS", "")
_api_key_single = os.environ.get("API_KEY", "")

_key_to_user: dict[str, str] = {}
_key_to_scope: dict[str, Scope] = {}

#: Entries API_KEYS declared that could not be used, in plain words. Warned here, where
#: logging may not be configured yet, and logged again from the app's lifespan, where it
#: is, so the line reaches Loki.
API_KEYS_PROBLEMS: list[str] = []

if _api_keys_json:
    _key_to_user, _key_to_scope, API_KEYS_PROBLEMS = _parse_api_keys(_api_keys_json)
elif _api_key_single:
    _key_to_user = {_api_key_single: "default"}
    _key_to_scope = {_api_key_single: DEFAULT_SCOPE}

if API_KEYS_PROBLEMS:
    logger.warning("API_KEYS: %s", " | ".join(API_KEYS_PROBLEMS))


def scope_by_user() -> dict[str, Scope]:
    """Which scope each configured key landed with, keyed by user id and never by key.

    An entry written in the flat shape silently gets "admin", so a muse key pasted into
    the wrong shape is a working admin key with no origin stamp. Nothing else in the
    running service reveals that: /api/auth-check echoes the CALLER's own scope, which an
    operator holding a different key cannot use. The startup log prints this map instead,
    so a mis-scoped entry is one query away.
    """
    return {_key_to_user[key]: scope for key, scope in _key_to_scope.items()}


async def get_current_user(authorization: str = Header(...)) -> AuthUser:
    token = authorization.removeprefix("Bearer ").strip()
    user_id = _key_to_user.get(token)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return AuthUser(user_id=user_id, scope=_key_to_scope.get(token, DEFAULT_SCOPE))
