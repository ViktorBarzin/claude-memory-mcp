"""Tests for the curated /muse/openapi.json document and the allowlist behind it.

Change 3 of the Muse memory connector design (2026-09-16): Muse's connector is pointed
at a document generated from the SAME frozenset the auth layer enforces, so the spec and
the enforcement cannot drift apart. The document is a strict subset of the already-open
/openapi.json, so it needs no authentication of its own.

Two hazards these tests pin, both of which look fine until they don't:
  * app.openapi() hands back the cached dict BY REFERENCE, so building the curated doc
    off it without a deepcopy corrupts /openapi.json for every later caller.
  * schema filtering has to be a transitive closure over $refs. Every filtered operation
    keeps its 422, which references HTTPValidationError, which references ValidationError
    and nothing else does. A one-level pass ships a dangling $ref.
"""

import copy
import importlib
import json
import os
import re
from unittest.mock import patch

import pytest
from fastapi.routing import APIRoute
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from claude_memory.api.models import (
    CANONICAL_CATEGORIES, CATEGORY_ENUM, MemoryRecall, MemoryStore, canonicalize_category,
)
from claude_memory.api.muse_spec import MUSE_SPEC_SERVER_URL, MUSE_SPEC_TITLE, build_muse_openapi
from claude_memory.api.scopes import (
    EXTERNAL_ALLOWED_OPERATIONS,
    EXTERNAL_PUBLIC_OPERATIONS,
    MUSE_SPEC_PATH,
)

# The full document as it stands today. Asserted rather than described, so the curated
# route cannot quietly change what every other client already reads.
FULL_SCHEMA_OPERATION_COUNT = 26
FULL_SCHEMA_NAMES = {
    "HTTPValidationError", "LinkCreate", "MemoryRecall", "MemoryResponse", "MemoryStore",
    "MemoryUpdate", "SecretResponse", "ShareMemory", "ShareTag", "SyncResponse",
    "UnshareTag", "ValidationError",
}

# The endpoints the design closes to an external key, by the string a reader of the design
# would grep for. The last three were named in a routing NOTE inside get_memory's
# docstring, which FastAPI renders as the operation description, so the curated document
# listed three endpoint names it exists to withhold — an LLM-driven connector reads that
# free text as readily as it reads the paths.
CLOSED_PATHS = (
    "/api/memories/import",
    "/api/memories/migrate-secrets",
    "/api/users",
    "/api/memories/{memory_id}/secret",
    "/api/memories/sync",
    "/api/memories/shared-with-me",
    "/api/memories/my-shares",
)


@pytest.fixture
def app_module():
    """A freshly imported app, so the document under test is this worktree's routes."""
    with patch.dict(
        os.environ,
        {"API_KEY": "spec-test-key", "API_KEYS": "", "DATABASE_URL": "postgresql://test"},
    ):
        import claude_memory.api.app as app_mod
        import claude_memory.api.auth as auth_mod

        importlib.reload(auth_mod)
        importlib.reload(app_mod)
        yield app_mod


@pytest.fixture
def curated(app_module):
    return build_muse_openapi(app_module.app.openapi())


def _live_operations(app):
    return {
        (method, route.path)
        for route in app.routes
        if isinstance(route, APIRoute)
        for method in route.methods
    }


def _document_operations(doc):
    return {(method.upper(), path) for path, item in doc["paths"].items() for method in item}


def _refs(node, found):
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            found.add(ref)
        for value in node.values():
            _refs(value, found)
    elif isinstance(node, list):
        for value in node:
            _refs(value, found)
    return found


# --- the allowlist names real routes ---


def test_every_allowlisted_operation_is_a_live_route(app_module):
    """The allowlist cannot name a path the app never serves — a typo there would
    silently grant nothing and silently exclude the operation from the spec."""
    missing = EXTERNAL_ALLOWED_OPERATIONS - _live_operations(app_module.app)
    assert not missing


def test_every_public_operation_is_a_live_route(app_module):
    missing = EXTERNAL_PUBLIC_OPERATIONS - _live_operations(app_module.app)
    assert not missing


def test_the_two_sets_do_not_overlap():
    assert not (EXTERNAL_ALLOWED_OPERATIONS & EXTERNAL_PUBLIC_OPERATIONS)


def test_no_closed_endpoint_is_allowlisted():
    """Derived from the allowlist, not from a second hand-kept list of denials."""
    allowed_paths = {path for _, path in EXTERNAL_ALLOWED_OPERATIONS}
    for path in CLOSED_PATHS:
        assert path not in allowed_paths


def test_link_deletion_stays_out_of_the_allowlist():
    """Deliberate asymmetry: the design grants link CREATION only, so Muse can create a
    link it cannot remove through its own connector. Pinned so it reads as a choice."""
    allowed_paths = {path for _, path in EXTERNAL_ALLOWED_OPERATIONS}
    assert "/api/memories/{memory_id}/links" in allowed_paths
    assert "/api/memories/{memory_id}/links/{dst_id}/{link_type}" not in allowed_paths


# --- the curated document ---


def test_curated_document_lists_exactly_the_allowlisted_operations(curated):
    assert _document_operations(curated) == set(EXTERNAL_ALLOWED_OPERATIONS)


def test_curated_document_omits_every_closed_endpoint(curated):
    for path in CLOSED_PATHS:
        assert path not in curated["paths"]


def test_no_closed_endpoint_is_named_anywhere_in_the_curated_document(curated):
    """Path keys are not the only place an endpoint name appears. Summaries, descriptions
    and examples are free text carried over from the source, and a docstring written as an
    internal routing note shipped three closed endpoint names inside the document."""
    text = json.dumps(curated)
    for path in CLOSED_PATHS:
        assert path not in text


def test_the_curated_document_mentions_no_path_it_does_not_serve(curated):
    """Stronger than the list above, and it needs no maintenance: every /api/... string
    anywhere in the document has to be one of the operations it actually grants."""
    mentioned = set(re.findall(r"/api/[A-Za-z0-9_{}/-]+", json.dumps(curated)))
    assert mentioned == {path for _, path in EXTERNAL_ALLOWED_OPERATIONS}


def test_the_spec_route_is_in_neither_document(app_module, curated):
    """include_in_schema=False is load-bearing: without it the route shows up as an
    operation inside /openapi.json and inside its own output."""
    assert MUSE_SPEC_PATH not in curated["paths"]
    assert MUSE_SPEC_PATH not in app_module.app.openapi()["paths"]


def test_every_ref_in_the_curated_document_resolves(curated):
    kept = set(curated["components"]["schemas"])
    for ref in _refs(curated, set()):
        assert ref.startswith("#/components/schemas/")
        assert ref.removeprefix("#/components/schemas/") in kept


def test_a_transitively_referenced_schema_is_kept(curated):
    """HTTPValidationError arrives via each operation's 422; ValidationError arrives only
    via HTTPValidationError. A one-level filter drops the second and ships a dangling
    $ref, so this is the test that fails loudly if the closure is ever flattened."""
    kept = set(curated["components"]["schemas"])
    assert "HTTPValidationError" in kept
    assert "ValidationError" in kept


def test_unreferenced_schemas_are_dropped(curated):
    kept = set(curated["components"]["schemas"])
    assert kept.isdisjoint({"SecretResponse", "ShareMemory", "ShareTag", "SyncResponse", "UnshareTag"})
    assert kept < FULL_SCHEMA_NAMES


def test_bearer_auth_replaces_the_authorization_header_parameter(curated):
    """A connector builder binds the pasted token to a declared security scheme. The
    generated document has none, and renders the token as a plain required header."""
    assert curated["components"]["securitySchemes"] == {"bearerAuth": {"type": "http", "scheme": "bearer"}}
    assert curated["security"] == [{"bearerAuth": []}]
    for path, item in curated["paths"].items():
        for method, operation in item.items():
            names = {str(p.get("name", "")).lower() for p in operation.get("parameters", [])}
            assert "authorization" not in names, f"{method} {path}"


def test_every_category_field_declares_the_closed_vocabulary(curated):
    """category is a closed 17-value set the server enforces with a 422, and it rendered
    as a bare string while link_type and sort_by carried enums in the same document. A
    connector builder reading a bare string offers a free-text box for a field that
    rejects free text."""
    schemas = curated["components"]["schemas"]
    for name in ("MemoryStore", "MemoryRecall", "MemoryUpdate"):
        enum = schemas[name]["properties"]["category"].get("enum")
        assert enum is not None, name
        assert [v for v in enum if v is not None] == CATEGORY_ENUM, name
        assert sorted(CANONICAL_CATEGORIES) == CATEGORY_ENUM


def test_the_enum_lists_only_values_the_server_accepts(curated):
    """The document is allowed to be stricter than the API — it omits the drift twins
    canonicalize_category folds — but never looser."""
    for value in CATEGORY_ENUM:
        assert canonicalize_category(value) == value


def test_an_optional_category_keeps_null_in_its_enum(curated):
    """MemoryRecall.category and MemoryUpdate.category render as an anyOf carrying null;
    an enum beside it that omitted null would contradict the type."""
    schemas = curated["components"]["schemas"]
    for name in ("MemoryRecall", "MemoryUpdate"):
        assert None in schemas[name]["properties"]["category"]["enum"], name
    assert None not in schemas["MemoryStore"]["properties"]["category"]["enum"]


def test_recall_folds_and_rejects_its_category_like_the_write_paths_do():
    """Recall matches with ``AND category = $4``, so before this an unfolded 'Gotcha' and
    an invented 'banana' both returned 200 and zero rows while store rejected them with a
    422 — a filter that silently matches nothing reads as "no memories", not as a typo."""
    assert MemoryRecall(context="x", category="Gotcha").category == "gotchas"
    assert MemoryRecall(context="x", category="gotcha").category == "gotchas"
    assert MemoryStore(content="x", category="Gotcha").category == "gotchas"
    with pytest.raises(ValidationError):
        MemoryRecall(context="x", category="banana")
    with pytest.raises(ValidationError):
        MemoryStore(content="x", category="banana")


def test_recall_still_treats_a_blank_category_as_no_filter():
    """recall.py gates the SQL clause on ``if category:``, so an empty string already
    meant "no filter" and has to keep meaning it rather than becoming a 422."""
    assert MemoryRecall(context="x").category is None
    assert MemoryRecall(context="x", category="").category is None
    assert MemoryRecall(context="x", category="   ").category is None


def test_servers_and_title_are_set(curated):
    assert curated["servers"] == [{"url": MUSE_SPEC_SERVER_URL}]
    assert curated["info"]["title"] == MUSE_SPEC_TITLE


def test_the_default_server_url_and_title_are_the_deployed_ones(app_module):
    """Asserting the document against the module's own constants can never fail whatever
    those constants say, so the values themselves are pinned here — with the environment
    variable cleared, since it is what the constant reads at import."""
    import claude_memory.api.muse_spec as muse_spec

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MUSE_SPEC_SERVER_URL", None)
        importlib.reload(muse_spec)
        try:
            doc = muse_spec.build_muse_openapi(app_module.app.openapi())
            assert doc["servers"] == [{"url": "https://claude-memory.viktorbarzin.me"}]
            assert doc["info"]["title"] == "Claude Memory (Muse connector)"
        finally:
            importlib.reload(muse_spec)


def test_the_server_url_can_be_overridden_for_another_deployment(app_module):
    import claude_memory.api.muse_spec as muse_spec

    with patch.dict(os.environ, {"MUSE_SPEC_SERVER_URL": "https://staging.example/api"}):
        importlib.reload(muse_spec)
        try:
            doc = muse_spec.build_muse_openapi(app_module.app.openapi())
            assert doc["servers"] == [{"url": "https://staging.example/api"}]
        finally:
            importlib.reload(muse_spec)


def test_path_parameters_survive_the_authorization_strip(curated):
    """Stripping the auth header must not take memory_id with it."""
    names = {p["name"] for p in curated["paths"]["/api/memories/{memory_id}"]["get"]["parameters"]}
    assert names == {"memory_id"}


# --- the shared-cache hazard ---


def test_building_the_curated_document_does_not_mutate_the_full_schema(app_module):
    full = app_module.app.openapi()
    before = copy.deepcopy(full)

    build_muse_openapi(full)

    assert full == before
    assert app_module.app.openapi() is full


def test_two_builds_are_identical(app_module):
    first = build_muse_openapi(app_module.app.openapi())
    second = build_muse_openapi(app_module.app.openapi())
    assert first == second


def test_the_full_document_is_unchanged_by_the_curated_route_existing(app_module):
    doc = app_module.app.openapi()
    assert len(_document_operations(doc)) == FULL_SCHEMA_OPERATION_COUNT
    assert set(doc["components"]["schemas"]) == FULL_SCHEMA_NAMES
    assert set(doc["components"]) == {"schemas"}
    assert "security" not in doc
    assert doc["info"]["title"] == "Claude Memory API"
    # The app-wide scope dependency takes a Request, which must add no parameter here.
    assert "parameters" not in doc["paths"]["/health"]["get"]


# --- the route ---


@pytest.mark.asyncio
async def test_the_route_serves_the_document_without_authentication(app_module):
    """A connector builder fetches the spec BEFORE the user has pasted a token, and the
    document only ever removes information from the already-open /openapi.json."""
    transport = ASGITransport(app=app_module.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(MUSE_SPEC_PATH)

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == build_muse_openapi(app_module.app.openapi())
