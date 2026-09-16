"""The curated OpenAPI document served at /muse/openapi.json (design change 3).

Generated from ``EXTERNAL_ALLOWED_OPERATIONS`` — the same frozenset the auth layer
enforces — so the document Muse's connector builder reads and the rule the API applies
cannot drift apart.

Two hazards this module exists to handle:

``FastAPI.openapi()`` caches its result and hands back THE SAME DICT BY REFERENCE on every
call, so filtering it in place would corrupt ``/openapi.json`` permanently for every other
caller. Hence the deepcopy, measured at 1.23 ms over 50 iterations on the full document.

Schema filtering has to be a transitive closure over ``$ref``. Every kept operation keeps
its 422, which references ``HTTPValidationError``, which references ``ValidationError``
and nothing else does — so a one-level pass ships a dangling reference.
"""

import copy
import os
from typing import Any

from claude_memory.api.scopes import EXTERNAL_ALLOWED_OPERATIONS

MUSE_SPEC_TITLE = "Claude Memory (Muse connector)"

#: Without a ``servers`` entry a strict client resolves paths against the document's own
#: URL, which is wrong the moment Muse fetches the spec through a redirect.
MUSE_SPEC_SERVER_URL = os.environ.get("MUSE_SPEC_SERVER_URL", "https://claude-memory.viktorbarzin.me")

_SCHEMA_REF_PREFIX = "#/components/schemas/"


def _schema_refs(node: object, found: set[str]) -> None:
    """Collect every component-schema name referenced anywhere under ``node``."""
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith(_SCHEMA_REF_PREFIX):
            found.add(ref[len(_SCHEMA_REF_PREFIX):])
        for value in node.values():
            _schema_refs(value, found)
    elif isinstance(node, list):
        for value in node:
            _schema_refs(value, found)


def _reachable_schemas(paths: dict[str, Any], schemas: dict[str, Any]) -> set[str]:
    """Every schema the kept operations reach, directly or through another schema."""
    reachable: set[str] = set()
    _schema_refs(paths, reachable)
    frontier = set(reachable)
    while frontier:
        nested: set[str] = set()
        _schema_refs(schemas.get(frontier.pop(), {}), nested)
        new = nested - reachable
        reachable |= new
        frontier |= new
    return reachable


def build_muse_openapi(full_schema: dict[str, Any]) -> dict[str, Any]:
    """Filter the app's OpenAPI document down to the operations an external key may call.

    Beyond filtering it does two things a connector builder needs and a human reader does
    not. It replaces the per-operation ``authorization`` header parameter with a declared
    ``bearerAuth`` security scheme, because the generated document has no security schemes
    at all and a builder then has no field to bind the pasted token to. And it sets
    ``servers`` and ``info.title``, neither of which the generated document carries.
    """
    doc: dict[str, Any] = copy.deepcopy(full_schema)

    paths: dict[str, Any] = {}
    for path, item in doc.get("paths", {}).items():
        kept: dict[str, Any] = {}
        for method, operation in item.items():
            if (method.upper(), path) not in EXTERNAL_ALLOWED_OPERATIONS:
                continue
            parameters = [
                p for p in operation.get("parameters", [])
                if not (p.get("in") == "header" and str(p.get("name", "")).lower() == "authorization")
            ]
            if parameters:
                operation["parameters"] = parameters
            else:
                operation.pop("parameters", None)
            kept[method] = operation
        if kept:
            paths[path] = kept
    doc["paths"] = paths

    schemas: dict[str, Any] = doc.get("components", {}).get("schemas", {})
    reachable = _reachable_schemas(paths, schemas)

    components: dict[str, Any] = doc.setdefault("components", {})
    components["schemas"] = {name: schema for name, schema in schemas.items() if name in reachable}
    components["securitySchemes"] = {"bearerAuth": {"type": "http", "scheme": "bearer"}}
    doc["security"] = [{"bearerAuth": []}]
    doc["servers"] = [{"url": MUSE_SPEC_SERVER_URL}]
    doc["info"] = dict(doc.get("info", {}), title=MUSE_SPEC_TITLE)
    return doc
