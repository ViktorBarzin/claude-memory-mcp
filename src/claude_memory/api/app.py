"""Claude Memory API -- shared persistent memory with PostgreSQL full-text search."""

import asyncio
import hashlib
import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

import asyncpg  # type: ignore[import-untyped]
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from claude_memory.api import auth as auth_module
from claude_memory.api import metrics
from claude_memory.api.auth import (
    DEFAULT_SCOPE, AuthUser, get_current_user, is_restricted_scope, _key_to_user,
)
from claude_memory.api.auth import Scope as KeyScope
from claude_memory.api.database import close_pool, get_pool, init_pool
from claude_memory.api.models import (
    LINK_TYPES, LinkCreate, MemoryRecall, MemoryResponse, MemoryStore, MemoryUpdate,
    SecretResponse, ShareMemory, ShareTag, SyncResponse, UnshareTag,
    canonicalize_category, validate_content_bound,
)
from claude_memory.api.muse_spec import build_muse_openapi
from claude_memory.api.permissions import check_memory_permission
from claude_memory.api.scopes import (
    MUSE_SPEC_PATH, is_allowed_for_external, stamp_origin, stamp_stored_origin,
)
from claude_memory.api.recall import (
    SUPERSEDES_DEPTH_CAP, _fused_recall, apply_link_semantics, embeddings_enabled,
    schedule_embedding,
)
from claude_memory.embeddings import select_embedder, set_embed_observer
from claude_memory.api.vault_service import (
    delete_secret,
    get_secret,
    is_vault_configured,
    store_secret,
)

logger = logging.getLogger(__name__)

# Context variable for MCP SSE multi-user support
_current_user: ContextVar[str] = ContextVar("_current_user", default="default")
# …and the calling key's scope alongside it, so the MCP write tools can stamp an external
# caller the same way the REST writes do (design change 4).
_current_scope: ContextVar[KeyScope] = ContextVar("_current_scope", default=DEFAULT_SCOPE)


def _token_of(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ").strip()


def _scope_of_token(token: str) -> KeyScope | None:
    """The scope of the key this token names, or None when it names no key.

    Read off the auth module rather than a dict bound at import: the test suite reloads
    ``claude_memory.api.auth`` to install a key set, which rebinds that dict.
    """
    return auth_module._key_to_scope.get(token)


async def enforce_key_scope(request: Request) -> None:
    """Refuse a restricted-scope key any operation outside the allowlist (design change 2).

    Registered as an app-wide dependency, not ASGI middleware: ``request.scope["route"]``
    is only populated once routing has matched, which happens after middleware and before
    dependencies (MCPAuthMiddleware below has to string-match the raw path for exactly
    this reason). Being app-wide is what makes it fail closed — a route added later is
    absent from ``EXTERNAL_ALLOWED_OPERATIONS``, so an external key gets 403 on it until
    someone deliberately lists it.

    One measured shape does not get that 403, and does not escalate either: a new
    SINGLE-segment literal under ``/api/memories/`` is shadowed by the allowlisted
    ``/api/memories/{memory_id}`` template, which is registered first and matches first,
    so the guard sees an allowed path. The request then fails int parsing with a 422 for
    admin and external keys alike and the new handler runs for nobody. A shape the path
    parameter cannot swallow (``POST /api/memories/{memory_id}/purge``, or any path
    outside the subtree) is refused normally.

    Which scopes are restricted is ``auth.UNRESTRICTED_SCOPES``, tested by membership
    rather than against the string "external", so a third scope is restricted from the
    moment it is added. Unrestricted keys, unknown tokens and unauthenticated requests
    pass straight through; each route's own ``get_current_user`` still decides those.
    This takes a ``Request`` rather than a ``Header`` parameter so it adds nothing to the
    OpenAPI document.
    """
    scope = _scope_of_token(_token_of(request))
    if scope is None or not is_restricted_scope(scope):
        return
    path = getattr(request.scope.get("route"), "path", None)
    if isinstance(path, str) and is_allowed_for_external(request.method, path):
        return
    raise HTTPException(status_code=403, detail="This API key's scope does not permit this endpoint")


def _record_embed(provider: str, seconds: float, fell_back_from: str | None) -> None:
    """Observer for the ONNX backend: which execution provider served, and how long.

    Registered at startup so ``claude_memory.embeddings`` never imports ``api.metrics``
    — that module has to keep importing with no optional extras installed (ADR-0002).
    A non-zero ``memory_embed_fallbacks_total`` rate is the signal that the GPU path is
    degraded and the CPU fallback is carrying recall.
    """
    metrics.EMBED_LATENCY.labels(provider=provider).observe(seconds)
    if fell_back_from is not None:
        metrics.EMBED_FALLBACKS.labels(from_provider=fell_back_from, to_provider=provider).inc()


async def _warm_embedder() -> None:
    """Load the dense-leg embedding model ONCE at startup (best-effort).

    ``select_embedder`` caches a process singleton, but the model still loads lazily on
    the first embed. Doing that here — before the pod serves traffic — keeps the first
    real recall from paying the ~2-4s cold model load and removes any first-hit load
    race between concurrent recalls. Guarded: recall already degrades to lexical on any
    dense-leg failure (api/recall.py), so a warmup failure must never block or crash
    startup. No-op when the dense leg is disabled.
    """
    if not embeddings_enabled():
        return
    try:
        embedder = select_embedder()
        await asyncio.to_thread(embedder.embed_query, "warmup")
        logger.info("dense-leg embedder warmed at startup (%s, model resident)", embedder.backend_label)
    except Exception as exc:  # noqa: BLE001 - best-effort; recall degrades to lexical
        logger.warning("embedder warmup failed; recall will degrade to lexical: %s", exc)


def _log_key_scopes() -> None:
    """Say at startup which scope each configured key landed with, and what was dropped.

    An API_KEYS entry written in the flat shape silently gets "admin", so a muse key
    pasted into the wrong shape is a working admin key whose writes carry no origin tag,
    and nothing else in the running service says so. This line does, by user id only —
    GET /api/users already lists those, and a key never goes to a log.
    """
    scopes = auth_module.scope_by_user()
    if scopes:
        logger.info("API key scopes: %s", ", ".join(f"{u}={s}" for u, s in sorted(scopes.items())))
    for problem in auth_module.API_KEYS_PROBLEMS:
        logger.error("%s", problem)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    _log_key_scopes()
    await init_pool()
    set_embed_observer(_record_embed)
    await _warm_embedder()
    async with streamable_session_mgr.run():
        yield
    await close_pool()


app = FastAPI(
    title="Claude Memory API",
    lifespan=lifespan,
    dependencies=[Depends(enforce_key_scope)],
)

UI_DIR = pathlib.Path(__file__).parent.parent / "ui" / "static"
_CACHE_BUST = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]


@app.get("/")
async def ui_root() -> Response:
    """Serve the UI single-page app with cache-busted static assets."""
    html = (UI_DIR / "index.html").read_text()
    html = html.replace('.js"', f'.js?v={_CACHE_BUST}"').replace(
        '.css"', f'.css?v={_CACHE_BUST}"'
    )
    return Response(content=html, media_type="text/html")


def _detect_sensitive(content: str) -> bool:
    """Check if content contains credentials using the credential detector."""
    try:
        from claude_memory.credential_detector import detect_credentials

        findings = detect_credentials(content)
        return len(findings) > 0
    except ImportError:
        return False


def _redact_content(content: str) -> str:
    """Redact sensitive content for storage in the main DB."""
    try:
        from claude_memory.credential_detector import detect_credentials, redact_credentials

        creds = detect_credentials(content)
        if creds:
            return redact_credentials(content, creds)
        return content
    except ImportError:
        return "[REDACTED]"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(RequestValidationError)
async def _validation_with_bound_metric(request: Request, exc: RequestValidationError) -> Response:
    """Default 422 behaviour + a counter when the ADR-0007 content bound rejects a write."""
    if any("1,400-char Memory bound" in str(e.get("msg", "")) for e in exc.errors()):
        metrics.BOUND_REJECTS.labels(surface="rest").inc()
    return await request_validation_exception_handler(request, exc)


@app.get("/metrics")
async def metrics_endpoint() -> Response:
    """Prometheus exposition (annotation-scraped in-cluster; aggregate counters only)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        pending = await conn.fetchval(
            "SELECT count(*) FROM memories"
            " WHERE embedding IS NULL AND NOT is_sensitive AND deleted_at IS NULL"
        )
        total = await conn.fetchval("SELECT count(*) FROM memories WHERE deleted_at IS NULL")
    metrics.EMBEDDINGS_PENDING.set(pending or 0)
    metrics.MEMORIES_TOTAL.set(total or 0)
    payload, content_type = metrics.exposition()
    return Response(content=payload, media_type=content_type)


@app.get(MUSE_SPEC_PATH, include_in_schema=False)
async def muse_openapi() -> JSONResponse:
    """The curated OpenAPI document Muse's connector is pointed at (design change 3).

    Unauthenticated on purpose, for two independent reasons. It is a strict subset of the
    already-unauthenticated /openapi.json, so gating the subset while the superset is open
    protects nothing. And a connector builder fetches the spec to learn the API's shape
    BEFORE the user has pasted a token, so a 401 here would break the first setup step.

    ``include_in_schema=False`` is load-bearing rather than cosmetic: without it this route
    appears as an operation inside /openapi.json and inside its own output.
    """
    return JSONResponse(build_muse_openapi(app.openapi()))


@app.get("/api/auth-check")
async def auth_check(user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    """Validate API key without doing any real work, and say what scope it carries.

    The scope is echoed because it is otherwise invisible to the holder: a key meant to
    be external but written in API_KEYS' flat shape is a silent admin key, and the only
    other signal was this endpoint answering 403 instead of 200.
    """
    return {"status": "ok", "user_id": user.user_id, "scope": user.scope}


@app.get("/api/users")
async def list_users(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return list of known user IDs (excluding current user) for sharing typeahead."""
    all_users = sorted(set(_key_to_user.values()))
    return {"users": [u for u in all_users if u != user.user_id]}


@app.get("/api/memories/sync", response_model=SyncResponse)
async def sync_memories(
    since: Optional[str] = None,
    user: AuthUser = Depends(get_current_user),
) -> SyncResponse:
    pool = await get_pool()
    server_time = datetime.now(timezone.utc).isoformat()

    async with pool.acquire() as conn:
        if since:
            since_dt = datetime.fromisoformat(since.replace(' ', '+'))
            rows = await conn.fetch(
                """
                SELECT id, content, category, tags, expanded_keywords, importance,
                       is_sensitive, created_at, updated_at, deleted_at
                FROM memories
                WHERE user_id = $1 AND updated_at > $2
                ORDER BY updated_at ASC
                """,
                user.user_id,
                since_dt,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, content, category, tags, expanded_keywords, importance,
                       is_sensitive, created_at, updated_at, deleted_at
                FROM memories
                WHERE user_id = $1 AND deleted_at IS NULL
                ORDER BY updated_at ASC
                """,
                user.user_id,
            )

    memories = []
    for row in rows:
        mem = {
            "id": row["id"],
            "content": row["content"],
            "category": row["category"],
            "tags": row["tags"],
            "expanded_keywords": row["expanded_keywords"],
            "importance": row["importance"],
            "is_sensitive": row["is_sensitive"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "deleted_at": row["deleted_at"].isoformat() if row["deleted_at"] else None,
        }
        memories.append(mem)

    return SyncResponse(memories=memories, server_time=server_time)


@app.post("/api/memories", response_model=MemoryResponse)
async def store_memory(body: MemoryStore, user: AuthUser = Depends(get_current_user)) -> MemoryResponse:
    pool = await get_pool()
    is_sensitive = body.force_sensitive or _detect_sensitive(body.content)
    # Design change 4: an external key's write carries source:<user_id>, server-side, with
    # no way for the client to opt out or to claim a different origin.
    tags = stamp_origin(body.tags, user.user_id) if is_restricted_scope(user.scope) else body.tags

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO memories (user_id, content, category, tags, expanded_keywords, importance, is_sensitive)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id, category, importance
            """,
            user.user_id,
            body.content if not is_sensitive else _redact_content(body.content),
            body.category,
            tags,
            body.expanded_keywords,
            body.importance,
            is_sensitive,
        )
        memory_id = row["id"]

        if is_sensitive and is_vault_configured():
            vault_path = await store_secret(user.user_id, memory_id, body.content)
            await conn.execute(
                "UPDATE memories SET vault_path = $1 WHERE id = $2",
                vault_path,
                memory_id,
            )

    # Embed-on-write OFF the hot path (shared helper): non-sensitive rows only, flag-gated.
    # Returns immediately; the response is never blocked on the embedding compute.
    schedule_embedding(pool, memory_id, body.content, is_sensitive=is_sensitive)

    metrics.STORES.labels(surface="rest", outcome="ok").inc()
    return MemoryResponse(id=row["id"], category=row["category"], importance=row["importance"])


@app.post("/api/memories/recall")
async def recall_memories(body: MemoryRecall, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    t0 = time.perf_counter()
    pool = await get_pool()

    query_text = f"{body.context} {body.expanded_query}".strip()

    # One shared retrieval helper for BOTH recall entry points (REST + FastMCP) so the
    # lexical→hybrid logic cannot drift. Flags off ⇒ verbatim current lexical SQL (no-op);
    # the embeddings flag adds the dense leg to a shared weighted-RRF pool with importance
    # as a post-fusion multiplier (api/recall.py).
    try:
        async with pool.acquire() as conn:
            all_rows = await _fused_recall(
                conn,
                user_id=user.user_id,
                query_text=query_text,
                sort_by=body.sort_by,
                category=body.category,
                limit=body.limit,
                pool=pool,
            )
            # ADR-0007 link semantics as a POST-ranking step, so every sort mode and
            # fusion path gets it: supersedes-redirect, resolved-by auto-attach, and
            # a links summary on every result (api/recall.py).
            linked = await apply_link_semantics(conn, user_id=user.user_id, rows=all_rows)
    except Exception:
        metrics.RECALL_ERRORS.labels(surface="rest").inc()
        raise

    results = []
    for item in linked:
        row = item["row"]
        content = row["content"]
        if row["is_sensitive"]:
            content = f"[SENSITIVE - use secret_get(id={row['id']})]"
        entry: dict[str, Any] = {
            "id": row["id"],
            "content": content,
            "category": row["category"],
            "tags": row["tags"],
            "importance": row["importance"],
            "is_sensitive": row["is_sensitive"],
            "rank": float(row["rank"]),
            "owner": row["owner"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "shared_by": row["shared_by"],
            "links": item["links"],
        }
        if item["redirected_from"] is not None:
            entry["redirected_from"] = item["redirected_from"]
        if item["attached_via"] is not None:
            entry["attached_via"] = item["attached_via"]
        results.append(entry)

    metrics.RECALL_REQUESTS.labels(surface="rest", sort=body.sort_by).inc()
    metrics.RECALL_LATENCY.labels(surface="rest").observe(time.perf_counter() - t0)
    return {"memories": results}


@app.get("/api/memories")
async def list_memories(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 10000,
    offset: int = 0,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    pool = await get_pool()

    # Build WHERE clauses dynamically — all memories are public
    where_clauses = ["deleted_at IS NULL"]
    count_params: list[Any] = []
    param_idx = 1

    if category:
        where_clauses.append(f"category = ${param_idx}")
        count_params.append(category)
        param_idx += 1

    if tag:
        where_clauses.append(
            f"${param_idx} = ANY(SELECT trim(t) FROM unnest(string_to_array(tags, ',')) AS t)"
        )
        count_params.append(tag)
        param_idx += 1

    where = " AND ".join(where_clauses)
    count_query = f"SELECT COUNT(*) FROM memories WHERE {where}"

    params: list[Any] = [*count_params, limit, offset]
    query = f"""
        SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at, user_id AS owner
        FROM memories WHERE {where}
        ORDER BY importance DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """

    async with pool.acquire() as conn:
        total = await conn.fetchval(count_query, *count_params)
        rows = await conn.fetch(query, *params)

    results = []
    for row in rows:
        content = row["content"]
        if row["is_sensitive"]:
            content = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append(
            {
                "id": row["id"],
                "content": content,
                "category": row["category"],
                "tags": row["tags"],
                "importance": row["importance"],
                "is_sensitive": row["is_sensitive"],
                "owner": row["owner"],
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
        )

    return {"memories": results, "total": total}


@app.get("/api/categories")
async def list_categories(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return the category values IN USE across all users.

    Not the list of legal values: a canonical category with no live rows is absent here.
    The vocabulary a write has to come from is the ``enum`` on every category field of
    the OpenAPI document (``models.CATEGORY_ENUM``).
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT category FROM memories WHERE deleted_at IS NULL ORDER BY category",
        )
    return {"categories": [r["category"] for r in rows]}


@app.get("/api/tags")
async def list_tags(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return all distinct tags with memory counts across all users."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT trim(t) as tag, COUNT(*) as count
            FROM memories, unnest(string_to_array(tags, ',')) AS t
            WHERE deleted_at IS NULL AND tags != '' AND tags IS NOT NULL
            GROUP BY trim(t)
            ORDER BY count DESC
            """,
        )
    return {"tags": [{"tag": r["tag"], "count": r["count"]} for r in rows]}


@app.get("/api/stats")
async def get_stats(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """Aggregated stats for the dashboard."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        total = await conn.fetchval(
            "SELECT COUNT(*) FROM memories WHERE user_id = $1 AND deleted_at IS NULL",
            user.user_id,
        )

        cat_rows = await conn.fetch(
            "SELECT category, COUNT(*) AS cnt FROM memories WHERE user_id = $1 AND deleted_at IS NULL GROUP BY category ORDER BY cnt DESC",
            user.user_id,
        )
        by_category = {r["category"]: r["cnt"] for r in cat_rows}

        imp_rows = await conn.fetch(
            """
            SELECT
                CASE
                    WHEN importance < 0.2 THEN '0.0-0.2'
                    WHEN importance < 0.4 THEN '0.2-0.4'
                    WHEN importance < 0.6 THEN '0.4-0.6'
                    WHEN importance < 0.8 THEN '0.6-0.8'
                    ELSE '0.8-1.0'
                END AS bucket,
                COUNT(*) AS cnt
            FROM memories WHERE user_id = $1 AND deleted_at IS NULL
            GROUP BY bucket ORDER BY bucket
            """,
            user.user_id,
        )
        by_importance = {r["bucket"]: r["cnt"] for r in imp_rows}

        activity_rows = await conn.fetch(
            """
            SELECT d::date AS date,
                   COUNT(*) FILTER (WHERE created_at::date = d::date) AS created,
                   COUNT(*) FILTER (WHERE updated_at::date = d::date AND updated_at > created_at + interval '1 second') AS updated
            FROM memories,
                 generate_series(CURRENT_DATE - interval '29 days', CURRENT_DATE, '1 day') AS d
            WHERE user_id = $1 AND deleted_at IS NULL
              AND (created_at::date = d::date OR (updated_at::date = d::date AND updated_at > created_at + interval '1 second'))
            GROUP BY d::date ORDER BY d::date
            """,
            user.user_id,
        )
        recent_activity = [
            {"date": r["date"].isoformat(), "created": r["created"], "updated": r["updated"]}
            for r in activity_rows
        ]

        shared_by_me = await conn.fetchval(
            "SELECT COUNT(*) FROM memory_shares WHERE owner_id = $1",
            user.user_id,
        )
        shared_with_me = await conn.fetchval(
            """SELECT COUNT(DISTINCT m.id) FROM memories m
               WHERE m.deleted_at IS NULL AND (
                 EXISTS (SELECT 1 FROM memory_shares ms WHERE ms.memory_id = m.id AND ms.shared_with = $1)
                 OR EXISTS (SELECT 1 FROM tag_shares ts WHERE ts.owner_id = m.user_id AND ts.shared_with = $1
                   AND EXISTS (SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t WHERE trim(t) = ts.tag))
               )""",
            user.user_id,
        )

    return {
        "total_memories": total,
        "by_category": by_category,
        "by_importance": by_importance,
        "recent_activity": recent_activity,
        "sharing_stats": {"shared_by_me": shared_by_me, "shared_with_me": shared_with_me},
    }


# NOTE: Literal-path routes (share-tag, shared-with-me, my-shares) must be
# registered before parameterized routes ({memory_id}) to prevent FastAPI
# from matching the literal segment as a path parameter.


@app.post("/api/memories/share-tag")
async def share_tag(body: ShareTag, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    if body.shared_with == user.user_id:
        raise HTTPException(status_code=400, detail="Cannot share with yourself")

    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO tag_shares (owner_id, tag, shared_with, permission)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (owner_id, tag, shared_with)
            DO UPDATE SET permission = EXCLUDED.permission
            """,
            user.user_id, body.tag, body.shared_with, body.permission,
        )
    return {"shared_tag": body.tag, "with": body.shared_with, "permission": body.permission}


@app.delete("/api/memories/share-tag")
async def unshare_tag(body: UnshareTag, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM tag_shares WHERE owner_id = $1 AND tag = $2 AND shared_with = $3",
            user.user_id, body.tag, body.shared_with,
        )
    return {"unshared_tag": body.tag, "from": body.shared_with}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()

    async with pool.acquire() as conn:
        # Only the owner can delete — even write-shared users cannot
        row = await conn.fetchrow(
            "SELECT id, vault_path, substr(content, 1, 50) AS preview FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL",
            memory_id,
            user.user_id,
        )
        if not row:
            # Check if memory exists but is owned by someone else
            exists = await conn.fetchrow(
                "SELECT id FROM memories WHERE id = $1 AND deleted_at IS NULL", memory_id
            )
            if exists:
                raise HTTPException(status_code=403, detail="Only the owner can delete a memory")
            # Idempotent: return success even if already deleted
            return {"deleted": memory_id, "preview": "[already deleted]"}

        if row["vault_path"]:
            await delete_secret(user.user_id, row["vault_path"])

        # Also clean up any shares for this memory
        await conn.execute("DELETE FROM memory_shares WHERE memory_id = $1", memory_id)
        await conn.execute(
            "UPDATE memories SET deleted_at = NOW(), updated_at = NOW() WHERE id = $1 AND user_id = $2",
            memory_id,
            user.user_id,
        )

    return {"deleted": memory_id, "preview": row["preview"]}


@app.post("/api/memories/{memory_id}/secret", response_model=SecretResponse)
async def get_memory_secret(memory_id: int, user: AuthUser = Depends(get_current_user)) -> SecretResponse:
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, content, is_sensitive, vault_path, encrypted_content
            FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL
            """,
            memory_id,
            user.user_id,
        )

    if not row:
        raise HTTPException(status_code=404, detail="Memory not found")

    if not row["is_sensitive"]:
        return SecretResponse(id=row["id"], content=row["content"], source="plaintext")

    if row["vault_path"]:
        secret = await get_secret(user.user_id, row["vault_path"])
        if secret:
            return SecretResponse(id=row["id"], content=secret, source="vault")

    if row["encrypted_content"]:
        return SecretResponse(
            id=row["id"],
            content="[ENCRYPTED - decryption not available]",
            source="encrypted",
        )

    return SecretResponse(id=row["id"], content=row["content"], source="plaintext")


@app.post("/api/memories/migrate-secrets")
async def migrate_secrets(user: AuthUser = Depends(get_current_user)) -> dict[str, int]:
    pool = await get_pool()
    migrated = 0

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, content FROM memories
            WHERE user_id = $1 AND is_sensitive = FALSE AND deleted_at IS NULL
            """,
            user.user_id,
        )

        for row in rows:
            if _detect_sensitive(row["content"]):
                original_content = row["content"]
                redacted = _redact_content(original_content)

                vault_path = None
                if is_vault_configured():
                    vault_path = await store_secret(user.user_id, row["id"], original_content)

                await conn.execute(
                    """
                    UPDATE memories
                    SET is_sensitive = TRUE, content = $1, vault_path = $2,
                        updated_at = NOW()
                    WHERE id = $3 AND user_id = $4
                    """,
                    redacted,
                    vault_path,
                    row["id"],
                    user.user_id,
                )
                migrated += 1

    return {"migrated": migrated}


@app.post("/api/memories/import")
async def import_memories(
    memories: list[MemoryStore], user: AuthUser = Depends(get_current_user)
) -> list[MemoryResponse]:
    pool = await get_pool()
    imported = []

    async with pool.acquire() as conn:
        for mem in memories:
            is_sensitive = mem.force_sensitive or _detect_sensitive(mem.content)
            content = mem.content if not is_sensitive else _redact_content(mem.content)

            row = await conn.fetchrow(
                """
                INSERT INTO memories (user_id, content, category, tags, expanded_keywords, importance, is_sensitive)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                RETURNING id, category, importance
                """,
                user.user_id,
                content,
                mem.category,
                mem.tags,
                mem.expanded_keywords,
                mem.importance,
                is_sensitive,
            )

            if is_sensitive and is_vault_configured():
                vault_path = await store_secret(user.user_id, row["id"], mem.content)
                await conn.execute(
                    "UPDATE memories SET vault_path = $1 WHERE id = $2",
                    vault_path,
                    row["id"],
                )

            imported.append(
                MemoryResponse(id=row["id"], category=row["category"], importance=row["importance"])
            )

    return imported


# --- Sharing Endpoints ---


@app.post("/api/memories/{memory_id}/share")
async def share_memory(memory_id: int, body: ShareMemory, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL",
            memory_id, user.user_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found or not owned by you")
        if body.shared_with == user.user_id:
            raise HTTPException(status_code=400, detail="Cannot share with yourself")

        await conn.execute(
            """
            INSERT INTO memory_shares (memory_id, owner_id, shared_with, permission)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (memory_id, shared_with)
            DO UPDATE SET permission = EXCLUDED.permission
            """,
            memory_id, user.user_id, body.shared_with, body.permission,
        )
    return {"shared": memory_id, "with": body.shared_with, "permission": body.permission}


@app.delete("/api/memories/{memory_id}/share/{target_user}")
async def unshare_memory(memory_id: int, target_user: str, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL",
            memory_id, user.user_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found or not owned by you")

        await conn.execute(
            "DELETE FROM memory_shares WHERE memory_id = $1 AND shared_with = $2",
            memory_id, target_user,
        )
    return {"unshared": memory_id, "from": target_user}


@app.get("/api/memories/shared-with-me")
async def shared_with_me(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Individual shares
        individual = await conn.fetch(
            """
            SELECT m.id, m.content, m.category, m.tags, m.importance, m.is_sensitive,
                   m.created_at, m.updated_at, m.user_id AS shared_by, ms.permission
            FROM memories m
            JOIN memory_shares ms ON ms.memory_id = m.id
            WHERE ms.shared_with = $1 AND m.deleted_at IS NULL
            ORDER BY m.importance DESC
            """,
            user.user_id,
        )

        # Tag shares
        tag_shared = await conn.fetch(
            """
            SELECT DISTINCT ON (m.id) m.id, m.content, m.category, m.tags, m.importance, m.is_sensitive,
                   m.created_at, m.updated_at, m.user_id AS shared_by, ts.permission
            FROM memories m
            JOIN tag_shares ts ON ts.owner_id = m.user_id
            WHERE ts.shared_with = $1 AND m.deleted_at IS NULL
              AND EXISTS (
                SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t
                WHERE trim(t) = ts.tag
              )
            ORDER BY m.id, m.importance DESC
            """,
            user.user_id,
        )

    seen_ids = set()
    results = []
    for row in list(individual) + list(tag_shared):
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        content = row["content"]
        if row["is_sensitive"]:
            content = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append({
            "id": row["id"], "content": content, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "shared_by": row["shared_by"], "permission": row["permission"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        })

    return {"memories": results}


@app.get("/api/memories/my-shares")
async def my_shares(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        memory_shares = await conn.fetch(
            """
            SELECT ms.memory_id, ms.shared_with, ms.permission, ms.created_at,
                   substr(m.content, 1, 80) AS preview
            FROM memory_shares ms
            JOIN memories m ON m.id = ms.memory_id
            WHERE ms.owner_id = $1 AND m.deleted_at IS NULL
            ORDER BY ms.created_at DESC
            """,
            user.user_id,
        )
        tag_shares = await conn.fetch(
            "SELECT tag, shared_with, permission, created_at FROM tag_shares WHERE owner_id = $1 ORDER BY created_at DESC",
            user.user_id,
        )

    return {
        "memory_shares": [
            {
                "memory_id": r["memory_id"], "shared_with": r["shared_with"],
                "permission": r["permission"], "preview": r["preview"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in memory_shares
        ],
        "tag_shares": [
            {
                "tag": r["tag"], "shared_with": r["shared_with"],
                "permission": r["permission"],
                "created_at": r["created_at"].isoformat(),
            }
            for r in tag_shares
        ],
    }


@app.put("/api/memories/{memory_id}")
async def update_memory(memory_id: int, body: MemoryUpdate, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        allowed, owner_id = await check_memory_permission(conn, memory_id, user.user_id, "write")
        if not allowed:
            if owner_id is None:
                raise HTTPException(status_code=404, detail="Memory not found")
            raise HTTPException(status_code=403, detail="Write permission required")

        # Design change 4: an external key's write carries source:<user_id> whether or not
        # it sent tags, so the value is resolved here and rides the same UPDATE. Gated on
        # the body asking for at least one change, so an empty PUT still gets its 400
        # rather than being turned into a successful tags-only write. Tags the client did
        # NOT send are stamped without being filtered: a write-share can put an external
        # key on a row it does not own, and rewriting that row's tags would delete
        # provenance the owner wrote.
        tags = body.tags
        if is_restricted_scope(user.scope) and body.model_dump(exclude_none=True):
            if tags is None:
                stored = await conn.fetchval("SELECT tags FROM memories WHERE id = $1", memory_id)
                tags = stamp_stored_origin(stored, user.user_id)
            else:
                tags = stamp_origin(tags, user.user_id)

        updates = []
        params: list[Any] = []
        idx = 1

        if body.content is not None:
            updates.append(f"content = ${idx}")
            params.append(body.content)
            idx += 1
        if body.category is not None:
            updates.append(f"category = ${idx}")
            params.append(body.category)
            idx += 1
        if tags is not None:
            updates.append(f"tags = ${idx}")
            params.append(tags)
            idx += 1
        if body.importance is not None:
            updates.append(f"importance = ${idx}")
            params.append(body.importance)
            idx += 1
        if body.expanded_keywords is not None:
            updates.append(f"expanded_keywords = ${idx}")
            params.append(body.expanded_keywords)
            idx += 1

        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")

        updates.append("updated_at = NOW()")
        params.append(memory_id)

        await conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ${idx}",
            *params,
        )

    return {"updated": memory_id}


# --- Link endpoints (ADR-0007) ---
#
# Typed, directed Memory→Memory edges, scoped to the calling user (each user
# writes their own edges). link_type is the closed enum of four; recall gives
# each type its behaviour (supersedes redirects, resolved-by auto-attaches,
# part-of / see-also are pointer-only) — see api/recall.apply_link_semantics.


@app.post("/api/memories/{memory_id}/links")
async def create_memory_link(
    memory_id: int, body: LinkCreate, user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    """Create a typed link from this memory (src) to body.target_id (dst)."""
    if body.target_id == memory_id:
        raise HTTPException(status_code=422, detail="A memory cannot link to itself")

    pool = await get_pool()
    async with pool.acquire() as conn:
        # Both ends must exist and be readable by the caller (ownership or shared-read).
        for mid in (memory_id, body.target_id):
            allowed, owner_id = await check_memory_permission(conn, mid, user.user_id, "read")
            if not allowed:
                if owner_id is None:
                    raise HTTPException(status_code=404, detail=f"Memory {mid} not found")
                raise HTTPException(status_code=403, detail=f"Read permission required on memory {mid}")

        if body.link_type == "supersedes":
            # Reject a cycle: walking the target's OUTGOING supersedes chain must never
            # reach the new source. Batched one query per hop, visited set, bounded depth
            # (same cap as the recall redirect walk).
            visited: set[int] = {body.target_id}
            frontier: list[int] = [body.target_id]
            for _ in range(SUPERSEDES_DEPTH_CAP):
                if not frontier:
                    break
                hop_rows = await conn.fetch(
                    """
                    SELECT dst_id FROM memory_links
                    WHERE user_id = $1 AND link_type = 'supersedes' AND src_id = ANY($2::int[])
                    """,
                    user.user_id,
                    frontier,
                )
                next_frontier: list[int] = []
                for r in hop_rows:
                    dst = r["dst_id"]
                    if dst == memory_id:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                f"supersedes cycle: memory {body.target_id} already "
                                f"(transitively) supersedes memory {memory_id}"
                            ),
                        )
                    if dst not in visited:
                        visited.add(dst)
                        next_frontier.append(dst)
                frontier = next_frontier

        try:
            row = await conn.fetchrow(
                """
                INSERT INTO memory_links (user_id, src_id, dst_id, link_type)
                VALUES ($1, $2, $3, $4)
                RETURNING id, created_at
                """,
                user.user_id,
                memory_id,
                body.target_id,
                body.link_type,
            )
        except asyncpg.UniqueViolationError:
            raise HTTPException(status_code=409, detail="Link already exists")

    metrics.LINKS_CREATED.labels(link_type=body.link_type).inc()
    return {
        "id": row["id"],
        "src_id": memory_id,
        "dst_id": body.target_id,
        "link_type": body.link_type,
        "created_at": row["created_at"].isoformat(),
    }


@app.delete("/api/memories/{memory_id}/links/{dst_id}/{link_type}")
async def delete_memory_link(
    memory_id: int, dst_id: int, link_type: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    """Delete one of the caller's typed links."""
    if link_type not in LINK_TYPES:
        raise HTTPException(
            status_code=422, detail=f"link_type must be one of: {', '.join(LINK_TYPES)}"
        )
    pool = await get_pool()
    async with pool.acquire() as conn:
        status = await conn.execute(
            "DELETE FROM memory_links WHERE user_id = $1 AND src_id = $2 AND dst_id = $3 AND link_type = $4",
            user.user_id,
            memory_id,
            dst_id,
            link_type,
        )
    if status == "DELETE 0":
        raise HTTPException(status_code=404, detail="Link not found")
    return {"unlinked": {"src_id": memory_id, "dst_id": dst_id, "link_type": link_type}}


# Registered AFTER the literal /api/memories/* GET routes (sync, shared-with-me,
# my-shares) so the path parameter cannot shadow them. A COMMENT rather than part of the
# docstring below: FastAPI renders a docstring as the operation description, and
# /muse/openapi.json is built from that document, so a routing note naming three
# endpoints the curated document exists to withhold would ship inside it.
@app.get("/api/memories/{memory_id}")
async def get_memory(memory_id: int, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """One full memory with the caller's links, both directions (ADR-0007 ``get <id>``)."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT id, content, category, tags, expanded_keywords, importance, is_sensitive,
                   created_at, updated_at, user_id AS owner,
                   CASE WHEN user_id = $2 THEN NULL ELSE user_id END AS shared_by
            FROM memories
            WHERE id = $1 AND deleted_at IS NULL
            """,
            memory_id,
            user.user_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Memory not found")
        edges = await conn.fetch(
            """
            SELECT src_id, dst_id, link_type FROM memory_links
            WHERE user_id = $1 AND (src_id = $2 OR dst_id = $2)
            ORDER BY created_at, id
            """,
            user.user_id,
            memory_id,
        )

    content = row["content"]
    if row["is_sensitive"]:
        content = f"[SENSITIVE - use secret_get(id={row['id']})]"

    return {
        "id": row["id"],
        "content": content,
        "category": row["category"],
        "tags": row["tags"],
        "expanded_keywords": row["expanded_keywords"],
        "importance": row["importance"],
        "is_sensitive": row["is_sensitive"],
        "owner": row["owner"],
        "shared_by": row["shared_by"],
        "created_at": row["created_at"].isoformat(),
        "updated_at": row["updated_at"].isoformat(),
        "links_out": [
            {"id": e["dst_id"], "type": e["link_type"]} for e in edges if e["src_id"] == memory_id
        ],
        "links_in": [
            {"id": e["src_id"], "type": e["link_type"]} for e in edges if e["dst_id"] == memory_id
        ],
    }


# --- MCP SSE Transport ---


def _resolve_user_from_token(token: str) -> str | None:
    """Resolve API key to user_id, reusing auth module's key map."""
    return _key_to_user.get(token)


mcp_server = FastMCP("claude-memory")


@mcp_server.tool()
async def memory_store(content: str, category: str = "facts", tags: str = "",
                       expanded_keywords: str = "", importance: float = 0.5) -> str:
    """Store a new memory. Content over 1,400 chars is rejected (ADR-0007): split it into
    a self-contained hub Memory plus part-of linked detail Memories."""
    # Same category canonicalization + content bound as the REST store path (ADR-0007);
    # this entry point's error convention is a JSON error, not an HTTP 422. The old
    # 500-char _split_content auto-chopper is retired — mechanical chopping is what
    # produced the part-N-of-M fragment pollution; splitting is a writing act.
    try:
        category = canonicalize_category(category)
        validate_content_bound(content)
    except ValueError as exc:
        if "Memory bound" in str(exc):
            metrics.BOUND_REJECTS.labels(surface="mcp").inc()
        else:
            metrics.STORES.labels(surface="mcp", outcome="category_rejected").inc()
        return json.dumps({"error": str(exc)})
    pool = await get_pool()
    user_id = _current_user.get()
    # Design change 4, same stamp as the REST store path. MCPAuthMiddleware refuses an
    # external key before it can reach a tool, so this is the second of two controls
    # rather than the only one — it keeps the invariant at the write itself.
    if is_restricted_scope(_current_scope.get()):
        tags = stamp_origin(tags, user_id)

    is_sensitive = _detect_sensitive(content)
    stored = content if not is_sensitive else _redact_content(content)
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO memories (user_id, content, category, tags, expanded_keywords, importance, is_sensitive)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               RETURNING id""",
            user_id, stored, category, tags, expanded_keywords, importance, is_sensitive,
        )
        memory_id = row["id"]

        if is_sensitive and is_vault_configured():
            vault_path = await store_secret(user_id, memory_id, content)
            await conn.execute("UPDATE memories SET vault_path = $1 WHERE id = $2", vault_path, memory_id)

        # Embed-on-write OFF the hot path (same shared helper as the REST store path):
        # non-sensitive content only, flag-gated; does not block the store response.
        schedule_embedding(pool, memory_id, content, is_sensitive=is_sensitive)

    metrics.STORES.labels(surface="mcp", outcome="ok").inc()
    return json.dumps({"id": memory_id, "category": category, "importance": importance})


@mcp_server.tool()
async def memory_recall(context: str, expanded_query: str = "",
                        category: str | None = None, sort_by: str = "relevance",
                        limit: int = 30) -> str:
    """Recall memories by semantic search."""
    t0 = time.perf_counter()
    pool = await get_pool()
    user_id = _current_user.get()
    query_text = f"{context} {expanded_query}".strip()
    if not query_text:
        return json.dumps({"error": "context is required"})
    # Same category canonicalization as the REST recall path, which app.py requires of
    # these two surfaces ("the two recall surfaces must not differ in what truth they
    # serve", below). The filter is an exact `AND category = $4` in SQL, so an unfolded
    # 'Gotcha' matched nothing and read as "no memories". This entry point's error
    # convention is a JSON error, the same as memory_store's.
    if category is not None and category.strip():
        try:
            category = canonicalize_category(category)
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
    else:
        category = None

    # SAME shared retrieval helper as the REST recall_memories endpoint — the two paths
    # cannot drift. Flags off ⇒ verbatim current lexical SQL (no-op); embeddings flag adds
    # the dense leg to a shared weighted-RRF pool (api/recall.py). Link semantics apply
    # here exactly as on REST (ADR-0007): supersedes-redirect, resolved-by attach,
    # links summary — the two recall surfaces must not differ in what truth they serve.
    try:
        async with pool.acquire() as conn:
            all_rows = await _fused_recall(
                conn,
                user_id=user_id,
                query_text=query_text,
                sort_by=sort_by,
                category=category,
                limit=limit,
                pool=pool,
            )
            linked = await apply_link_semantics(conn, user_id=user_id, rows=all_rows)
    except Exception:
        metrics.RECALL_ERRORS.labels(surface="mcp").inc()
        raise

    results = []
    for item in linked:
        row = item["row"]
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        entry: dict[str, Any] = {
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "rank": float(row["rank"]),
            "owner": row["owner"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
            "links": item["links"],
        }
        if item["redirected_from"] is not None:
            entry["redirected_from"] = item["redirected_from"]
        if item["attached_via"] is not None:
            entry["attached_via"] = item["attached_via"]
        if row["shared_by"]:
            entry["shared_by"] = row["shared_by"]
        results.append(entry)

    metrics.RECALL_REQUESTS.labels(surface="mcp", sort=sort_by).inc()
    metrics.RECALL_LATENCY.labels(surface="mcp").observe(time.perf_counter() - t0)
    return json.dumps({"memories": results})


@mcp_server.tool()
async def memory_list(category: str | None = None, limit: int = 10000) -> str:
    """List stored memories."""
    pool = await get_pool()

    if category:
        query = """SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at, user_id AS owner
                   FROM memories WHERE deleted_at IS NULL AND category = $1
                   ORDER BY importance DESC LIMIT $2"""
        params: list[Any] = [category, limit]
    else:
        query = """SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at, user_id AS owner
                   FROM memories WHERE deleted_at IS NULL
                   ORDER BY importance DESC LIMIT $1"""
        params = [limit]

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)

    results = []
    for row in rows:
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append({
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "owner": row["owner"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        })

    return json.dumps({"memories": results})


@mcp_server.tool()
async def memory_delete(memory_id: int) -> str:
    """Delete a memory by ID."""
    pool = await get_pool()
    user_id = _current_user.get()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, vault_path, substr(content, 1, 50) AS preview FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL",
            memory_id, user_id,
        )
        if not row:
            return json.dumps({"deleted": memory_id, "preview": "[already deleted]"})

        if row["vault_path"]:
            await delete_secret(user_id, row["vault_path"])

        await conn.execute(
            "UPDATE memories SET deleted_at = NOW(), updated_at = NOW() WHERE id = $1 AND user_id = $2",
            memory_id, user_id,
        )

    return json.dumps({"deleted": memory_id, "preview": row["preview"]})


@mcp_server.tool()
async def memory_count() -> str:
    """Count total memories."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL")
    return json.dumps({"count": count})


@mcp_server.tool()
async def secret_get(key: str) -> str:
    """Retrieve a secret value by key. Returns empty if not found."""
    return json.dumps({"error": "secret_get is not available via SSE transport"})


@mcp_server.tool()
async def memory_share(id: int, shared_with: str, permission: str = "read") -> str:
    """Share a memory with another user. Permission: 'read' or 'write'."""
    if permission not in ("read", "write"):
        return json.dumps({"error": "permission must be 'read' or 'write'"})
    pool = await get_pool()
    user_id = _current_user.get()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL",
            id, user_id,
        )
        if not row:
            return json.dumps({"error": "Memory not found or not owned by you"})
        await conn.execute(
            """INSERT INTO memory_shares (memory_id, owner_id, shared_with, permission)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (memory_id, shared_with) DO UPDATE SET permission = EXCLUDED.permission""",
            id, user_id, shared_with, permission,
        )
    return json.dumps({"shared": id, "with": shared_with, "permission": permission})


@mcp_server.tool()
async def memory_unshare(id: int, shared_with: str) -> str:
    """Revoke sharing of a memory from a user."""
    pool = await get_pool()
    user_id = _current_user.get()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM memory_shares WHERE memory_id = $1 AND owner_id = $2 AND shared_with = $3",
            id, user_id, shared_with,
        )
    return json.dumps({"unshared": id, "from": shared_with})


@mcp_server.tool()
async def memory_share_tag(tag: str, shared_with: str, permission: str = "read") -> str:
    """Share all memories with a given tag with another user. Future memories with this tag are automatically shared."""
    if permission not in ("read", "write"):
        return json.dumps({"error": "permission must be 'read' or 'write'"})
    pool = await get_pool()
    user_id = _current_user.get()
    async with pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO tag_shares (owner_id, tag, shared_with, permission)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT (owner_id, tag, shared_with) DO UPDATE SET permission = EXCLUDED.permission""",
            user_id, tag, shared_with, permission,
        )
    return json.dumps({"shared_tag": tag, "with": shared_with, "permission": permission})


@mcp_server.tool()
async def memory_unshare_tag(tag: str, shared_with: str) -> str:
    """Revoke tag-based sharing."""
    pool = await get_pool()
    user_id = _current_user.get()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM tag_shares WHERE owner_id = $1 AND tag = $2 AND shared_with = $3",
            user_id, tag, shared_with,
        )
    return json.dumps({"unshared_tag": tag, "from": shared_with})


@mcp_server.tool()
async def memory_update(id: int, content: str | None = None, tags: str | None = None,
                         importance: float | None = None, expanded_keywords: str | None = None) -> str:
    """Update an existing memory's content, tags, importance, or keywords."""
    pool = await get_pool()
    user_id = _current_user.get()
    async with pool.acquire() as conn:
        allowed, owner_id = await check_memory_permission(conn, id, user_id, "write")
        if not allowed:
            if owner_id is None:
                return json.dumps({"error": "Memory not found"})
            return json.dumps({"error": "Write permission required"})

        # Design change 4, same stamp and same gate as the REST update path.
        if is_restricted_scope(_current_scope.get()) and any(
            field is not None for field in (content, tags, importance, expanded_keywords)
        ):
            if tags is None:
                stored = await conn.fetchval("SELECT tags FROM memories WHERE id = $1", id)
                tags = stamp_stored_origin(stored, user_id)
            else:
                tags = stamp_origin(tags, user_id)

        updates = []
        params: list[Any] = []
        idx = 1
        if content is not None:
            updates.append(f"content = ${idx}")
            params.append(content)
            idx += 1
        if tags is not None:
            updates.append(f"tags = ${idx}")
            params.append(tags)
            idx += 1
        if importance is not None:
            updates.append(f"importance = ${idx}")
            params.append(importance)
            idx += 1
        if expanded_keywords is not None:
            updates.append(f"expanded_keywords = ${idx}")
            params.append(expanded_keywords)
            idx += 1

        if not updates:
            return json.dumps({"error": "No fields to update"})

        updates.append("updated_at = NOW()")
        params.append(id)
        await conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ${idx}",
            *params,
        )

    return json.dumps({"updated": id})


# Auth middleware for /mcp/* routes — pure ASGI to avoid BaseHTTPMiddleware
# buffering which breaks SSE streaming (responses never reach the client).
class MCPAuthMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["path"].startswith("/mcp"):
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            token = auth.removeprefix("Bearer ").strip()
            if not _resolve_user_from_token(token):
                response = Response(content="Unauthorized", status_code=401)
                await response(scope, receive, send)
                return
            # The MCP tools duplicate the REST writes and add memory_share and the tag
            # shares, none of which EXTERNAL_ALLOWED_OPERATIONS grants. An app-wide
            # Depends() cannot reach here — /mcp is a Mount, not an APIRoute — so the
            # allowlist is enforced for this transport by closing it outright, which also
            # keeps a tool added later out of an external key's reach.
            token_scope = _scope_of_token(token)
            if token_scope is not None and is_restricted_scope(token_scope):
                response = Response(content="Forbidden", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


app.add_middleware(MCPAuthMiddleware)

# Streamable HTTP transport — the only MCP transport (SSE is deprecated).
streamable_session_mgr = StreamableHTTPSessionManager(
    app=mcp_server._mcp_server,
    json_response=True,
    stateless=True,
)


class HandleStreamableHTTP:
    """ASGI wrapper that sets _current_user before delegating to the session manager."""

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        user_id = "default"
        key_scope: KeyScope = DEFAULT_SCOPE
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                token = value.decode().removeprefix("Bearer ").strip()
                resolved = _resolve_user_from_token(token)
                if resolved:
                    user_id = resolved
                    key_scope = _scope_of_token(token) or DEFAULT_SCOPE
                break
        _current_user.set(user_id)
        _current_scope.set(key_scope)
        await streamable_session_mgr.handle_request(scope, receive, send)


streamable_handler = HandleStreamableHTTP()

# Static files for UI (before MCP mount)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

# MCP streamable-http transport at /mcp/mcp
app.router.routes.insert(0, Mount("/mcp", routes=[
    Route("/mcp", endpoint=streamable_handler, methods=["GET", "POST", "DELETE"]),
]))
