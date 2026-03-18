"""Claude Memory API -- shared persistent memory with PostgreSQL full-text search."""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import Response
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount, Route

from claude_memory.api.auth import AuthUser, get_current_user, _key_to_user
from claude_memory.api.database import close_pool, get_pool, init_pool
from claude_memory.api.models import MemoryRecall, MemoryResponse, MemoryStore, SecretResponse, SyncResponse
from claude_memory.api.vault_service import (
    delete_secret,
    get_secret,
    is_vault_configured,
    store_secret,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="Claude Memory API", lifespan=lifespan)


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


@app.get("/api/auth-check")
async def auth_check(user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    """Validate API key without doing any real work."""
    return {"status": "ok", "user_id": user.user_id}


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
            body.tags,
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

    return MemoryResponse(id=row["id"], category=row["category"], importance=row["importance"])


@app.post("/api/memories/recall")
async def recall_memories(body: MemoryRecall, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()

    query_text = f"{body.context} {body.expanded_query}".strip()

    # Hybrid scoring: blend ts_rank relevance (0-1) with importance (0-1)
    hybrid_score = "(ts_rank(search_vector, query) * 0.7 + importance * 0.3)"
    if body.sort_by == "importance":
        hybrid_score = "(ts_rank(search_vector, query) * 0.4 + importance * 0.6)"

    order_clause = f"{hybrid_score} DESC"
    if body.sort_by == "recency":
        order_clause = "created_at DESC"

    category_filter = ""
    params: list[Any] = [user.user_id, query_text, body.limit]
    if body.category:
        category_filter = "AND category = $4"
        params.append(body.category)

    async with pool.acquire() as conn:
        # Try AND-match first (plainto_tsquery ANDs by default), fall back to
        # OR-match via individual word disjunction for broader results
        rows = await conn.fetch(
            f"""
            SELECT id, content, category, tags, importance, is_sensitive,
                   ts_rank(search_vector, query) AS rank,
                   created_at, updated_at
            FROM memories, plainto_tsquery('english', $2) query
            WHERE user_id = $1
              AND deleted_at IS NULL
              AND (search_vector @@ query OR $2 = '')
              {category_filter}
            ORDER BY {order_clause}
            LIMIT $3
            """,
            *params,
        )

        # If AND-match returned too few results, broaden to OR-match
        if len(rows) < body.limit and query_text:
            words = query_text.split()
            if len(words) > 1:
                or_tsquery = " | ".join(w for w in words if w)
                or_params: list[Any] = [user.user_id, or_tsquery, body.limit]
                or_cat_filter = ""
                if body.category:
                    or_cat_filter = "AND category = $4"
                    or_params.append(body.category)
                seen_ids = {r["id"] for r in rows}
                or_rows = await conn.fetch(
                    f"""
                    SELECT id, content, category, tags, importance, is_sensitive,
                           ts_rank(search_vector, query) AS rank,
                           created_at, updated_at
                    FROM memories, to_tsquery('english', $2) query
                    WHERE user_id = $1
                      AND deleted_at IS NULL
                      AND search_vector @@ query
                      {or_cat_filter}
                    ORDER BY {order_clause}
                    LIMIT $3
                    """,
                    *or_params,
                )
                rows = list(rows) + [r for r in or_rows if r["id"] not in seen_ids]
                rows = rows[:body.limit]

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
                "rank": float(row["rank"]),
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
        )

    return {"memories": results}


@app.get("/api/memories")
async def list_memories(
    category: Optional[str] = None,
    limit: int = 50,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    pool = await get_pool()

    if category:
        query = """
            SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at
            FROM memories WHERE user_id = $1 AND deleted_at IS NULL AND category = $2
            ORDER BY importance DESC LIMIT $3
        """
        params: list[Any] = [user.user_id, category, limit]
    else:
        query = """
            SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at
            FROM memories WHERE user_id = $1 AND deleted_at IS NULL
            ORDER BY importance DESC LIMIT $2
        """
        params = [user.user_id, limit]

    async with pool.acquire() as conn:
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
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
        )

    return {"memories": results}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: int, user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    pool = await get_pool()

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, vault_path, substr(content, 1, 50) AS preview FROM memories WHERE id = $1 AND user_id = $2 AND deleted_at IS NULL",
            memory_id,
            user.user_id,
        )
        if not row:
            # Idempotent: return success even if already deleted
            # Prevents old clients without 404-handling from infinite retry loops
            return {"deleted": memory_id, "preview": "[already deleted]"}

        if row["vault_path"]:
            await delete_secret(user.user_id, row["vault_path"])

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


# --- MCP SSE Transport ---


def _resolve_user_from_token(token: str) -> str | None:
    """Resolve API key to user_id, reusing auth module's key map."""
    return _key_to_user.get(token)


mcp_server = FastMCP("claude-memory")


@mcp_server.tool()
async def memory_store(content: str, category: str = "facts", tags: str = "",
                       expanded_keywords: str = "", importance: float = 0.5) -> str:
    """Store a new memory."""
    pool = await get_pool()
    # MCP SSE uses "default" user (single-user mode via middleware auth)
    user_id = "default"
    is_sensitive = _detect_sensitive(content)
    stored_content = content if not is_sensitive else _redact_content(content)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """INSERT INTO memories (user_id, content, category, tags, expanded_keywords, importance, is_sensitive)
               VALUES ($1, $2, $3, $4, $5, $6, $7)
               RETURNING id""",
            user_id, stored_content, category, tags, expanded_keywords, importance, is_sensitive,
        )
        memory_id = row["id"]

        if is_sensitive and is_vault_configured():
            vault_path = await store_secret(user_id, memory_id, content)
            await conn.execute("UPDATE memories SET vault_path = $1 WHERE id = $2", vault_path, memory_id)

    return json.dumps({"id": memory_id, "category": category, "importance": importance})


@mcp_server.tool()
async def memory_recall(context: str, expanded_query: str = "",
                        category: str | None = None, sort_by: str = "importance",
                        limit: int = 10) -> str:
    """Recall memories by semantic search."""
    pool = await get_pool()
    user_id = "default"
    query_text = f"{context} {expanded_query}".strip()
    if not query_text:
        return json.dumps({"error": "context is required"})

    hybrid_score = "(ts_rank(search_vector, query) * 0.7 + importance * 0.3)"
    if sort_by == "importance":
        hybrid_score = "(ts_rank(search_vector, query) * 0.4 + importance * 0.6)"

    order_clause = f"{hybrid_score} DESC"
    if sort_by == "recency":
        order_clause = "created_at DESC"

    category_filter = ""
    params: list[Any] = [user_id, query_text, limit]
    if category:
        category_filter = "AND category = $4"
        params.append(category)

    async with pool.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, content, category, tags, importance, is_sensitive,
                   ts_rank(search_vector, query) AS rank, created_at, updated_at
            FROM memories, plainto_tsquery('english', $2) query
            WHERE user_id = $1 AND deleted_at IS NULL
              AND (search_vector @@ query OR $2 = '')
              {category_filter}
            ORDER BY {order_clause}
            LIMIT $3
            """,
            *params,
        )

    results = []
    for row in rows:
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append({
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "rank": float(row["rank"]),
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        })

    return json.dumps({"memories": results})


@mcp_server.tool()
async def memory_list(category: str | None = None, limit: int = 20) -> str:
    """List stored memories."""
    pool = await get_pool()
    user_id = "default"

    if category:
        query = """SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at
                   FROM memories WHERE user_id = $1 AND deleted_at IS NULL AND category = $2
                   ORDER BY importance DESC LIMIT $3"""
        params: list[Any] = [user_id, category, limit]
    else:
        query = """SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at
                   FROM memories WHERE user_id = $1 AND deleted_at IS NULL
                   ORDER BY importance DESC LIMIT $2"""
        params = [user_id, limit]

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
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        })

    return json.dumps({"memories": results})


@mcp_server.tool()
async def memory_delete(memory_id: int) -> str:
    """Delete a memory by ID."""
    pool = await get_pool()
    user_id = "default"

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
    user_id = "default"
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE user_id = $1 AND deleted_at IS NULL", user_id)
    return json.dumps({"count": count})


@mcp_server.tool()
async def secret_get(key: str) -> str:
    """Retrieve a secret value by key. Returns empty if not found."""
    return json.dumps({"error": "secret_get is not available via SSE transport"})


# Auth middleware for /mcp/* routes
class MCPAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        if request.url.path.startswith("/mcp"):
            auth = request.headers.get("authorization", "")
            token = auth.removeprefix("Bearer ").strip()
            if not _resolve_user_from_token(token):
                return Response(content="Unauthorized", status_code=401)
        return await call_next(request)


app.add_middleware(MCPAuthMiddleware)

# Mount SSE transport
sse_transport = SseServerTransport("/messages/")


class HandleSSE:
    """ASGI app for SSE connections."""
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        async with sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
            await mcp_server._mcp_server.run(
                read_stream, write_stream, mcp_server._mcp_server.create_initialization_options()
            )


# Client connects to /mcp/sse, posts to /mcp/messages/
app.router.routes.insert(0, Mount("/mcp", routes=[
    Route("/sse", endpoint=HandleSSE()),
    Mount("/messages", app=sse_transport.handle_post_message),
]))
