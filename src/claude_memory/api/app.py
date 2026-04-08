"""Claude Memory API -- shared persistent memory with PostgreSQL full-text search."""

import hashlib
import json
import logging
import pathlib
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from claude_memory.api.auth import AuthUser, get_current_user, _key_to_user
from claude_memory.api.database import close_pool, get_pool, init_pool
from claude_memory.api.models import (
    MemoryRecall, MemoryResponse, MemoryStore, MemoryUpdate,
    SecretResponse, ShareMemory, ShareTag, SyncResponse, UnshareTag,
)
from claude_memory.api.permissions import check_memory_permission
from claude_memory.api.vault_service import (
    delete_secret,
    get_secret,
    is_vault_configured,
    store_secret,
)

logger = logging.getLogger(__name__)

# Context variable for MCP SSE multi-user support
_current_user: ContextVar[str] = ContextVar("_current_user", default="default")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="Claude Memory API", lifespan=lifespan)

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


@app.get("/api/auth-check")
async def auth_check(user: AuthUser = Depends(get_current_user)) -> dict[str, str]:
    """Validate API key without doing any real work."""
    return {"status": "ok", "user_id": user.user_id}


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
        # Own memories (AND-match)
        rows = await conn.fetch(
            f"""
            SELECT id, content, category, tags, importance, is_sensitive,
                   ts_rank(search_vector, query) AS rank,
                   created_at, updated_at,
                   NULL::text AS shared_by, NULL::text AS share_permission
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

        # Individually shared memories
        shared_rows = await conn.fetch(
            f"""
            SELECT m.id, m.content, m.category, m.tags, m.importance, m.is_sensitive,
                   ts_rank(m.search_vector, query) AS rank,
                   m.created_at, m.updated_at,
                   m.user_id AS shared_by, ms.permission AS share_permission
            FROM memories m
            JOIN memory_shares ms ON ms.memory_id = m.id,
                 plainto_tsquery('english', $2) query
            WHERE ms.shared_with = $1
              AND m.deleted_at IS NULL
              AND (m.search_vector @@ query OR $2 = '')
              {category_filter}
            ORDER BY {order_clause}
            LIMIT $3
            """,
            *params,
        )

        # Tag-shared memories
        tag_shared_rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (m.id)
                   m.id, m.content, m.category, m.tags, m.importance, m.is_sensitive,
                   ts_rank(m.search_vector, query) AS rank,
                   m.created_at, m.updated_at,
                   m.user_id AS shared_by, ts.permission AS share_permission
            FROM memories m
            JOIN tag_shares ts ON ts.owner_id = m.user_id,
                 plainto_tsquery('english', $2) query
            WHERE ts.shared_with = $1
              AND m.deleted_at IS NULL
              AND (m.search_vector @@ query OR $2 = '')
              AND EXISTS (
                SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t
                WHERE trim(t) = ts.tag
              )
              {category_filter}
            ORDER BY m.id
            LIMIT $3
            """,
            *params,
        )

        # Merge and deduplicate
        seen_ids: set[int] = set()
        all_rows = []
        for row in list(rows) + list(shared_rows) + list(tag_shared_rows):
            if row["id"] not in seen_ids:
                seen_ids.add(row["id"])
                all_rows.append(row)

        # Sort merged results by importance desc and trim
        all_rows.sort(key=lambda r: r["importance"], reverse=True)
        all_rows = all_rows[:body.limit]

        # If AND-match returned too few results, broaden to OR-match (own memories only)
        if len(all_rows) < body.limit and query_text:
            words = query_text.split()
            if len(words) > 1:
                or_tsquery = " | ".join(w for w in words if w)
                or_params: list[Any] = [user.user_id, or_tsquery, body.limit]
                or_cat_filter = ""
                if body.category:
                    or_cat_filter = "AND category = $4"
                    or_params.append(body.category)
                or_rows = await conn.fetch(
                    f"""
                    SELECT id, content, category, tags, importance, is_sensitive,
                           ts_rank(search_vector, query) AS rank,
                           created_at, updated_at,
                           NULL::text AS shared_by, NULL::text AS share_permission
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
                all_rows = all_rows + [r for r in or_rows if r["id"] not in seen_ids]
                all_rows = all_rows[:body.limit]

    results = []
    for row in all_rows:
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
                "shared_by": row["shared_by"],
                "share_permission": row["share_permission"],
            }
        )

    return {"memories": results}


@app.get("/api/memories")
async def list_memories(
    category: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    pool = await get_pool()

    # Build WHERE clauses dynamically
    where_clauses = ["user_id = $1", "deleted_at IS NULL"]
    count_params: list[Any] = [user.user_id]
    param_idx = 2

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
        SELECT id, content, category, tags, importance, is_sensitive, created_at, updated_at
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
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
            }
        )

    return {"memories": results, "total": total}


@app.get("/api/categories")
async def list_categories(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return distinct category values for the current user."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT DISTINCT category FROM memories WHERE user_id = $1 AND deleted_at IS NULL ORDER BY category",
            user.user_id,
        )
    return {"categories": [r["category"] for r in rows]}


@app.get("/api/tags")
async def list_tags(user: AuthUser = Depends(get_current_user)) -> dict[str, Any]:
    """Return all distinct tags with memory counts for the current user."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT trim(t) as tag, COUNT(*) as count
            FROM memories, unnest(string_to_array(tags, ',')) AS t
            WHERE user_id = $1 AND deleted_at IS NULL AND tags != '' AND tags IS NOT NULL
            GROUP BY trim(t)
            ORDER BY count DESC
            """,
            user.user_id,
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

        updates = []
        params: list[Any] = []
        idx = 1

        if body.content is not None:
            updates.append(f"content = ${idx}")
            params.append(body.content)
            idx += 1
        if body.tags is not None:
            updates.append(f"tags = ${idx}")
            params.append(body.tags)
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
    user_id = _current_user.get()
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
    user_id = _current_user.get()
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
                   ts_rank(search_vector, query) AS rank, created_at, updated_at,
                   NULL::text AS shared_by
            FROM memories, plainto_tsquery('english', $2) query
            WHERE user_id = $1 AND deleted_at IS NULL
              AND (search_vector @@ query OR $2 = '')
              {category_filter}
            ORDER BY {order_clause}
            LIMIT $3
            """,
            *params,
        )

        # Also fetch shared memories (individual + tag-based)
        shared_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (m.id) m.id, m.content, m.category, m.tags, m.importance,
                   m.is_sensitive, ts_rank(m.search_vector, query) AS rank,
                   m.created_at, m.updated_at, m.user_id AS shared_by
            FROM memories m, plainto_tsquery('english', $2) query
            WHERE m.deleted_at IS NULL
              AND (m.search_vector @@ query OR $2 = '')
              AND m.user_id != $1
              AND (
                EXISTS (SELECT 1 FROM memory_shares ms WHERE ms.memory_id = m.id AND ms.shared_with = $1)
                OR EXISTS (
                  SELECT 1 FROM tag_shares ts
                  WHERE ts.owner_id = m.user_id AND ts.shared_with = $1
                    AND EXISTS (SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t WHERE trim(t) = ts.tag)
                )
              )
            ORDER BY m.id
            LIMIT $3
            """,
            *params,
        )

    seen_ids = set()
    results = []
    for row in rows:
        seen_ids.add(row["id"])
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        entry: dict[str, Any] = {
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "rank": float(row["rank"]),
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        }
        results.append(entry)

    for row in shared_rows:
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append({
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "rank": float(row["rank"]),
            "shared_by": row["shared_by"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        })

    return json.dumps({"memories": results})


@mcp_server.tool()
async def memory_list(category: str | None = None, limit: int = 20) -> str:
    """List stored memories."""
    pool = await get_pool()
    user_id = _current_user.get()

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

    if category:
        shared_query = """
            SELECT DISTINCT ON (m.id) m.id, m.content, m.category, m.tags, m.importance,
                   m.is_sensitive, m.created_at, m.updated_at, m.user_id AS shared_by
            FROM memories m
            WHERE m.deleted_at IS NULL AND m.category = $2 AND m.user_id != $1
              AND (
                EXISTS (SELECT 1 FROM memory_shares ms WHERE ms.memory_id = m.id AND ms.shared_with = $1)
                OR EXISTS (
                  SELECT 1 FROM tag_shares ts
                  WHERE ts.owner_id = m.user_id AND ts.shared_with = $1
                    AND EXISTS (SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t WHERE trim(t) = ts.tag)
                )
              )
            ORDER BY m.id LIMIT $3"""
        shared_params: list[Any] = [user_id, category, limit]
    else:
        shared_query = """
            SELECT DISTINCT ON (m.id) m.id, m.content, m.category, m.tags, m.importance,
                   m.is_sensitive, m.created_at, m.updated_at, m.user_id AS shared_by
            FROM memories m
            WHERE m.deleted_at IS NULL AND m.user_id != $1
              AND (
                EXISTS (SELECT 1 FROM memory_shares ms WHERE ms.memory_id = m.id AND ms.shared_with = $1)
                OR EXISTS (
                  SELECT 1 FROM tag_shares ts
                  WHERE ts.owner_id = m.user_id AND ts.shared_with = $1
                    AND EXISTS (SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t WHERE trim(t) = ts.tag)
                )
              )
            ORDER BY m.id LIMIT $2"""
        shared_params = [user_id, limit]

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, *params)
        shared_rows = await conn.fetch(shared_query, *shared_params)

    seen_ids = set()
    results = []
    for row in rows:
        seen_ids.add(row["id"])
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append({
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "created_at": row["created_at"].isoformat(),
            "updated_at": row["updated_at"].isoformat(),
        })

    for row in shared_rows:
        if row["id"] in seen_ids:
            continue
        seen_ids.add(row["id"])
        c = row["content"]
        if row["is_sensitive"]:
            c = f"[SENSITIVE - use secret_get(id={row['id']})]"
        results.append({
            "id": row["id"], "content": c, "category": row["category"],
            "tags": row["tags"], "importance": row["importance"],
            "shared_by": row["shared_by"],
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
    user_id = _current_user.get()
    async with pool.acquire() as conn:
        count = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE user_id = $1 AND deleted_at IS NULL", user_id)
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
        await self.app(scope, receive, send)


app.add_middleware(MCPAuthMiddleware)

# Mount SSE transport
sse_transport = SseServerTransport("/messages/")


class HandleSSE:
    """ASGI app for SSE connections."""
    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Extract user from Authorization header for multi-user MCP
        user_id = "default"
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                token = value.decode().removeprefix("Bearer ").strip()
                resolved = _resolve_user_from_token(token)
                if resolved:
                    user_id = resolved
                break
        _current_user.set(user_id)
        async with sse_transport.connect_sse(scope, receive, send) as (read_stream, write_stream):
            await mcp_server._mcp_server.run(
                read_stream, write_stream, mcp_server._mcp_server.create_initialization_options()
            )


# Static files for UI (before MCP mount)
app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

# Client connects to /mcp/sse, posts to /mcp/messages/
app.router.routes.insert(0, Mount("/mcp", routes=[
    Route("/sse", endpoint=HandleSSE()),
    Mount("/messages", app=sse_transport.handle_post_message),
]))
