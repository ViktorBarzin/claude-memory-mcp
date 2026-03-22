"""Permission checking for shared memories."""

import asyncpg  # type: ignore[import-untyped]


async def check_memory_permission(
    conn: asyncpg.Connection, memory_id: int, user_id: str, required: str
) -> tuple[bool, str | None]:
    """Check if user_id has the required permission on memory_id.

    Returns (allowed, owner_id).
    - Owner always has full access.
    - Shared users checked via memory_shares and tag_shares.
    - required: "read" or "write". "read" is satisfied by either permission.
    """
    row = await conn.fetchrow(
        "SELECT user_id FROM memories WHERE id = $1 AND deleted_at IS NULL",
        memory_id,
    )
    if not row:
        return False, None

    owner_id = row["user_id"]

    # Owner always has access
    if owner_id == user_id:
        return True, owner_id

    # Check individual memory share
    share = await conn.fetchrow(
        "SELECT permission FROM memory_shares WHERE memory_id = $1 AND shared_with = $2",
        memory_id, user_id,
    )
    if share:
        if required == "read" or share["permission"] == "write":
            return True, owner_id
        return False, owner_id

    # Check tag-based shares
    tag_share = await conn.fetchrow(
        """
        SELECT ts.permission
        FROM tag_shares ts
        JOIN memories m ON m.user_id = ts.owner_id
        WHERE m.id = $1 AND ts.shared_with = $2
          AND EXISTS (
            SELECT 1 FROM unnest(string_to_array(m.tags, ',')) t
            WHERE trim(t) = ts.tag
          )
        ORDER BY CASE WHEN ts.permission = 'write' THEN 0 ELSE 1 END
        LIMIT 1
        """,
        memory_id, user_id,
    )
    if tag_share:
        if required == "read" or tag_share["permission"] == "write":
            return True, owner_id
        return False, owner_id

    return False, owner_id
