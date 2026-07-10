"""Add the ``memory_links`` table — typed, directed Memory→Memory edges.

Revision ID: 006
Revises: 005
Create Date: 2026-07-10

The ADR-0007 link substrate. This migration is **purely additive** and
**idempotent** — it adds exactly one table:

``memory_links(id, user_id, src_id, dst_id, link_type, created_at)``
    A typed, directed edge between two Memories, scoped per user (each user
    writes their own edges; ``UNIQUE(user_id, src_id, dst_id, link_type)``
    keeps one edge per user/pair/type). ``link_type`` is the **closed enum of
    four** — ``part-of`` / ``supersedes`` / ``see-also`` / ``resolved-by`` —
    enforced server-side by a CHECK constraint (the category-drift lesson:
    free vocabularies rot, so the enum is a constraint, not a convention).
    ``src_id``/``dst_id`` FK to ``memories(id)`` with ON DELETE CASCADE so a
    hard-deleted memory takes its edges with it (the app's soft delete leaves
    edges in place; recall filters ``deleted_at`` on the memory rows).

Two lookup indexes — ``(user_id, src_id)`` and ``(user_id, dst_id)`` — back
the recall post-processing step, which batch-walks edges in BOTH directions
(incoming ``supersedes`` for redirects, outgoing ``resolved-by`` for
auto-attach, all four types for the links summary).

Unlike 005 there is nothing pgvector-dependent here: the table is created
unconditionally (guarded only by ``_table_exists`` for idempotency) and no
``autocommit_block`` is needed — plain transactional DDL.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: The closed ADR-0007 link-type enum, enforced by the CHECK constraint below.
LINK_TYPES = ("part-of", "supersedes", "see-also", "resolved-by")


def _table_exists(conn: sa.engine.Connection, table_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = :tbl)"
        ),
        {"tbl": table_name},
    )
    return bool(result.scalar())


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "memory_links"):
        quoted_types = ", ".join(f"'{t}'" for t in LINK_TYPES)
        op.create_table(
            "memory_links",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("user_id", sa.Text, nullable=False),
            sa.Column(
                "src_id",
                sa.Integer,
                sa.ForeignKey("memories.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "dst_id",
                sa.Integer,
                sa.ForeignKey("memories.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("link_type", sa.Text, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
            sa.CheckConstraint(
                f"link_type IN ({quoted_types})",
                name="ck_memory_links_link_type",
            ),
            sa.UniqueConstraint(
                "user_id", "src_id", "dst_id", "link_type", name="uq_memory_links_edge"
            ),
        )
        op.create_index("idx_memory_links_user_src", "memory_links", ["user_id", "src_id"])
        op.create_index("idx_memory_links_user_dst", "memory_links", ["user_id", "dst_id"])


def downgrade() -> None:
    """Drop only the 006 addition; the 005-and-earlier schema stays intact."""
    op.execute("DROP TABLE IF EXISTS memory_links")
