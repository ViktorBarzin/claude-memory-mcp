"""Add memory sharing tables.

Revision ID: 004
Revises: 003
Create Date: 2026-03-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, table_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables "
            "WHERE table_name = :tbl)"
        ),
        {"tbl": table_name},
    )
    return result.scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if not _table_exists(conn, "memory_shares"):
        op.create_table(
            "memory_shares",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("memory_id", sa.Integer, sa.ForeignKey("memories.id"), nullable=False),
            sa.Column("owner_id", sa.String(100), nullable=False),
            sa.Column("shared_with", sa.String(100), nullable=False),
            sa.Column("permission", sa.String(10), nullable=False, server_default="read"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
            sa.UniqueConstraint("memory_id", "shared_with", name="uq_memory_shares_memory_user"),
        )
        op.create_index("idx_shares_shared_with", "memory_shares", ["shared_with"])
        op.create_index("idx_shares_memory_id", "memory_shares", ["memory_id"])

    if not _table_exists(conn, "tag_shares"):
        op.create_table(
            "tag_shares",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("owner_id", sa.String(100), nullable=False),
            sa.Column("tag", sa.String(100), nullable=False),
            sa.Column("shared_with", sa.String(100), nullable=False),
            sa.Column("permission", sa.String(10), nullable=False, server_default="read"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
            sa.UniqueConstraint("owner_id", "tag", "shared_with", name="uq_tag_shares_owner_tag_user"),
        )
        op.create_index("idx_tag_shares_shared_with", "tag_shares", ["shared_with"])


def downgrade() -> None:
    op.drop_table("tag_shares")
    op.drop_table("memory_shares")
