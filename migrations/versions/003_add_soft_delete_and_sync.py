"""Add soft delete and sync support.

Revision ID: 003
Revises: 002
Create Date: 2026-03-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(conn, column_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'memories' AND column_name = :col)"
        ),
        {"col": column_name},
    )
    return result.scalar()


def upgrade() -> None:
    conn = op.get_bind()

    if not _column_exists(conn, "deleted_at"):
        op.add_column("memories", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))

    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_updated ON memories(updated_at)")


def downgrade() -> None:
    op.drop_index("idx_memories_updated")
    op.drop_column("memories", "deleted_at")
