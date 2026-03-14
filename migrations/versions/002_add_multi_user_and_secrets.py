"""Add multi-user support and secret management columns.

Revision ID: 002
Revises: 001
Create Date: 2026-03-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "002"
down_revision: Union[str, None] = "001"
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

    if not _column_exists(conn, "user_id"):
        op.add_column("memories", sa.Column("user_id", sa.String(100), nullable=False, server_default="default"))

    if not _column_exists(conn, "is_sensitive"):
        op.add_column("memories", sa.Column("is_sensitive", sa.Boolean(), server_default="false"))

    if not _column_exists(conn, "vault_path"):
        op.add_column("memories", sa.Column("vault_path", sa.Text(), nullable=True))

    if not _column_exists(conn, "encrypted_content"):
        op.add_column("memories", sa.Column("encrypted_content", sa.LargeBinary(), nullable=True))

    op.execute("CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id)")


def downgrade() -> None:
    op.drop_index("idx_memories_user")
    op.drop_column("memories", "encrypted_content")
    op.drop_column("memories", "vault_path")
    op.drop_column("memories", "is_sensitive")
    op.drop_column("memories", "user_id")
