"""Initial schema with memories table.

Revision ID: 001
Revises:
Create Date: 2026-03-14
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    # Check if table already exists (handles pre-Alembic installations)
    result = conn.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'memories')")
    )
    if result.scalar():
        return

    op.execute("""
        CREATE TABLE memories (
            id SERIAL PRIMARY KEY,
            content TEXT NOT NULL,
            category VARCHAR(50) DEFAULT 'facts',
            tags TEXT DEFAULT '',
            expanded_keywords TEXT DEFAULT '',
            importance REAL DEFAULT 0.5,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            search_vector tsvector GENERATED ALWAYS AS (
                setweight(to_tsvector('english', coalesce(content, '')), 'A') ||
                setweight(to_tsvector('english', coalesce(expanded_keywords, '')), 'B') ||
                setweight(to_tsvector('english', coalesce(tags, '')), 'C') ||
                setweight(to_tsvector('english', coalesce(category, '')), 'D')
            ) STORED
        )
    """)
    op.execute("CREATE INDEX idx_memories_search ON memories USING GIN(search_vector)")


def downgrade() -> None:
    op.drop_index("idx_memories_search")
    op.drop_table("memories")
