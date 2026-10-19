"""Turn off the GIN pending list on ``idx_memories_search`` and flush it.

Revision ID: 007
Revises: 006
Create Date: 2026-10-19

With ``fastupdate`` on (the default), new rows' lexemes go to an unsorted pending list
that every GIN scan reads in full, and it is only merged into the index by VACUUM,
autoanalyze, or an insert that pushes it past ``gin_pending_list_limit`` (4 MB). On the
live store it held about 250 pages: a recall's AND-match for a word that matches nothing
read 251 buffers, all of them the pending list. When those pages had been evicted from
shared_buffers, each one was a separate read from a disk measured at 27-170 ms per read.

The store takes tens of writes a day, so paying the index update at insert time costs
nothing noticeable. Turning the option off does not touch an existing pending list, so
``gin_clean_pending_list`` merges it once. Both are result-preserving: the index answers
the same queries with the same rows.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX = "idx_memories_search"


def upgrade() -> None:
    op.execute(f"ALTER INDEX IF EXISTS {INDEX} SET (fastupdate = off)")
    op.execute(
        f"SELECT gin_clean_pending_list(to_regclass('{INDEX}')) WHERE to_regclass('{INDEX}') IS NOT NULL"
    )


def downgrade() -> None:
    op.execute(f"ALTER INDEX IF EXISTS {INDEX} RESET (fastupdate)")
