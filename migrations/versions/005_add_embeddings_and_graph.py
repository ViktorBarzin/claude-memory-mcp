"""Add dense-embedding column + HNSW index and the typed concept-graph tables.

Revision ID: 005
Revises: 004
Create Date: 2026-06-25

The hybrid-recall upgrade's production substrate (ADR-0004/0005/0006). This
migration is **purely additive** and **idempotent** — it adds:

1. the pgvector ``vector`` extension (``CREATE EXTENSION IF NOT EXISTS``);
2. ``memories.embedding halfvec(1024)`` — NULL-able; sensitive rows
   (``is_sensitive = 1``) keep it NULL and are never embedded (ADR-0003);
3. an **HNSW** index on ``embedding`` with ``halfvec_cosine_ops`` and the
   ADR-0006 build params (``m=16, ef_construction=64``), created
   **CONCURRENTLY**;
4. the three concept-graph tables — ``concepts`` / ``concept_edges`` /
   ``memory_concepts`` — the production mirror of the offline
   ``benchmarks/retrievers/graph_build.py`` dataclasses (phase-2; unused until
   ``MEMORY_GRAPH_ENABLED`` is on).

The migration **does not touch** the generated ``search_vector tsvector``
column or the ``idx_memories_search`` GIN index from migration 001 — lexical
recall and the SQLite-only degrade path stay byte-identical (ADR-0002).

Two correctness invariants, both verified against this repo:

* **pgvector-gated.** ``env.py`` may run this migration against a cluster where
  pgvector has not yet been enabled (the Terraform operand-image swap lands
  separately and is staged-only this run). The embedding column + HNSW index
  steps are therefore **gated on the ``vector`` extension being present**: if it
  is absent they no-op, so the migration is safe to run *before* infra enables
  pgvector. The graph tables carry no vector column on ``concept_edges`` /
  ``memory_concepts`` and are created unconditionally; ``concepts.embedding`` is
  the one graph-table vector column and is likewise gated.

* **CONCURRENTLY needs autocommit.** ``migrations/env.py`` wraps every migration
  in ``context.begin_transaction()``. ``CREATE INDEX CONCURRENTLY`` is illegal
  inside a transaction (PostgreSQL ``25001``), so the HNSW index is built inside
  ``op.get_context().autocommit_block()``, which temporarily suspends the
  surrounding transaction and runs in autocommit. ``if_not_exists=True`` keeps
  it idempotent.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: ``halfvec`` column width. Matches ``claude_memory.embeddings.EMBEDDING_DIM``
#: and the HNSW index dimension (ADR-0006 fixes this at 1024 for both Voyage-3.5
#: and bge-large-en-v1.5).
EMBEDDING_DIM = 1024

#: The dense-vector index name (read by the recall path's ``SET LOCAL
#: hnsw.ef_search`` txn and by the downgrade).
HNSW_INDEX_NAME = "idx_memories_embedding_hnsw"


def _table_exists(conn: sa.engine.Connection, table_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = :tbl)"
        ),
        {"tbl": table_name},
    )
    return bool(result.scalar())


def _column_exists(conn: sa.engine.Connection, table_name: str, column_name: str) -> bool:
    result = conn.execute(
        sa.text(
            "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :tbl AND column_name = :col)"
        ),
        {"tbl": table_name, "col": column_name},
    )
    return bool(result.scalar())


def _vector_extension_available(conn: sa.engine.Connection) -> bool:
    """True iff the pgvector ``vector`` extension can be created in this database.

    Checks ``pg_available_extensions`` — i.e. the extension's control files are
    installed on the server so ``CREATE EXTENSION vector`` will succeed. This is
    the gate that makes the migration **safe to run before infra enables
    pgvector**: on a stock operand image (no pgvector bundled) the extension is
    not available, so the embedding column + HNSW index steps no-op and only the
    (vector-free) graph tables land. Once the Terraform operand-image swap (a
    separate, staged-only change) ships a pgvector-bundled image, a re-run picks
    up the column + index. ``pg_extension`` (already-created) is a subset of this
    — both states return True, so the gate also covers a re-run after creation.
    """
    result = conn.execute(
        sa.text("SELECT EXISTS(SELECT 1 FROM pg_available_extensions WHERE name = 'vector')")
    )
    return bool(result.scalar())


def upgrade() -> None:
    conn = op.get_bind()

    # (1) pgvector extension — gated on AVAILABILITY so a run before infra enables
    # pgvector no-ops cleanly (a bare ``CREATE EXTENSION vector`` against a stock
    # image hard-fails with "extension is not available"; IF NOT EXISTS does NOT
    # save us there). When available the create is idempotent. The graph tables
    # below land regardless.
    has_vector = _vector_extension_available(conn)
    if has_vector:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # (2) embedding column — gated on pgvector. NULL-able: sensitive rows stay
    # NULL (never embedded; ADR-0003) and the column is empty until the
    # flag-gated embed-on-write path backfills it. SQLAlchemy has no first-class
    # ``halfvec`` type, so the DDL is issued directly (mirroring migration 001's
    # raw tsvector column). IF NOT EXISTS keeps re-runs a no-op.
    if has_vector and not _column_exists(conn, "memories", "embedding"):
        op.execute(f"ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding halfvec({EMBEDDING_DIM})")

    # (3) HNSW index — CONCURRENTLY, hence inside an autocommit_block (env.py
    # wraps the migration in a transaction; CONCURRENTLY there is illegal).
    # if_not_exists keeps re-runs a no-op. Gated on pgvector + column presence.
    if has_vector and _column_exists(conn, "memories", "embedding"):
        with op.get_context().autocommit_block():
            op.create_index(
                HNSW_INDEX_NAME,
                "memories",
                ["embedding"],
                unique=False,
                postgresql_using="hnsw",
                postgresql_with={"m": 16, "ef_construction": 64},
                postgresql_ops={"embedding": "halfvec_cosine_ops"},
                postgresql_concurrently=True,
                if_not_exists=True,
            )

    # (4) concept-graph tables — additive, phase-2, unused until MEMORY_GRAPH_ENABLED.
    # Production mirror of benchmarks/retrievers/graph_build.py's Concept /
    # ConceptEdge / MemoryConcept dataclasses.
    if not _table_exists(conn, "concepts"):
        op.create_table(
            "concepts",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            # canonical surface form (cluster representative); aliases is the set
            # of surface forms that collapsed into it (PG text[] array).
            sa.Column("canonical_name", sa.Text, nullable=False),
            sa.Column(
                "aliases",
                postgresql.ARRAY(sa.Text),
                nullable=False,
                server_default=sa.text("ARRAY[]::text[]"),
            ),
            sa.Column("category", sa.String(50), nullable=False, server_default="concept"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("NOW()")),
            sa.UniqueConstraint("canonical_name", name="uq_concepts_canonical_name"),
        )
        op.create_index("idx_concepts_canonical_name", "concepts", ["canonical_name"])
        # concepts.embedding is the concept node's representative vector (the NN
        # substrate for incremental entity resolution at write time). Gated on
        # pgvector like memories.embedding.
        if has_vector:
            op.execute(f"ALTER TABLE concepts ADD COLUMN embedding halfvec({EMBEDDING_DIM})")

    if not _table_exists(conn, "concept_edges"):
        op.create_table(
            "concept_edges",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "src_id",
                sa.Integer,
                sa.ForeignKey("concepts.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "dst_id",
                sa.Integer,
                sa.ForeignKey("concepts.id", ondelete="CASCADE"),
                nullable=False,
            ),
            # typed relation label from the bounded vocabulary (graph_build.py).
            sa.Column("relation", sa.String(50), nullable=False),
            # number of supporting triples (edges with the same (src,dst,relation) merge).
            sa.Column("weight", sa.Integer, nullable=False, server_default="0"),
            # de-duplicated memory ids that asserted the relation (evidence).
            sa.Column(
                "evidence_memory_ids",
                postgresql.ARRAY(sa.Integer),
                nullable=False,
                server_default=sa.text("ARRAY[]::integer[]"),
            ),
            sa.UniqueConstraint("src_id", "dst_id", "relation", name="uq_concept_edges_triple"),
        )
        op.create_index("idx_concept_edges_src", "concept_edges", ["src_id"])
        op.create_index("idx_concept_edges_dst", "concept_edges", ["dst_id"])

    if not _table_exists(conn, "memory_concepts"):
        op.create_table(
            "memory_concepts",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column(
                "memory_id",
                sa.Integer,
                sa.ForeignKey("memories.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "concept_id",
                sa.Integer,
                sa.ForeignKey("concepts.id", ondelete="CASCADE"),
                nullable=False,
            ),
            # the typed relation this memory played for this concept.
            sa.Column("relation", sa.String(50), nullable=False),
            sa.UniqueConstraint(
                "memory_id", "concept_id", "relation", name="uq_memory_concepts_link"
            ),
        )
        op.create_index("idx_memory_concepts_memory", "memory_concepts", ["memory_id"])
        op.create_index("idx_memory_concepts_concept", "memory_concepts", ["concept_id"])


def downgrade() -> None:
    """Drop only the 005 additions; leave the 004-and-earlier schema intact.

    The HNSW index is dropped CONCURRENTLY inside an autocommit_block for the
    same reason it was built that way (DROP INDEX CONCURRENTLY is illegal inside
    a transaction). All drops are ``IF EXISTS`` so the downgrade is safe even
    when pgvector was absent at upgrade time (the embedding column / index were
    never created in that case).
    """
    conn = op.get_bind()

    # graph tables (children before parent for the FKs).
    op.execute("DROP TABLE IF EXISTS memory_concepts")
    op.execute("DROP TABLE IF EXISTS concept_edges")
    op.execute("DROP TABLE IF EXISTS concepts")

    # HNSW index — CONCURRENTLY, hence autocommit_block.
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {HNSW_INDEX_NAME}")

    # embedding column.
    if _column_exists(conn, "memories", "embedding"):
        op.drop_column("memories", "embedding")

    # The ``vector`` extension is intentionally NOT dropped: other tenants on the
    # shared CNPG cluster may use it, and DROP EXTENSION would fail or cascade.
