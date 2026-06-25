"""BASELINE retriever: the product's CURRENT lexical recall (SQLite FTS5/BM25).

This is the "current system" the hybrid upgrade (dense embeddings + concept
graph, ADR-0001) must beat on recall@k / nDCG@10 / MRR. It is a *faithful*
reimplementation of the production local-store recall path, not an idealised
sketch — it mirrors ``src/claude_memory/mcp_server.py :: _sqlite_recall`` (and
the FTS5 schema/triggers in the same module) line-for-line where it matters:

Production recall (``sort_by="relevance"``) does ALL of the following, and so
does this retriever:

1. **Concatenate then split.** The MCP tool builds
   ``all_terms = f"{context} {expanded_query}"`` and splits it on whitespace,
   stripping any embedded ``"`` from each token. The harness already hands us
   one ``query`` string (the concatenation happens upstream of recall), so here
   ``query`` IS ``all_terms``; we split + strip identically.

2. **AND-first, then OR-broaden.** Production builds BOTH
   ``'"w1" AND "w2" ...'`` and ``'"w1" OR "w2" ...'`` and runs the **AND** match
   first; only if it returns zero rows does it fall back to the **OR** match.
   (The README's "Search Algorithm" prose shows only the OR form; the *code* is
   AND→OR, and the code is authoritative. We replicate the code.)

3. **Blended BM25+importance relevance ordering.** ``sort_by="relevance"`` is
   NOT a pure ``ORDER BY bm25()``. It is the blend
   ``(-bm25(memories_fts) * 0.7 + importance * 0.3) DESC`` (bm25 is negated
   because SQLite returns more-negative = better-match). We use the EXACT same
   expression. We deliberately evaluate ``relevance`` (not the production
   ``importance`` default) so the benchmark measures RETRIEVAL quality rather
   than the importance-sort prior — per the research brief.

4. **FTS5 default tokenizer.** The production virtual table is declared with no
   explicit tokenizer, i.e. ``unicode61`` — case-folding + unicode diacritic
   stripping, NO stemming and NO stop-word removal. We declare ours the same
   way, so "running" does not match "run" (a known lexical weakness the dense
   path is expected to fix on the *paraphrase* stratum).

5. **LIKE fallback.** If the FTS5 MATCH raises ``sqlite3.OperationalError``
   (e.g. a token that trips the FTS5 query grammar), production degrades to a
   ``content LIKE %context% OR tags LIKE %context%`` scan ordered by importance.
   We mirror that fallback (using the full query as the LIKE needle, since the
   harness query is the whole ``all_terms``).

DIFFERENCES FROM PRODUCTION (all immaterial to ranking, documented for honesty):
- The benchmark corpus has no per-user / soft-delete / category filtering, so we
  drop the ``user_id``/``deleted_at``/``category`` predicates. No category is
  passed by the harness, so the category branch is never taken anyway.
- We build a fresh in-memory FTS5 index over ``data/corpus.jsonl`` rather than
  reading the live ``memory.db``; same schema, same tokenizer, same columns
  (content/category/tags/expanded_keywords), so BM25 statistics match what the
  product would compute over the same documents.

The harness reference ``harness.baselines.SqliteFtsRetriever`` implements the
*README* ordering (pure ``ORDER BY bm25(), importance``). This module is the
faithful-to-the-CODE variant and is the one the RUN reports as ``retriever="fts"``.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

# Import the corpus dataclass from the sibling harness package. run_eval.py and
# run_benchmark put the benchmarks/ root on sys.path; support direct execution
# (python retrievers/fts.py) too by adding it ourselves if the import fails.
try:  # pragma: no cover - exercised by both import paths
    from harness.types import Memory, MemoryId
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.types import Memory, MemoryId

# Mirror production token extraction: split ``all_terms`` on whitespace and strip
# any embedded double-quote from each token (mcp_server uses
# ``w.replace(chr(34), "")``). We lowercase as well; FTS5 unicode61 case-folds
# regardless, so this only normalises the quoted MATCH literals we emit.
_DQUOTE = '"'


class FtsRetriever:
    """Faithful reimplementation of the production SQLite FTS5/BM25 recall.

    Mirrors ``_sqlite_recall(sort_by="relevance")``: AND-first then OR-broaden
    over an FTS5(content, category, tags, expanded_keywords) index, ranked by
    the blended ``(-bm25*0.7 + importance*0.3)`` score, with a LIKE fallback.
    """

    #: Label surfaced in benchmark reports / the RUN schema.
    name = "fts"

    def __init__(self, sort_by: str = "relevance") -> None:
        # We benchmark "relevance" so the metric reflects retrieval quality, not
        # the importance prior. "importance" is kept for parity / diagnostics.
        if sort_by not in ("relevance", "importance"):
            raise ValueError(f"sort_by must be 'relevance' or 'importance', got {sort_by!r}")
        self.sort_by = sort_by
        self._con: sqlite3.Connection | None = None

    # ── lifecycle hooks (duck-typed by the runner) ───────────────────────────

    def build_index(self, corpus: Sequence[Memory]) -> None:
        """Build a fresh in-memory FTS5 index over the corpus.

        Same virtual-table shape and (default ``unicode61``) tokenizer as the
        production ``memories_fts`` table. We carry ``memory_id`` and
        ``importance`` as UNINDEXED columns so the relevance blend can read
        importance without a join — semantically identical to the production
        ``memories m JOIN memories_fts fts ON m.id = fts.rowid`` read.
        """
        con = sqlite3.connect(":memory:")
        con.execute(
            """
            CREATE VIRTUAL TABLE memories_fts USING fts5(
                content, category, tags, expanded_keywords,
                memory_id UNINDEXED, importance UNINDEXED
            )
            """
        )
        con.executemany(
            "INSERT INTO memories_fts"
            "(content, category, tags, expanded_keywords, memory_id, importance)"
            " VALUES (?,?,?,?,?,?)",
            [
                (
                    m.content,
                    m.category,
                    m.tags,
                    m.expanded_keywords,
                    int(m.id),
                    float(m.importance),
                )
                for m in corpus
            ],
        )
        con.commit()
        self._con = con

    def index_size_bytes(self) -> int:
        """Approximate on-disk index size (sum of FTS5 shadow-table page bytes).

        The index is in-memory, so this is the SQLite page accounting for the
        FTS5 shadow tables — reported for the storage column, non-gating per
        ADR-0001.
        """
        if self._con is None:
            return 0
        try:
            page_count = self._con.execute("PRAGMA page_count").fetchone()[0]
            page_size = self._con.execute("PRAGMA page_size").fetchone()[0]
            return int(page_count) * int(page_size)
        except sqlite3.Error:
            return 0

    # ── query construction (mirrors _sqlite_recall) ──────────────────────────

    @staticmethod
    def _tokens(query: str) -> list[str]:
        """Split ``all_terms`` exactly as production does: whitespace split,
        drop embedded double-quotes, drop empties."""
        return [w.replace(_DQUOTE, "").lower() for w in query.split() if w.strip()]

    @classmethod
    def _and_or_queries(cls, query: str) -> tuple[str, str]:
        """Build the ('"w1" AND "w2" ...', '"w1" OR "w2" ...') MATCH pair."""
        words = cls._tokens(query)
        if not words:
            return "", ""
        quoted = [f'"{w}"' for w in words]
        return " AND ".join(quoted), " OR ".join(quoted)

    def _order_clause(self) -> str:
        # bm25() is negative (more-negative = better), so negate before blending.
        if self.sort_by == "relevance":
            return "(-bm25(memories_fts) * 0.7 + importance * 0.3) DESC"
        return "(-bm25(memories_fts) * 0.4 + importance * 0.6) DESC"

    # ── retrieve ──────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int) -> list[MemoryId]:
        """Return up to ``k`` memory ids, ranked best-first.

        AND-match first (precise); if it yields nothing, OR-broaden. On an FTS5
        grammar error, fall back to a LIKE scan ordered by importance — exactly
        the production degradation path.
        """
        assert self._con is not None, "call build_index first"
        and_query, or_query = self._and_or_queries(query)
        if not or_query:  # no usable tokens
            return []

        order = self._order_clause()
        base_select = "SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? "
        try:
            rows: list[tuple[int]] = []
            # AND first for precise matches, fall back to OR for broader recall.
            for fts_query in (and_query, or_query):
                rows = self._con.execute(
                    f"{base_select}ORDER BY {order} LIMIT ?",
                    (fts_query, k),
                ).fetchall()
                if rows:
                    break
        except sqlite3.OperationalError:
            # Mirror production LIKE fallback: full query as the needle,
            # ordered by importance.
            like = f"%{query}%"
            rows = self._con.execute(
                "SELECT memory_id FROM memories_fts "
                "WHERE content LIKE ? OR tags LIKE ? "
                "ORDER BY importance DESC LIMIT ?",
                (like, like, k),
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None


# Convenience for `run_eval.py --retriever retrievers.fts:FtsRetriever`
# and a no-arg default instantiation (sort_by="relevance").
