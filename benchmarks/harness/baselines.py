"""Reference LEXICAL baseline retrievers that mirror the production system.

These exist so (a) the eval-set author can VERIFY a query's labels and check
that paraphrase queries genuinely defeat lexical matching, and (b) later agents
have an honest "current system" to beat.

`SqliteFtsRetriever` builds an in-memory SQLite FTS5 index over the corpus and
runs the SAME query shape the production local store uses:
    words -> '"w1" OR "w2" ...' MATCH, ORDER BY bm25(), importance as tiebreak.
(README "SQLite: FTS5 with BM25".) This is the closest faithful, dependency-free
baseline. The Postgres tsvector path is documented in the README; its ranking
differs (weighted A/B/C/D + importance-first default) but for a quality ceiling
comparison the FTS5/BM25 relevance ordering is the right lexical reference.
"""
from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence

from .types import Memory, MemoryId

# FTS5 reserved-ish tokens; we quote every term anyway, but strip embedded quotes.
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class SqliteFtsRetriever:
    """Faithful FTS5/BM25 lexical baseline (mirrors local_store search)."""

    name = "sqlite_fts5_bm25"

    def __init__(self, sort_by: str = "relevance") -> None:
        # "relevance": ORDER BY bm25(), importance DESC  (best for quality eval)
        # "importance": ORDER BY importance DESC, ... (production default)
        self.sort_by = sort_by
        self._con: sqlite3.Connection | None = None

    def build_index(self, corpus: Sequence[Memory]) -> None:
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
            "INSERT INTO memories_fts(content, category, tags, expanded_keywords, memory_id, importance)"
            " VALUES (?,?,?,?,?,?)",
            [
                (m.content, m.category, m.tags, m.expanded_keywords, m.id, m.importance)
                for m in corpus
            ],
        )
        con.commit()
        self._con = con

    def _fts_query(self, query: str) -> str:
        words = _WORD_RE.findall(query.lower())
        if not words:
            return ""
        return " OR ".join(f'"{w}"' for w in words)

    def retrieve(self, query: str, k: int) -> list[MemoryId]:
        assert self._con is not None, "call build_index first"
        match = self._fts_query(query)
        if not match:
            return []
        if self.sort_by == "importance":
            order = "importance DESC, bm25(memories_fts)"
        else:
            order = "bm25(memories_fts), importance DESC"
        try:
            rows = self._con.execute(
                f"SELECT memory_id FROM memories_fts WHERE memories_fts MATCH ? "
                f"ORDER BY {order} LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.OperationalError:
            # mirror production LIKE fallback on FTS syntax errors
            like = f"%{query}%"
            rows = self._con.execute(
                "SELECT memory_id FROM memories_fts WHERE content LIKE ? OR tags LIKE ? "
                "ORDER BY importance DESC LIMIT ?",
                (like, like, k),
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
