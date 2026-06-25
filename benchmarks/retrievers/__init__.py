"""Pluggable retrievers for the claude-memory recall benchmark.

Each retriever implements the harness `retrieve(query, k) -> list[int]` contract
(see ``harness/types.py`` :: ``Retriever``) and, optionally, the ``build_index`` /
``index_size_bytes`` lifecycle hooks the runner duck-types.

``fts.FtsRetriever`` is the LEXICAL BASELINE — the product's current local-store
recall (SQLite FTS5/BM25). It is the "current system" any hybrid retriever must
beat on recall@k / nDCG@10 / MRR (ADR-0001).
"""
