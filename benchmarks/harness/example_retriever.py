"""Worked example: how a later agent plugs a retriever into the harness.

A retriever needs only one method:

    retrieve(self, query: str, k: int) -> list[int]   # ranked memory ids

Optionally it may implement lifecycle hooks the runner will use if present:

    build_index(self, corpus: list[Memory]) -> None    # timed separately
    index_size_bytes(self) -> int                      # reported

Run this file directly for a smoke test against the local eval set:
    .venv/bin/python -m harness.example_retriever
"""
from __future__ import annotations

from collections.abc import Sequence

from .types import Memory, MemoryId


class SubstringRetriever:
    """Trivial baseline: rank by count of query-word occurrences in content.

    Deliberately weak — exists only to demonstrate the interface. The real
    lexical baseline is harness.baselines.SqliteFtsRetriever.
    """

    name = "substring_demo"

    def __init__(self) -> None:
        self._corpus: list[Memory] = []

    def build_index(self, corpus: Sequence[Memory]) -> None:
        self._corpus = list(corpus)

    def retrieve(self, query: str, k: int) -> list[MemoryId]:
        words = [w for w in query.lower().split() if len(w) > 2]
        scored: list[tuple[int, float]] = []
        for m in self._corpus:
            hay = (m.content + " " + m.expanded_keywords + " " + m.tags).lower()
            score = sum(hay.count(w) for w in words)
            if score:
                scored.append((m.id, score + m.importance))  # importance tiebreak
        scored.sort(key=lambda t: t[1], reverse=True)
        return [mid for mid, _ in scored[:k]]


def _smoke() -> None:
    from .dataset import load_dataset
    from .runner import run_benchmark

    ds = load_dataset()
    res = run_benchmark(SubstringRetriever(), ds)
    print(res.summary())


if __name__ == "__main__":
    _smoke()
