"""Core dataclasses and the pluggable Retriever protocol."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

MemoryId = int


@dataclass(frozen=True)
class Memory:
    """One corpus entry (mirrors corpus.jsonl)."""

    id: MemoryId
    content: str
    category: str = "facts"
    tags: str = ""
    expanded_keywords: str = ""
    importance: float = 0.5


@dataclass(frozen=True)
class Query:
    """One eval query (mirrors queries.jsonl)."""

    query_id: str
    text: str
    stratum: str  # "exact" | "paraphrase" | "multihop"
    # convenience copy of relevant ids; authoritative source is Qrels
    relevant_ids: tuple[MemoryId, ...] = field(default_factory=tuple)


# query_id -> set of relevant memory ids (binary relevance)
Qrels = dict[str, set[MemoryId]]


@runtime_checkable
class Retriever(Protocol):
    """Pluggable retriever contract.

    Implementations rank corpus memories for a query and return the top-k
    memory ids, best match first. The harness will call `retrieve` once per
    query and compare against qrels.

    Optional lifecycle hooks let a retriever build an index from the corpus
    and report index build time / on-disk size; the runner uses them if
    present (duck-typed), so a minimal retriever need only implement
    `retrieve`.
    """

    def retrieve(self, query: str, k: int) -> list[MemoryId]:
        """Return up to k memory ids, ranked best-first."""
        ...
