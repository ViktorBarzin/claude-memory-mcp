"""Load corpus / queries / qrels JSONL into typed objects."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import Memory, Query, Qrels, MemoryId

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Dataset:
    corpus: list[Memory]
    queries: list[Query]
    qrels: Qrels

    @property
    def corpus_by_id(self) -> dict[MemoryId, Memory]:
        return {m.id: m for m in self.corpus}

    def strata(self) -> set[str]:
        return {q.stratum for q in self.queries}


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus(path: Path | None = None) -> list[Memory]:
    path = path or (_DATA_DIR / "corpus.jsonl")
    rows = _read_jsonl(path)
    return [
        Memory(
            id=r["id"],
            content=r["content"],
            category=r.get("category", "facts"),
            tags=r.get("tags", "") or "",
            expanded_keywords=r.get("expanded_keywords", "") or "",
            importance=r.get("importance", 0.5),
        )
        for r in rows
    ]


def load_queries(path: Path | None = None) -> list[Query]:
    path = path or (_DATA_DIR / "queries.jsonl")
    rows = _read_jsonl(path)
    return [
        Query(
            query_id=r["query_id"],
            text=r["text"],
            stratum=r["stratum"],
            relevant_ids=tuple(r.get("relevant_ids", [])),
        )
        for r in rows
    ]


def load_qrels(path: Path | None = None) -> Qrels:
    path = path or (_DATA_DIR / "qrels.jsonl")
    rows = _read_jsonl(path)
    qrels: Qrels = {}
    for r in rows:
        qid = r["query_id"]
        rel = set(r["relevant_ids"])
        qrels.setdefault(qid, set()).update(rel)
    return qrels


def load_dataset(
    corpus_path: Path | None = None,
    queries_path: Path | None = None,
    qrels_path: Path | None = None,
    *,
    validate: bool = True,
) -> Dataset:
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    if validate:
        _validate(corpus, queries, qrels)

    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


def _validate(corpus: list[Memory], queries: list[Query], qrels: Qrels) -> None:
    corpus_ids = {m.id for m in corpus}
    q_ids = {q.query_id for q in queries}

    # Every query must have a qrels entry, and vice versa.
    missing_qrels = q_ids - set(qrels)
    if missing_qrels:
        raise ValueError(f"queries without qrels: {sorted(missing_qrels)[:10]}")
    orphan_qrels = set(qrels) - q_ids
    if orphan_qrels:
        raise ValueError(f"qrels without queries: {sorted(orphan_qrels)[:10]}")

    # Every relevant id must exist in the corpus and the set must be non-empty.
    for qid, rels in qrels.items():
        if not rels:
            raise ValueError(f"empty qrels for query {qid}")
        unknown = rels - corpus_ids
        if unknown:
            raise ValueError(
                f"query {qid} references non-corpus ids {sorted(unknown)[:10]}"
            )
