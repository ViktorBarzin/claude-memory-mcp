"""Retrieval metrics with BINARY relevance.

Conventions
-----------
- `ranked`: list of memory ids, best-first, as returned by a retriever.
- `relevant`: set of relevant memory ids for the query (from qrels).
- All functions are pure and operate on a single query; the runner aggregates
  (macro-average over queries).

Definitions
-----------
recall@k   = |relevant ∩ ranked[:k]| / |relevant|
             (fraction of all relevant items retrieved within the top k)
MRR        = 1 / rank_of_first_relevant  (0 if none retrieved at all)
nDCG@k     = DCG@k / IDCG@k  with binary gains (gain=1 for relevant)
             DCG@k = sum over i in [1..k] of rel_i / log2(i + 1)
             IDCG@k is the DCG of the ideal ranking (all relevant first),
             capped at min(|relevant|, k) ones.

Notes
-----
- nDCG uses the standard log2(rank+1) discount (Järvelin & Kekäläinen 2002);
  with binary gains this is the common IR convention also used by BEIR/pytrec_eval.
- MRR is reported as the reciprocal rank of the FIRST relevant hit, which for a
  single query equals the per-query reciprocal-rank that the runner averages.
- Duplicate ids in `ranked` are de-duplicated keeping first occurrence, so a
  retriever cannot inflate recall by repeating an id.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

MemoryId = int


def _dedup_keep_order(ranked: Sequence[MemoryId]) -> list[MemoryId]:
    seen: set[MemoryId] = set()
    out: list[MemoryId] = []
    for x in ranked:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def recall_at_k(ranked: Sequence[MemoryId], relevant: Iterable[MemoryId], k: int) -> float:
    rel = set(relevant)
    if not rel:
        # Undefined; treat as 0 contribution. Runner should never pass empty.
        return 0.0
    top = _dedup_keep_order(ranked)[:k]
    hits = sum(1 for x in top if x in rel)
    return hits / len(rel)


def reciprocal_rank(ranked: Sequence[MemoryId], relevant: Iterable[MemoryId]) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    for i, x in enumerate(_dedup_keep_order(ranked), start=1):
        if x in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked: Sequence[MemoryId], relevant: Iterable[MemoryId], k: int) -> float:
    rel = set(relevant)
    top = _dedup_keep_order(ranked)[:k]
    dcg = 0.0
    for i, x in enumerate(top, start=1):
        if x in rel:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[MemoryId], relevant: Iterable[MemoryId], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    dcg = dcg_at_k(ranked, rel, k)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def per_query_metrics(ranked: Sequence[MemoryId], relevant: Iterable[MemoryId]) -> dict[str, float]:
    """All headline metrics for one query."""
    rel = set(relevant)
    return {
        "recall@5": recall_at_k(ranked, rel, 5),
        "recall@10": recall_at_k(ranked, rel, 10),
        "ndcg@10": ndcg_at_k(ranked, rel, 10),
        "mrr": reciprocal_rank(ranked, rel),
    }


METRIC_NAMES = ("recall@5", "recall@10", "ndcg@10", "mrr")
