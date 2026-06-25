"""Unit tests for metrics + runner. No real corpus needed (synthetic data).

Run:  .venv/bin/python -m pytest harness/test_harness.py -q
"""
from __future__ import annotations

import math

from harness import metrics
from harness.dataset import Dataset
from harness.runner import run_benchmark, _percentile
from harness.types import Memory, Query


# ---------------- metrics ----------------

def test_recall_at_k_basic():
    ranked = [9, 8, 3, 7, 1]
    rel = {3, 1, 99}  # 99 never retrieved
    assert metrics.recall_at_k(ranked, rel, 5) == 2 / 3
    assert metrics.recall_at_k(ranked, rel, 2) == 0.0  # neither in top2
    assert metrics.recall_at_k(ranked, rel, 3) == 1 / 3  # only id 3 in top3


def test_recall_perfect_and_zero():
    assert metrics.recall_at_k([1, 2, 3], {1, 2, 3}, 5) == 1.0
    assert metrics.recall_at_k([4, 5, 6], {1, 2, 3}, 5) == 0.0


def test_reciprocal_rank():
    assert metrics.reciprocal_rank([5, 4, 3], {3}) == 1 / 3
    assert metrics.reciprocal_rank([3, 4, 5], {3}) == 1.0
    assert metrics.reciprocal_rank([7, 8], {3}) == 0.0
    # first relevant wins
    assert metrics.reciprocal_rank([9, 3, 1], {1, 3}) == 1 / 2


def test_ndcg_perfect():
    # all relevant at the top -> nDCG == 1
    assert math.isclose(metrics.ndcg_at_k([1, 2, 3, 4], {1, 2, 3}, 10), 1.0)


def test_ndcg_known_value():
    # single relevant doc at rank 2: DCG = 1/log2(3); IDCG = 1/log2(2)=1
    ranked = [9, 1, 8]
    val = metrics.ndcg_at_k(ranked, {1}, 10)
    assert math.isclose(val, (1 / math.log2(3)) / 1.0)


def test_ndcg_two_relevant_suboptimal_order():
    # relevant {1,2}; retrieved at ranks 1 and 3
    ranked = [1, 9, 2]
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # ranks 1 and 3
    idcg = 1 / math.log2(2) + 1 / math.log2(3)  # ideal ranks 1 and 2
    assert math.isclose(metrics.ndcg_at_k(ranked, {1, 2}, 10), dcg / idcg)


def test_dedup_does_not_inflate():
    # repeating a relevant id must not increase recall beyond 1 hit's worth
    ranked = [3, 3, 3, 3]
    assert metrics.recall_at_k(ranked, {3, 7}, 5) == 0.5
    assert metrics.reciprocal_rank(ranked, {3}) == 1.0


def test_empty_relevant_is_zero():
    assert metrics.recall_at_k([1, 2], set(), 5) == 0.0
    assert metrics.ndcg_at_k([1, 2], set(), 5) == 0.0


# ---------------- percentile ----------------

def test_percentile():
    vals = [10, 20, 30, 40]
    assert _percentile(vals, 50) == 25.0  # interpolated median
    assert _percentile(vals, 0) == 10
    assert _percentile(vals, 100) == 40
    assert _percentile([5.0], 95) == 5.0
    assert _percentile([], 50) == 0.0


# ---------------- runner ----------------

def _toy_dataset() -> Dataset:
    corpus = [Memory(id=i, content=f"memory {i}") for i in range(1, 11)]
    queries = [
        Query("q_exact_1", "find 1", "exact", (1,)),
        Query("q_para_1", "restate 5", "paraphrase", (5,)),
        Query("q_multi_1", "join 3 and 4", "multihop", (3, 4)),
    ]
    qrels = {"q_exact_1": {1}, "q_para_1": {5}, "q_multi_1": {3, 4}}
    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


class _PerfectRetriever:
    """Returns exactly the relevant ids first (oracle) — for runner plumbing."""

    def __init__(self, qrels):
        self._qrels = qrels
        self._by_text = None

    def build_index(self, corpus):
        self._n = len(corpus)

    def index_size_bytes(self):
        return 1234

    def retrieve(self, query, k):
        # map query text back via the toy queries' known answers
        mapping = {"find 1": [1], "restate 5": [5], "join 3 and 4": [3, 4]}
        ids = mapping.get(query, [])
        # pad with distractors
        pad = [x for x in range(100, 100 + k)]
        return (ids + pad)[:k]


def test_runner_perfect_retriever():
    ds = _toy_dataset()
    r = _PerfectRetriever(ds.qrels)
    res = run_benchmark(r, ds, retriever_name="perfect")
    assert res.n_queries == 3
    assert math.isclose(res.overall["recall@10"], 1.0)
    assert math.isclose(res.overall["mrr"], 1.0)
    assert math.isclose(res.overall["ndcg@10"], 1.0)
    # per-stratum present
    assert set(res.per_stratum) == {"exact", "paraphrase", "multihop"}
    assert res.per_stratum["multihop"].n_queries == 1
    # lifecycle hooks captured
    assert res.index_build_seconds is not None
    assert res.index_size_bytes == 1234
    # latency recorded
    assert res.latency_ms["p95"] >= 0.0


def test_runner_callable_retriever_and_misses():
    ds = _toy_dataset()

    def retrieve(query, k):  # always wrong
        return [999][:k]

    res = run_benchmark(retrieve, ds, retriever_name="bad", warmup=False)
    assert res.overall["recall@10"] == 0.0
    assert res.overall["mrr"] == 0.0
    assert res.index_build_seconds is None  # no hook on a bare callable
    assert "perfect" not in res.summary()
    assert "bad" in res.summary()
