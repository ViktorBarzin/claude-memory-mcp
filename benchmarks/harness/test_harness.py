"""Unit tests for metrics + runner (synthetic data, always run) PLUS the S0
preserved-artifact wiring tests (skip unless the local, gitignored eval set +
cached bge-large matrix are symlinked into benchmarks/data and benchmarks/cache).

The synthetic tests need no real corpus. The S0 tests assert that:
  * load_dataset() resolves the PRESERVED eval set (119 queries, strata
    exact:40 / paraphrase:40 / multihop:39) byte-identically — not regenerated;
  * HybridRetriever.build_index takes the dense CACHE-HIT branch against the
    preserved bge-large matrix (fingerprint ca7b1d4ed22672e8) with NO model load
    — verified by patching _embed_local to raise and confirming build still wins.
They skip cleanly where numpy/networkx or the preserved artifacts are absent
(e.g. CI), so `pytest tests/` and the harness suite stay green everywhere.

Run:  .venv/bin/python -m pytest harness/test_harness.py -q
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

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


# ---------------- S0: preserved eval set + cached bge-large matrix ----------
#
# These exercise the REAL (local, gitignored) artifacts wired in via symlinks
# (benchmarks/data -> preserved data, benchmarks/cache -> preserved cache). They
# skip when those are absent or numpy/networkx aren't installed, so they never
# break CI; locally they prove the harness reuses the preserved set + matrix
# WITHOUT regenerating or re-embedding.

# Preserved-artifact fingerprint of the eval corpus (id+content hash). The cached
# bge-large matrix is named emb_BAAI_bge-large-en-v1.5_<this>.npy. If the corpus
# ever changes this constant must change with it — that is the point of the test.
_PRESERVED_FINGERPRINT = "ca7b1d4ed22672e8"
_EXPECTED_STRATA = {"exact": 40, "paraphrase": 40, "multihop": 39}

_BENCH_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _BENCH_ROOT / "data"
_CACHE_DIR = _BENCH_ROOT / "cache"
_PRESERVED_EMB = _CACHE_DIR / f"emb_BAAI_bge-large-en-v1.5_{_PRESERVED_FINGERPRINT}.npy"

# Preserved data/cache live OUTSIDE the repo (gitignored symlinks). If a teammate
# has not wired them in, skip the whole module-tail rather than fail.
_artifacts_present = (
    (_DATA_DIR / "corpus.jsonl").exists()
    and (_DATA_DIR / "queries.jsonl").exists()
    and (_DATA_DIR / "qrels.jsonl").exists()
    and _PRESERVED_EMB.exists()
)
_needs_artifacts = pytest.mark.skipif(
    not _artifacts_present,
    reason="preserved eval set / cached bge-large matrix not symlinked into benchmarks/{data,cache}",
)


@_needs_artifacts
def test_preserved_eval_set_loads_with_expected_strata():
    """load_dataset() resolves the PRESERVED eval set: exactly 119 queries with
    strata exact:40 / paraphrase:40 / multihop:39, all referentially valid."""
    pytest.importorskip("numpy")
    from collections import Counter

    from harness import load_dataset

    ds = load_dataset(validate=True)  # raises on any referential-integrity break
    assert len(ds.corpus) == 5452
    assert len(ds.queries) == 119
    assert dict(Counter(q.stratum for q in ds.queries)) == _EXPECTED_STRATA
    # qrels cover every query (validate=True already enforces this; assert the
    # count too so a regenerated, differently-sized set trips the test).
    assert len(ds.qrels) == 119


@_needs_artifacts
def test_preserved_eval_set_is_byte_identical_not_regenerated():
    """The wired-in eval files are BYTE-IDENTICAL to the preserved artifacts — we
    reuse them, never regenerate. Reading through the symlink and hashing both
    ends proves no in-repo copy diverged."""
    import hashlib

    preserved = Path(
        "/home/wizard/.claude/claude-memory/benchmark-artifacts/data"
    )
    if not preserved.exists():  # pragma: no cover - environment-specific
        pytest.skip("preserved artifact directory not present on this host")
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl"):
        via_harness = (_DATA_DIR / name).read_bytes()
        canonical = (preserved / name).read_bytes()
        assert hashlib.sha256(via_harness).hexdigest() == hashlib.sha256(canonical).hexdigest(), (
            f"{name} read through benchmarks/data differs from the preserved artifact"
        )


@_needs_artifacts
def test_hybrid_build_index_takes_dense_cache_hit_without_model_load():
    """HybridRetriever.build_index must take the dense CACHE-HIT branch against the
    preserved bge-large matrix: it loads emb_..._ca7b1d4ed22672e8.npy and NEVER
    loads the model. We patch _embed_local to raise — if the cache-miss branch
    were taken the dense leg would be disabled and self._emb would stay None.
    Because build_index swallows dense failures into self.errors, the assertion is
    the absence of that error AND a populated matrix, not an exception."""
    np = pytest.importorskip("numpy")
    pytest.importorskip("networkx")

    from harness.dataset import load_corpus
    from retrievers.hybrid import HybridRetriever, _corpus_fingerprint

    corpus = load_corpus()
    # Sanity: the preserved corpus still fingerprints to the cached matrix's name,
    # i.e. the matrix on disk genuinely matches this corpus (no silent staleness).
    assert _corpus_fingerprint(corpus) == _PRESERVED_FINGERPRINT

    r = HybridRetriever()
    # Booby-trap BOTH embed paths: a cache HIT must touch neither.
    def _boom(*_a, **_k):  # noqa: ANN002, ANN003
        raise AssertionError("re-embedding attempted — cache-hit branch was NOT taken")

    r._embed_local = _boom  # type: ignore[method-assign]
    r._embed_hosted = _boom  # type: ignore[method-assign]
    r._load_local_model = _boom  # type: ignore[method-assign]

    r.build_index(corpus)

    # Dense leg succeeded via the cache — no "dense leg disabled" error recorded.
    assert not any("dense leg disabled" in e for e in r.errors), r.errors
    # Matrix loaded from cache: 5452 rows × 1024 dims, float32, model never touched.
    assert r._emb is not None
    assert r._emb.shape == (5452, 1024)
    assert r._emb.dtype == np.float32
    assert r.embedding_dim == 1024
    assert r._model is None  # the cache-hit branch returns before loading the model
    assert r.embedding_backend == "local:BAAI/bge-large-en-v1.5"
    # Row order matches the corpus order (the cache .ids.npy gate passed).
    assert r._ids == [m.id for m in corpus]
