"""Unit tests for the HYBRID retriever's pure logic: concept normalisation, the
concept-graph build + 1-hop expansion, weighted RRF fusion, and graceful
degradation when the dense leg is unavailable.

These tests are MODEL-FREE on purpose — they never load sentence-transformers (a
~1.3 GB / multi-minute CPU load). The dense leg is exercised by monkeypatching the
ranking method, so the fusion + graph behaviour is verified deterministically and
fast. The full end-to-end quality run is done via scripts/run_eval.py against the
real (local, gitignored) corpus.

Run:  .venv/bin/python -m pytest retrievers/test_hybrid.py -q
"""
from __future__ import annotations

import math

from harness.types import Memory
from retrievers.hybrid import (
    _RRF_K,
    HybridRetriever,
    _concepts_for,
    _normalise_concept,
)


# ---------------- concept normalisation ----------------

def test_normalise_concept_depluralisation():
    cases = {
        "Decisions": "decision",
        "policies": "policy",
        "addresses": "address",
        "boxes": "box",
        "tags": "tag",
        # invariants: don't over-strip
        "access": "access",
        "class": "class",
        "status": "status",
        "analysis": "analysis",
        "kubernetes": "kubernete",  # heuristic, acceptable (collapses consistently)
        "k8s": "k8s",
        "GPU": "gpu",
    }
    for inp, exp in cases.items():
        assert _normalise_concept(inp) == exp, f"{inp!r} -> {_normalise_concept(inp)!r}"


def test_normalise_concept_is_stable_under_repetition():
    # normalising an already-normalised token must be a no-op (idempotent), so the
    # graph collapses variants consistently no matter the source field.
    for tok in ["decision", "policy", "address", "tag", "gpu", "access"]:
        assert _normalise_concept(_normalise_concept(tok)) == _normalise_concept(tok)


def test_concepts_for_unions_tags_keywords_content():
    m = Memory(
        id=1,
        content="The Postgres cluster uses pgvector for embeddings.",
        tags="database,postgres",
        expanded_keywords="cnpg vector search",
    )
    cs = _concepts_for(m)
    # from tags (note: 'postgres' de-plurals to 'postgre' — a consistent heuristic
    # collapse; what matters is every memory mentioning it lands on the SAME node).
    assert "database" in cs and "postgre" in cs
    # from expanded_keywords
    assert "cnpg" in cs and "vector" in cs and "search" in cs
    # from content (salient tokens, stop-words removed)
    assert "pgvector" in cs and "embedding" in cs  # 'embeddings' -> 'embedding'
    assert "the" not in cs and "for" not in cs  # stop-words excluded


# ---------------- graph build + expansion ----------------

def _shared_concept_corpus() -> list[Memory]:
    # Three memories share concept "alpha" (df=3); two share "beta" (df=2); "gamma"
    # is unique (df=1, links nothing). With min_df=2 and a generous max_df, alpha
    # and beta both form edges.
    return [
        Memory(id=10, content="alpha topic one", tags="alpha", expanded_keywords="beta"),
        Memory(id=20, content="alpha topic two", tags="alpha", expanded_keywords="beta"),
        Memory(id=30, content="alpha topic three", tags="alpha", expanded_keywords="gamma"),
        Memory(id=40, content="unrelated delta", tags="delta", expanded_keywords="delta"),
    ]


def test_graph_build_links_shared_concepts():
    r = HybridRetriever()
    # widen max_df so small-corpus concepts aren't pruned as "hubs"
    import retrievers.hybrid as H

    old = H._CONCEPT_MAX_DF_FRAC
    H._CONCEPT_MAX_DF_FRAC = 1.0
    try:
        r._build_graph(_shared_concept_corpus())
    finally:
        H._CONCEPT_MAX_DF_FRAC = old

    g = r._graph
    assert g is not None
    # alpha links 10-20-30 (a triangle); beta links 10-20; "topic" links 10-20-30
    # too (shared content token). So the triangle exists and 10-20 is the heaviest
    # edge (they additionally share 'beta').
    assert g.has_edge(10, 20)
    assert g.has_edge(10, 30)
    assert g.has_edge(20, 30)
    # 10-20 share alpha + beta + topic (=3); 10-30 share alpha + topic (=2). The
    # exact counts aren't load-bearing — the INVARIANT is w(10,20) > w(10,30).
    assert g[10][20]["weight"] > g[10][30]["weight"]
    # the unrelated memory 40 (concept 'delta', df=1) links nothing.
    assert g.degree(40) == 0
    stats = r.graph_stats()
    assert stats["nodes"] == 4 and stats["edges"] >= 3


def test_graph_rank_expands_from_seeds_by_weight():
    r = HybridRetriever()
    import retrievers.hybrid as H

    old = H._CONCEPT_MAX_DF_FRAC
    H._CONCEPT_MAX_DF_FRAC = 1.0
    try:
        r._build_graph(_shared_concept_corpus())
    finally:
        H._CONCEPT_MAX_DF_FRAC = old

    # Seed from memory 10; neighbours 20 (w=2) and 30 (w=1) should both surface,
    # with 20 ranked above 30 (heavier shared-concept edge).
    nbrs = r._graph_rank([10], exclude={10}, k=10)
    assert nbrs[:2] == [20, 30]
    # excluded seeds are never returned
    assert 10 not in nbrs


def test_graph_rank_empty_without_graph_or_seeds():
    r = HybridRetriever()  # no graph built
    assert r._graph_rank([1, 2], exclude=set(), k=5) == []
    r._graph = object.__new__(type("G", (), {}))  # truthy but unused
    assert r._graph_rank([], exclude=set(), k=5) == []  # no seeds


# ---------------- RRF fusion ----------------

def test_rrf_accumulate_formula():
    scores: dict[int, float] = {}
    from collections import defaultdict

    scores = defaultdict(float)
    HybridRetriever._rrf_accumulate(scores, [7, 8, 9], weight=1.0)
    assert math.isclose(scores[7], 1.0 / (_RRF_K + 1))
    assert math.isclose(scores[8], 1.0 / (_RRF_K + 2))
    assert math.isclose(scores[9], 1.0 / (_RRF_K + 3))
    # a second weighted list adds on top
    HybridRetriever._rrf_accumulate(scores, [8], weight=0.5)
    assert math.isclose(scores[8], 1.0 / (_RRF_K + 2) + 0.5 / (_RRF_K + 1))


def test_retrieve_fuses_all_three_legs_and_degrades():
    """End-to-end fusion with the dense leg STUBBED (no model). Verifies (a) FTS +
    dense agreement floats a doc to the top, (b) the graph leg can introduce a doc
    neither base leg returned, and (c) dense-disabled degrades to FTS(+graph)."""
    corpus = [
        Memory(id=1, content="alpha shared concept", tags="alpha", expanded_keywords="alpha"),
        Memory(id=2, content="alpha shared concept too", tags="alpha", expanded_keywords="alpha"),
        Memory(id=3, content="beta unrelated", tags="beta", expanded_keywords="beta"),
    ]
    import retrievers.hybrid as H

    old = H._CONCEPT_MAX_DF_FRAC
    H._CONCEPT_MAX_DF_FRAC = 1.0
    try:
        r = HybridRetriever()
        # Stub the dense BUILD so the test never loads the ~1.3 GB model nor writes
        # to the shared cache/ dir; build_index then only does FTS + graph.
        r._build_dense = lambda _c: None  # type: ignore[method-assign]
        r.build_index(corpus)  # FTS + graph build only
        # Stub the dense RANKER deterministically to "agree" with FTS on doc 1.
        r._dense_rank = lambda q, k: [1]  # type: ignore[method-assign]

        # query matching doc 1 lexically; doc 2 shares concept 'alpha' with doc 1
        # (graph neighbour) even if FTS ranks it lower.
        out = r.retrieve("alpha shared concept", k=3)
        assert out, "should return something"
        assert out[0] == 1  # FTS+dense agreement puts doc 1 first
        assert 2 in out  # graph expansion (shares 'alpha') pulls doc 2 in
    finally:
        H._CONCEPT_MAX_DF_FRAC = old


def test_graceful_degradation_records_error(monkeypatch):
    """If the dense build raises, the retriever records it and still serves FTS."""
    corpus = [Memory(id=i, content=f"doc number {i} content", tags="t") for i in range(1, 6)]
    r = HybridRetriever()

    def boom(_corpus):
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(r, "_build_dense", boom)
    r.build_index(corpus)
    assert any("dense leg disabled" in e for e in r.errors)
    assert r._emb is None
    # FTS still answers
    out = r.retrieve("doc number 3 content", k=5)
    assert 3 in out
