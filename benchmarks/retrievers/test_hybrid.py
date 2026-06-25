"""Unit tests for the HYBRID retriever's pure logic: concept normalisation, the
concept-graph build + 1-hop expansion, weighted RRF fusion, and — the focus of
slice S1 — the FUSION FIX that lets GRAPH-ONLY candidates compete in the shared
fused pool.

These tests are MODEL-FREE on purpose — they never load sentence-transformers (a
~1.3 GB / multi-minute CPU load). The dense leg and (in the fusion-fix tests) the
graph leg are exercised by monkeypatching the ranking methods, so fusion + graph
behaviour is verified deterministically and fast. The full end-to-end quality run
is done via scripts/run_eval.py against the real (local, gitignored) corpus.

THE FUSION FLAW THIS SLICE FIXES (ADR-0005, the run's reason to exist)
======================================================================
The prior ``retrieve()`` computed ``base_set = FTS∪dense`` seeds and called
``_graph_rank(seeds, exclude=base_set)`` at a fixed ``w_graph=0.35`` — so the
graph leg was structurally barred from REINFORCING base-pool docs, and a graph-only
hit at ``w_graph=0.35`` scored only ``0.35/61 ≈ 0.0057`` — below ANY realistic
fused top-k boundary. "Graph adds nothing" was a MATH ARTIFACT, never tested.

THE FIX (verified numerically in scratchpad/rrf_sim.py):
  * ``_graph_rank`` drops ``exclude``; ``retrieve()`` RRF-accumulates the FULL
    ``graph_ranked`` into the SAME shared pool as FTS+dense (graph candidates that
    also appear in a base leg are reinforced, exactly like FTS∪dense overlap).
  * ``w_graph`` is a SWEPT attribute, so the sparse graph leg gets a genuine shot.
  * The boundary a graph-only hit must clear is the TOP-10 FUSED score — in
    realistic fusion (FTS 50 + dense 50, ~30 overlapping → strong docs
    double-counted) that bar is ≈ 0.0286 (top-5 ≈ 0.0308), NOT the weakest tail
    (0.0091). A graph-only rank-1 hit scores ``w_graph/61``: 0.0164 at w_graph=1.0
    (BARRED) → 0.0328 at w_graph≈2.0 (ENTERS). The sweep must reach ≥ 2.0.

Run:  .venv/bin/python -m pytest retrievers/test_hybrid.py -q
"""
from __future__ import annotations

import math
from collections import defaultdict

from harness.types import Memory, MemoryId
from retrievers.hybrid import (
    _RRF_K,
    HybridRetriever,
    _concepts_for,
    _normalise_concept,
)


# ---------------- concept normalisation ----------------

def test_normalise_concept_depluralisation() -> None:
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


def test_normalise_concept_is_stable_under_repetition() -> None:
    # normalising an already-normalised token must be a no-op (idempotent), so the
    # graph collapses variants consistently no matter the source field.
    for tok in ["decision", "policy", "address", "tag", "gpu", "access"]:
        assert _normalise_concept(_normalise_concept(tok)) == _normalise_concept(tok)


def test_concepts_for_unions_tags_keywords_content() -> None:
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


def _build_graph_no_pruning(r: HybridRetriever, corpus: list[Memory]) -> None:
    """Build the concept graph with the df-ceiling disabled so the tiny test
    corpora aren't pruned away as non-discriminative hubs."""
    import retrievers.hybrid as H

    old = H._CONCEPT_MAX_DF_FRAC
    H._CONCEPT_MAX_DF_FRAC = 1.0
    try:
        r._build_graph(corpus)
    finally:
        H._CONCEPT_MAX_DF_FRAC = old


def test_graph_build_links_shared_concepts() -> None:
    r = HybridRetriever()
    _build_graph_no_pruning(r, _shared_concept_corpus())

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


def test_graph_rank_expands_from_seeds_by_weight() -> None:
    # REWRITTEN for the fusion fix: _graph_rank no longer takes `exclude`. Seeds
    # are expanded to their neighbours; the seed itself is not its own neighbour
    # (no self-loops), so it does not reappear via its own edges.
    r = HybridRetriever()
    _build_graph_no_pruning(r, _shared_concept_corpus())

    # Seed from memory 10; neighbours 20 (heavier shared-concept edge) and 30
    # should both surface, with 20 ranked above 30.
    nbrs = r._graph_rank([10], k=10)
    assert nbrs[:2] == [20, 30]
    # 10 has no self-edge, so seeding only from 10 never re-emits 10.
    assert 10 not in nbrs


def test_graph_rank_includes_other_seeds_neighbours() -> None:
    # NEW (fusion-fix contract): with `exclude` gone, a node reachable from ANY
    # seed enters the graph ranking — graph candidates are no longer carved out of
    # the base pool. Seeding from 30 surfaces its neighbours 10 and 20.
    r = HybridRetriever()
    _build_graph_no_pruning(r, _shared_concept_corpus())

    nbrs = r._graph_rank([30], k=10)
    assert 10 in nbrs and 20 in nbrs
    assert 30 not in nbrs  # no self-loop


def test_graph_rank_empty_without_graph_or_seeds() -> None:
    # REWRITTEN: new `_graph_rank(seeds, k)` signature (no `exclude`).
    r = HybridRetriever()  # no graph built
    assert r._graph_rank([1, 2], k=5) == []
    r._build_graph(_shared_concept_corpus())  # real graph, but no seeds
    assert r._graph_rank([], k=5) == []  # no seeds -> empty leg


# ---------------- RRF fusion ----------------

def test_rrf_accumulate_formula() -> None:
    scores: dict[MemoryId, float] = defaultdict(float)
    HybridRetriever._rrf_accumulate(scores, [7, 8, 9], weight=1.0)
    assert math.isclose(scores[7], 1.0 / (_RRF_K + 1))
    assert math.isclose(scores[8], 1.0 / (_RRF_K + 2))
    assert math.isclose(scores[9], 1.0 / (_RRF_K + 3))
    # a second weighted list adds on top
    HybridRetriever._rrf_accumulate(scores, [8], weight=0.5)
    assert math.isclose(scores[8], 1.0 / (_RRF_K + 2) + 0.5 / (_RRF_K + 1))


# ---------------- THE FUSION FIX (slice S1) ----------------

def _stub_three_legs(
    r: HybridRetriever,
    fts_ranked: list[MemoryId],
    dense_ranked: list[MemoryId],
    graph_ranked: list[MemoryId],
) -> None:
    """Stub all three legs of an already-instantiated retriever, model-free, so
    retrieve() exercises ONLY the fusion arithmetic. The graph leg is stubbed at
    `_graph_rank` (its new no-`exclude` signature) — whatever the seeds, it returns
    the fixed `graph_ranked`, simulating a built graph."""
    r._fts.retrieve = lambda q, k: list(fts_ranked[:k])  # type: ignore[method-assign]
    r._dense_rank = lambda q, k: list(dense_ranked[:k])  # type: ignore[method-assign]
    r._graph_rank = lambda seeds, k: list(graph_ranked[:k])  # type: ignore[method-assign]


def test_fusion_graph_only_hit_absent_at_w0_present_when_swept() -> None:
    """POOL-INCLUSION (slice S1.a): a graph-only id G must be ABSENT from the fused
    output at w_graph=0, and ENTER it once w_graph is swept up — proving the
    base-set exclusion is gone and graph candidates compete in the shared pool."""
    G = 999  # graph-only id, returned by NO base leg
    fts = [1, 2, 3, 4, 5]
    dense = [1, 2, 3, 6, 7]  # overlaps fts on 1,2,3; G appears in neither
    r = HybridRetriever()
    _stub_three_legs(r, fts, dense, graph_ranked=[G])

    # w_graph = 0 : the graph leg contributes nothing; G cannot appear.
    r.w_graph = 0.0
    out0 = r.retrieve("q", k=10)
    assert G not in out0, "at w_graph=0 a graph-only id must not appear"
    # sanity: the base legs still fuse normally (agreement floats 1,2,3 up).
    assert out0[:3] == [1, 2, 3]

    # w_graph swept up : G enters the fused output (it competes in the shared pool).
    r.w_graph = 5.0
    out_swept = r.retrieve("q", k=10)
    assert G in out_swept, "swept w_graph must let the graph-only id compete"


def _realistic_base_legs() -> tuple[list[MemoryId], list[MemoryId]]:
    """The realistic-fusion construction from scratchpad/rrf_sim.py: FTS returns 50
    ids (1..50); dense returns 50 ids of which 30 OVERLAP fts (1..30, the strong
    double-counted docs) and 20 are dense-only (101..120). This makes the top-10
    fused boundary ≈ 0.0286 (top-5 ≈ 0.0308), dominated by dual-leg docs — NOT the
    weakest single-leg tail (0.0091)."""
    fts = list(range(1, 51))
    dense = list(range(1, 31)) + list(range(101, 121))
    return fts, dense


def test_fusion_real_boundary_graph_only_clears_top10_only_when_weight_high() -> None:
    """REAL-BOUNDARY (slice S1.b; challenger must_fix #1/#2). Stub FTS+dense to the
    realistic 50+50 lists with ~30 overlapping ids, plus a graph-only G at
    graph-rank 1. Assert G does NOT reach top-10 at w_graph=1.0 (score 0.0164 <
    top-10 bar ≈ 0.0286) and DOES at w_graph=2.0 (0.0328 > 0.0286) — the assertion
    the prior run could NEVER satisfy, now made against the CORRECT fused boundary
    rather than the weakest tail id."""
    G = 999  # graph-only: in neither base leg
    fts, dense = _realistic_base_legs()
    r = HybridRetriever()
    _stub_three_legs(r, fts, dense, graph_ranked=[G])  # G at graph-rank 1

    # --- nail the boundary the test asserts against, from the SAME fusion math ---
    base_scores: dict[MemoryId, float] = defaultdict(float)
    HybridRetriever._rrf_accumulate(base_scores, fts, r.w_fts)
    HybridRetriever._rrf_accumulate(base_scores, dense, r.w_dense)
    ranked_base = sorted(base_scores.values(), reverse=True)
    top10_bar = ranked_base[9]   # 10th-best base score = the recall@10 entry bar
    top5_bar = ranked_base[4]
    # These match scratchpad/rrf_sim.py (top-10 ≈ 0.02857, top-5 ≈ 0.03077).
    assert math.isclose(top10_bar, 1.0 / (_RRF_K + 10) + 1.0 / (_RRF_K + 10), rel_tol=1e-9)
    assert top5_bar > top10_bar  # dual-leg docs dominate the head

    def graph_only_score(w: float) -> float:  # G is at graph-rank 1
        return w / (_RRF_K + 1)

    assert graph_only_score(1.0) < top10_bar       # 0.0164 < 0.0286  → barred
    assert graph_only_score(2.0) > top10_bar       # 0.0328 > 0.0286  → enters

    # --- BARRED at w_graph=1.0: G must not reach the fused top-10 ---
    r.w_graph = 1.0
    out_w1 = r.retrieve("q", k=10)
    assert G not in out_w1, (
        "at w_graph=1.0 a graph-only rank-1 hit scores 0.0164 < the 0.0286 top-10 "
        "bar and MUST stay out — asserting against the real dual-leg boundary, not "
        "the weak tail the prior run mistakenly used"
    )

    # --- ENTERS at w_graph=2.0: G clears the top-10 (and even the top-5) bar ---
    r.w_graph = 2.0
    out_w2 = r.retrieve("q", k=10)
    assert G in out_w2, (
        "at w_graph=2.0 the graph-only hit scores 0.0328 > the 0.0286 top-10 bar "
        "and MUST enter — the assertion the prior (flawed-fusion) run could never "
        "satisfy"
    )


# ---------------- end-to-end fusion + degradation ----------------

def test_retrieve_fuses_all_three_legs_and_degrades() -> None:
    """REWRITTEN (slice S1): end-to-end fusion with the dense leg STUBBED (no
    model) and a REAL concept graph. Verifies (a) FTS + dense agreement floats a
    doc to the top, (b) the graph leg — now in the SHARED pool, no exclusion —
    introduces a doc that shares a concept with a base hit but that the base legs
    ranked too low to surface, and (c) the run does not crash with dense disabled.

    Replaces the prior keyword-co-occurrence assertion that relied on the now-gone
    `exclude` carve-out; here the graph leg competes in the shared pool and a high
    w_graph lets its candidate enter."""
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
        # Stub the dense RANKER deterministically to "agree" with FTS on doc 1, and
        # restrict FTS to doc 1 so doc 2 reaches the pool ONLY via graph expansion.
        r._dense_rank = lambda q, k: [1]  # type: ignore[method-assign]
        r._fts.retrieve = lambda q, k: [1]  # type: ignore[method-assign]
        # Default (down-weighted) graph weight: FTS+dense agreement on doc 1 scores
        # 2/61 ≈ 0.0328, well above the graph-only candidate doc 2 at 0.35/61 ≈
        # 0.0057 — so doc 1 stays first WHILE doc 2 still enters the shared pool
        # (the no-exclusion fix; under the old carve-out doc 2, a seed, was barred).
        r.w_graph = 0.35

        out = r.retrieve("alpha shared concept", k=3)
        assert out, "should return something"
        assert out[0] == 1  # FTS+dense agreement keeps doc 1 first at the default w_graph
        assert 2 in out  # graph expansion (shares 'alpha') pulls doc 2 into the pool
    finally:
        H._CONCEPT_MAX_DF_FRAC = old


def test_graceful_degradation_records_error(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """If the dense build raises, the retriever records it and still serves FTS."""
    corpus = [Memory(id=i, content=f"doc number {i} content", tags="t") for i in range(1, 6)]
    r = HybridRetriever()

    def boom(_corpus: object) -> None:
        raise RuntimeError("simulated embedding failure")

    monkeypatch.setattr(r, "_build_dense", boom)
    r.build_index(corpus)
    assert any("dense leg disabled" in e for e in r.errors)
    assert r._emb is None
    # FTS still answers
    out = r.retrieve("doc number 3 content", k=5)
    assert 3 in out
