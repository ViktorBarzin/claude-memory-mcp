"""Tests for the comparison-matrix helpers (slice S5): the 4-config builder, the
w_graph sweep, the GRAPH-ONLY-IN-TOP10 diagnostic, and the entity-bridged multihop
sub-cut. These are the pieces ``run_eval.py`` orchestrates; isolating them here
keeps the verdict logic testable and model-free.

THE GRAPH-ONLY-IN-TOP10 DIAGNOSTIC (the metric the prior run lacked)
-------------------------------------------------------------------
Per query: take the retriever's FTS∪dense base pool (each leg pulled to
``base_depth``) and its fused ``top-k``; COUNT the fused-top-k ids that are absent
from that base pool — i.e. ids the graph leg (or pure fusion reshuffling of a
graph-surfaced id) lifted INTO the answer that neither base leg had at depth.
If this count stays ≈0 across the whole w_graph sweep, the graph genuinely cannot
compete — and THAT, on a valid (shared-pool) test, is the verdict.

These tests stub the retriever legs (model-free) exactly as test_hybrid.py does,
so the diagnostic arithmetic is verified deterministically without loading any
embedding model.

Run:  .venv/bin/python -m pytest harness/test_matrix.py -q
"""
from __future__ import annotations

from harness.matrix import (
    CONFIG_NAMES,
    GRAPH_ONLY_K,
    W_GRAPH_SWEEP,
    build_configs,
    count_graph_only_in_topk,
    entity_bridged_multihop_ids,
    graph_only_ids_for_query,
    graph_verdict,
)
from harness.significance import BootstrapResult
from harness.types import MemoryId, Query
from retrievers.hybrid import HybridRetriever


def _stub(r: HybridRetriever, fts: list[MemoryId], dense: list[MemoryId], graph: list[MemoryId]) -> None:
    """Stub all three legs model-free (mirrors test_hybrid._stub_three_legs)."""
    r._fts.retrieve = lambda q, k: list(fts[:k])  # type: ignore[method-assign]
    r._dense_rank = lambda q, k: list(dense[:k])  # type: ignore[method-assign]
    r._graph_rank = lambda seeds, k: list(graph[:k])  # type: ignore[method-assign]


# ---------------- config matrix ----------------

def test_build_configs_has_four_named_configs() -> None:
    """The matrix is exactly FTS / +dense / +graph / +both (ADR comparison plan)."""
    factories = build_configs()
    assert set(factories) == set(CONFIG_NAMES)
    assert CONFIG_NAMES == ("fts", "dense", "graph", "both")


def test_build_configs_set_correct_leg_weights() -> None:
    """Each config ablates legs by RRF weight, holding the FTS leg on throughout:
    - fts:   dense OFF (w_dense=0), graph OFF (w_graph=0)
    - dense: dense ON,  graph OFF (w_graph=0)
    - graph: dense OFF (w_dense=0), graph ON
    - both:  dense ON,  graph ON
    The FTS leg always carries weight (it is the lexical baseline every config
    must beat). The graph weight for the ON configs is set by the sweep; here we
    only assert the ON/OFF gating."""
    factories = build_configs(w_graph=2.0)

    fts = factories["fts"]()
    assert fts.w_fts > 0 and fts.w_dense == 0.0 and fts.w_graph == 0.0

    dense = factories["dense"]()
    assert dense.w_fts > 0 and dense.w_dense > 0 and dense.w_graph == 0.0

    graph = factories["graph"]()
    assert graph.w_fts > 0 and graph.w_dense == 0.0 and graph.w_graph == 2.0

    both = factories["both"]()
    assert both.w_fts > 0 and both.w_dense > 0 and both.w_graph == 2.0


def test_build_configs_threads_w_graph_into_on_configs() -> None:
    """The swept w_graph reaches only the graph-ON configs (graph, both)."""
    for w in (0.5, 1.0, 5.0):
        f = build_configs(w_graph=w)
        assert f["graph"]().w_graph == w
        assert f["both"]().w_graph == w
        assert f["fts"]().w_graph == 0.0
        assert f["dense"]().w_graph == 0.0


# ---------------- graph-only-in-top10 diagnostic ----------------

def test_graph_only_ids_for_query_flags_ids_outside_base_pool() -> None:
    """A graph-only id (in neither FTS nor dense at base_depth) that the fused
    output surfaces is reported; ids that came from a base leg are not."""
    G = 999
    fts = [1, 2, 3, 4, 5]
    dense = [1, 2, 3, 6, 7]  # base pool = {1,2,3,4,5,6,7}
    r = HybridRetriever()
    _stub(r, fts, dense, graph=[G])
    r.w_graph = 5.0  # high enough that G reaches the fused top-k

    fused = r.retrieve("q", k=10)
    assert G in fused  # precondition: the graph lifted G into the answer
    only = graph_only_ids_for_query(r, "q", k=10, base_depth=50)
    assert only == {G}  # exactly the graph-only id, nothing from the base legs


def test_graph_only_count_is_zero_when_graph_off() -> None:
    """With the graph leg ablated (w_graph=0) nothing outside the base pool can
    enter — the diagnostic must read 0 (the 'graph cannot compete' signal)."""
    G = 999
    r = HybridRetriever()
    _stub(r, fts=[1, 2, 3, 4, 5], dense=[1, 2, 3, 6, 7], graph=[G])
    r.w_graph = 0.0
    assert graph_only_ids_for_query(r, "q", k=10, base_depth=50) == set()


def test_graph_only_count_zero_when_graph_overlaps_base() -> None:
    """If the graph leg only re-surfaces ids already in the base pool, the
    diagnostic is 0 — the graph reinforced but contributed no NEW candidate."""
    r = HybridRetriever()
    _stub(r, fts=[1, 2, 3, 4, 5], dense=[1, 2, 3, 6, 7], graph=[2, 3])  # all in base
    r.w_graph = 5.0
    assert graph_only_ids_for_query(r, "q", k=10, base_depth=50) == set()


def test_count_graph_only_in_topk_aggregates_over_queries() -> None:
    """count_graph_only_in_topk sums per-query graph-only counts across the eval
    queries (the headline diagnostic number per config × w_graph)."""
    G = 999
    r = HybridRetriever()
    _stub(r, fts=[1, 2, 3, 4, 5], dense=[1, 2, 3, 6, 7], graph=[G])
    r.w_graph = 5.0
    queries = [
        Query(query_id="q1", text="a", stratum="multihop"),
        Query(query_id="q2", text="b", stratum="exact"),
    ]
    # Each query surfaces the same graph-only G → total 2 (1 per query),
    # and the per-query breakdown is reported too.
    total, per_query = count_graph_only_in_topk(r, queries, k=10, base_depth=50)
    assert total == 2
    assert per_query == {"q1": 1, "q2": 1}


def test_graph_only_k_default_matches_recall_at_10() -> None:
    """The diagnostic's default cutoff is 10 — the gating recall@10 boundary the
    challenger analysis fixed (NOT the retrieve_k=20 headroom)."""
    assert GRAPH_ONLY_K == 10


# ---------------- entity-bridged multihop sub-cut ----------------

def test_entity_bridged_multihop_excludes_part_n_of_m_chunks() -> None:
    """The entity-bridged subset keeps multihop queries whose qrels point at
    DISTINCT memories (a genuine multi-entity bridge) and drops part-N-of-M chunk
    queries whose relevant ids are near-contiguous fragments of ONE memory — the
    cut where a typed graph is theorized to help (vs the chunk queries dense
    already shortcuts).

    A query reads as chunk-style when it has ≥2 relevant ids that form a TIGHT
    numeric cluster (id span ≈ count). The real chunk queries are near-contiguous,
    not strictly consecutive (e.g. 85/86/88/89/91 spans only 6 over 5 ids,
    67/68/70/71/72 spans 5 over 5), so the heuristic must tolerate a small gap, not
    require a perfect run. Entity-bridged targets sit far apart (a large span)."""
    queries = [
        # entity-bridged: two unrelated ids, far apart (span 872 ≫ 2) → KEEP
        Query(query_id="mh1", text="bridge", stratum="multihop", relevant_ids=(12, 884)),
        # chunk-style, near-contiguous (span 6 over 5 ids, like the real 85..91) → DROP
        Query(query_id="mh2", text="chunks", stratum="multihop", relevant_ids=(85, 86, 88, 89, 91)),
        # chunk-style, near-contiguous (span 5 over 5 ids, like the real 67..72) → DROP
        Query(query_id="mh4", text="chunks2", stratum="multihop", relevant_ids=(67, 68, 70, 71, 72)),
        # single-relevant multihop counts as entity-bridged (one target entity) → KEEP
        Query(query_id="mh3", text="single", stratum="multihop", relevant_ids=(500,)),
        # two ids moderately apart (span 40 ≫ 2) → genuine bridge → KEEP
        Query(query_id="mh5", text="bridge2", stratum="multihop", relevant_ids=(100, 140)),
        # non-multihop strata are never in the cut
        Query(query_id="ex1", text="exact", stratum="exact", relevant_ids=(1, 2)),
    ]
    ids = entity_bridged_multihop_ids(queries)
    assert ids == {"mh1", "mh3", "mh5"}


def test_entity_bridged_multihop_empty_when_no_multihop() -> None:
    queries = [Query(query_id="ex1", text="x", stratum="exact", relevant_ids=(1,))]
    assert entity_bridged_multihop_ids(queries) == set()


# ---------------- sweep grid + verdict rule ----------------

def test_w_graph_sweep_reaches_5() -> None:
    """The sweep must extend to 5.0 (challenger-corrected ceiling): the true top-10
    dual-leg bar (~0.0286) needs w_graph≈2.0 to enter and higher to displace, so
    the grid spans 0 → 5.0 with the {1.5, 2.0, 3.0, 4.0, 5.0} high tail."""
    assert W_GRAPH_SWEEP[0] == 0.0
    assert W_GRAPH_SWEEP[-1] == 5.0
    for w in (1.5, 2.0, 3.0, 4.0, 5.0):
        assert w in W_GRAPH_SWEEP
    # monotone increasing, no duplicates
    assert list(W_GRAPH_SWEEP) == sorted(set(W_GRAPH_SWEEP))


def _boot(delta: float, lo: float, hi: float, *, n: int = 39) -> BootstrapResult:
    p = 1.0 if hi <= 0 else (0.0 if lo > 0 else 0.5)
    return BootstrapResult(delta=delta, ci_low=lo, ci_high=hi, p_le_zero=p, n=n)


def test_graph_verdict_promote_when_both_vs_dense_multihop_clears_zero() -> None:
    """The DECISIVE rule: promote the graph iff the +both-vs-+dense MULTIHOP CI
    clears zero (ci_low > 0). A positive, zero-clearing CI → PROMOTE."""
    decisive = _boot(0.05, 0.01, 0.09)  # CI clears zero
    verdict = graph_verdict(both_vs_dense_multihop=decisive, graph_only_total_max=42)
    assert verdict["promote"] is True
    assert "promote" in verdict["verdict"].lower()


def test_graph_verdict_stay_gated_when_ci_crosses_zero() -> None:
    """CI crosses zero → graph stays GATED, but on a VALID test (it competed)."""
    decisive = _boot(0.01, -0.02, 0.04)  # CI straddles zero
    verdict = graph_verdict(both_vs_dense_multihop=decisive, graph_only_total_max=7)
    assert verdict["promote"] is False
    assert "gated" in verdict["verdict"].lower()
    # the valid-test framing must be recorded (graph candidates DID compete)
    assert "valid" in verdict["verdict"].lower()


def test_graph_verdict_notes_when_graph_never_competed() -> None:
    """If the graph-only-in-top10 count stayed 0 across the whole sweep, the verdict
    must SAY the graph never reached the top-k (it genuinely cannot compete), not
    merely that the CI failed."""
    decisive = _boot(0.0, -0.03, 0.03)
    verdict = graph_verdict(both_vs_dense_multihop=decisive, graph_only_total_max=0)
    assert verdict["promote"] is False
    assert verdict["graph_competed"] is False
    assert "never" in verdict["verdict"].lower() or "could not" in verdict["verdict"].lower()


def test_graph_verdict_records_graph_competed_when_count_positive() -> None:
    decisive = _boot(0.02, 0.005, 0.05)
    verdict = graph_verdict(both_vs_dense_multihop=decisive, graph_only_total_max=15)
    assert verdict["graph_competed"] is True
