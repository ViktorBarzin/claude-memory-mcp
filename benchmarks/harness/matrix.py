"""The S5 comparison MATRIX over the four retriever configs, the w_graph sweep,
and the GRAPH-ONLY-IN-TOP10 diagnostic that finally tests whether the concept
graph can compete.

``run_eval.py`` orchestrates these helpers; keeping them here makes the verdict
logic importable and unit-testable (model-free), rather than buried in a CLI.

THE FOUR CONFIGS (everything else — corpus, bge-large dense matrix, typed graph,
fusion — held FIXED so dense-vs-graph is the only variable):
    fts    lexical baseline           w_dense=0   w_graph=0    (FTS only)
    dense  FTS + dense                w_dense=1   w_graph=0    (reconfirm phase-1)
    graph  FTS + graph                w_dense=0   w_graph=w    (graph vs lexical)
    both   FTS + dense + graph        w_dense=1   w_graph=w    (full hybrid)
The FTS leg always carries weight — it is the lexical baseline every config must
beat (ADR-0001). The graph weight ``w`` is set by the SWEEP.

THE GRAPH-ONLY-IN-TOP10 DIAGNOSTIC (the number the prior run lacked):
per query, count the fused top-k ids that are ABSENT from the FTS∪dense base pool
(each base leg pulled to ``base_depth``) — ids the graph leg lifted into the
answer that neither base leg had at depth. If this stays ≈0 across the whole
w_graph sweep, the graph genuinely cannot compete, and that — on a valid
shared-pool test — is the verdict.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

from .significance import BootstrapResult
from .types import MemoryId, Query

# The matrix is exactly these four configs, in this order.
CONFIG_NAMES: tuple[str, ...] = ("fts", "dense", "graph", "both")

# Per-config leg gating: (w_dense, graph_on). w_fts is always 1.0 (the lexical
# baseline is never ablated). The graph weight, when on, is the swept value.
_DENSE_ON: dict[str, bool] = {"fts": False, "dense": True, "graph": False, "both": True}
_GRAPH_ON: dict[str, bool] = {"fts": False, "dense": False, "graph": True, "both": True}

# Default cutoff for the graph-only diagnostic: the gating recall@10 boundary
# (NOT the retrieve_k=20 headroom) — the boundary the challenger analysis fixed.
GRAPH_ONLY_K = 10

# The w_graph sweep grid. Extended ABOVE 2.0 (challenger-corrected ceiling): the
# true top-10 fused boundary is dual-leg-dominated (~0.0286), so a graph-only rank-1
# hit (score w_graph/61) needs w_graph≈2.0 just to ENTER and higher to DISPLACE
# strong dual-leg docs. If the graph-only-in-top10 diagnostic is still ≈0 at 5.0,
# the graph genuinely cannot compete — a valid verdict, not a math artifact.
W_GRAPH_SWEEP: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0)

# Default base-leg depth for the diagnostic's "base pool" — matches the per-leg
# retrieval depth hybrid.retrieve() uses before fusion (max(k, 50)).
DEFAULT_BASE_DEPTH = 50

# A multihop query is treated as a part-N-of-M CHUNK query (NOT entity-bridged)
# when its ≥2 relevant ids form a tight numeric cluster: span (max−min) ≤
# count + this slack. The real chunk queries are near-contiguous, not perfect runs
# (85/86/88/89/91 → span 6 over 5; 67/68/70/71/72 → span 5 over 5), so a small
# slack tolerates the skipped id without admitting genuinely-distant entity targets.
_CHUNK_SPAN_SLACK = 2


def _make_retriever(*, w_dense: float, w_graph: float):  # type: ignore[no-untyped-def]
    """Build an UNBUILT HybridRetriever with the given leg weights (w_fts always
    on). Imported lazily so importing this module doesn't pull the retriever stack
    (and its heavy optional deps) until a config is actually instantiated."""
    from retrievers.hybrid import HybridRetriever

    r = HybridRetriever()
    r.w_fts = 1.0
    r.w_dense = w_dense
    r.w_graph = w_graph
    return r


def build_configs(w_graph: float = 0.35) -> dict[str, Callable[[], object]]:
    """Return ``name -> factory`` for the four configs. Each factory creates a
    FRESH, UNBUILT retriever with the config's leg weights (the graph-ON configs
    get ``w_graph``; the others get 0.0). Cheap — the expensive index build /
    graph injection happens later, once, on a shared retriever via
    :func:`apply_config` in the real run."""
    factories: dict[str, Callable[[], object]] = {}
    for name in CONFIG_NAMES:
        wd = 1.0 if _DENSE_ON[name] else 0.0
        wg = w_graph if _GRAPH_ON[name] else 0.0
        # functools.partial (not a default-arg lambda) so the per-config weights are
        # captured by value AND the factory's type is inferable under mypy --strict.
        factories[name] = functools.partial(_make_retriever, w_dense=wd, w_graph=wg)
    return factories


def apply_config(retriever: object, config_name: str, *, w_graph: float) -> None:
    """Mutate a PRE-BUILT retriever's leg weights to a named config in place.

    The real run builds ONE fully-equipped retriever (FTS + cached dense matrix +
    injected typed graph), then re-runs it under each config × weight by flipping
    weights only — avoiding 4× the expensive index build per sweep point. Ablation
    is exact: a zero-weight leg is a true no-op in ``_rrf_accumulate``.
    """
    if config_name not in CONFIG_NAMES:
        raise ValueError(f"unknown config {config_name!r} (expected one of {CONFIG_NAMES})")
    retriever.w_fts = 1.0  # type: ignore[attr-defined]
    retriever.w_dense = 1.0 if _DENSE_ON[config_name] else 0.0  # type: ignore[attr-defined]
    retriever.w_graph = w_graph if _GRAPH_ON[config_name] else 0.0  # type: ignore[attr-defined]


def graph_only_ids_for_query(
    retriever: object,
    query: str,
    *,
    k: int = GRAPH_ONLY_K,
    base_depth: int = DEFAULT_BASE_DEPTH,
) -> set[MemoryId]:
    """Ids in the fused top-``k`` that are ABSENT from the FTS∪dense base pool.

    The base pool is each base leg pulled to ``base_depth`` (matching retrieve()'s
    pre-fusion depth). Any fused-top-k id outside it was contributed by the graph
    leg (a graph-only candidate, or a graph-reinforced id that pure base fusion did
    not have at depth). This is the per-query graph-only diagnostic.
    """
    # base pool: FTS∪dense at base_depth, using the retriever's own legs.
    fts_ranked = retriever._fts.retrieve(query, base_depth)  # type: ignore[attr-defined]
    dense_ranked = retriever._dense_rank(query, base_depth)  # type: ignore[attr-defined]
    base_pool: set[MemoryId] = set(fts_ranked) | set(dense_ranked)

    fused = retriever.retrieve(query, k)  # type: ignore[attr-defined]
    return {mid for mid in fused[:k] if mid not in base_pool}


def count_graph_only_in_topk(
    retriever: object,
    queries: Sequence[Query],
    *,
    k: int = GRAPH_ONLY_K,
    base_depth: int = DEFAULT_BASE_DEPTH,
) -> tuple[int, dict[str, int]]:
    """Aggregate the graph-only-in-top-k diagnostic over the eval queries.

    Returns ``(total, per_query)`` where ``total`` is the summed count of graph-only
    ids across all queries and ``per_query`` maps query_id → that query's count.
    The headline number per (config, w_graph) cell of the sweep.
    """
    per_query: dict[str, int] = {}
    total = 0
    for q in queries:
        only = graph_only_ids_for_query(retriever, q.text, k=k, base_depth=base_depth)
        per_query[q.query_id] = len(only)
        total += len(only)
    return total, per_query


def entity_bridged_multihop_ids(queries: Sequence[Query]) -> set[str]:
    """Query ids of the ENTITY-BRIDGED multihop subset.

    Keeps multihop queries whose relevant ids point at DISTINCT memories (a genuine
    multi-entity bridge — where a typed graph is theorized to help) and drops
    part-N-of-M CHUNK queries whose relevant ids are near-contiguous fragments of
    ONE source memory (which dense already shortcuts). A single-relevant multihop
    query is entity-bridged (one target entity). Non-multihop strata are never in
    the cut.

    Heuristic: a multihop query with ≥2 relevant ids is a chunk query when its id
    span (max − min) ≤ count + slack — a tight numeric cluster. This tolerates the
    skipped id in near-contiguous real chunk runs without admitting distant targets.
    """
    out: set[str] = set()
    for q in queries:
        if q.stratum != "multihop":
            continue
        ids = sorted(set(q.relevant_ids))
        if len(ids) <= 1:
            out.add(q.query_id)  # single target entity → bridged
            continue
        span = ids[-1] - ids[0]
        is_chunk = span <= len(ids) + _CHUNK_SPAN_SLACK
        if not is_chunk:
            out.add(q.query_id)
    return out


def graph_verdict(
    *,
    both_vs_dense_multihop: BootstrapResult,
    graph_only_total_max: int,
) -> dict[str, Any]:
    """Settle the concept-graph verdict from the DECISIVE test.

    The graph is promoted iff the ``+both`` vs ``+dense`` MULTIHOP 95% CI clears
    zero (``ci_low > 0`` — a measurable gain OVER dense on the stratum where a typed
    graph is theorized to help). Otherwise it stays GATED — but, this run, on a
    VALID test: graph candidates genuinely competed in the shared fused pool at a
    swept-up weight (the prior "graph adds nothing" was a math artifact).

    ``graph_only_total_max`` is the maximum, over the whole w_graph sweep, of the
    graph-only-in-top10 diagnostic. When it is 0 the graph NEVER reached the fused
    top-k at any weight — so it could not have competed, and the verdict says so
    explicitly (a stronger statement than "the CI failed").

    Returns a JSON-serialisable dict (``promote``, ``graph_competed``, ``verdict``
    prose, and the decisive numbers) for the matrix summary + the report.
    """
    promote = both_vs_dense_multihop.significant
    graph_competed = graph_only_total_max > 0

    if promote:
        verdict = (
            "PROMOTE the concept graph: the +both-vs-+dense multihop 95% CI clears "
            f"zero (Δ={both_vs_dense_multihop.delta:+.4f}, "
            f"CI=[{both_vs_dense_multihop.ci_low:+.4f}, {both_vs_dense_multihop.ci_high:+.4f}]), "
            "so the graph adds measurable value OVER dense on the multihop stratum."
        )
    elif not graph_competed:
        verdict = (
            "Graph STAYS GATED on a VALID test: across the entire w_graph sweep NOT "
            "ONE graph-only id ever reached the fused top-10 (graph-only-in-top10 max "
            "= 0), so the graph COULD NOT compete even with the exclusion removed and "
            "the weight swept to the ceiling. The leg is built and shared-pool-fair, "
            "but it surfaces nothing the base legs lack — a genuine result, not the "
            "prior math artifact."
        )
    else:
        verdict = (
            "Graph STAYS GATED on a VALID test: graph-only candidates DID reach the "
            f"fused top-10 (max {graph_only_total_max} over the sweep) — they genuinely "
            "competed in the shared pool — but the decisive +both-vs-+dense multihop "
            f"95% CI does NOT clear zero (Δ={both_vs_dense_multihop.delta:+.4f}, "
            f"CI=[{both_vs_dense_multihop.ci_low:+.4f}, {both_vs_dense_multihop.ci_high:+.4f}]), "
            "so the graph adds no measurable value OVER dense. Gated, not excluded."
        )

    return {
        "promote": promote,
        "graph_competed": graph_competed,
        "graph_only_in_top10_max": graph_only_total_max,
        "decisive_test": both_vs_dense_multihop.to_dict(),
        "verdict": verdict,
    }
