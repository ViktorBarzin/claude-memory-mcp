#!/usr/bin/env python3
"""Run the recall benchmark — either a SINGLE retriever or the full S5 COMPARISON
MATRIX (the four configs + w_graph sweep + graph-only diagnostic + per-stratum
paired-bootstrap significance + the graph verdict).

Single retriever (unchanged):
    .venv/bin/python scripts/run_eval.py --retriever fts5
    .venv/bin/python scripts/run_eval.py --retriever fts5 --json results/fts5.json
    .venv/bin/python scripts/run_eval.py --retriever mypkg.mymod:MyRetriever

The matrix (slice S5) — holds bge-large FIXED so dense-vs-graph is the only
variable; reuses the preserved eval set + cached embedding matrix UNCHANGED:
    .venv/bin/python scripts/run_eval.py matrix --outdir results
    .venv/bin/python scripts/run_eval.py matrix --extract        # run haiku triple extraction (cached)
    .venv/bin/python scripts/run_eval.py matrix --w-graph 2.0     # headline weight for the 4-config persist

The matrix:
  * persists FTS / +dense / +graph / +both result JSONs (every cell reproducible);
  * sweeps w_graph ∈ {0..5.0} for the graph-ON configs and reports the
    overall+per-stratum nDCG@10/recall@10 curve PLUS the graph-only-in-top10
    diagnostic per weight;
  * computes per-stratum paired-bootstrap (B=10000) CIs for +graph-vs-FTS,
    +both-vs-+dense (DECISIVE) and +dense-vs-FTS, plus the entity-bridged multihop
    sub-cut;
  * settles the verdict: promote the graph iff +both-vs-+dense multihop CI clears
    zero, else stay gated on a VALID test.

Outputs under results/ are LOCAL-ONLY (gitignored): they may echo retrieved ids
(never content), and derive from real personal memories. Docs/commits use
AGGREGATE numbers + synthetic examples only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import BenchmarkResult, load_dataset, metrics, run_benchmark  # noqa: E402
from harness.baselines import SqliteFtsRetriever  # noqa: E402
from harness.dataset import Dataset  # noqa: E402
from harness.example_retriever import SubstringRetriever  # noqa: E402
from harness.matrix import (  # noqa: E402
    CONFIG_NAMES,
    GRAPH_ONLY_K,
    W_GRAPH_SWEEP,
    apply_config,
    count_graph_only_in_topk,
    entity_bridged_multihop_ids,
    graph_verdict,
)
from harness.significance import paired_bootstrap_from_rows  # noqa: E402

ALIASES = {
    "fts5": lambda: SqliteFtsRetriever(sort_by="relevance"),
    "fts5_importance": lambda: SqliteFtsRetriever(sort_by="importance"),
    "substring": SubstringRetriever,
}

# Default headline weight for the persisted +graph/+both JSONs. The sweep explores
# the whole grid; this is just the single weight the 4 named result files are cut at
# (chosen near the boundary the challenger analysis identified). Overridable.
_DEFAULT_HEADLINE_W = 2.0

# The three comparisons the verdict rests on: (label, config_A, config_B). B−A.
_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("dense_vs_fts", "fts", "dense"),       # reconfirm the phase-1 dense win
    ("graph_vs_fts", "fts", "graph"),       # graph vs lexical
    ("both_vs_dense", "dense", "both"),     # DECISIVE: graph OVER dense
)
_STRATA = ("exact", "paraphrase", "multihop")
_SIG_SEED = 1234  # fixed → reproducible CIs


def resolve(spec: str):  # type: ignore[no-untyped-def]
    if spec in ALIASES:
        return ALIASES[spec]()
    if ":" not in spec:
        raise SystemExit(f"unknown retriever alias '{spec}' (use module:Class or one of {list(ALIASES)})")
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


# ── single-retriever mode (original behaviour) ──────────────────────────────

def run_single(args: argparse.Namespace) -> None:
    ds = load_dataset(validate=True)
    retr = resolve(args.retriever)
    res = run_benchmark(retr, ds, retrieve_k=args.k)
    print(res.summary())
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res.to_dict(), indent=2))
        print(f"\nwrote {args.json}")


# ── matrix mode (slice S5) ──────────────────────────────────────────────────

def _graph_build_memory_ids(ds: Dataset, graph_sample: int | None) -> tuple[set[int], int]:
    """Which memory ids feed the GRAPH build. The graph build's canonicalizer is
    O(distinct_forms × clusters) pure-Python, so the full 24k-form corpus vocabulary
    is wall-clock-prohibitive offline. When ``graph_sample`` is set we cap the build
    to that many memories — but ALWAYS include every qrels-target memory first, so
    each eval query still HAS a reachable target in the graph (a sampled-out target
    would make the graph trivially unable to help that query). The remainder is a
    seeded random fill for bridging context. Returns (ids, n_total_corpus).

    This is the brief's sanctioned "if time-bound, sample with a documented note";
    the eval still runs over the FULL corpus + FULL dense matrix — only the graph's
    coverage is sampled, and a query whose neighbours fall outside the sample simply
    gets no graph lift (an honest degrade, never a crash)."""
    import random

    all_ids = [m.id for m in ds.corpus]
    if graph_sample is None or graph_sample >= len(all_ids):
        return set(all_ids), len(all_ids)

    qrel_targets: set[int] = set()
    for rels in ds.qrels.values():
        qrel_targets |= rels
    chosen = set(qrel_targets)
    fill = [i for i in all_ids if i not in chosen]
    random.Random(1234).shuffle(fill)
    for i in fill:
        if len(chosen) >= graph_sample:
            break
        chosen.add(i)
    return chosen, len(all_ids)


def _build_equipped_retriever(ds: Dataset, *, extract: bool, graph_sample: int | None = None) -> object:
    """Build ONE fully-equipped HybridRetriever: FTS + cached bge-large dense matrix
    + (if a typed graph can be built) the injected typed concept graph. Reused across
    every config × weight by flipping leg weights only — so the expensive build runs
    once. The graph leg is wired only when triples are available (or --extract runs
    them); otherwise the graph-ON configs degrade to FTS-only and the verdict reports
    that the graph was unavailable.

    ``graph_sample`` caps the GRAPH build to that many memories (qrel targets always
    included) to keep the canonicalizer tractable; the FTS + dense legs always see
    the full corpus."""
    from retrievers.graph_build import build_concept_graph_fast
    from retrievers.graph_extract import default_haiku_extractor, extract_triples
    from retrievers.hybrid import HybridRetriever, _corpus_fingerprint

    r = HybridRetriever()
    t0 = time.perf_counter()
    r.build_index(ds.corpus)  # FTS + cached dense (no model load — S0 cache-hit)
    print(f"[matrix] built FTS+dense index in {time.perf_counter() - t0:.1f}s "
          f"(backend={r.embedding_backend}, errors={r.errors or 'none'})")

    fp = _corpus_fingerprint(ds.corpus)
    triples_path = Path(__file__).resolve().parents[1] / "cache" / f"triples_{fp}.jsonl"
    records = [
        {"id": m.id, "content": m.content, "category": m.category,
         "tags": m.tags, "importance": m.importance}
        for m in ds.corpus
    ]

    if extract or triples_path.exists():
        t0 = time.perf_counter()
        triples_by_mem = extract_triples(
            records, default_haiku_extractor, cache_path=triples_path
        )
        graph_ids, n_total = _graph_build_memory_ids(ds, graph_sample)
        triples_for_build = {mid: tr for mid, tr in triples_by_mem.items() if mid in graph_ids}
        n_triples = sum(len(v) for v in triples_for_build.values())
        cov = 100.0 * len(graph_ids) / n_total if n_total else 0.0
        print(f"[matrix] graph build over {len(graph_ids)}/{n_total} memories "
              f"({cov:.0f}% corpus coverage; all qrel targets included), "
              f"{n_triples} triples, loaded in {time.perf_counter() - t0:.1f}s "
              f"(cache: {triples_path.name})")
        # Build the typed graph with the SAME bge-large encoder the dense leg uses,
        # so concept surface forms canonicalise under the fixed model. The ~24k
        # distinct surface forms are NOT in the cached corpus matrix (that holds the
        # 5452 memory CONTENTS), so they must be embedded with the bge model — a
        # one-time ~minutes cost we CACHE to a gitignored .npy keyed by the ordered-
        # form hash, so a rerun of the matrix re-embeds NOTHING. Adapts the
        # retriever's numpy batch embedder to EmbedFn's list[list[float]].
        import hashlib as _hashlib

        import numpy as _np

        def _embed_forms(texts: list[str]) -> list[list[float]]:
            key = _hashlib.sha256(
                "\x00".join(texts).encode("utf-8", "replace")
            ).hexdigest()[:16]
            forms_cache = (
                Path(__file__).resolve().parents[1] / "cache" / f"forms_{fp}_{key}.npy"
            )
            if forms_cache.exists():
                mat = _np.load(forms_cache)
            else:
                mat = _np.asarray(r._embed_local(texts))  # type: ignore[attr-defined]
                _np.save(forms_cache, mat)
            return [list(map(float, row)) for row in mat]

        t0 = time.perf_counter()
        # numpy-vectorised build: the pure-Python single-linkage is O(forms×clusters)
        # interpreted cosines (~72s/1000 forms measured → hours on the full ~24k-form
        # vocabulary), wall-clock-prohibitive offline. build_concept_graph_fast is
        # byte-equivalent (pinned by test_graph_build) but matmul-vectorised, so the
        # FULL corpus builds without the sampling fallback.
        cgraph = build_concept_graph_fast(triples_for_build, _embed_forms)
        r.set_concept_graph(cgraph)
        stats = r.graph_stats()
        print(f"[matrix] built typed concept graph in {time.perf_counter() - t0:.1f}s: "
              f"{stats['typed_concepts']} concepts, {stats['typed_edges']} typed edges "
              f"(prior keyword prototype: 2,095,624 edges)")
    else:
        print(f"[matrix] no triples cache at {triples_path.name} and --extract not set: "
              "graph leg UNAVAILABLE; graph/both configs degrade to FTS-only. "
              "Run with --extract to build the graph.")
    return r


def _measure_graph_latency(retriever: object, ds: Dataset) -> dict[str, float]:
    """Per-query graph-leg (PPR) latency — non-gating, but the read-path PPR is on
    the production hot path so we MEASURE it (challenger note: ~2ms is unproven)."""
    import statistics

    from harness.runner import _percentile

    if getattr(retriever, "_cgraph", None) is None:
        return {}
    times: list[float] = []
    # seed the graph leg from each query's base fusion, time the graph rank only.
    for q in ds.queries:
        fts_ranked = retriever._fts.retrieve(q.text, 50)  # type: ignore[attr-defined]
        dense_ranked = retriever._dense_rank(q.text, 50)  # type: ignore[attr-defined]
        seeds = list(dict.fromkeys([*fts_ranked, *dense_ranked]))
        t0 = time.perf_counter()
        retriever._graph_rank(seeds, k=50)  # type: ignore[attr-defined]
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean": statistics.fmean(times) if times else 0.0,
        "p50": _percentile(times, 50),
        "p95": _percentile(times, 95),
        "max": max(times) if times else 0.0,
    }


class _NoRebuild:
    """Forward retrieve() to a PRE-BUILT retriever while HIDING build_index, so
    run_benchmark (which duck-types build_index and would otherwise rebuild the FTS
    + dense index on EVERY config eval) reuses the one already-built index. The
    shared-retriever optimisation is the whole point of mutating weights in place;
    without this each of the ~24 config evals would re-load the dense matrix and
    rebuild the keyword-fallback graph. The injected typed concept graph
    (set_concept_graph) is unaffected either way."""

    def __init__(self, inner: object, name: str) -> None:
        self._inner = inner
        self.name = name

    def retrieve(self, query: str, k: int) -> list[int]:
        return self._inner.retrieve(query, k)  # type: ignore[attr-defined,no-any-return]


def _eval_config(retriever: object, ds: Dataset, name: str, *, w_graph: float, k: int) -> BenchmarkResult:
    apply_config(retriever, name, w_graph=w_graph)
    # wrap so run_benchmark does NOT rebuild the shared index per config eval.
    return run_benchmark(_NoRebuild(retriever, name), ds, retrieve_k=k, retriever_name=name, warmup=False)


def _sweep(retriever: object, ds: Dataset, *, k: int) -> dict[str, Any]:
    """w_graph sweep over the graph-ON configs. For each weight report overall +
    per-stratum nDCG@10/recall@10 AND the graph-only-in-top10 diagnostic."""
    curve: list[dict[str, Any]] = []
    for w in W_GRAPH_SWEEP:
        row: dict[str, Any] = {"w_graph": w, "configs": {}}
        for name in ("graph", "both"):
            res = _eval_config(retriever, ds, name, w_graph=w, k=k)
            apply_config(retriever, name, w_graph=w)  # ensure weights set for diagnostic
            total, _per = count_graph_only_in_topk(retriever, ds.queries, k=GRAPH_ONLY_K)
            row["configs"][name] = {
                "overall": {m: res.overall[m] for m in ("ndcg@10", "recall@10")},
                "per_stratum": {
                    s: {m: res.per_stratum[s].metrics[m] for m in ("ndcg@10", "recall@10")}
                    for s in res.per_stratum
                },
                "graph_only_in_top10": total,
            }
        curve.append(row)
        g = row["configs"]["graph"]
        b = row["configs"]["both"]
        print(f"[sweep] w_graph={w:<4} graph nDCG@10={g['overall']['ndcg@10']:.4f} "
              f"(graph-only@10={g['graph_only_in_top10']})  "
              f"both nDCG@10={b['overall']['ndcg@10']:.4f} "
              f"(graph-only@10={b['graph_only_in_top10']})")
    return {"grid": list(W_GRAPH_SWEEP), "curve": curve}


def _significance(
    per_query_by_config: dict[str, list[dict[str, Any]]],
    entity_bridged: set[str],
) -> dict[str, Any]:
    """Per-stratum + overall + entity-bridged paired-bootstrap CIs for the three
    comparisons, on nDCG@10 (the headline metric) and recall@10."""
    out: dict[str, Any] = {}
    for label, a_name, b_name in _COMPARISONS:
        rows_a = per_query_by_config[a_name]
        rows_b = per_query_by_config[b_name]
        comp: dict[str, Any] = {}
        for metric in ("ndcg@10", "recall@10"):
            by_stratum: dict[str, Any] = {}
            for stratum in (None, *_STRATA):
                res = paired_bootstrap_from_rows(
                    rows_a, rows_b, metric=metric, stratum=stratum,
                    seed=_SIG_SEED, comparison=label,
                )
                by_stratum[stratum or "overall"] = res.to_dict()
            # entity-bridged multihop sub-cut: restrict to the bridged query ids.
            eb_a = [r for r in rows_a if r["query_id"] in entity_bridged]
            eb_b = [r for r in rows_b if r["query_id"] in entity_bridged]
            eb = paired_bootstrap_from_rows(
                eb_a, eb_b, metric=metric, stratum=None,
                seed=_SIG_SEED, comparison=label,
            )
            eb_dict = eb.to_dict()
            eb_dict["stratum"] = "multihop_entity_bridged"
            by_stratum["multihop_entity_bridged"] = eb_dict
            comp[metric] = by_stratum
        out[label] = comp
    return out


def run_matrix(args: argparse.Namespace) -> None:
    ds = load_dataset(validate=True)
    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] dataset: {len(ds.corpus)} memories, {len(ds.queries)} queries, "
          f"strata={ {s: sum(1 for q in ds.queries if q.stratum == s) for s in ds.strata()} }")

    graph_sample = getattr(args, "graph_sample", None)
    retriever = _build_equipped_retriever(ds, extract=args.extract, graph_sample=graph_sample)
    graph_available = getattr(retriever, "_cgraph", None) is not None
    graph_ids, n_total_corpus = _graph_build_memory_ids(ds, graph_sample)
    graph_coverage = len(graph_ids) / n_total_corpus if n_total_corpus else 0.0

    # 1) persist the four headline result JSONs at the headline weight.
    headline_w = args.w_graph
    per_query_by_config: dict[str, list[dict[str, Any]]] = {}
    summary_metrics: dict[str, Any] = {}
    for name in CONFIG_NAMES:
        res = _eval_config(retriever, ds, name, w_graph=headline_w, k=args.k)
        path = outdir / f"{name}.json"
        path.write_text(json.dumps(res.to_dict(), indent=2))
        per_query_by_config[name] = res.per_query
        summary_metrics[name] = {
            "overall": res.overall,
            "per_stratum": {s: res.per_stratum[s].metrics for s in res.per_stratum},
            "latency_ms": res.latency_ms,
        }
        print(f"[matrix] {name:<6} → {path.name}  "
              f"nDCG@10={res.overall['ndcg@10']:.4f} recall@10={res.overall['recall@10']:.4f} "
              f"recall@5={res.overall['recall@5']:.4f} mrr={res.overall['mrr']:.4f}")

    # 2) w_graph sweep + graph-only diagnostic.
    sweep = _sweep(retriever, ds, k=args.k) if graph_available else {"grid": list(W_GRAPH_SWEEP), "curve": []}
    graph_only_max = max(
        (c["configs"]["both"]["graph_only_in_top10"] for c in sweep["curve"]),
        default=0,
    )

    # 3) per-stratum + entity-bridged significance.
    entity_bridged = entity_bridged_multihop_ids(ds.queries)
    significance = _significance(per_query_by_config, entity_bridged)

    # 4) the verdict (decisive = +both-vs-+dense multihop on nDCG@10).
    decisive = paired_bootstrap_from_rows(
        per_query_by_config["dense"], per_query_by_config["both"],
        metric="ndcg@10", stratum="multihop", seed=_SIG_SEED, comparison="both_vs_dense",
    )
    verdict = graph_verdict(both_vs_dense_multihop=decisive, graph_only_total_max=graph_only_max)

    graph_latency = _measure_graph_latency(retriever, ds)

    summary = {
        "dataset": {
            "n_corpus": len(ds.corpus),
            "n_queries": len(ds.queries),
            "strata": {s: sum(1 for q in ds.queries if q.stratum == s) for s in ds.strata()},
            "n_entity_bridged_multihop": len(entity_bridged),
        },
        "config": {
            "headline_w_graph": headline_w,
            "retrieve_k": args.k,
            "graph_only_k": GRAPH_ONLY_K,
            "embedding_backend": getattr(retriever, "embedding_backend", "unknown"),
            "graph_available": graph_available,
            "graph_build_memories": len(graph_ids),
            "graph_build_corpus_coverage": round(graph_coverage, 4),
            "graph_stats": retriever.graph_stats() if graph_available else {},  # type: ignore[attr-defined]
            "graph_leg_latency_ms": graph_latency,
            "metric_names": list(metrics.METRIC_NAMES),
            "significance": {"method": "paired_bootstrap", "n_boot": 10_000, "ci": 0.95, "seed": _SIG_SEED},
        },
        "results": summary_metrics,
        "sweep": sweep,
        "significance": significance,
        "verdict": verdict,
    }
    summary_path = outdir / "matrix_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict['verdict']}")
    print("=" * 78)
    print(f"wrote {summary_path} (+ {', '.join(n + '.json' for n in CONFIG_NAMES)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    # matrix subcommand
    m = sub.add_parser("matrix", help="run the S5 4-config comparison matrix + verdict")
    m.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    m.add_argument("--k", type=int, default=20, help="retrieve depth per query")
    m.add_argument("--w-graph", dest="w_graph", type=float, default=_DEFAULT_HEADLINE_W,
                   help="headline graph weight for the four persisted JSONs")
    m.add_argument("--extract", action="store_true",
                   help="run (cached) haiku triple extraction to build the graph leg")
    m.add_argument("--graph-sample", dest="graph_sample", type=int, default=None,
                   help="cap the GRAPH build to N memories (qrel targets always included) to keep "
                        "the canonicalizer tractable; FTS+dense always see the full corpus")
    m.set_defaults(func=run_matrix)

    # default single-retriever mode (back-compat: no subcommand)
    ap.add_argument("--retriever", default="fts5")
    ap.add_argument("--k", type=int, default=20, help="depth requested from retriever")
    ap.add_argument("--json", type=Path, default=None, help="write full result JSON here")

    args = ap.parse_args()
    if args.cmd == "matrix":
        run_matrix(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
