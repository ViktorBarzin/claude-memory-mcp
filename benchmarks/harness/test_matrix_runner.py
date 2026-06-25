"""End-to-end (model-free, LLM-free) tests for the S5 matrix ORCHESTRATOR in
scripts/run_eval.py: the w_graph sweep, the per-stratum significance assembly, the
graph-leg latency probe, and the full run_matrix() persist + verdict path.

The retriever is fully stubbed (FTS+dense legs monkeypatched, a tiny REAL typed
concept graph injected via the S3 builder with a deterministic embedder), so the
orchestration glue is verified deterministically without loading bge-large or
calling haiku. This is what makes the script's logic — not just the helper modules
— covered.

Run:  .venv/bin/python -m pytest harness/test_matrix_runner.py -q
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from harness.dataset import Dataset
from harness.matrix import W_GRAPH_SWEEP, apply_config
from harness.types import Memory, Query
from retrievers.graph_build import ConceptGraph, build_concept_graph
from retrievers.hybrid import HybridRetriever

# Import the script-under-test as a module (scripts/ is not a package).
_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval_under_test", _SCRIPT)
assert _spec is not None and _spec.loader is not None
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


def _embed_stub(texts: list[str]) -> list[list[float]]:
    """Orthogonal-axis embedder so each distinct surface form is its own concept
    (no over-merge) — enough to drive PPR deterministically."""
    axes = {"viktor": 0, "svelte": 1, "postgres": 2, "redis": 3, "kafka": 4}
    out: list[list[float]] = []
    for t in texts:
        v = [0.0] * 5
        v[axes.get(t.strip().lower(), sum(ord(c) for c in t.lower()) % 5)] = 1.0
        out.append(v)
    return out


def _typed_graph() -> ConceptGraph:
    """Memories 200 and 300 both mention 'Svelte'; 400 mentions only 'Kafka'. So
    300 is one concept hop from 200 (a graph-only bridge if 300 is in no base leg)."""
    triples = {
        200: [("Viktor", "prefers", "Svelte")],
        300: [("Svelte", "runs-on", "Postgres")],
        400: [("service", "uses", "Kafka")],
    }
    return build_concept_graph(triples, _embed_stub, threshold=0.9)


def _dataset() -> Dataset:
    """Tiny eval set spanning all three strata. Memory 300 is the multihop target
    that ONLY the graph can bridge to (it is in neither base leg, see _stub_legs)."""
    corpus = [Memory(id=i, content=f"m{i}", importance=0.5) for i in (1, 2, 3, 200, 300, 400)]
    queries = [
        Query(query_id="ex1", text="exact-q", stratum="exact", relevant_ids=(1,)),
        Query(query_id="pp1", text="para-q", stratum="paraphrase", relevant_ids=(2,)),
        # entity-bridged multihop: target 300 sits far from the base hits → bridged
        Query(query_id="mh1", text="multi-q", stratum="multihop", relevant_ids=(300,)),
    ]
    qrels = {q.query_id: set(q.relevant_ids) for q in queries}
    return Dataset(corpus=corpus, queries=queries, qrels=qrels)


def _equipped_retriever() -> HybridRetriever:
    """A retriever whose base legs are stubbed and whose graph leg is a real typed
    graph. FTS+dense return {1,2,200} (never 300); the graph bridges 200→300, so at
    a high w_graph the multihop target 300 becomes a graph-only top-k hit."""
    r = HybridRetriever()
    r.set_concept_graph(_typed_graph())
    base = [1, 2, 200]
    r._fts.retrieve = lambda q, k: list(base[:k])  # type: ignore[method-assign]
    r._dense_rank = lambda q, k: list(base[:k])  # type: ignore[method-assign]
    return r


# ---------------- sweep ----------------

def test_sweep_covers_full_grid_with_diagnostic() -> None:
    r = _equipped_retriever()
    ds = _dataset()
    sweep = run_eval._sweep(r, ds, k=20)
    assert sweep["grid"] == list(W_GRAPH_SWEEP)
    assert len(sweep["curve"]) == len(W_GRAPH_SWEEP)
    # each cell reports both graph-ON configs with overall + per-stratum + diagnostic
    for cell in sweep["curve"]:
        assert set(cell["configs"]) == {"graph", "both"}
        for cfg in cell["configs"].values():
            assert "ndcg@10" in cfg["overall"] and "recall@10" in cfg["overall"]
            assert "graph_only_in_top10" in cfg
    # at w_graph=0 nothing graph-only can enter; at the top of the sweep the
    # bridged target 300 IS surfaced → the diagnostic climbs above zero.
    at_zero = sweep["curve"][0]["configs"]["both"]["graph_only_in_top10"]
    at_top = sweep["curve"][-1]["configs"]["both"]["graph_only_in_top10"]
    assert at_zero == 0
    assert at_top >= 1, "the swept-up graph must lift the bridged-only target into top-10"


# ---------------- significance assembly ----------------

def test_significance_has_all_comparisons_strata_and_entity_cut() -> None:
    r = _equipped_retriever()
    ds = _dataset()
    # produce per_query rows for each config at a high weight (so the graph fires)
    from harness import run_benchmark

    per_query_by_config = {}
    for name in ("fts", "dense", "graph", "both"):
        apply_config(r, name, w_graph=5.0)
        res = run_benchmark(r, ds, retrieve_k=20, retriever_name=name, warmup=False)
        per_query_by_config[name] = res.per_query

    from harness.matrix import entity_bridged_multihop_ids

    eb = entity_bridged_multihop_ids(ds.queries)
    sig = run_eval._significance(per_query_by_config, eb)

    # all three comparisons present
    assert set(sig) == {"dense_vs_fts", "graph_vs_fts", "both_vs_dense"}
    for comp in sig.values():
        assert set(comp) == {"ndcg@10", "recall@10"}
        for metric_block in comp.values():
            # overall + 3 strata + the entity-bridged sub-cut
            assert "overall" in metric_block
            assert {"exact", "paraphrase", "multihop"} <= set(metric_block)
            assert "multihop_entity_bridged" in metric_block
            # each cell is a serialised BootstrapResult
            cell = metric_block["overall"]
            assert {"delta", "ci_low", "ci_high", "p_le_zero", "n", "significant"} <= set(cell)


# ---------------- graph-leg latency probe ----------------

def test_measure_graph_latency_reports_when_graph_present() -> None:
    r = _equipped_retriever()
    ds = _dataset()
    lat = run_eval._measure_graph_latency(r, ds)
    assert set(lat) == {"mean", "p50", "p95", "max"}
    assert all(v >= 0.0 for v in lat.values())


def test_measure_graph_latency_empty_without_graph() -> None:
    r = HybridRetriever()  # no typed graph injected
    assert run_eval._measure_graph_latency(r, _dataset()) == {}


# ---------------- full run_matrix persist + verdict ----------------

class _Args:
    def __init__(self, outdir: Path) -> None:
        self.outdir = outdir
        self.k = 20
        self.w_graph = 5.0
        self.extract = False
        self.graph_sample = None


def test_run_matrix_persists_four_jsons_and_summary(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """run_matrix writes fts/dense/graph/both JSONs + matrix_summary.json with a
    verdict, on a stubbed retriever (no model, no LLM). We patch the dataset loader
    and the equipped-retriever builder to the tiny stubbed pair."""
    ds = _dataset()
    monkeypatch.setattr(run_eval, "load_dataset", lambda validate=True: ds)
    monkeypatch.setattr(
        run_eval,
        "_build_equipped_retriever",
        lambda dataset, extract, graph_sample=None: _equipped_retriever(),
    )

    run_eval.run_matrix(_Args(tmp_path))

    for name in ("fts", "dense", "graph", "both"):
        p = tmp_path / f"{name}.json"
        assert p.exists()
        doc = json.loads(p.read_text())
        assert doc["retriever_name"] == name
        assert "overall" in doc and "per_query" in doc

    summary = json.loads((tmp_path / "matrix_summary.json").read_text())
    assert summary["config"]["graph_available"] is True
    assert summary["dataset"]["n_queries"] == 3
    # sweep + significance + verdict all present
    assert summary["sweep"]["grid"] == list(W_GRAPH_SWEEP)
    assert "both_vs_dense" in summary["significance"]
    v = summary["verdict"]
    assert {"promote", "graph_competed", "verdict"} <= set(v)
    # the bridged target 300 reaches top-10 under the swept graph → graph competed
    assert v["graph_competed"] is True
