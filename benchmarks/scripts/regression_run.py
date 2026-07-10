#!/usr/bin/env python3
"""POST-CLEANUP retrieval-regression gate: re-run the benchmark retrievers against a
NEW store snapshot + the PRESERVED 119-query eval set, and PASS/FAIL against the
stored baseline numbers.

    .venv/bin/python scripts/regression_run.py --snapshot 2026-07-10 \
        --bridge data/corpus.jsonl --id-map ../cleanup-report.json \
        --json results/regression-2026-07-10.json

``--bridge`` translates the preserved eval set's gold ids into the snapshot's id
space by exact content match — REQUIRED for API snapshots: the preserved set carries
local-SQLite ids, the API store remote ids (verified 2026-07-10: 0/5,452 id-stable,
137/139 gold ids content-bridgeable). See POST_CLEANUP_GATE.md.

Retrievers (the matrix configs the baseline numbers were cut at):
  * ``fts``   — retrievers.fts.FtsRetriever, the faithful production lexical path
                (always runs; stdlib sqlite only).
  * ``dense`` — the +dense config (HybridRetriever, w_graph=0) under the SAME fixed
                local bge-large-en-v1.5 model the baseline held fixed. Runs only when
                the model is already in the local HF cache (``--retrievers auto``);
                a changed corpus re-embeds once on CPU (~minutes) and caches to
                ``cache/`` keyed by the new corpus fingerprint. Hosted-API keys are
                deliberately ignored: the baseline is bge-large, and the whole store
                must not transit a hosted embedding API for a validation run.

Gold-id mapping (ADR-0007): the cleanup rewrites the store, superseding/merging
entries — a preserved gold id may now be a tombstone whose SUCCESSOR is the right
answer. ``--id-map FILE`` (emitted by ``store_cleanup.py --report``) is applied in
BOTH directions of the eval:
  * gold ids in qrels/queries are remapped old→successor (chains collapse; a cycle
    is an error in the report);
  * retrieved ids are redirected old→successor before scoring — emulating the
    production supersedes-redirect ("the successor is served in place of the old
    memory whenever the old one would rank"), so a retriever surfacing the tombstone
    is scored exactly as production would deliver it.

Exit codes: 0 = PASS, 1 = regression beyond threshold, 2 = usage/data error.
Outputs under results/ stay LOCAL-ONLY (gitignored) like every other run artifact.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

_BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH_ROOT))

from harness.dataset import Dataset, load_corpus, load_qrels, load_queries  # noqa: E402
from harness.runner import run_benchmark  # noqa: E402
from harness.types import MemoryId, Query  # noqa: E402

DEFAULT_SNAPSHOT_ROOT = _BENCH_ROOT / "snapshots"
DEFAULT_QUERIES = _BENCH_ROOT / "data" / "queries.jsonl"
DEFAULT_QRELS = _BENCH_ROOT / "data" / "qrels.jsonl"

# The gate metric + default threshold (task brief): fail when overall recall@10
# drops MORE THAN this vs baseline.
GATE_METRIC = "recall@10"
DEFAULT_THRESHOLD = 0.02

# Comparison metrics reported per slice (overall + per stratum).
_COMPARE_METRICS = ("recall@10", "ndcg@10")

# The fixed local embedding model the baseline was produced with (hybrid-build-report:
# "Held fixed so dense-vs-graph is the only variable: bge-large-en-v1.5").
_BGE_HUB_DIRNAME = "models--BAAI--bge-large-en-v1.5"

BASELINE_SOURCE = (
    "docs/research/hybrid-build-report.md §2 — preserved eval set "
    "(corpus fingerprint ca7b1d4ed22672e8, 5,452 memories / 119 queries)"
)

# The stored baseline numbers: the committed aggregate table from the build report.
# ``fts`` is the lexical production path; ``dense`` is the +dense config (w_graph=0)
# under fixed bge-large. Same shape as this script's --json "results" entries, so a
# previous regression_run output can be passed via --baseline to chain gates.
DEFAULT_BASELINES: dict[str, dict[str, Any]] = {
    "fts": {
        "overall": {"recall@10": 0.6952, "ndcg@10": 0.6507},
        "per_stratum": {
            "exact": {"recall@10": 1.0000, "ndcg@10": 0.9908},
            "paraphrase": {"recall@10": 0.3750, "ndcg@10": 0.3123},
            "multihop": {"recall@10": 0.7111, "ndcg@10": 0.6491},
        },
    },
    "dense": {
        "overall": {"recall@10": 0.8338, "ndcg@10": 0.7284},
        "per_stratum": {
            "exact": {"recall@10": 1.0000, "ndcg@10": 0.9723},
            "paraphrase": {"recall@10": 0.7250, "ndcg@10": 0.5023},
            "multihop": {"recall@10": 0.7748, "ndcg@10": 0.7101},
        },
    },
}


# ── id-map: loading, resolution, application (pure) ──────────────────────────


def _pairs_from_json(data: Any) -> list[tuple[Any, Any]] | None:
    """Extract (old, new) pairs from the tolerated JSON shapes, or None."""
    if isinstance(data, Mapping):
        # a full cleanup report: the map may sit under a conventional key.
        for key in ("id_map", "superseded_by", "supersessions"):
            if key in data:
                return _pairs_from_json(data[key])
        return list(data.items())
    if isinstance(data, list):
        pairs: list[tuple[Any, Any]] = []
        for item in data:
            if isinstance(item, Mapping) and "old" in item and "new" in item:
                pairs.append((item["old"], item["new"]))
            elif isinstance(item, Sequence) and not isinstance(item, str) and len(item) == 2:
                pairs.append((item[0], item[1]))
            else:
                return None
        return pairs
    return None


def load_id_map(path: Path) -> dict[int, int]:
    """Load an old→new id map from ``store_cleanup.py --report`` output.

    Accepted shapes: a flat JSON object ``{"old": new, ...}``; a JSON list of
    ``[old, new]`` pairs or ``{"old":…, "new":…}`` objects; either of those nested
    under an ``id_map`` / ``superseded_by`` / ``supersessions`` key of a larger
    report; or plain text lines ``old new`` / ``old,new`` (``#`` comments ignored).
    """
    text = path.read_text(encoding="utf-8")
    try:
        pairs = _pairs_from_json(json.loads(text))
    except json.JSONDecodeError:
        pairs = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 2:
                pairs = None
                break
            pairs.append((parts[0], parts[1]))
    if pairs is None:
        raise ValueError(f"unrecognised id-map shape in {path}")
    try:
        return {int(old): int(new) for old, new in pairs}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-integer id in id-map {path}: {exc}") from exc


def resolve_id_map(raw: Mapping[int, int]) -> dict[int, int]:
    """Collapse supersession chains to the FINAL successor (a→b, b→c ⇒ a→c, b→c).

    Identity entries are dropped; a cycle means the cleanup report is inconsistent
    and raises. The result maps every superseded id directly to its living successor.
    """
    resolved: dict[int, int] = {}
    for start in raw:
        seen = {start}
        cur = start
        while cur in raw and raw[cur] != cur:
            cur = raw[cur]
            if cur in seen:
                raise ValueError(f"cycle in id-map involving id {cur}")
            seen.add(cur)
        if cur != start:
            resolved[start] = cur
    return resolved


def apply_id_map_to_qrels(
    qrels: Mapping[str, set[MemoryId]], id_map: Mapping[int, int]
) -> dict[str, set[MemoryId]]:
    """Remap gold ids to their successors (duplicates merge via set semantics)."""
    return {qid: {id_map.get(mid, mid) for mid in rels} for qid, rels in qrels.items()}


def apply_id_map_to_queries(queries: Sequence[Query], id_map: Mapping[int, int]) -> list[Query]:
    """Remap the queries' convenience relevant_ids copies (qrels stay authoritative)."""
    return [
        Query(
            query_id=q.query_id,
            text=q.text,
            stratum=q.stratum,
            relevant_ids=tuple(dict.fromkeys(id_map.get(mid, mid) for mid in q.relevant_ids)),
        )
        for q in queries
    ]


def redirect_ranked(ranked: Sequence[MemoryId], id_map: Mapping[int, int]) -> list[MemoryId]:
    """Emulate the ADR-0007 supersedes-redirect on a retrieved ranking.

    Each superseded id is replaced by its successor AT ITS RANK ("served in place
    of"); if the successor also appears later, the first occurrence wins (dedup
    keeping best rank). Ids outside the map pass through unchanged.
    """
    out: list[MemoryId] = []
    seen: set[MemoryId] = set()
    for mid in ranked:
        target = id_map.get(mid, mid)
        if target not in seen:
            seen.add(target)
            out.append(target)
    return out


# ── content bridge between id spaces (pure) ──────────────────────────────────
#
# Verified live (2026-07-10): the PRESERVED eval set carries LOCAL-SQLite ids while
# API snapshots — and the cleanup that rewrites the API store, so its report too —
# carry REMOTE ids. 0/5,452 preserved (id, content) pairs matched the live store,
# yet 5,340 contents matched verbatim under a different id (137/139 gold ids).
# Scoring an API snapshot against the preserved qrels therefore REQUIRES this
# bridge; without it 70/119 queries reference absent ids and the remaining 49
# would silently score against colliding, unrelated rows.


def build_content_bridge(
    reference: Sequence[Any], snapshot: Sequence[Any]
) -> tuple[dict[int, tuple[int, ...]], dict[str, int]]:
    """Map reference-corpus ids → snapshot ids by EXACT (stripped) content match.

    Exact-only on purpose: a fuzzy bridge could silently attach gold judgments to
    the wrong memory. Content shared by several snapshot rows (near-duplicate
    twins) maps to ALL of them — the qrels twin precedent: any twin satisfies the
    query. Reference ids with no match are absent from the bridge (reported).
    """
    by_content: dict[str, list[int]] = {}
    for m in snapshot:
        by_content.setdefault(m.content.strip(), []).append(m.id)
    bridge: dict[int, tuple[int, ...]] = {}
    ambiguous = 0
    unbridged = 0
    for m in reference:
        hits = by_content.get(m.content.strip())
        if hits:
            bridge[m.id] = tuple(sorted(hits))
            if len(hits) > 1:
                ambiguous += 1
        else:
            unbridged += 1
    report = {
        "reference_ids": len(reference),
        "bridged": len(bridge),
        "ambiguous": ambiguous,
        "unbridged": unbridged,
    }
    return bridge, report


def apply_bridge_to_qrels(
    qrels: Mapping[str, set[MemoryId]], bridge: Mapping[int, tuple[int, ...]]
) -> tuple[dict[str, set[MemoryId]], dict[str, set[MemoryId]]]:
    """Translate gold ids through the bridge (twins EXPAND the gold set).

    Application is TOTAL: an unbridged gold id is dropped and reported — never
    passed through raw, where it could collide with an unrelated snapshot id and
    silently score the wrong row. Returns (bridged_qrels, unbridged_by_query).
    """
    out: dict[str, set[MemoryId]] = {}
    unbridged: dict[str, set[MemoryId]] = {}
    for qid, rels in qrels.items():
        mapped: set[MemoryId] = set()
        miss: set[MemoryId] = set()
        for mid in rels:
            targets = bridge.get(mid)
            if targets is None:
                miss.add(mid)
            else:
                mapped.update(targets)
        out[qid] = mapped
        if miss:
            unbridged[qid] = miss
    return out, unbridged


def apply_bridge_to_queries(
    queries: Sequence[Query], bridge: Mapping[int, tuple[int, ...]]
) -> list[Query]:
    """Translate the queries' convenience relevant_ids copies (unbridged ids drop)."""
    out: list[Query] = []
    for q in queries:
        mapped: list[MemoryId] = []
        for mid in q.relevant_ids:
            mapped.extend(bridge.get(mid, ()))
        out.append(
            Query(
                query_id=q.query_id,
                text=q.text,
                stratum=q.stratum,
                relevant_ids=tuple(dict.fromkeys(mapped)),
            )
        )
    return out


# ── gold coverage against the snapshot corpus (pure) ─────────────────────────


def gold_coverage(
    qrels: Mapping[str, set[MemoryId]], corpus_ids: set[MemoryId]
) -> dict[str, set[MemoryId]]:
    """query_id → gold ids MISSING from the corpus (only non-empty entries)."""
    missing = {qid: rels - corpus_ids for qid, rels in qrels.items()}
    return {qid: m for qid, m in missing.items() if m}


def prune_missing_gold(
    qrels: Mapping[str, set[MemoryId]], corpus_ids: set[MemoryId]
) -> tuple[dict[str, set[MemoryId]], set[str], dict[str, set[MemoryId]]]:
    """Drop missing gold ids; a query whose gold set empties is dropped entirely.

    Returns (pruned_qrels, dropped_query_ids, missing_report).
    """
    report = gold_coverage(qrels, corpus_ids)
    pruned: dict[str, set[MemoryId]] = {}
    dropped: set[str] = set()
    for qid, rels in qrels.items():
        kept = rels & corpus_ids
        if kept:
            pruned[qid] = kept
        else:
            dropped.add(qid)
    return pruned, dropped, report


# ── retriever construction ────────────────────────────────────────────────────


class RedirectingRetriever:
    """Wrap a retriever so its rankings pass through the supersedes-redirect.

    Forwards the ``build_index`` / ``index_size_bytes`` lifecycle hooks (unlike the
    matrix's _NoRebuild wrapper, the index HAS to be built here). Pulls extra depth
    so post-redirect dedup can still fill k results.
    """

    def __init__(self, inner: object, id_map: Mapping[int, int], name: str) -> None:
        self._inner = inner
        self._id_map = dict(id_map)
        self.name = name

    def build_index(self, corpus: Sequence[Any]) -> None:
        build = getattr(self._inner, "build_index", None)
        if callable(build):
            build(corpus)

    def index_size_bytes(self) -> int:
        size = getattr(self._inner, "index_size_bytes", None)
        return int(size()) if callable(size) else 0

    def retrieve(self, query: str, k: int) -> list[MemoryId]:
        # Over-fetch so redirect dedup (which only ever shrinks the list) can still
        # fill k. Capped at 2k: that keeps the hybrid's per-leg fusion depth at the
        # baseline run's max(k, 50) for the standard retrieve_k=20, so the fused
        # ordering stays comparable to the stored baseline numbers.
        raw = self._inner.retrieve(query, k + min(len(self._id_map), k))  # type: ignore[attr-defined]
        return redirect_ranked(raw, self._id_map)[:k]


def _hf_hub_cache_dir(env: Mapping[str, str]) -> Path:
    """Resolve the huggingface hub cache dir the way huggingface_hub does."""
    if env.get("HF_HUB_CACHE"):
        return Path(env["HF_HUB_CACHE"])
    if env.get("HF_HOME"):
        return Path(env["HF_HOME"]) / "hub"
    return Path(env.get("HOME", str(Path.home()))) / ".cache" / "huggingface" / "hub"


def _bge_in_hub_cache(env: Mapping[str, str]) -> bool:
    return (_hf_hub_cache_dir(env) / _BGE_HUB_DIRNAME).is_dir()


def local_bge_available(env: Mapping[str, str] | None = None) -> bool:
    """True when the dense leg can run OFFLINE: sentence-transformers is importable
    and bge-large-en-v1.5 is already in the local HF hub cache (no download)."""
    import importlib.util as _ilu

    if _ilu.find_spec("sentence_transformers") is None:
        return False
    return _bge_in_hub_cache(os.environ if env is None else env)


def _build_fts_retriever() -> object:
    from retrievers.fts import FtsRetriever

    return FtsRetriever(sort_by="relevance")


def _build_dense_retriever() -> object:
    """The +dense matrix config: FTS + dense at full weight, graph ablated to 0.

    Pins the LOCAL bge-large backend by scrubbing hosted-API keys from THIS process:
    the baseline held the model fixed, and a validation run must not ship the whole
    store to a hosted embedding API.
    """
    for key in ("VOYAGE_API_KEY", "OPENAI_API_KEY", "CO_API_KEY"):
        if os.environ.pop(key, None) is not None:
            print(f"[regression] ignoring {key}: dense leg pinned to local bge-large "
                  "(the fixed baseline model)")
    from retrievers.hybrid import HybridRetriever

    r = HybridRetriever()
    r.w_fts = 1.0
    r.w_dense = 1.0
    r.w_graph = 0.0
    return r


_RETRIEVER_BUILDERS = {"fts": _build_fts_retriever, "dense": _build_dense_retriever}


def select_retrievers(spec: str) -> list[str]:
    """Parse --retrievers: 'auto' → fts (+dense when locally available), or an
    explicit comma list of fts/dense (dense forced even if the model needs a
    download)."""
    if spec == "auto":
        names = ["fts"]
        if local_bge_available():
            names.append("dense")
        else:
            print("[regression] dense leg SKIPPED: sentence-transformers or the cached "
                  f"bge-large model ({_BGE_HUB_DIRNAME}) is not available locally. "
                  "Pass --retrievers fts,dense to force (may download the model).")
        return names
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in _RETRIEVER_BUILDERS]
    if unknown:
        raise ValueError(f"unknown retriever(s) {unknown}; choose from {sorted(_RETRIEVER_BUILDERS)}")
    return names


# ── the regression run ────────────────────────────────────────────────────────


def run_regression(
    *,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    id_map: Mapping[int, int] | None = None,
    bridge_corpus_path: Path | None = None,
    retrievers: Iterable[str] = ("fts",),
    retrieve_k: int = 20,
    redirect: bool = True,
    drop_missing_gold: bool = False,
) -> dict[str, Any]:
    """Evaluate the named retrievers on (snapshot corpus × preserved queries).

    When ``bridge_corpus_path`` is given (the reference corpus the queries/qrels
    were built against — for the preserved set, ``data/corpus.jsonl``), gold ids are
    first translated into the snapshot's id space by exact content match. The
    resolved id-map is then applied to the gold side (qrels + queries) and — when
    ``redirect`` — to every retrieved ranking (the ADR-0007 supersedes-redirect
    emulation; the id-map is in the SNAPSHOT id space). Gold ids still missing from
    the snapshot after both steps are a hard error unless ``drop_missing_gold``
    (they mean the cleanup deleted a gold memory or the id-map is incomplete).
    """
    resolved = resolve_id_map(id_map or {})

    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)

    bridge_report: dict[str, int] | None = None
    unbridged: dict[str, set[MemoryId]] = {}
    if bridge_corpus_path is not None:
        bridge, bridge_report = build_content_bridge(load_corpus(bridge_corpus_path), corpus)
        qrels, unbridged = apply_bridge_to_qrels(qrels, bridge)
        queries = apply_bridge_to_queries(queries, bridge)

    queries = apply_id_map_to_queries(queries, resolved)
    qrels = apply_id_map_to_qrels(qrels, resolved)

    corpus_ids = {m.id for m in corpus}
    missing = gold_coverage(qrels, corpus_ids)
    for qid, ids in unbridged.items():  # unbridged golds are missing too
        missing.setdefault(qid, set()).update(ids)
    dropped: set[str] = set()
    if missing:
        if not drop_missing_gold:
            detail = ", ".join(
                f"{qid}→{sorted(m)}" for qid, m in sorted(missing.items())
            )
            hint = (
                "Extend the id-map (superseded gold ids must map to their successors) "
                "or pass --drop-missing-gold."
            )
            if bridge_corpus_path is None:
                hint += (
                    " If the eval set's gold ids live in a DIFFERENT id space than the "
                    "snapshot (the preserved set carries local-SQLite ids; API snapshots "
                    "carry remote ids), pass --bridge <reference corpus.jsonl>."
                )
            raise ValueError(
                f"{len(missing)} queries reference gold ids ABSENT from the snapshot "
                f"after bridge/id-map application: {detail}. {hint}"
            )
        qrels, dropped, missing = prune_missing_gold(qrels, corpus_ids)
        for qid, ids in unbridged.items():  # keep unbridged golds in the report
            missing.setdefault(qid, set()).update(ids)
        queries = [q for q in queries if q.query_id not in dropped]

    dataset = Dataset(corpus=corpus, queries=queries, qrels=qrels)

    results: dict[str, dict[str, Any]] = {}
    for name in retrievers:
        retriever = _RETRIEVER_BUILDERS[name]()
        if redirect and resolved:
            retriever = RedirectingRetriever(retriever, resolved, name)
        res = run_benchmark(retriever, dataset, retrieve_k=retrieve_k, retriever_name=name)
        results[name] = {
            "overall": res.overall,
            "per_stratum": {s: sr.metrics for s, sr in res.per_stratum.items()},
            "latency_ms": res.latency_ms,
            "n_queries": res.n_queries,
            "index_build_seconds": res.index_build_seconds,
        }
        if name == "dense":
            emb = getattr(getattr(retriever, "_inner", retriever), "_emb", None)
            errors = getattr(getattr(retriever, "_inner", retriever), "errors", [])
            if emb is None:
                raise RuntimeError(
                    f"dense leg did not build (errors: {errors or 'unknown'}); refusing "
                    "to report lexical-only numbers under the 'dense' label"
                )

    return {
        "dataset": {
            "corpus_path": str(corpus_path),
            "n_corpus": len(corpus),
            "n_queries": len(queries),
            "strata": {s: sum(1 for q in queries if q.stratum == s) for s in dataset.strata()},
        },
        "bridge": bridge_report,
        "id_map": {
            "entries": len(resolved),
            "redirect_emulated": bool(redirect and resolved),
        },
        "gold": {
            "missing_after_map": {qid: sorted(m) for qid, m in sorted(missing.items())},
            "dropped_queries": sorted(dropped),
        },
        "results": results,
    }


# ── baseline comparison + gate (pure) ─────────────────────────────────────────


def compare_to_baseline(
    results: Mapping[str, Mapping[str, Any]],
    baselines: Mapping[str, Mapping[str, Any]],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, Any]:
    """Compare each retriever's numbers to its baseline; gate on overall recall@10.

    The gate: FAIL iff any evaluated retriever WITH a baseline has
    ``current − baseline < −threshold`` on overall recall@10 (a drop of exactly the
    threshold still passes — the brief gates on "drops >threshold"). Per-slice
    recall@10/nDCG@10 deltas are reported for the table but do not gate.
    """
    retrievers: dict[str, Any] = {}
    overall_pass = True
    for name, res in results.items():
        base = baselines.get(name)
        cells: list[dict[str, Any]] = []
        if base is not None:
            slices: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = [
                ("overall", res["overall"], base["overall"])
            ]
            base_strata = base.get("per_stratum", {})
            for stratum in sorted(res.get("per_stratum", {})):
                if stratum in base_strata:
                    slices.append((stratum, res["per_stratum"][stratum], base_strata[stratum]))
            for slice_name, cur_m, base_m in slices:
                for metric in _COMPARE_METRICS:
                    if metric not in cur_m or metric not in base_m:
                        continue
                    cells.append(
                        {
                            "slice": slice_name,
                            "metric": metric,
                            "baseline": base_m[metric],
                            "current": cur_m[metric],
                            "delta": cur_m[metric] - base_m[metric],
                        }
                    )
            delta = res["overall"][GATE_METRIC] - base["overall"][GATE_METRIC]
            gate = {
                "metric": GATE_METRIC,
                "baseline": base["overall"][GATE_METRIC],
                "current": res["overall"][GATE_METRIC],
                "delta": delta,
                "threshold": threshold,
            }
            passed = delta >= -threshold
            overall_pass = overall_pass and passed
        else:
            gate = None
            passed = True  # nothing to gate against; reported, not gating
        retrievers[name] = {"passed": passed, "gate": gate, "cells": cells}
    return {"passed": overall_pass, "threshold": threshold, "retrievers": retrievers}


def format_comparison_table(comparison: Mapping[str, Any]) -> str:
    """Human-readable comparison table + per-retriever and overall verdicts."""
    lines: list[str] = []
    header = f"{'retriever':<10}{'slice':<14}{'metric':<11}{'baseline':>10}{'current':>10}{'delta':>9}"
    lines.append(header)
    lines.append("-" * len(header))
    for name, entry in comparison["retrievers"].items():
        for cell in entry["cells"]:
            lines.append(
                f"{name:<10}{cell['slice']:<14}{cell['metric']:<11}"
                f"{cell['baseline']:>10.4f}{cell['current']:>10.4f}{cell['delta']:>+9.4f}"
            )
        gate = entry["gate"]
        if gate is None:
            lines.append(f"{name:<10}no stored baseline — reported only, not gating")
        else:
            verdict = "PASS" if entry["passed"] else "FAIL"
            lines.append(
                f"{name:<10}gate: overall {gate['metric']} {gate['current']:.4f} vs "
                f"{gate['baseline']:.4f} (Δ{gate['delta']:+.4f}, allowed drop "
                f"{gate['threshold']:.2f}) → {verdict}"
            )
        lines.append("")
    lines.append(f"GATE: {'PASS' if comparison['passed'] else 'FAIL'}")
    return "\n".join(lines)


def load_baselines(path: Path) -> dict[str, Any]:
    """Load baselines from a previous regression_run --json output (its "results"
    key), or a bare ``{retriever: {overall, per_stratum}}`` object of the same shape."""
    data = json.loads(path.read_text(encoding="utf-8"))
    baselines = data.get("results", data)
    for name, entry in baselines.items():
        if "overall" not in entry:
            raise ValueError(f"baseline entry {name!r} in {path} has no 'overall' metrics")
    return baselines


# ── CLI ───────────────────────────────────────────────────────────────────────


def _resolve_snapshot(spec: str) -> Path:
    """--snapshot accepts a dir path, a corpus.jsonl path, or a name under
    benchmarks/snapshots/."""
    p = Path(spec)
    for candidate in (p, DEFAULT_SNAPSHOT_ROOT / spec):
        if candidate.is_file():
            return candidate
        if (candidate / "corpus.jsonl").is_file():
            return candidate / "corpus.jsonl"
    raise ValueError(
        f"snapshot {spec!r} not found (looked for {p} and {DEFAULT_SNAPSHOT_ROOT / spec}); "
        "create one with scripts/snapshot_corpus.py"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True,
                    help="snapshot name under snapshots/, or a dir/corpus.jsonl path")
    ap.add_argument("--queries", type=Path, default=DEFAULT_QUERIES,
                    help="eval queries (default: the preserved 119-query set)")
    ap.add_argument("--qrels", type=Path, default=DEFAULT_QRELS,
                    help="eval qrels (default: the preserved set)")
    ap.add_argument("--id-map", type=Path, default=None,
                    help="old→new id map from store_cleanup.py --report (superseded gold "
                         "ids → successors; also drives the supersedes-redirect emulation)")
    ap.add_argument("--bridge", type=Path, default=None,
                    help="reference corpus.jsonl the queries/qrels were built against "
                         "(data/corpus.jsonl for the preserved set): gold ids are translated "
                         "into the snapshot's id space by exact content match — REQUIRED when "
                         "scoring an API snapshot with the preserved (local-SQLite-id) eval set")
    ap.add_argument("--no-redirect", action="store_true",
                    help="do NOT emulate the supersedes-redirect on retrieved rankings")
    ap.add_argument("--drop-missing-gold", action="store_true",
                    help="drop gold ids (and emptied queries) absent from the snapshot "
                         "instead of erroring")
    ap.add_argument("--retrievers", default="auto",
                    help="'auto' (fts + dense-if-cached) or a comma list of fts,dense")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"max allowed overall {GATE_METRIC} drop vs baseline "
                         f"(default {DEFAULT_THRESHOLD})")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="baseline JSON (a previous regression_run --json output); "
                         "default: the stored build-report numbers")
    ap.add_argument("--k", type=int, default=20, help="retrieve depth per query")
    ap.add_argument("--json", type=Path, default=None, help="write the full run JSON here")
    args = ap.parse_args()

    try:
        corpus_path = _resolve_snapshot(args.snapshot)
        id_map = load_id_map(args.id_map) if args.id_map else {}
        retrievers = select_retrievers(args.retrievers)
        baselines = load_baselines(args.baseline) if args.baseline else DEFAULT_BASELINES
        baseline_source = str(args.baseline) if args.baseline else BASELINE_SOURCE

        out = run_regression(
            corpus_path=corpus_path,
            queries_path=args.queries,
            qrels_path=args.qrels,
            id_map=id_map,
            bridge_corpus_path=args.bridge,
            retrievers=retrievers,
            retrieve_k=args.k,
            redirect=not args.no_redirect,
            drop_missing_gold=args.drop_missing_gold,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    comparison = compare_to_baseline(out["results"], baselines, threshold=args.threshold)

    print(f"[regression] snapshot: {corpus_path}  ({out['dataset']['n_corpus']} memories, "
          f"{out['dataset']['n_queries']} queries)")
    if out["bridge"]:
        b = out["bridge"]
        print(f"[regression] bridge: {b['bridged']}/{b['reference_ids']} reference ids matched "
              f"({b['ambiguous']} twin-expanded, {b['unbridged']} unmatched)")
    if out["gold"]["dropped_queries"]:
        print(f"[regression] WARNING: dropped {len(out['gold']['dropped_queries'])} queries with "
              f"no surviving gold ids: {out['gold']['dropped_queries']}")
    print(f"[regression] baseline: {baseline_source}\n")
    print(format_comparison_table(comparison))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                    "snapshot": str(corpus_path),
                    "baseline_source": baseline_source,
                    **out,
                    "comparison": comparison,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    raise SystemExit(0 if comparison["passed"] else 1)


if __name__ == "__main__":
    main()
