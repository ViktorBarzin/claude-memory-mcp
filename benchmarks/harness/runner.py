"""Benchmark runner: drive a pluggable retriever over the eval set and report
overall + per-stratum quality metrics, plus per-query latency and (optional)
index build time / size.

Quality decides adoption (recall@k, nDCG@10, MRR). Latency and storage are
measured and reported but DO NOT gate the decision (ADR-0001 success metric).
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from typing import Any

from . import metrics
from .dataset import Dataset
from .types import MemoryId, Query, Retriever

# A retriever may be the Protocol object or a bare callable retrieve(query, k).
RetrieverLike = Retriever | Callable[[str, int], list[MemoryId]]

# k used for the retrieve() call. We request enough depth to compute all
# metrics (max cutoff is 10) with headroom so ties past k=10 don't distort.
DEFAULT_RETRIEVE_K = 20


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (pct in [0,100]). Empty -> 0.0."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


@dataclass
class StratumResult:
    stratum: str
    n_queries: int
    metrics: dict[str, float]  # macro-averaged metric -> value


@dataclass
class BenchmarkResult:
    retriever_name: str
    n_queries: int
    retrieve_k: int
    overall: dict[str, float]
    per_stratum: dict[str, StratumResult]
    latency_ms: dict[str, float]  # mean / p50 / p95 / max
    index_build_seconds: float | None = None
    index_size_bytes: int | None = None
    per_query: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_stratum"] = {k: asdict(v) for k, v in self.per_stratum.items()}
        return d

    def summary(self) -> str:
        lines = [
            f"Retriever: {self.retriever_name}",
            f"Queries: {self.n_queries}  (retrieve_k={self.retrieve_k})",
        ]
        if self.index_build_seconds is not None:
            lines.append(f"Index build: {self.index_build_seconds:.3f}s")
        if self.index_size_bytes is not None:
            lines.append(f"Index size: {self.index_size_bytes / 1e6:.2f} MB")
        lat = self.latency_ms
        lines.append(
            "Latency/query: "
            f"p50={lat['p50']:.2f}ms  p95={lat['p95']:.2f}ms  "
            f"mean={lat['mean']:.2f}ms  max={lat['max']:.2f}ms"
        )
        cols = metrics.METRIC_NAMES
        header = "  ".join(f"{c:>10}" for c in cols)
        lines.append("")
        lines.append(f"{'stratum':<12}{'n':>5}  {header}")
        lines.append("-" * (19 + len(header)))
        for name in ("overall", *sorted(self.per_stratum)):
            if name == "overall":
                m, n = self.overall, self.n_queries
            else:
                sr = self.per_stratum[name]
                m, n = sr.metrics, sr.n_queries
            row = "  ".join(f"{m[c]:>10.4f}" for c in cols)
            lines.append(f"{name:<12}{n:>5}  {row}")
        return "\n".join(lines)


def _get_retrieve_fn(retriever: RetrieverLike) -> Callable[[str, int], list[MemoryId]]:
    if hasattr(retriever, "retrieve"):
        return retriever.retrieve  # type: ignore[attr-defined]
    if callable(retriever):
        return retriever
    raise TypeError("retriever must implement retrieve(query, k) or be callable")


def _maybe_build_index(retriever: RetrieverLike, dataset: Dataset) -> tuple[float | None, int | None]:
    """Call optional lifecycle hooks if present (duck-typed).

    - build_index(corpus) -> None : measured wall-clock build time.
    - index_size_bytes() -> int   : reported on-disk/in-memory index size.
    Returns (build_seconds_or_None, size_bytes_or_None).
    """
    build_seconds: float | None = None
    size_bytes: int | None = None

    build = getattr(retriever, "build_index", None)
    if callable(build):
        t0 = time.perf_counter()
        build(dataset.corpus)
        build_seconds = time.perf_counter() - t0

    size_fn = getattr(retriever, "index_size_bytes", None)
    if callable(size_fn):
        try:
            size_bytes = int(size_fn())
        except Exception:
            size_bytes = None

    return build_seconds, size_bytes


def run_benchmark(
    retriever: RetrieverLike,
    dataset: Dataset,
    *,
    retrieve_k: int = DEFAULT_RETRIEVE_K,
    retriever_name: str | None = None,
    warmup: bool = True,
    collect_per_query: bool = True,
) -> BenchmarkResult:
    """Evaluate `retriever` over `dataset`.

    The retriever is asked for `retrieve_k` ids per query (>= max metric
    cutoff of 10). Metrics are macro-averaged over queries, overall and per
    stratum. Latency is measured around each retrieve() call only (index build
    is timed separately via the optional build_index hook).
    """
    name = retriever_name or getattr(retriever, "name", None) or type(retriever).__name__
    retrieve = _get_retrieve_fn(retriever)
    qrels = dataset.qrels

    build_seconds, size_bytes = _maybe_build_index(retriever, dataset)

    # Optional warmup (first call can pay import/JIT/connection costs that would
    # skew p95). Excluded from latency stats. Uses the first query if any.
    if warmup and dataset.queries:
        try:
            retrieve(dataset.queries[0].text, retrieve_k)
        except Exception:
            pass  # warmup failures surface on the real call below

    per_query_rows: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    # accumulate per-stratum metric sums for macro-average
    strata: dict[str, dict[str, float]] = {}
    strata_counts: dict[str, int] = {}
    overall_sums = {m: 0.0 for m in metrics.METRIC_NAMES}

    for q in dataset.queries:
        rel = qrels[q.query_id]
        t0 = time.perf_counter()
        ranked = list(retrieve(q.text, retrieve_k))
        dt_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(dt_ms)

        m = metrics.per_query_metrics(ranked, rel)
        for key, val in m.items():
            overall_sums[key] += val
        strata.setdefault(q.stratum, {mm: 0.0 for mm in metrics.METRIC_NAMES})
        strata_counts[q.stratum] = strata_counts.get(q.stratum, 0) + 1
        for key, val in m.items():
            strata[q.stratum][key] += val

        if collect_per_query:
            per_query_rows.append(
                {
                    "query_id": q.query_id,
                    "stratum": q.stratum,
                    "n_relevant": len(rel),
                    "latency_ms": round(dt_ms, 3),
                    "retrieved": ranked[:retrieve_k],
                    **{k: round(v, 6) for k, v in m.items()},
                }
            )

    n = len(dataset.queries)
    overall = {k: (overall_sums[k] / n if n else 0.0) for k in metrics.METRIC_NAMES}
    per_stratum: dict[str, StratumResult] = {}
    for s, sums in strata.items():
        c = strata_counts[s]
        per_stratum[s] = StratumResult(
            stratum=s,
            n_queries=c,
            metrics={k: (sums[k] / c if c else 0.0) for k in metrics.METRIC_NAMES},
        )

    latency_stats = {
        "mean": statistics.fmean(latencies_ms) if latencies_ms else 0.0,
        "p50": _percentile(latencies_ms, 50),
        "p95": _percentile(latencies_ms, 95),
        "max": max(latencies_ms) if latencies_ms else 0.0,
    }

    return BenchmarkResult(
        retriever_name=name,
        n_queries=n,
        retrieve_k=retrieve_k,
        overall=overall,
        per_stratum=per_stratum,
        latency_ms=latency_stats,
        index_build_seconds=build_seconds,
        index_size_bytes=size_bytes,
        per_query=per_query_rows,
    )
