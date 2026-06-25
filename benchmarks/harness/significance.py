"""Paired-bootstrap significance test over per-query retrieval-metric deltas.

This is how slice S5 SETTLES the concept-graph verdict. Quality differences
between two retriever configs (e.g. ``+both`` vs ``+dense``) are tested PER
STRATUM with a paired bootstrap over the per-query metric deltas, matching the
benchmark-report §2.1 methodology. We report, for each comparison × stratum:

    Δ           the observed mean delta  (metric_B − metric_A, macro over queries)
    95% CI      [2.5th, 97.5th] percentile of the bootstrap distribution of Δ
    P(Δ ≤ 0)    fraction of bootstrap replicates with mean delta ≤ 0
                (a one-sided "no improvement" mass; small ⇒ confident improvement)

A comparison is DECISIVE in favour of B when the 95% CI clears zero (ci_low > 0).
For the graph: promote iff the ``+both`` vs ``+dense`` MULTIHOP CI clears zero;
otherwise the graph stays gated — but now on a VALID test (graph candidates
genuinely competed in the shared fused pool at a swept-up weight), not the prior
math artifact.

Why PAIRED: both configs are evaluated on the SAME queries, so the per-query
deltas remove cross-query difficulty variance — the bootstrap resamples query
INDICES (not the two configs independently), preserving the pairing. This is the
standard paired bootstrap for IR metric comparisons (Sakai 2006; the procedure
BEIR/ranx use for significance).

Determinism: the estimator is SEEDED (numpy ``default_rng(seed)``), so a given
(deltas, n_boot, seed) yields a byte-identical result — CI numbers are
reproducible in CI and in the report.

Inputs are the runner's ``per_query`` rows (``BenchmarkResult.per_query`` /
``to_dict()["per_query"]``): each row is a dict carrying ``query_id``,
``stratum`` and the headline metric keys (``recall@5``/``recall@10``/``ndcg@10``/
``mrr``). We pair rows by ``query_id`` so a reordered results file still pairs
correctly, and slice by ``stratum`` for the per-stratum cuts.

numpy backs the resampling (the ``benchmarks`` optional extra); the harness is
never imported by the product or SQLite-only mode (ADR-0002).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# 95% CI is the reporting convention (benchmark-report §2.1). Two-sided percentile
# interval: the 2.5th and 97.5th percentiles of the bootstrap delta distribution.
_CI_LOW_PCT = 2.5
_CI_HIGH_PCT = 97.5

# Bootstrap replicates. B=10000 per the slice spec — enough that the 2.5/97.5
# percentile estimates are stable to ~3 decimals for n≈40 queries.
DEFAULT_N_BOOT = 10_000


@dataclass(frozen=True)
class BootstrapResult:
    """Outcome of one paired bootstrap (one comparison, one stratum / overall).

    All four headline numbers plus enough context to render a report row. ``n`` is
    the number of PAIRED queries the test ran over (shared query ids in the chosen
    stratum); a comparison with ``n == 0`` is a non-significant sentinel, never a
    crash. ``significant`` is the decision rule: the 95% CI clears zero (improvement)
    — exposed as a property so callers don't re-derive it.
    """

    delta: float
    ci_low: float
    ci_high: float
    p_le_zero: float
    n: int
    n_boot: int = DEFAULT_N_BOOT
    metric: str = ""
    comparison: str = ""
    stratum: str = "overall"

    @property
    def significant(self) -> bool:
        """True iff the 95% CI clears zero on the positive side (B beats A)."""
        return self.n > 0 and self.ci_low > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": self.comparison,
            "stratum": self.stratum,
            "metric": self.metric,
            "n": self.n,
            "n_boot": self.n_boot,
            "delta": self.delta,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "p_le_zero": self.p_le_zero,
            "significant": self.significant,
        }

    def summary(self) -> str:
        star = " *" if self.significant else ""
        return (
            f"{self.comparison:<22} {self.stratum:<10} {self.metric:<10} "
            f"n={self.n:<3} Δ={self.delta:+.4f}  "
            f"95%CI=[{self.ci_low:+.4f}, {self.ci_high:+.4f}]  "
            f"P(Δ≤0)={self.p_le_zero:.4f}{star}"
        )


def paired_bootstrap(
    deltas: list[float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    metric: str = "",
    comparison: str = "",
    stratum: str = "overall",
) -> BootstrapResult:
    """Paired bootstrap over per-query metric deltas ``d_i = metric_B − metric_A``.

    Resamples the ``n`` query indices WITH REPLACEMENT ``n_boot`` times; each
    replicate's statistic is the mean of the resampled deltas. The observed Δ is the
    mean of all deltas; the 95% CI is the [2.5, 97.5] percentile interval of the
    replicate means; ``P(Δ≤0)`` is the fraction of replicate means ≤ 0.

    Empty ``deltas`` (no paired queries) returns a degenerate non-significant
    sentinel (Δ=0, CI=[0,0], P=1.0, n=0) rather than raising — a stratum with no
    shared ids should read as "no evidence", not crash the matrix.
    """
    n = len(deltas)
    if n == 0:
        return BootstrapResult(
            delta=0.0,
            ci_low=0.0,
            ci_high=0.0,
            p_le_zero=1.0,
            n=0,
            n_boot=n_boot,
            metric=metric,
            comparison=comparison,
            stratum=stratum,
        )

    d = np.asarray(deltas, dtype=np.float64)
    observed = float(d.mean())

    rng = np.random.default_rng(seed)
    # (n_boot, n) matrix of resampled indices; row-mean = one replicate statistic.
    idx = rng.integers(0, n, size=(n_boot, n))
    replicate_means = d[idx].mean(axis=1)

    ci_low = float(np.percentile(replicate_means, _CI_LOW_PCT))
    ci_high = float(np.percentile(replicate_means, _CI_HIGH_PCT))
    p_le_zero = float(np.count_nonzero(replicate_means <= 0.0) / n_boot)

    return BootstrapResult(
        delta=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_le_zero=p_le_zero,
        n=n,
        n_boot=n_boot,
        metric=metric,
        comparison=comparison,
        stratum=stratum,
    )


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map query_id → row. Later rows win on a duplicate id (defensive; the runner
    emits one row per query)."""
    return {str(r["query_id"]): r for r in rows}


def paired_deltas_from_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    metric: str,
    stratum: str | None = None,
) -> list[float]:
    """Pair runner ``per_query`` rows by ``query_id`` and return B−A deltas.

    - Pairs by ``query_id`` (NOT positional), so a reordered results file is fine.
    - Drops ids present in only one config (an unmatched query carries no pair).
    - If ``stratum`` is given, keeps only rows whose ``stratum`` matches (the
      stratum is read from config A's row — both configs evaluate the same query,
      so the stratum is identical).
    - Order follows config A's row order, made deterministic.

    The returned list feeds :func:`paired_bootstrap`.
    """
    b_by_id = _index_rows(rows_b)
    deltas: list[float] = []
    for ra in rows_a:
        qid = str(ra["query_id"])
        if stratum is not None and ra.get("stratum") != stratum:
            continue
        rb = b_by_id.get(qid)
        if rb is None:
            continue  # unmatched id → no pair
        deltas.append(float(rb[metric]) - float(ra[metric]))
    return deltas


def paired_bootstrap_from_rows(
    rows_a: list[dict[str, Any]],
    rows_b: list[dict[str, Any]],
    *,
    metric: str,
    stratum: str | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 0,
    comparison: str = "",
) -> BootstrapResult:
    """Convenience: pair two configs' per-query rows (by id, optionally filtered to
    a stratum) and run the paired bootstrap on the resulting deltas. This is the
    exact call the matrix runner makes per (comparison, stratum)."""
    deltas = paired_deltas_from_rows(rows_a, rows_b, metric=metric, stratum=stratum)
    return paired_bootstrap(
        deltas,
        n_boot=n_boot,
        seed=seed,
        metric=metric,
        comparison=comparison,
        stratum=(stratum or "overall"),
    )
