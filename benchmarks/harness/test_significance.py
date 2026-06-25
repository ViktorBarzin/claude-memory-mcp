"""Tests for the paired-bootstrap significance module (slice S5).

The S5 verdict on the concept graph is settled by a PER-STRATUM paired bootstrap
over per-query metric deltas (B=10000), matching benchmark-report §2.1. These
tests pin the estimator's behaviour against ANALYTIC expectations on synthetic
per-query deltas (known-positive / zero / negative), and prove it reads the
runner's ``per_query`` rows (it pairs by ``query_id`` and slices by ``stratum``).

Verified analytically in scratchpad/boot_sim.py:
  * constant delta c on every query  → Δ = c, CI = [c, c], P(Δ≤0) = (1 if c≤0 else 0)
  * all-zero deltas                   → Δ = 0, CI = [0, 0], P(Δ≤0) = 1.0
  * symmetric ±1 deltas               → Δ ≈ 0, CI straddles 0, P(Δ≤0) ≈ 0.5
  * mostly-positive (strong signal)   → Δ > 0, CI lower bound > 0, P(Δ≤0) ≈ 0

These are MODEL-FREE and fast (no retriever, no corpus): they synthesise per_query
rows directly. The estimator is SEEDED so results are deterministic in CI.

Run:  .venv/bin/python -m pytest harness/test_significance.py -q
"""
from __future__ import annotations

import math

from harness.significance import (
    BootstrapResult,
    paired_bootstrap,
    paired_bootstrap_from_rows,
    paired_deltas_from_rows,
)


# ---------------- core paired_bootstrap on synthetic deltas ----------------

def test_bootstrap_all_positive_constant_delta() -> None:
    """A constant positive delta on every query: every resample mean equals the
    constant, so Δ = c, the CI is degenerate [c, c], and P(Δ≤0) = 0."""
    res = paired_bootstrap([0.2] * 39, n_boot=10_000, seed=0)
    assert isinstance(res, BootstrapResult)
    assert res.n == 39
    assert math.isclose(res.delta, 0.2, abs_tol=1e-12)
    assert math.isclose(res.ci_low, 0.2, abs_tol=1e-9)
    assert math.isclose(res.ci_high, 0.2, abs_tol=1e-9)
    assert res.p_le_zero == 0.0


def test_bootstrap_all_zero_delta() -> None:
    """All-zero deltas: Δ = 0, CI = [0, 0], and since 0 ≤ 0 every replicate
    counts toward P(Δ≤0) → 1.0 (no improvement at all)."""
    res = paired_bootstrap([0.0] * 40, n_boot=10_000, seed=0)
    assert res.delta == 0.0
    assert res.ci_low == 0.0
    assert res.ci_high == 0.0
    assert res.p_le_zero == 1.0


def test_bootstrap_all_negative_constant_delta() -> None:
    """A constant negative delta: Δ = -0.1, degenerate CI, P(Δ≤0) = 1.0
    (a confident REGRESSION)."""
    res = paired_bootstrap([-0.1] * 39, n_boot=10_000, seed=0)
    assert math.isclose(res.delta, -0.1, abs_tol=1e-12)
    assert math.isclose(res.ci_low, -0.1, abs_tol=1e-9)
    assert math.isclose(res.ci_high, -0.1, abs_tol=1e-9)
    assert res.p_le_zero == 1.0


def test_bootstrap_symmetric_straddles_zero() -> None:
    """Symmetric ±1 deltas: Δ ≈ 0, the CI straddles zero (lo < 0 < hi), and
    P(Δ≤0) ≈ 0.5 — the textbook "no significant difference" outcome."""
    deltas = [1.0] * 50 + [-1.0] * 50
    res = paired_bootstrap(deltas, n_boot=10_000, seed=7)
    assert math.isclose(res.delta, 0.0, abs_tol=1e-9)
    assert res.ci_low < 0.0 < res.ci_high
    assert 0.3 < res.p_le_zero < 0.7


def test_bootstrap_mostly_positive_clears_zero() -> None:
    """A strong-but-imperfect positive signal (35×+0.5, 4×-0.1): Δ > 0, the CI
    lower bound clears zero, and P(Δ≤0) ≈ 0 — a SIGNIFICANT improvement (the
    'promote' shape for the graph verdict)."""
    deltas = [0.5] * 35 + [-0.1] * 4
    res = paired_bootstrap(deltas, n_boot=10_000, seed=3)
    assert res.delta > 0.0
    assert res.ci_low > 0.0  # 95% CI clears zero → significant
    assert res.p_le_zero < 0.01


def test_bootstrap_is_deterministic_under_fixed_seed() -> None:
    """Same deltas + same seed → byte-identical result (CI reproducible in CI)."""
    deltas = [0.5] * 35 + [-0.1] * 4
    a = paired_bootstrap(deltas, n_boot=2_000, seed=99)
    b = paired_bootstrap(deltas, n_boot=2_000, seed=99)
    assert a == b


def test_bootstrap_ci_brackets_delta() -> None:
    """For any sample the observed Δ must lie within (or on) the reported CI."""
    deltas = [0.3, -0.1, 0.0, 0.25, 0.4, -0.05, 0.1, 0.2, 0.15, -0.2]
    res = paired_bootstrap(deltas, n_boot=5_000, seed=1)
    assert res.ci_low <= res.delta <= res.ci_high


def test_bootstrap_empty_deltas_is_degenerate_not_crash() -> None:
    """No paired observations (e.g. a stratum with no shared query ids) must not
    crash: Δ=0, CI=[0,0], P=1.0, n=0 — a clearly non-significant sentinel."""
    res = paired_bootstrap([], n_boot=1_000, seed=0)
    assert res.n == 0
    assert res.delta == 0.0
    assert res.ci_low == 0.0 and res.ci_high == 0.0
    assert res.p_le_zero == 1.0


# ---------------- pairing from runner per_query rows ----------------

def _rows(metric_by_qid: dict[str, float], stratum_by_qid: dict[str, str]) -> list[dict]:
    """Synthesise runner-shaped per_query rows: each row carries query_id, stratum
    and the headline metric keys. We vary one metric and hold the others fixed so
    the pairing logic is unambiguous."""
    rows = []
    for qid, val in metric_by_qid.items():
        rows.append(
            {
                "query_id": qid,
                "stratum": stratum_by_qid[qid],
                "n_relevant": 1,
                "latency_ms": 1.0,
                "retrieved": [],
                "recall@5": val,
                "recall@10": val,
                "ndcg@10": val,
                "mrr": val,
            }
        )
    return rows


def test_paired_deltas_pairs_by_query_id() -> None:
    """paired_deltas_from_rows must pair rows by query_id (NOT positional) and
    return B-minus-A deltas in a STABLE order. Rows are deliberately shuffled
    between the two configs to prove the pairing is by id."""
    strata = {"q1": "exact", "q2": "exact", "q3": "multihop"}
    a = _rows({"q1": 0.2, "q2": 0.5, "q3": 0.1}, strata)
    # config B improves q1 by +0.3, q2 unchanged, q3 by +0.4 — rows reordered.
    b = _rows({"q3": 0.5, "q1": 0.5, "q2": 0.5}, strata)
    deltas = paired_deltas_from_rows(a, b, metric="ndcg@10")
    # order follows config A's row order: q1, q2, q3 (paired by id, not position).
    assert [round(d, 6) for d in deltas] == [0.3, 0.0, 0.4]


def test_paired_deltas_filters_by_stratum() -> None:
    """When a stratum is given, only that stratum's shared query ids contribute."""
    strata = {"q1": "exact", "q2": "multihop", "q3": "multihop"}
    a = _rows({"q1": 0.2, "q2": 0.1, "q3": 0.3}, strata)
    b = _rows({"q1": 0.9, "q2": 0.6, "q3": 0.3}, strata)
    multihop = paired_deltas_from_rows(a, b, metric="ndcg@10", stratum="multihop")
    assert [round(d, 6) for d in multihop] == [0.5, 0.0]  # q2:+0.5, q3:0; q1 excluded
    exact = paired_deltas_from_rows(a, b, metric="ndcg@10", stratum="exact")
    assert [round(d, 6) for d in exact] == [0.7]


def test_paired_deltas_ignores_unmatched_ids() -> None:
    """Query ids present in only one config are dropped from the pairing."""
    strata = {"q1": "exact", "q2": "exact", "q3": "exact"}
    a = _rows({"q1": 0.2, "q2": 0.5}, strata)
    b = _rows({"q2": 0.5, "q3": 0.9}, strata)  # q1 only in A, q3 only in B
    deltas = paired_deltas_from_rows(a, b, metric="ndcg@10")
    assert [round(d, 6) for d in deltas] == [0.0]  # only q2 shared, delta 0


def test_paired_bootstrap_from_rows_end_to_end() -> None:
    """paired_bootstrap_from_rows pairs runner rows by id, filters to a stratum,
    and runs the bootstrap — the exact call the runner makes per (comparison,
    stratum). Multihop config B beats A on every shared query → CI clears zero."""
    strata = {f"m{i}": "multihop" for i in range(20)}
    strata["e1"] = "exact"
    a_vals = {f"m{i}": 0.2 for i in range(20)}
    a_vals["e1"] = 0.5
    b_vals = {f"m{i}": 0.6 for i in range(20)}  # +0.4 on every multihop query
    b_vals["e1"] = 0.5
    a = _rows(a_vals, strata)
    b = _rows(b_vals, strata)
    res = paired_bootstrap_from_rows(a, b, metric="ndcg@10", stratum="multihop", seed=0)
    assert res.n == 20
    assert math.isclose(res.delta, 0.4, abs_tol=1e-9)
    assert res.ci_low > 0.0  # decisive: clears zero
    assert res.p_le_zero == 0.0
