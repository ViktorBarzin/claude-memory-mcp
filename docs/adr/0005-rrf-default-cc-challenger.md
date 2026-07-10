# Weighted RRF as the fusion default; convex combination as a benchmark challenger

> **Amendment (2026-07-09, storage-model grilling):** the API's *default* `sort_by` flips
> from `importance` to `relevance`. Importance keeps exactly the role this ADR gives it — a
> post-fusion prior multiplier — and stops being the default rank axis; `sort_by=importance`
> stays available explicitly. Rationale: a month-long session audit measured importance-sorted
> recall as the largest rediscovery driver (55% of entries at ≥0.8 make the axis
> non-discriminating), and a correct default protects every caller that forgets a flag — the
> failure mode actually observed. See [ADR-0007](0007-bounded-self-contained-memories-with-typed-links.md).

The hybrid read path must fuse three retrieval signals on **incomparable scales** — unbounded
`ts_rank` (BM25), bounded cosine, and (phase 2) an arbitrary graph-proximity score — where the graph
list is **sparse and often empty**. We adopt **weighted Reciprocal Rank Fusion** as the default
fusion function: `score(d) = Σ_s w_s/(60 + rank_s(d))`, default `w_lex = w_dense = 1.0`,
`w_graph = 0.35`, with the existing **importance** value applied as a *post-fusion prior multiplier*
(`final = fused × (0.7 + 0.3·importance)` for `sort_by="relevance"`) — importance is a prior, **not**
a fourth fused list.

RRF is the right default because it is **score-scale-free** (no BM25-vs-cosine calibration to
maintain), treats a missing leg as a clean **0** contribution (no missing-modality bias), is
near-parameter-free (`k=60` is demonstrably uncritical across [10,100]), and **collapses to today's
exact lexical ordering** when the dense/graph legs are empty — which is the SQLite-only
graceful-degrade path ([ADR-0002](0002-api-postgres-first-sqlite-stays-lexical.md)) running the *same*
code.

## Considered options

- **Convex combination / TM2C2** (Bruch et al., min-max normalization) — the literature consistently
  shows it **edges RRF on nDCG/recall** when scores are calibratable (Weaviate switched its default to it).
  It is the **standing challenger**. ⚠️ **Correction:** an earlier draft claimed "the benchmark ran CC
  against RRF and RRF was chosen on our eval set" — **no CC results were actually produced or persisted in
  this run.** RRF was adopted on *principled* grounds (scale-free, treats a missing/empty leg as a clean 0,
  collapses to today's exact lexical ordering for the SQLite-only degrade path), **not** a measured
  head-to-head. Benchmarking CC vs RRF on our eval set is an open follow-up — do it before locking fusion,
  and especially if the graph is ever adopted or score distributions shift.
- **Cross-encoder stage-2 re-rank** (e.g. `bge-reranker-v2-m3` over the fused top ~20–30) — a
  *separate*, independently-gated stage, not a fusion function. Deferred; ship only if it clears both
  the quality bar and the hot-path p95 budget on the GPU node.

## Consequences

- Fusion is ~30 lines over three top-N queries; the lexical leg reuses the existing
  `plainto_tsquery`/`ts_rank` query + OR-broaden fallback verbatim.
- The exact-stratum nDCG/MRR dip (~0.018/0.025, recall unaffected) is the known RRF cost of blending
  one perfect hit with near-ties; a small **exact-match rank bonus** is the tunable recovery and a
  cheap follow-up.
- `k=60` is borrowed from TREC ad-hoc IR; a quick re-sweep on the eval set is worthwhile but the
  literature says it is insensitive.
