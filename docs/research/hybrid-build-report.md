# Hybrid-recall build report: what was implemented, the fair comparison, and the graph verdict

Status: **the build + decision instrument for the hybrid-recall upgrade.** It completes the
[benchmark report](benchmark-report.md), which adopted lexical+dense but flagged the concept
graph as **UNEVALUATED** (the prototype's fusion made it mathematically impossible for a
graph-only result to enter the fused top-k). This run **built both legs as real, tested,
flag-gated code**, **fixed the fusion so graph candidates genuinely compete**, and **settled
the graph question on a valid test**.

> **Privacy.** The corpus, queries, qrels, embedding cache, and extracted graph all derive from
> real personal memories and stay **local + gitignored**. Only **aggregate numbers and synthetic
> illustrations** appear in this document.

## Bottom line

1. **Dense (semantic) recall is a robust win and ships on.** FTS+dense beats lexical FTS on every
   overall metric — paraphrase recall@10 **+0.350**, overall nDCG@10 **+0.078** — with paired-bootstrap
   CIs that clear zero. This reconfirms the benchmark report's phase-1 result **byte-identically**
   (same fixed `bge-large-en-v1.5`, same eval set).
2. **The concept graph adds no measurable value over dense — on a VALID test — so it stays gated
   OFF.** This run fixed the prototype's fusion flaw (graph candidates now compete in the shared
   fused pool) and swept the graph weight. Graph-only candidates **genuinely reached the fused
   top-10** (up to **737** of them over the sweep), yet the decisive **+both-vs-+dense** comparison
   **never cleared zero on any stratum at any weight**. Verdict: **graph-null**. Unlike the prior
   run, this is a finding, not an artifact.
3. **The graph competes only at weights that destroy precision.** Where the graph is quiet
   (w_graph ≤ 0.25, zero graph-only ids in the top-10) it is harmless but adds nothing; where it
   speaks (w_graph ≥ 1.0) it floods low-precision PPR-bridged neighbours and demotes the genuinely
   relevant dense/lexical hits. There is **no weight that both surfaces graph-only ids and beats
   dense.**

**Recommended production defaults:** `MEMORY_EMBEDDINGS_ENABLED` → **on** (after the
[promotion runbook](../runbooks/hybrid-recall-promotion.md)); `MEMORY_GRAPH_ENABLED` → **off**.

---

## 1. What was implemented

Both legs are real, tested, flag-gated code on the `wizard/hybrid-recall-impl` branch. SQLite-only
mode stays purely lexical with **no new required dependencies** — every embedding/graph dependency
is an **optional extra** and the code degrades cleanly to lexical when the extra/flag is absent
(ADR-0002).

### 1.1 Product code (`src/claude_memory/`, shipped behind flags)

- **Dense embedding backends** (`embeddings.py`): an `Embedder` Protocol with two backends —
  `VoyageEmbedder` (voyage-3.5, hosted, 1024-d, non-sensitive rows only) selected when
  `VOYAGE_API_KEY` is set, else `BgeEmbedder` (`bge-large-en-v1.5`, local, sensitive-safe + no-key
  fallback). Both emit L2-normalised `list[float]` of dim 1024. **Sensitive rows (`is_sensitive=1`)
  short-circuit to `None` before any model load or API call** (ADR-0003). Heavy deps are lazy-imported
  inside the backends, so the base install pulls none.
- **Fused recall + embed-on-write via one shared helper** (`api/recall.py`): a single `_fused_recall`
  that **both** recall entry points (REST `recall_memories` + FastMCP `memory_recall`) call, and a single
  `schedule_embedding` that **both** store paths call — the anti-drift guarantee. With **both flags off
  (default), `_fused_recall` is a TRUE no-op**: it runs the exact current `ts_rank` SQL and returns rows
  verbatim, so existing behaviour and tests are byte-identical (it is **not** an RRF collapse — an
  additive blend and a multiplicative-on-RRF order differently, and the no-op path avoids that). When the
  embeddings flag is on, a dense CTE leg (`embedding <=> $qvec` over HNSW, depth 50) joins a shared
  weighted-RRF pool (k=60) and **importance becomes a post-fusion multiplier** (`final = fused ×
  (0.7 + 0.3·importance)`), per ADR-0005. The dense leg embeds the query off the event loop and **degrades
  to lexical on any backend failure** (a hung hosted API or a model-load error never stalls or 500s recall).
  Embed-on-write runs **off the hot path** (background task) and **refuses sensitive rows**. The graph leg is
  marked phase-2 and is deliberately not wired into the production read path (see §5).
- **`sync.py` is untouched and proven embedding-free.** The embedding column lives **only in Postgres**;
  SQLite gets no vector column and the sync payload is unchanged — asserted by tests.

### 1.2 Migration & staged production artifacts (additive, idempotent, NOT applied)

- **Alembic migration 005** (`migrations/versions/005_add_embeddings_and_graph.py`, down_revision 004):
  additive + idempotent + PG-only. `CREATE EXTENSION IF NOT EXISTS vector` **gated on
  `pg_available_extensions`**; `memories.embedding halfvec(1024)` NULL-able; an HNSW index
  (`halfvec_cosine_ops`, m=16, ef_construction=64) built `CONCURRENTLY` inside an `autocommit_block()`
  (env.py wraps every migration in a transaction, so a bare `CONCURRENTLY` would raise 25001); and the
  three concept-graph tables `concepts` / `concept_edges` / `memory_concepts`. The vector steps **no-op
  cleanly when pgvector is unavailable**, so the migration is safe to run before infra enables it. The
  lexical `search_vector`/GIN schema from migration 001 is untouched.
- **pgvector-on-CNPG Terraform + Dockerfile** (`deploy/infra/`): `Dockerfile.pgvector` (FROM the CNPG
  PostGIS operand image + `postgresql-16-pgvector`, additive so PostGIS survives, build self-gates on
  `vector.control`) and `dbaas-pg-cluster-pgvector.tf` (an operand-image swap behind a
  `pg_cluster_image` variable that **defaults to the current image** — landing it in the GitOps repo is a
  no-op until an operator promotes). **Nothing applied; the live cluster is untouched.**
- **Promotion runbooks** (`docs/runbooks/`): the full-hybrid
  [`hybrid-recall-promotion.md`](../runbooks/hybrid-recall-promotion.md) and the dense-only
  [`promote-pgvector-dense-recall.md`](promote-pgvector-dense-recall.md) — gated steps (presence claim →
  operand-image swap → live additive migration → backfill → flag flip → verify → optional prod compare) +
  per-layer rollback + the full multi-tenant blast radius.

### 1.3 Offline benchmark legs (`benchmarks/`, the comparison substrate)

- **Fixed-fusion hybrid retriever** (`retrievers/hybrid.py`): the fusion flaw is fixed — `_graph_rank`
  dropped the base-set `exclude` carve-out so the **full graph ranking flows into the SAME shared RRF
  pool** as FTS+dense, and `w_graph` is a swept attribute. A graph-only hit now competes; a zero-weight leg
  is a true no-op (for the ablations). The graph leg is **Personalized PageRank** (HippoRAG-2 idea, no
  per-query LLM) over the typed bipartite memory↔concept graph, with the transition matrix **cached once
  per graph** (a typed-1-hop variant remains behind `graph_mode='1hop'`).
- **Batched + cached LLM triple extraction** (`retrievers/graph_extract.py`): ~15–25 id-tagged memories
  per `claude -p --model haiku` call, triples cached to a gitignored `triples_<fp>.jsonl` keyed by
  `(id, content)` — reruns cost zero LLM calls; sensitive rows filtered before any external call.
- **Canonicalisation → typed concept graph** (`retrievers/graph_build.py`): EDC define+canonicalize —
  each distinct surface form embedded once (cached bge-large), clustered by cosine-NN + threshold into
  canonical concepts (alias forms collapse), open relations canonicalised into a bounded typed vocabulary.
  A vectorised canonicaliser (byte-equivalent to the pure-Python path) builds the **full-corpus** typed
  graph in ~minute-scale, enabling 100% coverage without sampling.
- **Comparison apparatus** (`harness/significance.py`, `harness/matrix.py`, `scripts/run_eval.py`): the
  4-config matrix, the weight sweep, the **graph-only-in-top10 diagnostic** the prior run lacked, the
  paired-bootstrap significance, and the entity-bridged multi-hop sub-cut.

### 1.4 Verification (CI gates, exactly as specified)

`ruff check src/ tests/` → clean. `mypy src/claude_memory/` → success, 17 source files. `pytest tests/`
→ **236 passed**. Benchmark suite (`benchmarks/` retrievers + harness) → **100 passed**. No `Any`, no
type-suppression comments, no skipped hooks.

---

## 2. The fair comparison

Held **fixed** so dense-vs-graph is the **only** variable: the embedding model (`bge-large-en-v1.5`),
the preserved eval set (corpus fingerprint `ca7b1d4ed22672e8`, **5,452** memories, **119** queries —
**40 exact / 40 paraphrase / 39 multi-hop**, of which **28** multi-hop are entity-bridged), and the
cached bge-large corpus matrix (5452×1024, **reused — no re-embedding**). A standing test asserts the
cache-hit branch is taken (the local embedder is booby-trapped to raise) and the fingerprint resolves to
`ca7b1d4ed22672e8`, so a silent re-embed cannot slip in.

Four configs, fused identically (graph candidates in the shared pool, only `w_graph` varied within a
config), all 7 result JSONs persisted to the gitignored `benchmarks/results/`:

- **FTS** — lexical baseline (the production code path, reused verbatim as the hybrid's lexical leg).
- **+dense** — FTS+dense (`w_graph=0`).
- **+graph** — FTS+graph (`w_dense=0`), isolating graph vs lexical.
- **+both** — the full hybrid.

`+dense` and `+graph` headline columns are the persisted configs; `+graph (w.25)` and `+both (w.25)` are
shown at the **selected sweep weight w_graph=0.25** — the only weight where the graph does not degrade
quality (see §4). Δ columns: vs FTS for `+dense`/`+graph`, vs `+dense` for `+both`.

| Metric | Stratum | FTS | +dense | +graph (w.25) | +both (w.25) | Δ dense−FTS | Δ both−dense |
|---|---|---|---|---|---|---|---|
| recall@5 | overall | 0.6663 | 0.7415 | 0.6283 | 0.7376 | +0.0752 | −0.0039 |
| recall@5 | exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| recall@5 | paraphrase | 0.3500 | 0.5500 | 0.3250 | 0.5500 | +0.2000 | 0.0000 |
| recall@5 | multihop | 0.6485 | 0.6726 | 0.5581 | 0.6609 | +0.0241 | −0.0117 |
| recall@10 | overall | 0.6952 | 0.8338 | 0.6887 | 0.8226 | +0.1385 | −0.0111 |
| recall@10 | exact | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| recall@10 | paraphrase | 0.3750 | 0.7250 | 0.3750 | 0.7000 | +0.3500 | −0.0250 |
| recall@10 | multihop | 0.7111 | 0.7748 | 0.6912 | 0.7665 | +0.0637 | −0.0083 |
| nDCG@10 | overall | 0.6507 | 0.7284 | 0.5665 | 0.7214 | +0.0777 | −0.0070 |
| nDCG@10 | exact | 0.9908 | 0.9723 | 0.9815 | 0.9723 | −0.0185 | 0.0000 |
| nDCG@10 | paraphrase | 0.3123 | 0.5023 | 0.2403 | 0.4885 | +0.1900 | −0.0138 |
| nDCG@10 | multihop | 0.6491 | 0.7101 | 0.4752 | 0.7030 | +0.0610 | −0.0072 |
| MRR | overall | 0.6737 | 0.7297 | 0.5332 | 0.7263 | +0.0560 | −0.0034 |
| MRR | exact | 0.9875 | 0.9625 | 0.9750 | 0.9625 | −0.0250 | 0.0000 |
| MRR | paraphrase | 0.2958 | 0.4343 | 0.1998 | 0.4257 | +0.1385 | −0.0086 |
| MRR | multihop | 0.7393 | 0.7940 | 0.4221 | 0.7924 | +0.0547 | −0.0016 |

**Read:** `+dense` decisively beats FTS — paraphrase recall@10 **+0.350**, overall nDCG@10 **+0.078**.
`+graph` **alone** (FTS+graph, `w_dense=0`) is **worse than FTS on every stratum** even at the gentle
w=0.25. `+both` at w=0.25 ≈ `+dense` (every Δ ≤ 0, mostly tiny-negative); **at no weight does `+both`
exceed `+dense`.** At higher weights `+both` collapses (overall nDCG@10 = 0.193 at w=2.0, 0.004 at w=5.0).

## 3. Significance (paired bootstrap, per stratum)

B=10000, seed=1234, over the runner per-query rows; the dense baseline is the `w_graph=0` cut (FTS+dense).
A 95% CI clearing zero ⇒ significant.

**Reconfirm +dense vs FTS (significant wins):** overall nDCG@10 Δ=+0.0777 CI[+0.0408,+0.1161]; overall
recall@10 Δ=+0.1385 CI[+0.0794,+0.2031]; paraphrase recall@10 Δ=+0.3500 CI[+0.2000,+0.5000]. The known
small **exact** dip is real: nDCG@10 Δ=−0.0185 CI[−0.0461,0]. **Multi-hop dense-vs-FTS stays
marginal/inconclusive** (nDCG@10 P(Δ≤0)=0.044, recall@10 P=0.064) — matching the prior report.

**Decisive +both vs +dense (the graph's marginal value over dense) — never clears zero on any stratum at
any weight:**

- **multi-hop nDCG@10** (the stratum a typed graph should help): at w=0.25 Δ=−0.0072
  CI[−0.0228,+0.0093] P(Δ≤0)=0.814 — straddles zero, point estimate negative; and monotonically worse
  with weight — w=0.5 Δ=−0.067 (significantly negative), w=1.0 Δ=−0.191, **w=2.0 Δ=−0.589** (CI entirely
  below zero).
- **w=0.25 full per-stratum** +both−dense: overall nDCG@10 Δ=−0.0070 CI[−0.0158,+0.0006] P=0.964;
  paraphrase nDCG@10 Δ=−0.0138 CI[−0.0346,−0.0004] (**significantly negative even at the gentlest
  weight**); exact Δ=0 exactly.
- **Entity-bridged multi-hop sub-cut** (28/39 multi-hop queries; the 11 part-N-of-M chunk queries
  excluded — the theorised graph sweet spot) at w=0.25: nDCG@10 Δ=−0.0051 CI[−0.0261,+0.0167] P=0.703;
  recall@10 Δ=−0.0045 CI[−0.0580,+0.0491] P=0.653. **Even here, no positive signal** — CI straddles zero,
  point estimate negative.

## 4. The weight sweep — why this verdict is valid, not a math artifact

**The prior flaw is fixed and confirmed.** Two changes made the test valid: dropping the base-set
exclusion so graph candidates enter the shared fused pool, and a PPR seed-lockout fix so a bridged
graph-only memory lands at graph-rank ~1. Model-free regression tests prove a graph-only hit reaches the
fused top-10 at the correct dual-leg-dominated boundary (barred at w=1.0 with score 0.0164 < the
~0.0286 top-10 bar; enters at w=2.0 with 0.0328) — assertions the prior run could never satisfy.

**Graph-only-in-top10 diagnostic** (count of fused-top-10 ids absent from the FTS∪dense depth-50 base
pool, summed over 119 queries, for `+both`):

| w_graph | 0.00 | 0.25 | 0.50 | 0.75 | 1.0 | 1.5 | 2.0 | 3.0 | 4.0 | 5.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| graph-only in top-10 | 0 | 0 | 0 | 0 | 89 | 221 | 381 | 601 | 682 | **737** |

The graph **first** reaches the fused top-10 at w_graph≈1.0 and rises to **737** by w=5.0 — empirical
proof the graph **can** now compete (max graph-only ≫ 0, so the verdict is **not** the prior math
artifact). Below w≈1.0 a graph-only rank-1 hit (score w/61) cannot clear the ~0.0286 top-10 bar, so it
contributes 0.

**The fundamental tension (the real finding).** `+both` overall nDCG@10 vs w_graph:
0.7284 (w0) → 0.7214 (.25) → 0.6930 (.5) → 0.6159 (1.0) → 0.1929 (2.0) → 0.0043 (5.0). recall@10:
0.8338 → 0.8226 → 0.7516 (1.0) → 0.4702 (2.0) → 0.0060 (5.0). The graph competes **only** at weights that
simultaneously destroy precision: where it is quiet (w ≤ 0.25, graph-only = 0) it is harmless but adds
nothing; where it speaks (w ≥ 1.0) it floods low-precision PPR-bridged neighbours and demotes the
genuinely relevant dense/lexical hits. **The single weight where +both is statistically
indistinguishable from +dense (w=0.25) is exactly the weight where the graph contributes zero ids to the
top-10 — i.e. "no harm" only when the graph is silent.** Headline w_graph=0.25 selected on joint
overall+multi-hop nDCG@10 (least regression); RRF k=60 fixed.

**The typed graph itself:** 14,871 canonical concepts, 17,644 typed edges (vs the prior keyword
prototype's 2,095,624 — a **119× reduction**), a 19,824-node / 48,209-edge bipartite PPR index, **100%
corpus coverage** (full 5,452 memories, all qrel targets included; no sampling).

## 5. The graph verdict

**graph-null.** On a valid shared-pool test with the weight swept to 5.0, graph candidates **genuinely
competed** (max 737 graph-only ids in the fused top-10) yet **never beat dense** — the decisive
+both-vs-+dense CI does not clear zero on any stratum at any weight, including the entity-bridged
multi-hop sweet spot. The graph **stays gated**, this time on a **valid test**, not the prior fusion
math-artifact.

This is a stronger and more defensible result than the prior report could offer: the benchmark report
explicitly listed the graph as **UNEVALUATED** and demanded a retest that "put graph ids in the fused
candidate pool (no base-set exclusion) and/or sweep the graph weight." This run did exactly that and the
graph still did not help.

**Production posture.** Ship the **dense** leg (significant, robust win). Keep the concept-graph leg
**built and flag-gated** (`MEMORY_GRAPH_ENABLED` default **off**) but **do not enable it**, and **do not
wire a graph leg into the production read path** — the API's `_fused_recall` deliberately leaves the
graph leg as phase-2 with no production module. The graph code, migration tables, and PPR substrate are
staged for a future revisit (e.g. a precision-filtered graph leg, or graph-as-reranker rather than
graph-as-RRF-leg), but the current RRF-leg formulation is **settled as graph-null**.

## 6. Latency & storage (measured, non-gating per ADR-0001)

All offline, CPU, non-gating. Per-query end-to-end retrieve (dominated by the bge-large query embedding
on CPU): FTS p50 214 ms / p95 405 ms; +dense p50 185 ms / p95 227 ms (cached corpus matrix → no per-query
corpus embed).

**Graph-leg PPR latency** (the design demanded a real measurement; the survey's "~2 ms" was unproven):
over the typed graph's 19,824-node / 48,209-edge bipartite index, **with the transition matrix cached**
(built once, 269 ms), per-query PPR run = **p50 13.6 ms / p95 14.5 ms / max 16 ms**. **Critical
operational finding:** the un-cached PPR rebuilds that sparse matrix on **every call** — which made the
full sweep (~4,760 PPR-bearing retrieves) take >20 min and never finish; **caching the matrix is
mandatory** for any PPR hot path. 14 ms at 48k edges is far above the survey's 2 ms but well below the
~140–280 ms a 2.1M-edge graph would cost — consistent with **edge count, not node count, driving PPR
cost**. The cheaper typed-1-hop variant remains available behind `graph_mode='1hop'` as a recursive-CTE-
portable substrate, but is moot given the graph-null verdict.

## 7. Recommended flag defaults (and why)

| Flag | Production default | Justification (from the numbers) |
|---|---|---|
| `MEMORY_EMBEDDINGS_ENABLED` | **on** (after promotion) | +dense is a significant, robust win: paraphrase recall@10 +0.350 CI[+0.20,+0.50], overall nDCG@10 +0.078 CI[+0.041,+0.116]; the only downside is a small, bounded exact-stratum dip (nDCG@10 −0.0185) that exact still serves at recall@10 = 1.000. Net clearly positive. Promote via the [runbook](../runbooks/hybrid-recall-promotion.md). |
| `MEMORY_GRAPH_ENABLED` | **off** | graph-null on a valid test: +both-vs-+dense never clears zero (multi-hop nDCG@10 Δ=−0.0072 CI[−0.0228,+0.0093] at the only non-degrading weight), and at any weight where the graph actually reaches the top-10 it degrades quality (overall nDCG@10 collapses to 0.193 at w=2.0). Enabling it cannot help and can only hurt. The schema/code ship for a future revisit. |

Until both infra (pgvector on CNPG) **and** the embeddings flag are on, claude-memory serves **lexical-
only** recall exactly as today — both legs are dormant and the degrade path is byte-identical.

## 8. Threats to validity

- **Binary, un-pooled qrels** bias absolute recall/nDCG *levels* low (per the benchmark report §6); only
  the **deltas** here are trustworthy. This affects all configs equally, so it does not change the
  relative dense-vs-graph or hybrid-vs-FTS conclusions.
- **Multi-hop is the graph's entire rationale**, but many of its 39 queries are part-N-of-M chunks of one
  memory that dense already shortcuts; the 28-query entity-bridged sub-cut isolates the genuine
  multi-entity case — and even there the graph shows no positive signal.
- **The verdict is specific to the graph-as-RRF-leg formulation** with PPR seeding from fused base hits.
  It does **not** rule out a differently-shaped graph contribution (precision-filtered candidates,
  reranking rather than fusion, or a graph tuned for a different query distribution). Those are future
  work, gated behind a fresh comparison.
- **Model held fixed at bge-large.** Voyage-3.5 in production may shift absolute dense numbers, but the
  comparison's purpose was dense-vs-graph with the embedding model as a controlled constant; that
  control is sound.

## Related

- [`benchmark-report.md`](benchmark-report.md) — the phase-1 lexical+dense decision; flagged the graph
  UNEVALUATED.
- [`survey.md`](survey.md) / [`integration-design.md`](integration-design.md) — landscape and design.
- [`../runbooks/hybrid-recall-promotion.md`](../runbooks/hybrid-recall-promotion.md) — gated production
  promotion + rollback.
- `docs/adr/0001`–`0006` — the decisions this build realises.
- `benchmarks/results/*.json` (gitignored, local) — the reproducible result artifacts behind every number
  here.
