# Integration design: hybrid recall (production + prototype-as-built)

Status: the design the benchmark validated. This document specifies **(A) the production design**
for the API/Postgres deployment and **(B) the prototype as actually built** for the benchmark.
Read the [survey](survey.md) for *why* these choices, the
[benchmark report](benchmark-report.md) for whether they cleared the gate, and
[ADR-0001–0003](../adr/) for the fixed constraints.

**Headline architectural decision (recorded in
[ADR-0004](../adr/0004-phase-the-hybrid-lexical-dense-first-graph-gated.md)):** ship
**lexical + dense fusion** first (a statistically-robust paraphrase win cleared ADR-0001's gate); the
**concept graph is deferred behind a gate** because it was **never validly tested** — the prototype's
fusion config structurally barred it from the top-k (see [benchmark report §1 + §3](benchmark-report.md)),
so it is *unevaluated*, not disproven. Deferral is on cost + uncertainty grounds. The design below is
structured around that phasing.

---

## A. Production design (API/Postgres deployment)

### A.0 Where it plugs in

The semantic layer targets the **API/Postgres path only** (ADR-0002). The authoritative store is
the CNPG Postgres behind the FastAPI server (`src/claude_memory/api/app.py`); the local SQLite
cache stays **lexical (FTS5) only** and degrades gracefully. Recall fires on the **hot path** of
every prompt (an auto-recall hook before each turn), so the read path must stay within a
per-prompt latency budget even though latency is non-gating for *adoption* (ADR-0001).

The current production recall (`app.py::recall_memories`, `POST /api/memories/recall`) is a single
`plainto_tsquery('english')` + `ts_rank(search_vector, query)` ordered by a blend
`ts_rank*0.7 + importance*0.3` (or `*0.4/*0.6` for `sort_by="importance"`), with an OR-broaden
fallback gated at `ts_rank > OR_BROADEN_MIN_RANK` when AND-match under-fills. The live schema
(`migrations/001`) has a generated `search_vector tsvector` (setweight A=content, B=expanded_keywords,
C=tags, D=category) + a GIN index `idx_memories_search`. **The hybrid design is purely additive
to this.**

### A.1 Schema delta (additive, one migration)

```sql
-- new Alembic migration (Postgres only; SQLite path unchanged)
ALTER TABLE memories ADD COLUMN embedding halfvec(1024);   -- NULL for is_sensitive=1
CREATE INDEX CONCURRENTLY idx_memories_embedding
  ON memories USING hnsw (embedding halfvec_cosine_ops)
  WITH (m = 16, ef_construction = 64);
```

- **1024-d** matches both production (Voyage-3.5) and the prototype (bge-large-en-v1.5), so the
  column dimension and all fusion code are identical whichever model runs.
- **halfvec** (fp16) halves index size at ~no recall loss; 1024-d halfvec = 2048 bytes/row →
  single-digit MB for the whole corpus.
- The existing `search_vector` + GIN index are **untouched**. Lexical behaviour is unchanged, so
  NULL-embedding rows (sensitive memories) and SQLite-only mode degrade to exactly today's FTS.
- `CONCURRENTLY` avoids locking the shared table during backfill.
- The concept-graph tables (§A.5) ship **only if/when the graph clears its gate** — phase 2.

### A.2 Write path (store / update) — all LLM work here, off the recall hot path

On `memory_store` / `memory_update`, for **non-sensitive** rows (`is_sensitive=0`, hard ADR-0003
gate):

1. **Embed** `content` (optionally `content + expanded_keywords`) → one `halfvec(1024)` vector,
   written to the new column. Voyage-3.5 `input_type="document"` for stores; bge-large
   `encode_document` for sensitive/no-key fallback. `is_sensitive=1` rows get `embedding=NULL` —
   never embedded, never egressed; they still match via FTS.
2. **(Phase 2, gated) Extract** concepts/edges for the new memory and incrementally merge into the
   graph tables (entity resolution via pgvector nearest-neighbour + threshold, LLM tie-break only
   on ambiguity — Graphiti-style fast-path).
3. **(Optional, flagged) Curate** — the Mem0-style ADD/UPDATE/DELETE/NOOP loop, run async, never
   physically deleting (supersede to `[SUPERSEDED]` tombstone). Isolated behind a flag so it never
   confounds the benchmark.

The existing **background sync engine** already moves rows SQLite↔Postgres in a daemon thread; the
embedding is just another column it carries (authoritative vector in Postgres). Extraction/curation
ride the same off-hot-path lane. The synchronous store call must **not** block on embedding/
extraction if it would delay the response — these run async.

### A.3 Read path (hot path) — three CTEs, RRF fusion, importance prior

Replace the single ts_rank ORDER BY with three top-N legs over the **same `memories` table**, fused
in the handler:

1. **Lexical leg** — the *existing* query verbatim: `plainto_tsquery('english', $q)` +
   `ts_rank(search_vector, query)`, with the existing OR-broaden fallback
   (`OR_BROADEN_MIN_RANK`) kept intact. `rank_lex` = position in this list. (LIMIT ~50.)
2. **Dense leg** — `ORDER BY embedding <=> $qvec LIMIT 50` using the HNSW index. `rank_dense` =
   position. Sensitive rows (NULL embedding) never enter this list.
3. **Graph leg (phase 2, gated, currently disabled)** — seed concept nodes by reusing the
   lexical+dense match, traverse 1–2 hops via a recursive CTE over the edge table, score reachable
   memories by hop-decay → `rank_graph`. List allowed to be empty.

**Fuse** in Python: `fused(d) = Σ_{s} w_s / (60 + rank_s(d))`, default `w_lex = w_dense = 1.0`,
`w_graph ≈ 0.35` (down-weighted per the negative-prior finding — see benchmark). Missing leg ⇒ 0
contribution, no special-casing.

**Preserve the importance prior** (the current code is *not* pure relevance): apply it as a
post-fusion multiplier — `final(d) = fused(d) * (0.7 + 0.3*importance)` for `sort_by="relevance"`,
or use importance as the tie-break for `sort_by="importance"`. Importance is a *prior*, **not** a
fourth fused list. `sort_by="recency"` stays a pure `ORDER BY created_at`, untouched.

> **Why RRF, not convex combination, as the default:** we fuse three incomparable scales
> (unbounded ts_rank, bounded cosine, arbitrary graph proximity), one of them sparse/often-empty.
> RRF is scale-agnostic and treats a missing leg as a clean 0, where CC would force a maintained
> normalization per signal plus a decision for "absent." RRF also collapses to today's exact
> lexical ordering when dense/graph are empty (the SQLite degrade path, **same code**). The
> benchmark ran CC/TM2C2 as a challenger (ADR-0001 is quality-gated); on our set RRF was chosen.

**Single-query alternative (future):** the three legs can be expressed as CTEs + a FULL OUTER JOIN
on `id` with RRF computed in SQL (Supabase hybrid-search pattern), saving a round-trip. The
prototype and initial production both fuse in Python for clarity; in-DB fusion is an optimization,
not a correctness change.

### A.4 SQLite-only graceful degrade

With only the lexical leg present, RRF reduces to ranking by `rank_lex` — **identical ordering to
today's FTS5 `bm25()`**. Zero behaviour change offline; the *same* fusion code path runs in both
modes (dense/graph legs simply empty). Satisfies ADR-0002.

### A.5 Concept graph (phase 2 — designed, gated, NOT shipped in v1)

If a future benchmark justifies it (the prototype's did not — §B.4):

```sql
CREATE TABLE concepts (
  id            bigserial PRIMARY KEY,
  canonical_name text NOT NULL,
  aliases       text[],
  embedding     halfvec(1024),   -- for canonicalization + query seeding
  category      text
);
CREATE TABLE concept_edges (
  src_id        bigint REFERENCES concepts(id),
  dst_id        bigint REFERENCES concepts(id),
  relation      text NOT NULL,
  weight        real,
  valid_from    timestamptz,     -- bi-temporal (Graphiti-style supersede)
  valid_to      timestamptz,
  evidence_memory_ids bigint[]
);
CREATE TABLE memory_concepts (   -- the "mentions" link
  memory_id     bigint REFERENCES memories(id),
  concept_id    bigint REFERENCES concepts(id),
  relation      text
);
```

- **Construction** (backfill): batched open LLM triple-extraction (~10–25 calls for the whole
  corpus, each memory id-tagged) → global embedding-cluster canonicalization (EDC/KGGEN style) →
  write the three tables. Off the hot path; `is_sensitive=1` filtered *before* any call.
- **Incremental** (per new memory): extract its triples, resolve entities against `concepts` via
  pgvector NN + threshold (LLM tie-break only on ambiguity), set-merge — never re-cluster.
- **Traversal at recall:** plain recursive SQL CTE (1–2 hops). **No Apache AGE** — our multi-hop is
  shallow. If a future need for PPR arises, compute it in Python over a cached `scipy.sparse`
  transition matrix loaded from the edge table (Postgres has no native PPR), rebuilt only on graph
  mutation.
- **Bi-temporal edges** realize our "supersede, don't accumulate" rule as a queryable timeline:
  contradicted edges get `valid_to` set, not deleted.

### A.6 Optional stage-2 cross-encoder (gated separately)

`bge-reranker-v2-m3` on the GPU node over the fused top ~20–30, `sort_by="relevance"` only,
sensitive rows excluded, with a hard-timeout fallback to fused order. Ship only if it clears both
the quality bar **and** the p95 hot-path budget. Not in v1.

### A.7 Infrastructure (production deploy)

- **pgvector enablement on CNPG:** the cluster already runs pgvector for Immich, so the legacy
  custom-operand-image path is in place; `CREATE EXTENSION vector` + the additive migration. Any
  extension add triggers a rolling restart of the shared cluster — coordinate via presence/GitOps.
- **All cluster changes via Terraform/Terragrunt** in `infra/stacks/...` (GitOps, never kubectl).
- **Embedding/extraction compute:** in-cluster **llama-cpp on the GPU node** for sensitive-safe
  local processing (and the no-key fallback); **Voyage-3.5** (hosted) for the non-sensitive batch
  (ADR-0003). Sensitive memories are routed locally or left lexical-only — enforced, not
  best-effort.
- **PgBouncer:** set `hnsw.ef_search` via `SET LOCAL` inside the recall transaction (transaction
  pooling).
- **pgvectorscale/DiskANN deferred** — not needed below ~1–5M vectors.

---

## B. Prototype as built (the benchmark harness)

The prototype validates **retrieval quality cheaply, in-process** — *not* pgvector/Postgres
(standing up CNPG just to benchmark would burn days before knowing if hybrid even beats FTS). It is
a faithful stand-in: the lexical leg is the *exact* production code path, and the fusion is the same
weighted RRF the production design specifies.

### B.1 Files (committable code only; data/cache/results gitignored)
- `benchmarks/retrievers/fts.py` — `FtsRetriever`, the lexical baseline.
- `benchmarks/retrievers/hybrid.py` — `HybridRetriever`, the three-leg fusion.
- `benchmarks/retrievers/test_hybrid.py` — 9 model-free tests (synthetic content only).
- `benchmarks/scripts/run_eval.py`, `benchmarks/harness/` — runner + metrics.
- Eval data (`benchmarks/data/{corpus,queries,qrels}.jsonl`), embedding cache
  (`benchmarks/cache/*.npy`), and full results (`benchmarks/results/*.json`) are **gitignored**
  (verified via `git check-ignore`) — privacy rule: no real memory content committed.

### B.2 Lexical leg = the real product

`hybrid.py` **reuses `retrievers.fts.FtsRetriever` verbatim**, which is itself a faithful
reimplementation of `src/claude_memory/mcp_server.py::_sqlite_recall` (`sort_by="relevance"`): a
fresh in-memory FTS5 index over the 5,452-memory corpus with the production virtual-table shape and
default `unicode61` tokenizer; query handling mirrors production (AND-match first, OR-broaden if
zero rows; rank by `-bm25()*0.7 + importance*0.3`; LIKE fallback on operational errors). **So the
hybrid's lexical component *is* the exact production system it must beat — no drift.**

### B.3 Dense leg

- **Model:** `BAAI/bge-large-en-v1.5`, 1024-d, L2-normalized. Passages raw; the query gets the BGE
  instruction prefix `"Represent this sentence for searching relevant passages: "`. Similarity =
  cosine via numpy matmul over the normalized matrix (faiss unnecessary at N=5452).
- **Embeddings:** all 5,452 memories embedded once (one-time ≈ 31.5 min on a CPU-only box),
  **cached** fingerprint-keyed to `cache/emb_BAAI_bge-large-en-v1.5_<corpusfp>.npy` (+ `.ids.npy`)
  → reruns skip the embed (cache-hit rebuild ≈ 8.3 s).
- Hosted Voyage/OpenAI/Cohere paths are implemented and key-gated but were **untriggered** (no key
  in env) — so the prototype ran the local default, which is also the sensitive-only production
  fallback. **Production maps this matrix to pgvector `halfvec(1024)`.**

### B.4 Graph leg (built, but structurally excluded from the ranking — UNMEASURED)

- **Construction with ZERO LLM calls** (the tractable shortcut): concepts = union of each memory's
  `tags` + the already-LLM-generated `expanded_keywords` field + a lightweight regex/stop-word
  noun-phrase proxy over `content`, normalized + de-pluralized. 37,075 concepts extracted; **19,907
  kept** after document-frequency pruning (df ∈ [2, 2%·N=109]: df<2 links nothing, df>109 are
  non-discriminative hubs). Concept cliques → weighted memory–memory edges. Result: **5,452 nodes,
  2,095,624 edges**, built in ~9 s (in-memory networkx).
- **Traversal:** 1 hop from the top-10 RRF seeds (capped 25 neighbours/seed), contributing **only ids
  not already in the base legs** — this exclusion is the bug (next bullet).
- **Result: the graph was structurally barred from the ranking, so its value is UNMEASURED.** Because
  graph hits are restricted to ids *outside* the FTS∪dense base set and weighted 0.35, a graph-only id's
  max RRF (`0.35/61 ≈ 0.0057`) sits below any base-leg id's min (`1.0/110 ≈ 0.0091`) — it can never enter
  the fused top-k. The "graph ≡ nothing" ablation (§B.6) was therefore **guaranteed by construction, not
  an empirical finding** (a post-run review found a relevant graph-surfaced memory that fusion discarded).

### B.5 Fusion (the production recipe, exactly)

Weighted RRF, `RRF(d) = Σ_leg w_leg/(60 + rank_leg(d))`, chosen over convex combination because it
is score-scale-free (no BM25-vs-cosine calibration). Weights `w_fts = 1.0, w_dense = 1.0,
w_graph = 0.35`. Each leg pulled to depth 50 before fusion, truncated to k.

### B.6 Decision-relevant ablation (this is what informs ADR-0004)

| Config | What | Overall recall@10 | Para recall@10 | Multi recall@10 |
|---|---|---|---|---|
| **A** full hybrid (FTS+dense+graph) | the prototype | 0.834 | 0.725 | 0.775 |
| **B** FTS+dense (w_graph=0) | graph removed | **0.834** | **0.725** | **0.775** |
| **C** dense-only | | 0.748 | — | — |
| **D** FTS-only (= baseline) | | 0.695 | 0.375 | 0.711 |

**A and B are identical to three decimals on every metric — but this is a structural artifact (§B.4),
not a test of the graph.** The valid signal here is the **FTS-vs-dense decomposition**: dense-only (C)
and FTS-only (D) each lose to the fusion (B) — dense recovers paraphrase, lexical recovers exact, fusion
gets the best of both. The concept graph itself is **unevaluated** (it could never affect top-k under
this fusion config). **This still supports phasing — ship lexical+dense (phase 1), the robust measured
win — but the graph is gated pending a *valid* retest, not because it failed.** (Configs B/C/D were not
persisted as result JSONs; only A and D are reproducible from committed artifacts.)

### B.7 Prototype → production mapping

| Prototype (in-process) | Production (ADR-0002) |
|---|---|
| numpy cosine over normalized matrix | pgvector `halfvec(1024)` + HNSW ANN |
| `.npy` embedding cache, fingerprint-keyed | `embedding` column on `memories`, synced |
| in-memory networkx graph (phase 2) | `concepts` / `concept_edges` / `memory_concepts` tables |
| `FtsRetriever` (FTS5 in-memory) | existing `search_vector` + GIN (`plainto_tsquery`/`ts_rank`) |
| weighted RRF in Python | same RRF (Python handler, or CTE+FULL OUTER JOIN in SQL) |
| bge-large local | Voyage-3.5 hosted (non-sensitive) / bge-large local (sensitive, no-key) |

### B.8 Prototype caveats (carried into the report's limitations)
1. **Graph result is INVALID, not merely "null."** The fusion config barred the graph leg from the
   top-k by construction (§B.4), so the benchmark did not actually test it. A valid retest must include
   graph candidates in the fused pool (drop the base-set exclusion) and/or sweep the weight, ideally with
   a typed-relation graph from real LLM extraction and multi-hop queries whose hops are *not* semantically
   adjacent.
2. **Exact-stratum nDCG/MRR dip ~0.018/0.025** vs FTS (recall unaffected) is the standard RRF cost
   of blending one perfect hit with near-ties; a small exact-match rank bonus could recover it.
3. **Latency** (p50 ≈ 230 ms) is CPU-bound on the local query embed; non-gating, GPU/hosted ~10×
   faster. Baseline FTS was p50 ≈ 15.7 ms (pure SQLite).
4. **No pgvector/Postgres** in the prototype — the production substrate is design-only here; the
   numbers measure *retrieval quality*, which transfers, not the production latency profile.

---

## C. Open questions (for production rollout)

1. **pgvector enablement mechanism** — confirm whether the live CNPG is on the legacy
   custom-operand-image (likely, since Immich uses pgvector) or the modern image-volume-extensions
   path; either way the migration is additive, but the enablement DDL/Terraform differs.
2. **Graph gate** — what evidence would re-open the concept graph? Candidate: a multi-hop eval slice
   whose hops are *not* semantically adjacent (where dense can't shortcut), built from real
   LLM-extracted typed relations rather than keyword co-occurrence. Until then, graph stays off.
3. **Voyage vs bge-large in production** — the benchmark ran bge-large (local). A cheap follow-up:
   re-run the dense leg with Voyage-3.5 on the non-sensitive corpus to confirm the hosted model's
   higher quality ceiling holds on *our* content before committing the production default.
