# Benchmark report: FTS (lexical baseline) vs Hybrid (lexical + dense + graph)

Status: **the ADR-0001 decision instrument.** This is the head-to-head that gates hybrid
adoption. Read the [survey](survey.md) for the landscape and the
[integration design](integration-design.md) for what was built.

**Bottom line:** Hybrid (lexical FTS ⊕ dense embeddings, RRF-fused) beats the lexical FTS baseline on
**every overall metric**, driven by the **paraphrase** stratum (recall@10 **+0.350**, robust under a
paired bootstrap) — precisely the gap embeddings were meant to close, with **no recall regression on
exact**. **Recommendation: adopt the lexical + dense fusion;** the ADR-0001 quality gate is met by
that fusion alone.

> **⚠️ Post-review corrections — read before citing this report.** An adversarial completeness review
> after the run found that two prominent claims in the original draft did NOT survive scrutiny. They are
> corrected in place below; the full review is in the [Appendix](#appendix--adversarial-completeness-review).
>
> 1. **The concept graph was NOT shown to be useless — it was never validly tested.** The prototype's
>    fusion restricted the graph leg to candidate ids *outside* the FTS∪dense base set and weighted it at
>    0.35, which makes it **mathematically impossible** for a graph-only result to enter the fused top-k
>    (max graph RRF `0.35/(60+1)≈0.0057` < any base-leg minimum `1.0/(60+50)≈0.0091`). So the ablation
>    "full-hybrid ≡ FTS+dense to three decimals" was **guaranteed by the fusion config, not an empirical
>    finding** — the review even found a genuinely-relevant memory the graph surfaced (and both base legs
>    missed) that fusion then discarded. The concept graph is therefore **UNEVALUATED**, and is deferred on
>    **cost + invalid-test** grounds, *not* because it failed. A valid retest must put graph ids in the
>    fused candidate pool (no base-set exclusion) and/or sweep the graph weight.
> 2. **The multi-hop "win" is not statistically significant.** A paired bootstrap (B=10000) puts 3 of the
>    4 multi-hop metric CIs across zero (recall@10 Δ+0.064, P(Δ≤0)≈0.06). Only the **overall** and
>    **paraphrase** deltas are robust. Multi-hop deltas are de-bolded below. Since multi-hop is the entire
>    rationale for a concept graph, there is currently **no statistically distinguishable evidence** that
>    *anything* (dense or graph) helps the multi-hop stratum.
>
> Also: absolute recall/nDCG *levels* are biased low (binary, un-pooled qrels — §6); only FTS-vs-hybrid
> **deltas** are trustworthy. None of this changes the adopt-lexical+dense recommendation, which rests on
> the robust paraphrase/overall win.

---

## 1. Methodology

### 1.1 Test collection
- **Corpus:** 5,452 memories (`benchmarks/data/corpus.jsonl`, gitignored — privacy). Sensitive
  memories (`is_sensitive=1`) excluded entirely per ADR-0003. The corpus is the user's real
  personal memories; **only aggregate numbers and synthetic illustrations appear in this document.**
- **Queries:** 119, stratified — **40 exact / 40 paraphrase / 39 multi-hop**
  (`benchmarks/data/queries.jsonl`).
- **Qrels:** `benchmarks/data/qrels.jsonl`, LLM-generated (the LongMemEval pipeline inverted for a
  memory store: seed memory → exact/paraphrase/multi-hop query → relevant ids).

### 1.2 The two systems
- **`fts` (baseline)** — `benchmarks/retrievers/fts.py::FtsRetriever`. A **faithful
  reimplementation of the production code path** `src/claude_memory/mcp_server.py::_sqlite_recall`
  (`sort_by="relevance"`): in-memory FTS5 over the full corpus with the production virtual-table
  shape + default `unicode61` tokenizer (no stemming/stop-words); AND-match first, OR-broaden only
  if AND returns zero; rank by the production blend `-bm25()*0.7 + importance*0.3`; LIKE fallback on
  operational errors. **This is "the current system" the hybrid must beat — not the simplified
  README prose.**
- **`hybrid`** — `benchmarks/retrievers/hybrid.py::HybridRetriever`. Three legs fused with weighted
  RRF: (1) lexical = **`FtsRetriever` reused verbatim** (so the hybrid's lexical component *is* the
  baseline — no drift); (2) dense = `BAAI/bge-large-en-v1.5` (1024-d, local, L2-normalized, BGE
  query-instruction prefix), cosine via numpy; (3) graph = a keyword-co-occurrence memory-node graph
  (5,452 nodes / 2,095,624 edges), 1-hop expansion from the top-10 seeds.
  Fusion: `RRF(d) = Σ_leg w_leg/(60 + rank_leg(d))`, `w_fts = w_dense = 1.0, w_graph = 0.35`, each
  leg to depth 50.

### 1.3 Metrics & protocol
- **recall@5, recall@10** (hot-path "did we surface it"), **nDCG@10** (graded, position-aware
  headline), **MRR** (first-hit). Per stratum and overall.
- `retrieve_k=20`, run via `scripts/run_eval.py --retriever retrievers.{fts,hybrid}:…`. Deterministic;
  both invocation paths (programmatic + CLI) verified identical.
- **`sort_by="relevance"` pinned across both arms** (not the production `importance` default), so the
  benchmark measures *retrieval* quality, not the importance prior — and everything else (corpus,
  queries, OR-broaden behaviour) is held fixed.
- Full result JSONs written only to gitignored `benchmarks/results/{fts,hybrid}.json`; retriever code
  contains no embedded corpus content (safe to commit).

---

## 2. Head-to-head results

### 2.1 Comparison table (FTS vs Hybrid, with deltas)

| Stratum | Metric | FTS | Hybrid | Δ |
|---|---|---:|---:|---:|
| **Overall** (n=119) | recall@5 | 0.6663 | 0.7415 | **+0.0752** |
| | recall@10 | 0.6952 | 0.8338 | **+0.1386** |
| | nDCG@10 | 0.6507 | 0.7284 | **+0.0777** |
| | MRR | 0.6737 | 0.7297 | **+0.0560** |
| **Exact** (n=40) | recall@5 | 1.0000 | 1.0000 | +0.0000 |
| | recall@10 | 1.0000 | 1.0000 | +0.0000 |
| | nDCG@10 | 0.9908 | 0.9723 | −0.0185 |
| | MRR | 0.9875 | 0.9625 | −0.0250 |
| **Paraphrase** (n=40) | recall@5 | 0.3500 | 0.5500 | **+0.2000** |
| | recall@10 | 0.3750 | 0.7250 | **+0.3500** |
| | nDCG@10 | 0.3123 | 0.5023 | **+0.1900** |
| | MRR | 0.2958 | 0.4343 | **+0.1385** |
| **Multi-hop** (n=39) | recall@5 | 0.6485 | 0.6726 | +0.0241 ¹ |
| | recall@10 | 0.7111 | 0.7748 | +0.0637 ¹ |
| | nDCG@10 | 0.6491 | 0.7101 | +0.0610 ¹ |
| | MRR | 0.7393 | 0.7940 | +0.0547 ¹ |

¹ **Multi-hop deltas are NOT statistically significant.** Paired bootstrap (B=10000): recall@5
`CI[−0.046,+0.095]` (P(Δ≤0)≈0.25), recall@10 `CI[−0.020,+0.143]` (P≈0.06), MRR `CI[−0.038,+0.143]`
(P≈0.12); only nDCG@10 is marginal (P≈0.04). Treat the multi-hop stratum as **inconclusive**. The
overall and paraphrase deltas, by contrast, have CIs well clear of zero (P≤0.003).

### 2.2 Reading the strata

- **Exact — recall held at 1.0, but that is partly circular.** Exact queries were *generated* as "a
  salient phrase whose top FTS hit is memory X," then X was labelled relevant — so FTS recall@5/@10 = 1.0
  is substantially **guaranteed by how the stratum was built**, not an independent property. The one
  genuine signal here is hybrid's small **degradation**: nDCG@10 −0.018 `CI[−0.046,0]`, MRR −0.025
  `CI[−0.063,0]` — a real, consistent rank demotion from blending one perfect lexical hit with dense
  near-ties. The proposed exact-match rank bonus (ADR-0005) would recover it, **but that fix is asserted,
  not measured.** Recall is unaffected; the cost is rank-position only.

- **Paraphrase — the LARGEST WIN, the dense leg's payoff.** recall@10 **+0.350** (0.375 → 0.725),
  recall@5 **+0.200**, nDCG@10 **+0.190**. This is exactly the low-lexical-overlap stratum embeddings
  were predicted to fix: lexical FTS finds barely a third of paraphrased answers; adding dense nearly
  doubles recall@10. **This stratum alone justifies the hybrid.**

- **Multi-hop — INCONCLUSIVE (deltas not significant).** recall@10 +0.064, nDCG@10 +0.061, MRR +0.055,
  but 3 of 4 CIs cross zero (footnote ¹). So we **cannot claim** a multi-hop win for hybrid. We also
  cannot attribute anything to the graph: per the §3 caveat, the fusion config structurally barred the
  graph leg from the top-k, so the multi-hop stratum tests only FTS vs FTS+dense — and that difference is
  not statistically distinguishable here. Multi-hop is exactly the stratum a *properly tested* concept
  graph is meant to win, and it remains an open question.

---

## 3. Ablation — what each leg contributes

Four configs, everything else fixed:

| Config | Legs | Overall recall@10 | Overall nDCG@10 | Para recall@10 | Multi recall@10 | Exact nDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| **A** | FTS + dense + graph (full hybrid) | 0.834 | 0.728 | 0.725 | 0.775 | 0.972 |
| **B** | FTS + dense (`w_graph=0`) | **0.834** | **0.728** | **0.725** | **0.775** | **0.972** |
| **C** | dense only | 0.748 | — | — | — | 0.861 |
| **D** | FTS only (= baseline) | 0.695 | 0.651 | 0.375 | 0.711 | 0.991 |

**A ≡ B to three decimals on every metric — but this is a STRUCTURAL ARTIFACT, not a test of the graph.**
The fusion restricts the graph leg to ids *outside* the FTS∪dense base set and weights it at 0.35, so a
graph-only id's maximum RRF score is `0.35/(60+1) ≈ 0.0057`, strictly below the *minimum* score of any id
from a base leg (`1.0/(60+50) ≈ 0.0091`). Since both base legs return 50 ids, **every base-leg id outranks
every graph-only id** — a graph-only result can never enter the fused top-k, regardless of corpus, query
set, or graph quality. A ≡ B was **mathematically guaranteed before any data ran.** The honest reading is
"the graph cannot affect top-k *under this fusion config*," NOT "the concept graph contributes nothing." (A
spot check found a genuinely-relevant memory the graph surfaced and both base legs missed at depth 50 —
fusion discarded it.) **The graph is therefore unevaluated.**

What the ablation *does* validly show is the **FTS-vs-dense decomposition**, which stands:
- **Dense recovers paraphrase** (C beats D's 0.375 para recall@10 decisively) but is weaker on exact
  (C exact nDCG 0.861 vs D's 0.991).
- **Lexical recovers exact** (D exact nDCG 0.991, the best) but collapses on paraphrase.
- **Fusion (B) gets the best of both** — exact recall stays perfect, paraphrase nearly doubles.
- *Caveat:* configs B/C/D were not persisted as result JSONs, so these specific numbers are not
  independently reproducible from committed artifacts (only A = full hybrid and D = FTS are).

**To actually test the graph** (deferred follow-up — [ADR-0004](../adr/0004-phase-the-hybrid-lexical-dense-first-graph-gated.md)):
put graph candidates in the fused pool (drop the base-set exclusion) and/or sweep `w_graph` upward, on a
multi-hop slice whose hops are *not* semantically adjacent (so the dense leg can't shortcut them), using
real typed-relation extraction rather than the prototype's zero-LLM keyword co-occurrence graph.

---

## 4. Latency & storage (measured, NON-GATING per ADR-0001)

### 4.1 Latency (per-query `retrieve()`, CPU-only box, no GPU)

| System | p50 | p95 | mean | max |
|---|---:|---:|---:|---:|
| FTS (pure SQLite) | 15.7 ms | 27.8 ms | 12.8 ms | 31.9 ms |
| Hybrid | 229.6 ms | 344.5 ms | 249.3 ms | 640.0 ms |

The hybrid's ~230 ms p50 is **dominated by the local bge-large query embedding** (one CPU forward
pass). On the production GPU node or via a hosted API this drops ~10× to low tens of ms. The FTS,
dense-ANN, and RRF-merge costs themselves are negligible. **Latency does not gate adoption** (the
success metric is quality-first), and the production read path (pgvector HNSW + GPU/hosted query
embed) is far faster than this prototype's CPU profile.

### 4.2 Storage

| Component | Size | Notes |
|---|---|---|
| FTS5 in-memory index | ~8.3 MB | SQLite shadow tables over 5,452 memories |
| Dense matrix | 22.3 MB | 5,452 × 1024 float32; cached `.npy`, fingerprint-keyed |
| Concept graph (in-memory) | ~202 MB (Python-object estimate) | 5,452 nodes + 2,095,624 edges, networkx; **not persisted, not shipped** |
| Total reported index | 232.2 MB | matrix + FTS + graph estimate |

Production maps the dense matrix to pgvector `halfvec(1024)` (~2 KB/row, single-digit MB total for
the corpus) and — only if the graph is ever adopted — three Postgres node/edge tables. No
pgvector/docker in the prototype; in-process numpy cosine (faiss unnecessary at N=5452). Storage is
reported, not gating.

---

## 5. Recommendation

**ADOPT — ship lexical + dense fusion (phase 1); defer the concept graph behind a gate.**

Rationale:
1. **The ADR-0001 quality gate is met.** Hybrid beats FTS on all four overall metrics, with the
   decisive, statistically-robust win on **paraphrase** (recall@10 +0.350) — precisely the gap embeddings
   were meant to close — with **no recall regression on exact**. (The multi-hop deltas are *not*
   statistically significant — §2.2, footnote ¹ — so they are not part of the case for adoption.)
2. **The gain is the FTS+dense fusion, not the graph.** The ablation shows full-hybrid ≡ FTS+dense.
   So phase 1 = embeddings (pgvector) fused with the existing FTS via weighted RRF, preserving the
   importance prior — exactly the [integration design](integration-design.md) §A.
3. **The concept graph stays GATED — because it was not validly tested, not because it failed**
   ([ADR-0004](../adr/0004-phase-the-hybrid-lexical-dense-first-graph-gated.md)). The prototype's fusion
   config structurally barred it from the top-k (§3), so this benchmark says nothing about its value.
   Deferral is justified by operational cost (LLM extraction + two extra tables + traversal) **plus the
   remaining uncertainty** — not by evidence of uselessness. Re-open with a valid retest: graph candidates
   in the fused pool (no base-set exclusion) and/or a swept weight, on non-semantically-adjacent multi-hop
   queries built from real typed-relation extraction.
4. **The small exact-stratum nDCG/MRR dip** is the known RRF blending cost on recall-perfect queries;
   an exact-match rank bonus is a cheap follow-up ([ADR-0005](../adr/0005-rrf-default-cc-challenger.md)
   records RRF as the default with the bonus as a tunable).

Adopt with the changes already designed: pgvector `halfvec(1024)` + HNSW, weighted-RRF fusion with
the importance prior preserved, sensitive memories excluded from embedding (ADR-0003), and SQLite-
only mode unchanged (ADR-0002).

---

## 6. Limitations (stated honestly)

1. **Label noise / qrels quality.** Qrels were **LLM-generated** with **lighter hand-verification
   than the ideal protocol** (no measured Cohen's κ between LLM and human judgments). LLM judges are
   systematically lenient, and the eval is **119 queries (~40/stratum)** — *below* the ~50/stratum
   the literature (Voorhees & Buckley) recommends for confident per-stratum ranking, and **no
   bootstrap CIs or paired significance test** were computed. The overall and paraphrase deltas
   (+0.14, +0.35 recall@10) are large enough to be robust to plausible label noise; the multi-hop
   (+0.06) and the exact-stratum dip (~0.02) are within the range where label noise could matter, so
   treat those as directional, not precise.

2. **Pooling / "holes."** It is not confirmed that qrels pooled the top-k of *all* arms (FTS, dense,
   hybrid) before judging. If the pool was lexical-biased, the metric is **biased against** the dense
   and hybrid arms (they retrieve unjudged relevant memories scored as misses) — meaning the true
   hybrid uplift could be *larger* than reported, not smaller. This does not threaten the "adopt"
   conclusion but caveats the exact magnitudes.

3. **Snapshot vs pgvector (substrate mismatch).** The prototype used an **in-process numpy** dense
   index over a static corpus snapshot, **not** the production pgvector HNSW on live CNPG Postgres.
   Retrieval *quality* transfers (cosine is cosine; HNSW recall at this scale is ~exact), but the
   production **latency profile, ANN approximation, and filtered-top-k behaviour are unmeasured here**
   and must be validated post-migration.

4. **Extraction shortcuts AND a structurally-excluded graph leg.** The concept graph was built with
   **zero LLM calls** — concepts = `tags` ∪ `expanded_keywords` ∪ a regex noun-phrase proxy, edges from
   keyword co-occurrence — **not** the typed-relation, LLM-extracted graph the production design (§A.5)
   specifies. More importantly, the fusion config **structurally barred even this cheap graph from the
   top-k** (§1 corrections, §3), so this run is **not a valid test of the graph at all** — not of the
   cheap construction, and certainly not of a properly LLM-extracted typed-relation graph. The graph is
   *gated and unevaluated*, not *killed*.

5. **Embedding model is the prototype default, not the production pick.** Numbers are for
   **bge-large-en-v1.5** (local). Production should use **Voyage-3.5** (also 1024-d) for
   non-sensitive memories (ADR-0003); its higher quality ceiling on *our* content is unverified — a
   cheap, recommended follow-up (re-run the dense leg with Voyage).

6. **`sort_by="relevance"` not the production default.** The benchmark pins `relevance` to isolate
   retrieval quality; production defaults to `importance`-blended ranking. The design preserves
   importance as a post-fusion prior, but the *user-visible* ranking under the default sort was not
   benchmarked.

7. **Single user, dense corpus.** ~5,452 memories from one author are topically adjacent with many
   near-duplicates, so "the one relevant id" is sometimes fuzzy; graded judgments over a pool mitigate
   this but it remains a property of the corpus that may not generalize.

---

## Appendix — adversarial completeness review

An independent critic reviewed this report after the run (verdict: **usable-with-caveats**). Its findings
drove the §1 corrections; the full list is recorded here so the review is part of the permanent record.

1. **Graph-null claim is circular, not empirical (most serious).** Proven that the fusion config
   (graph leg excluded from the base set; `w_graph=0.35`) caps a graph-only id's RRF at `0.35/61≈0.0057`,
   below any base-leg id's `1.0/110≈0.0091`. `A≡B` was guaranteed before any data. Honest statement: "the
   graph cannot affect top-k under this fusion config," not "the graph contributes nothing."
2. **Graph mechanism mis-attributed, and contradicted by the data.** Reconstructing the legs (graph =
   5,452 nodes / 2,095,624 edges) on a 15-query sample found `rel_only_via_graph=1`: a relevant memory the
   graph surfaced that was absent from BOTH base legs at depth 50 — dense did NOT already cover it, and
   fusion discarded it. The "dense already retrieves them" explanation is false in ≥1 observed case.
3. **Multi-hop wins are not statistically supported — yet were bolded.** Paired bootstrap (B=10000):
   overall & paraphrase robust (P≤0.003); multi-hop 3/4 CIs cross zero (recall@10 P≈0.06). Multi-hop is the
   sole rationale for the graph, and it shows no statistically distinguishable hybrid advantage.
4. **Exact stratum is circular by construction.** All 40 exact queries were generated as "top FTS hit for a
   salient phrase of ⟨id⟩," so FTS's 1.0 is largely tautological. The only genuine exact signal is hybrid's
   small nDCG/MRR demotion, whose claimed rank-bonus fix is asserted, never measured.
5. **qrels are binary and un-pooled, so nDCG is mislabeled and absolute numbers unreliable.** Binary labels
   make nDCG degenerate to position-discounted recall (no graded info). Un-pooled, author-assigned labels
   bias absolute recall/nDCG **low** (esp. for dense/hybrid); only the FTS-vs-hybrid **delta** is trustworthy.
6. **Underpowered and single-corpus.** 119 queries (~40/stratum) is below the cited ~50/stratum standard;
   65% of queries have exactly one relevant id (recall@5≈recall@10 for most). One author, no second corpus,
   no inter-annotator agreement: external validity asserted, not demonstrated.
7. **Headline metric overstates hot-path value.** recall@10 leads everywhere, but the auto-recall hook
   injects a top-k into the prompt; recall@5 (+0.075) and MRR/first-hit are the decision-relevant metrics.
   Hybrid's recall@10 edge partly comes from pushing answers into ranks 6–10. The hook's effective `k` is
   never stated.
8. **Production substrate and model wholly unmeasured.** Numbers are local exact numpy cosine over
   bge-large; production is pgvector HNSW (approximate; recall depends on `ef_search`) with a filtered top-k
   (NULL-embedding sensitive rows — a partial-index interaction that can hurt HNSW recall), using Voyage-3.5
   (never run). "Quality transfers" and "~10× faster on GPU/hosted" are assumptions, not measurements.
9. **Ablation configs A/B/C/D are not reproducible from artifacts.** Only `results/{fts,hybrid}.json`
   (= A and D) are persisted; B/C have no saved JSON, and there's no run-manifest/seed/version capture
   beyond the embedding-cache fingerprint. The decision-critical `A≡B` and the C/D numbers must be re-run.
10. **Several deferred fixes are asserted, not tested.** The exact-match rank bonus, the importance
    post-fusion prior (the benchmark pinned `sort_by="relevance"` without it, so the user-visible production
    ranking was never measured), and the "CC ran on our set and RRF was chosen" claim (no CC results exist
    anywhere) are all unverified.
