# Landscape survey: semantic + concept-graph memory for hybrid recall

Status: research input for the ADR-0001 hybrid upgrade. Scope: how the agent-memory and
graph-RAG literature builds and retrieves over a personal-memory store, which embedding
model to use, how to fuse lexical + dense + graph signals, and how to evaluate the result.

**Read this with the decisions already fixed in [ADR-0001](../adr/0001-pursue-hybrid-retrieval-embeddings-and-concept-graph.md)
–[0003](../adr/0003-external-embedding-apis-allowed-for-non-sensitive-memories.md):** we pursue
hybrid (gated on a benchmark beating FTS), embeddings live in pgvector on the existing CNPG
Postgres, the concept graph is node/edge tables in Postgres, sensitive memories never egress,
and adoption is decided **quality-first** (recall@k / nDCG@10 / MRR; latency & storage are
reported, not gating).

The recurring conclusion below: **borrow the ideas, not the engines.** None of the four
systems surveyed is a drop-in for our stack, but each contributes a mechanism we re-implement
natively on Postgres + pgvector.

---

## 1. Our workload is the opposite of GraphRAG's design target

Before comparing systems it helps to state what we are *not*. The graph-RAG family was built
for **global sensemaking** ("what are the themes across this corpus") over a **static document
collection**. Our workload is the reverse:

| Dimension | GraphRAG target | claude-memory-mcp |
|---|---|---|
| Unit | Long documents, chunked | Atomic, already-curated memories (avg ~500 chars) |
| Corpus dynamics | Static, indexed once | Append-heavy: a few hundred memories/day arriving |
| Query type | Corpus-wide summarization | Point / multi-hop recall ("what did I decide about X") |
| Hot path | Offline batch | **Every user prompt** (auto-recall hook fires before each turn) |
| Scale | 10k–1M+ chunks | ~5k memories today → tens of thousands |

This mismatch is the lens for everything that follows. The expensive part of GraphRAG —
community detection + hierarchical LLM summaries — is the *wrong retrieval unit* for atomic
point lookups, and re-summarizing communities on a sustained append stream is its dominant,
unbounded cost. We want a design whose index-time work is **proportional to new content**, and
whose retrieval path has **no LLM call** (so it fits the per-prompt budget).

---

## 2. The GraphRAG family — Microsoft GraphRAG, LightRAG, nano-graphrag, LazyGraphRAG

All four turn text into an entity–relation knowledge graph via LLM extraction; they differ on
the expensive part (community detection + hierarchical summarization), which is exactly where
**incremental cost** lives.

### Microsoft GraphRAG (Edge et al. 2024, arXiv 2404.16130)
Pipeline: chunk → LLM extracts entities + relationships per chunk (with multi-round
"gleanings" to catch misses) → summarize duplicate element instances into node/edge
descriptions → build graph → **Leiden** community detection producing a *hierarchy*
(levels C0..C3) → an LLM writes a **community report** for every community at every level.
Two query modes: **global** (map-reduce over all community reports — corpus-wide
sensemaking) and **local** (start from query-relevant entities, fan out). Indexing is
LLM-heavy: ~4,000 LLM calls / ~35 min for one textbook; ~$20–40 per 1M tokens with gpt-4o.

**Incremental:** the `graphrag update` command (GraphRAG 1.0) computes deltas and places new
entities into existing communities "rather than re-running Leiden," re-summarizing only
changed communities — **but** maintainers warn that once drift crosses a threshold "the worst
case degrades to the same performance as a normal indexing." A periodic, unpredictable
full-reindex cliff on a sustained append stream. Parquet/file-pipeline oriented, not
Postgres-native.

### LightRAG (HKUDS, arXiv 2410.05779, EMNLP 2025)
Pipeline: chunk → LLM extracts entities + relations → "profiling" generates a key-value text
summary per node/edge → **deduplicate** merges identical entities/relations across chunks.
**No community detection.** Retrieval is **dual-level**: the LLM splits the query into
low-level keywords (specific entities) and high-level keywords (broad themes via relationship
chains); each set is matched by *vector* similarity against an entity-vector index and a
relation-vector index, then one-hop neighbours are pulled from the graph. Modes:
naive / local / global / hybrid / mix (mix = default).

**Incremental (the crux):** a new document goes through the same local indexing to produce a
small local graph, then is integrated by **set union** of node-sets and edge-sets into the
existing graph — "eliminating the need to rebuild the entire index graph." No communities ⇒
**no global re-clustering or re-summarization, ever** ⇒ O(new content) per insert, the only
genuinely-incremental member. Ships a PostgreSQL all-in-one backend (PGVectorStorage on
pgvector + PGGraphStorage on Apache AGE + KV + doc-status in one DB, PG ≥16.6).

### nano-graphrag (~1100 LOC)
A faithful minimal reimplementation of Microsoft GraphRAG and an excellent compact *reference*
for the exact extraction/community/report prompts. **Hard NO for incremental:** README states
plainly "each time you insert, the communities of graph will be re-computed and the community
reports will be re-generated" — O(whole graph) LLM cost per append.

### LazyGraphRAG (Microsoft Research, 2024)
Defers **all** LLM work to query time: index time uses only NLP noun-phrase extraction + graph
statistics — "indexing costs are identical to vector RAG and 0.1% of the costs of full
GraphRAG." Sidesteps the incremental-re-summarization problem entirely by never pre-summarizing
communities. The **defer-LLM-cost principle** is the one to borrow.

### Verdict for us
**Adopt none wholesale; steal LightRAG's architecture + LazyGraphRAG's defer-LLM principle.**
LightRAG is the only one whose incremental model (pure set-union, no re-clustering) structurally
fits an append-heavy stream, and whose retrieval path (vector + one-hop graph, no query-time
map-reduce) is hot-path-viable. But adopting LightRAG-the-product is not recommended: its
Postgres graph path needs the **Apache AGE** extension (not on our CNPG), and that path has
documented concurrency/entity-merge instability under append-heavy load (asyncpg pool timeouts
at the merge stage; slow upgrades). Our multi-hop is shallow (1–2 hops), expressible in plain
recursive SQL CTEs over node/edge tables — no AGE needed.

---

## 3. Zep / Graphiti — temporal knowledge graph for agent memory

Zep (arXiv 2501.13956) is an agent-memory service; **Graphiti** is its open engine
(Neo4j / FalkorDB / Kuzu backend, ~20k stars, MIT). It is the **closest conceptual analog** to
our hybrid goal — it fuses exactly the three signals ADR-0001 wants.

**Three-tier graph:** episode subgraph (raw ingested data, the provenance layer ≈ our Memory
rows) → semantic entity subgraph (entity nodes + typed relationship edges, each linking back to
its source episodes) → community subgraph (clusters with LLM summaries — the GraphRAG "global"
layer).

**Bi-temporal model:** every semantic edge carries **four** timestamps on two timelines —
*valid time* (`t_valid`/`t_invalid`: when the fact held true in the world) and *transaction
time* (`t'_created`/`t'_expired`: when Zep learned/retracted it). Facts are never deleted;
superseded facts get their validity window closed. This is a principled, queryable version of
our **"supersede, don't accumulate"** memory discipline.

**Incremental ingestion (per episode):** a *sequence* of LLM calls — entity extraction →
entity resolution/dedup (embed + cosine + BM25 search against existing nodes, then an LLM
judges merge vs create) → fact (edge) extraction → fact dedup → temporal extraction (resolve
"two weeks ago" against a reference time) → edge invalidation (LLM compares each new edge
against related existing edges; on a temporally-overlapping contradiction it closes the old
edge). Cost is **heavy on write**, paid back on reads.

**Retrieval (sub-second, NO LLM at query time):** three parallel searches fused, then reranked.
- `φ_cos`: cosine over embeddings of fact text / entity names / community summaries (BGE-m3, 1024-d).
- `φ_bm25`: BM25 full-text over the same fields.
- `φ_bfs`: breadth-first n-hop graph traversal from seed nodes — the genuinely graph-native signal.
- Rerank: pluggable — **RRF**, MMR (diversity), episode-mentions (frequency), node-distance, or a cross-encoder (most accurate, slowest).

**Published quality (the strongest evidence in this family):** on **LongMemEval**, Zep reports
**+18.5%** accuracy over a baseline (71.2% vs 60.2% with gpt-4o) *and* ~90% query-latency
reduction; on **MemGPT DMR**, 94.8% vs 93.4%. These are conversational long-context QA, not
personal-fact recall@k — so the headline numbers won't transfer directly, but the *fusion
recipe* is exactly what we benchmark.

### Verdict for us
**Primary design blueprint for the concept-graph half — but not an adopted dependency.**
Graphiti has **no pgvector backend**; adopting the engine forces a new Neo4j/FalkorDB graph DB
into the cluster, conflicting with ADR-0002 and reuse-before-building. We borrow four mechanisms,
re-implemented on Postgres: (1) the episodic(=Memory rows)/semantic(=new node+edge tables)
split; (2) the parallel-search + RRF fusion read path; (3) resolution-via-search to dedupe the
graph using our existing FTS+vector; (4) bi-temporal edge invalidation as the queryable form of
our supersede discipline. We de-scope the community/summarization tier and the default
cross-encoder. Two hard caveats: keep the multi-LLM-call extraction **off the hot path**
(background, like our sync engine), and route extraction through in-cluster llama-cpp / filter
`is_sensitive` per ADR-0003.

---

## 4. Mem0 / Mem0g — extraction-based, LLM-curated memory

Mem0 (arXiv 2504.19413) is a **write-side memory curator**, not a retrieval algorithm — it
solves a *different axis* than our gated problem, and is **complementary**.

**Two-phase pipeline.** *Extraction:* on each new message pair, an LLM (fed an async
conversation summary + the last ~10 messages) emits a set of concise "candidate facts."
*Update (the curation step):* for each candidate fact, retrieve the top ~10 semantically-similar
existing memories, then a function-calling LLM picks one of four ops — **ADD** (new), **UPDATE**
(merge richer detail into an existing id, gated on information content), **DELETE** (a
contradicted memory), **NOOP**. Net effect: the store self-deduplicates, self-merges, and
self-supersedes instead of accumulating. Two LLM calls per write (extract + decide) + a vector
search; **async by default** (off the user hot path); the **read/search path is pure vector
similarity with no LLM**.

**Mem0g (graph variant):** a directed labeled entity graph (Alice –lives_in→ SF) on Neo4j; a
conflict-detection + LLM update-resolver marks superseded relationships *invalid* rather than
deleting them.

**Published quality:** on **LOCOMO**, Mem0 J=66.88 / Mem0g 68.44 beats OpenAI memory (52.90),
A-Mem (48.38), LangMem (58.10), ties Zep (65.99), at ~1/15th the tokens of full-context; Mem0g
specifically wins temporal reasoning. Reference latencies (gpt-4o-mini): search p95 ≈ 0.20s,
total p95 ≈ 1.44s, vs full-context ≈ 17s.

### Verdict for us
**Adopt the curation loop as a separate, flagged subsystem — it does NOT move the ADR-0001
retrieval metric by itself** (its search is vector-only, no lexical+graph fusion). The
ADD/UPDATE/DELETE/NOOP loop is the highest-leverage idea Mem0 offers: it automates a discipline
our own rules already mandate (every correction stored, supersede-don't-accumulate, tombstones)
but currently leave to manual human effort. It is cheap to build against our existing Memory
model + `update_memory` endpoint, runs async off the recall hot path, and respects the
`is_sensitive` boundary. **Hard guardrails required:** never physically DELETE — supersede to a
`[SUPERSEDED]` tombstone (importance ~0.3, per our convention); log every op; gate behind the
non-sensitive filter. Keep extraction *optional* (our memories are already atomic, so usually
only the single UPDATE-decision call is needed). Mem0g's "mark invalid, not delete" and triplet
schema (source, relation, dest) are reusable ideas, but implemented on pgvector/Postgres, not
Neo4j. **Critically: isolate curation behind a flag so the benchmark measures retrieval quality
independently of any curation behaviour change.**

---

## 5. HippoRAG / HippoRAG 2 — Personalized PageRank over a concept graph

HippoRAG (NeurIPS 2024, arXiv 2405.14831) and HippoRAG 2 (ICML 2025, arXiv 2502.14802) are the
**strongest published evidence that a concept graph wins on multi-hop** — precisely the query
class ADR-0001 says the graph must beat lexical on.

**Mechanism (hippocampal indexing analogy):** LLM = neocortex; retrieval encoder =
parahippocampal region (detects synonyms); open KG = hippocampal index. *Offline:* an LLM runs
OpenIE on each passage → a schema-free KG of noun-phrase nodes joined by relation edges; the
encoder adds **synonym edges** between phrase nodes with cosine > τ=0.8. *Online, per query:*
(1) ONE LLM call does NER on the query; (2) the encoder links query entities to nearest KG
nodes = **seed nodes**; (3) each seed weight is scaled by node specificity (`|P_i|⁻¹`, an
IDF-like rare-phrase boost) and written into the **Personalized PageRank** reset vector;
(4) PPR runs to convergence (damping 0.5); (5) the phrase-node probability vector scores
passages. Multi-hop emerges because the random walk reaches passages sharing **no** query tokens
— in **one** retrieval step instead of iterative retrieve-reason loops.

**HippoRAG 2** makes passages first-class nodes (linked to their phrases by "contains" context
edges), shifts linking to query→triple + **LLM triple-filtering** ("recognition memory"), and
seeds *all* passage nodes by embedding similarity (small weight ~0.05) so dense and graph blend
in one PPR. Net effect: a single PPR fuses lexical-ish phrase matching, dense passage
similarity, and multi-hop traversal into one ranked list.

**Published quality (passage recall@5):** HippoRAG 2 beats the strongest 7B embedding baseline
(NV-Embed-v2) on every multi-hop set — 2Wiki **90.4 vs 76.5** (+13.9), MuSiQue **74.7 vs 69.7**
(+5.0), HotpotQA **96.3 vs 94.5** — and is the only structure-augmented method that *doesn't
regress* simple QA (NQ 78.0).

### Verdict for us
**Adopt the idea (PPR spreading activation over our concept graph), not the framework.** Two
hard adaptations, both fitting our stack:
1. **Drop the per-query LLM** (v1 NER / v2 triple-filtering) — the only thing that would blow
   the hot-path budget — and **seed PPR from our existing FTS top-k ∪ pgvector top-k**, weighted
   by fused score × importance × node-specificity. This turns PPR into the *fusion layer*
   ADR-0001 wants, with zero added LLM latency.
2. **Prefer a memory-node graph** (memories as nodes, our typed Relationships as edges) over
   HippoRAG's phrase explosion (it turns 11.6k passages into ~92k nodes; at our scale that'd be
   ~43k phrase nodes). Leaner and native to ADR-0002's node/edge tables.

A reproducible PPR latency micro-benchmark on a 5,400-memory graph measured **~2 ms** (memory-node
graph, transition matrix cached) to **~21 ms** (full phrase graph), ~105 ms even at 3× growth —
PPR is **not** the bottleneck; the stock recipe's online LLM is (which we remove). Postgres can
*store* the graph but has no native PPR (pgrouting = shortest-path only), so PPR is computed in
Python over a cached `scipy.sparse` transition matrix loaded from the node/edge tables, rebuilt
only on graph mutation. **Caveat for the gate:** our LLM-free seeding variant is *not* validated
by the papers, and our 5.4k personal corpus is far smaller and less multi-hop-dense than their
90k-node Wikipedia graphs — so the benchmark must confirm the multi-hop win transfers.

---

## 6. Embedding model survey

Our `content` and `expanded_keywords` are **short** prose (capped ~500 chars), so a model's
max-token limit is effectively a non-constraint — quality, dimensionality, and deploy
feasibility decide.

### Self-hostable (sentence-transformers on the GPU node, or GGUF via llama-cpp; pgvector stores the vector)

| Model | Dim | Params / VRAM (fp16) | MTEB(en) avg | License |
|---|---|---|---|---|
| nomic-embed-text-v1.5 | 768 (Matryoshka 64–768) | 0.1B / <1 GB | 62.28 | Apache-2.0 |
| bge-base-en-v1.5 | 768 | 109M / ~0.5 GB | ~63.5 | MIT |
| **bge-large-en-v1.5** | **1024** | 335M / ~1.3 GB | **64.23** | MIT |
| e5-large-v2 | 1024 | 0.3B / ~1.3 GB | ~62.25 | MIT |
| bge-m3 | 1024 dense (+sparse +ColBERT) | 568M / ~1–2.4 GB | en ~59–60 (strong multiling/BEIR) | MIT |
| gte-Qwen2-1.5B-instruct | 1536 | 1.5B / ~3.4 GB | **67.16** (top of set) | Apache-2.0 |

### Hosted (API call, NON-SENSITIVE memories only per ADR-0003)

| Model | Dim | MTEB(en) avg | Price /1M tok | License |
|---|---|---|---|---|
| OpenAI text-embedding-3-small | 1536 (Matryoshka→256) | 62.3 | $0.02 | proprietary |
| OpenAI text-embedding-3-large | 3072 (Matryoshka) | 64.6 | $0.13 | proprietary |
| **Voyage-3.5** | **1024** (+256/512/2048, int8/binary) | beats OpenAI-3-large ~7.5% on Voyage's eval | $0.06 (first 200M free) | proprietary |
| Voyage-3.5-lite | 1024 | beats OpenAI-3-large ~2–3.8% | $0.02–0.03 | proprietary |
| Cohere embed-english-v3.0 | 1024 (native int8/binary) | ~64.5 | ~$0.10 (sales-quoted) | proprietary |

**Implementation notes that matter.** Use **asymmetric** prompting (query vs document):
sentence-transformers `encode_query`/`encode_document`, always `normalize_embeddings=True` so
pgvector cosine == dot product. e5-large-v2 *requires* manual `"query: "`/`"passage: "` prefixes
or quality collapses; bge prepends a query instruction; gte-Qwen2 prepends a task instruction to
queries only. Pick the dimension **once** — changing it later forces a full re-embed + HNSW
rebuild.

### Recommendation (one of each, quality-first)
- **Local: BAAI/bge-large-en-v1.5** (1024-d, MIT) — best quality-per-complexity in the
  self-hostable set for short English memories: strong retrieval, ~1.3 GB VRAM (runs on CPU at
  ~100 ms), no `trust_remote_code`, mature ST support. The 512-token cap is irrelevant for our
  content. (gte-Qwen2-1.5B-instruct is the explicit upgrade candidate if the benchmark says
  bge-large leaves quality on the table; nomic is the fallback if a long context or sub-768
  Matryoshka dims are ever wanted.)
- **Hosted: Voyage-3.5** (1024-d) — highest measured retrieval quality of the hosted options,
  **same 1024-d as the local pick** so the pgvector column and fusion code are identical whether
  local or hosted (clean A/B), and our whole corpus embeds inside the free tier. Non-sensitive
  only; sensitive rows go to bge-large locally. (OpenAI text-embedding-3-small is the pragmatic
  fallback if no Voyage key.)

> **Prototype note:** the prototype as built used **bge-large-en-v1.5** (1024-d, local default,
> no API key in env). Production should adopt **Voyage-3.5** (also 1024-d) for non-sensitive
> memories per ADR-0003, keeping bge-large as the sensitive-only / no-key fallback. Both 1024-d
> means the pgvector schema and fusion code are unchanged across the choice.

---

## 7. Fusion of lexical + dense + graph signals

Three retrieval families produce candidate lists per recall; one fusion function merges them.

### Reciprocal Rank Fusion (RRF) — rank-based, scale-agnostic
Cormack/Clarke/Büttcher (SIGIR'09): `score(d) = Σ_s 1/(k + rank_s(d))`, summed over every signal
the doc appears in. `k` is a smoothing constant — their sweep found **k=60 near-optimal but
uncritical** (≤0.3% MAP swing across [10,100]; k=0 or k=500 costs 3–4%). A doc present in one
list but absent from another contributes `1/(k+rank)` where it fires and **0** elsewhere — so
multi-signal agreement is rewarded and single-signal hits are penalized (the hybrid behaviour we
want). Extends to N lists trivially (just sum), which makes a 3-way fuse a one-liner.
**Weighted RRF:** `Σ_s w_s/(k + rank_s(d))` — bias a stronger signal, no pre-normalization.

### Weighted score fusion / convex combination (CC) — score-based, needs normalization
Bruch et al. (arXiv 2210.11934, TOIS 2023): `f = α·φ(semantic) + (1-α)·φ(lexical)` with
**theoretical** min-max normalization (TM2C2): cosine min = -1, BM25 min = 0, per-query max —
stable across queries. Findings: **CC/TM2C2 beats RRF on nDCG and recall** in- and out-of-domain
(RRF ~3.86% lower nDCG@10 in one replication); Weaviate switched its default from rankedFusion to
relativeScoreFusion (min-max CC) for ~6% recall on FIQA. CC is sample-efficient (α tunes from a
small labeled set) but requires calibratable scores.

### Folding in graph hits
Build a graph candidate list, then feed it to the fuser as just another ranked list:
**seed** (match the query to concept nodes via the same FTS + dense over node labels) →
**traverse** (1–2 hops to reachable memories) → **score** each reachable memory. Three documented
scorings: hop-decay `Σ_paths β^hops` (β≈0.5–0.7); Personalized PageRank seeded on matched nodes
(HippoRAG); or node-degree priority (GraphRAG local search). The
*Calibrated-Fusion-for-Graph-Vector* paper (arXiv 2603.28886) is explicit: naive graph+vector
fusion fails on **scale incompatibility**, so convert graph traversal into a probability-like
normalized score before fusing. Crucial consequence: the graph list is **sparse** (often a
handful of memories, sometimes zero). Under RRF that's handled automatically; under CC you must
explicitly treat "absent" as the theoretical min or the missing-modality term silently biases the
sum.

### Cross-encoder re-rank — a separate stage-2, not a fusion function
Retrieve top-N each → fuse → take fused top ~20–30 → score each (query, memory) pair jointly with
a cross-encoder (e.g. bge-reranker-v2-m3) → re-sort. Reported lift +5 to +15 nDCG@10 on
BEIR/MTEB; cost scales with pair count so it is only ever a small-candidate-set stage.

### Recommendation
**Weighted RRF over three lists (FTS, dense, graph), k=60, equal weights to start**, with
importance applied as a deterministic post-fusion prior and a cross-encoder as an optional,
benchmark-gated stage-2. RRF is the right *default* because we fuse three incompatible scales,
one of them sparse/often-empty; it is near-parameter-free; and it collapses to exactly today's
lexical ordering when dense/graph are empty (the SQLite graceful-degrade path). **But because
adoption is quality-gated, the benchmark must also run CC/TM2C2 as a challenger** — the
literature is consistent that CC edges RRF on quality when scores are calibratable. (See the
[benchmark report](benchmark-report.md) for which won on *our* eval set.)

---

## 8. Concept-graph construction from memories

Turning flat Memory rows into nodes (concepts/entities) + typed directed edges + memory→concept
"mentions" edges. Three extraction families:

- **(A) Open LLM triple extraction** (schema-free) — prompt an LLM to emit `[subject, relation,
  object]` triples. High recall, but relation labels proliferate ("prefers"/"likes"/"favors"), so
  it **requires** downstream canonicalization. GraphRAG is the canonical implementation
  (extract + gleaning + cross-chunk entity summarization).
- **(B) Schema-guided** — constrain to a fixed ontology. Cleaner, but a fixed schema misses
  surprises in a heterogeneous personal corpus. **EDC** (Zhang & Soh, EMNLP 2024) bridges the two:
  *extract* open triples → *define* (LLM writes a one-sentence definition per distinct relation) →
  *canonicalize* (embed definitions, retrieve nearest existing relations, LLM verifies map-vs-add).
  Two modes: target-alignment (fixed schema) and self-canonicalization (grow schema dynamically).
- **(C) Entity resolution / canonicalization** (the dedup problem — "Svelte"/"SvelteKit"/"svelte
  framework" are one node): cluster-then-refine on the *aggregated* graph — embed every surface
  string, cluster by cosine (HDBSCAN / connected-components over a threshold), optional
  LLM-as-judge per cluster. KGGEN (arXiv 2502.09956) does iterative LLM-guided clustering;
  Graphiti uses MinHash+LSH fast-path with LLM fallback. **Cost scales with distinct entities (low
  thousands), not with memory count.**
- **(D) Lightweight non-LLM** — spaCy NER + noun-chunks + co-occurrence edges, or **ReLiK**
  (Sapienza, ACL 2024) for *typed* relations on CPU at up to 40× LLM speed, zero per-doc LLM cost.
  The natural ablation baseline and sqlite-only fallback.

**The tractable recipe for our corpus.** Measured: 5,452 non-sensitive memories ≈ 683K content
tokens total — *tiny*. At ~125 content-tokens/memory, ~570 memories pack into one 100K-token
request, so the **entire corpus extracts in ~10–25 batched LLM calls, not 5,452 sequential
calls**. Pipeline: (1) batch-extract open triples (each memory tagged with its `memory_id` so
triples map back), parallelized — LangExtract / KGGEN style; (2) aggregate + canonicalize globally
*once* (embed distinct entities, cluster, LLM-judge only ambiguous clusters — tens of calls);
(3) optionally one batched LLM "define relations" pass for EDC-style relation canonicalization.
Total budget: low tens of calls, minutes of wall-clock, a few dollars hosted or one GPU-node
llama-cpp session. **Canonicalization quality (the similarity threshold / cluster granularity) is
where this lives or dies and must be tuned against held-out data, not eyeballed.** Write-time /
Graphiti-style per-memory extraction is for *incremental updates only* — the wrong tool for the
one-shot backfill.

---

## 9. Vector storage in Postgres (production substrate)

`pgvector` is a **proven capability on our exact CNPG cluster** (Immich already does vector search
there, and claude-memory-mcp is already a tenant of the shared `pg-cluster-rw.dbaas` behind
PgBouncer) — zero new infrastructure, reuse-before-building satisfied.

- **HNSW** (recommended default): `USING hnsw (embedding halfvec_cosine_ops) WITH (m=16,
  ef_construction=64)`; query knob `SET hnsw.ef_search` (default 40). Best speed-recall tradeoff;
  buildable on an empty table; graph in RAM. **IVFFlat** is rejected (must be built *after* data
  exists — an empty-table footgun — and has a lower recall ceiling).
- **halfvec** (fp16, 2 bytes/dim) halves index size at ~no recall loss; indexable ≤4000 dims.
  768-d halfvec = 1536 bytes/row; at our scale total embedding storage is single-digit MB.
- **Filtered ANN:** we always filter `deleted_at IS NULL` (often `category`). Post-filtering can
  under-fill top-k; enable `hnsw.iterative_scan='relaxed_order'`, and **always add a tie-breaker**
  (`, id`) since approximate indexes give non-deterministic order.
- **Hybrid in one query:** each retriever is a CTE producing a per-ranker rank; fuse with RRF via
  FULL OUTER JOIN on memory id — no score calibration needed across the incomparable ts_rank and
  cosine scales.
- **pgvectorscale / StreamingDiskANN** (bounded-RAM disk graph, SBQ compression) is **deferred** —
  Rust/pgrx must be compiled into the operand image, and it only earns its keep above ~1–5M
  vectors. Our corpus is orders of magnitude below that.
- **PgBouncer gotcha:** per-query GUCs (`hnsw.ef_search`) must be `SET LOCAL` inside the recall
  transaction, not session-level, under transaction pooling.

**Not for the prototype** (the prototype uses an in-process numpy index); this is the production
adoption path *if* the benchmark clears the gate — an additive Alembic migration (one nullable
`halfvec(1024)` column + HNSW index) plus a Terraform change to the CNPG stack.

---

## 10. Evaluation methodology

A retrieval test collection = corpus + query set + **qrels** (relevance judgments). For each
query, call recall, take the ordered list of returned memory ids, score against qrels — measuring
the *retriever in isolation*, exactly what ADR-0001's gate needs.

**Metrics (compute all; pick one primary):**
- **Recall@k** — "did we surface the right memory at all?" *The* hot-path metric (auto-recall
  injects top-N; if the memory isn't in top-k it can't help). Report @5/@10/@20/@30.
- **nDCG@k** — graded + position-aware; the best single summary (BEIR standard is nDCG@10).
  Headline quality number for the gate.
- **MRR** — only the first hit matters; relevant for the exact-lookup stratum.
- **MAP** — broad binary recall+precision blend; secondary, stable for significance tests.

**Stratification (the ADR-0001 hypothesis-targeted design):** *exact/lexical* (FTS already wins —
the **regression guard**); *paraphrase/semantic* (disjoint vocabulary — the value-of-embeddings
test); *multi-hop* (≥2 memories or a concept link — the graph test).

**Qrels generation (the LongMemEval pipeline, inverted for memories):** sample seed memories
stratified by category + importance → an LLM generates exact / paraphrase / multi-hop queries →
label relevant ids, with **pooling** (union the top-k of every arm, TREC-style) and an LLM-judge
on the **UMBRELA 0–3 scale**. **Separate the generator model from the judge model** to avoid
self-preference leakage. **Hand-verify** a ≥15–20% sample (oversample multi-hop) and require
Cohen's κ(LLM, human) ≥ ~0.6 before trusting auto-labels; always hand-author multi-hop
relevant-id sets.

**Pitfalls with standard mitigations (all baked into the protocol):** LLM judges are
systematically *lenient* (κ gate); "holes" (new arms retrieve unjudged docs — must pool *all*
arms before judging, else the gate is rigged against semantic/hybrid); generator-as-judge leakage
(model separation); too-easy self-generated queries (check paraphrase shares no content tokens);
adversarial/unanswerable queries have no relevant id and **must be kept out of the ranked metrics**
(mixing them corrupted the disputed Zep-vs-mem0 LOCOMO comparison — 84%→58%).

**Sizing:** Voorhees & Buckley (2002) — ≥25 topics is the floor, 50 yield reliable rankings, and a
~5–6% absolute gap at n=50 is needed for 95% confidence the ordering holds on a different query
set. Since the gate is *per stratum*, each stratum wants its own ~50 queries.

> **Honest note on what we actually built:** our eval set is **119 queries (40 exact / 40
> paraphrase / 39 multihop)** — just below the ~50/stratum ideal, and qrels were LLM-generated
> with lighter hand-verification than the full protocol prescribes. This is a real limitation,
> tracked in the [benchmark report](benchmark-report.md).

---

## 11. Synthesis — what we borrow, from whom

| Source | Borrowed mechanism | Re-implemented as | Adopted? |
|---|---|---|---|
| LightRAG | Incremental set-union graph merge; dual-level retrieve | Native node/edge tables, no AGE; FTS+dense+graph fuse | Idea only |
| LazyGraphRAG | Defer LLM cost; index-time work ∝ new content | Store-time extraction off hot path | Principle |
| Zep / Graphiti | Episodic/semantic split; 3-signal RRF read path; bi-temporal invalidation | Memory rows + Postgres node/edge tables; pgvector+FTS+CTE | **Blueprint** |
| Mem0 | ADD/UPDATE/DELETE/NOOP write-side curation | Flagged async curator over existing `update_memory` | Complementary, flagged |
| HippoRAG 2 | PPR spreading activation for multi-hop | LLM-free FTS+vector-seeded PPR over memory-node graph (phase 2, gated) | Idea only |
| Bruch et al. / Cormack | RRF default + CC/TM2C2 challenger | Weighted RRF k=60, post-fusion importance prior | **Direct** |
| EDC / KGGEN | Open-extract → define → canonicalize globally | Batched extraction + embedding-cluster canonicalization | **Direct** |
| pgvector / Supabase | HNSW + halfvec + RRF hybrid in one SQL query | Additive migration to CNPG (production only) | **Production design** |
| LongMemEval / UMBRELA / Voorhees | Stratified LLM-qrels + pooling + κ gate | Our exact/paraphrase/multi-hop eval | **Direct** |

The through-line: **a memory-node concept graph, dense pgvector embeddings, and the existing
lexical FTS, fused with weighted RRF, with all LLM work pushed to store time** — sized for an
append-heavy personal store and gated on a benchmark that beats FTS.

---

## Sources

**GraphRAG family**
- Edge et al., "From Local to Global: A Graph RAG Approach…" (Microsoft, 2024) — arXiv 2404.16130
- Microsoft GraphRAG incremental-indexing design — github.com/microsoft/graphrag/issues/741; GraphRAG 1.0 blog (microsoft.com/en-us/research/blog/moving-to-graphrag-1-0…)
- Guo et al., "LightRAG: Simple and Fast RAG" (HKUDS, EMNLP 2025) — arXiv 2410.05779; github.com/HKUDS/LightRAG (incl. issue #2122, PG+AGE concurrency)
- nano-graphrag — github.com/gusye1234/nano-graphrag
- LazyGraphRAG — microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/

**Temporal KG memory**
- Rasmussen et al., "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (2025) — arXiv 2501.13956
- Graphiti — github.com/getzep/graphiti; Neo4j writeup (neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)

**Extraction-based memory**
- "Mem0: Building Production-Ready AI Agents…" (2025) — arXiv 2504.19413; github.com/mem0ai/mem0 (configs/prompts.py)

**Graph-PPR retrieval**
- Gutiérrez et al., "HippoRAG" (NeurIPS 2024) — arXiv 2405.14831
- "From RAG to Memory" = HippoRAG 2 (ICML 2025) — arXiv 2502.14802; github.com/OSU-NLP-Group/HippoRAG

**Embeddings**
- bge-large-en-v1.5 — huggingface.co/BAAI/bge-large-en-v1.5; gte-Qwen2-1.5B — huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct; nomic — huggingface.co/nomic-ai/nomic-embed-text-v1.5; bge-m3 — huggingface.co/BAAI/bge-m3; e5-large-v2 — huggingface.co/intfloat/e5-large-v2
- Voyage-3/3.5 — blog.voyageai.com/2024/09/18/voyage-3/; docs.voyageai.com/docs/pricing
- OpenAI text-embedding-3 — developers.openai.com/api/docs/guides/embeddings; Cohere embed v3 — docs.cohere.com/docs/cohere-embed

**Fusion**
- Cormack, Clarke, Büttcher (SIGIR'09) — cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Bruch et al., "An Analysis of Fusion Functions for Hybrid Retrieval" (TOIS 2023) — arXiv 2210.11934
- Elastic weighted RRF; Weaviate hybrid-search fusion algorithms; "Calibrated Fusion for Heterogeneous Graph-Vector Retrieval" — arXiv 2603.28886
- bge-reranker — huggingface.co/BAAI/bge-reranker-base

**Concept-graph construction**
- Zhang & Soh, "Extract-Define-Canonicalize" (EDC, EMNLP 2024) — arXiv 2404.03868; github.com/clear-nus/edc
- KGGEN — arXiv 2502.09956; ReLiK (ACL 2024) — arXiv 2408.00103; LightKGG — arXiv 2510.23341; Google LangExtract — github.com/google/langextract

**Postgres vector storage**
- pgvector — github.com/pgvector/pgvector; pgvectorscale — github.com/timescale/pgvectorscale; CNPG image-volume extensions — cloudnative-pg.io/docs/devel/imagevolume_extensions/; Supabase hybrid search — supabase.com/docs/guides/ai/hybrid-search
- This cluster: `infra/docs/architecture/databases.md` (claude-memory-mcp is a CNPG tenant); this repo: `migrations/versions/001_initial_schema.py`, `src/claude_memory/api/app.py`

**Evaluation**
- LoCoMo — arXiv 2402.17753; LongMemEval — arXiv 2410.10813; UMBRELA — arXiv 2406.06519; "Judging the Judges" / LLMJudge — arXiv 2502.13908; Voorhees & Buckley "Topic Set Size" (2002); Buckley & Voorhees "Bias and the Limits of Pooling"; BEIR — arXiv 2104.08663
