# claude-memory recall benchmark

Stratified retrieval benchmark gating the hybrid-recall adoption decision
(ADR-0001): does dense-vector semantic recall + a concept graph beat the current
lexical FTS on **recall@5, recall@10, nDCG@10, MRR**? Quality decides adoption;
latency/storage are measured but non-gating.

> **PRIVACY — read first.** The corpus is the operator's REAL personal memories.
> `data/` (corpus/queries/qrels), `.venv/`, `cache/`, `results/`, and
> `scripts/build_eval_set.py` (the generator embeds memory-derived query text)
> are **gitignored and must never be committed**. Everything else here contains
> only code / aggregate numbers and is safe to commit. Sensitive memories
> (`is_sensitive=1`) are excluded from the corpus entirely.

## Layout

```
benchmarks/
  harness/                 # importable package (committable; no real content)
    types.py               # Memory, Query, Qrels, Retriever protocol
    metrics.py             # recall@k, nDCG@k, MRR (binary relevance)
    dataset.py             # load_dataset() + referential-integrity validation
    runner.py              # run_benchmark() -> overall + per-stratum + latency
    baselines.py           # SqliteFtsRetriever (faithful FTS5/BM25 reference)
    example_retriever.py   # worked example of the plug-in interface
    test_harness.py        # unit tests (pytest)
  scripts/
    export_corpus.py       # SQLite -> data/corpus.jsonl (non-sensitive only)
    snapshot_corpus.py     # API -> snapshots/<date>/corpus.jsonl (non-sensitive only)
    regression_run.py      # post-cleanup gate: snapshot vs stored baseline (POST_CLEANUP_GATE.md)
    build_eval_set.py      # -> data/queries.jsonl + qrels.jsonl  [GITIGNORED]
    dataset_stats.py       # validate + print AGGREGATE stats (safe)
    run_eval.py            # CLI: run a retriever, print/save metrics
  data/                    # [GITIGNORED] corpus.jsonl, queries.jsonl, qrels.jsonl (PRESERVED eval set)
  snapshots/               # [GITIGNORED] dated post-cleanup corpus snapshots
  .venv/                   # [GITIGNORED]
```

## Dataset schema (JSONL, one object per line)

**`corpus.jsonl`** — every non-sensitive memory:
```json
{"id": 137, "content": "...", "category": "decisions", "tags": "memory,architecture",
 "expanded_keywords": "...", "importance": 0.85}
```
`id` (int) is the join key everywhere. `tags` is comma-separated; `expanded_keywords`
space-separated (matches the production schema).

**`queries.jsonl`** — eval queries, three strata:
```json
{"query_id": "para_006", "text": "...", "stratum": "paraphrase", "relevant_ids": [380],
 "_note": "author rationale", "_jaccard": 0.023}
```
- `stratum` ∈ `exact` | `paraphrase` | `multihop`.
- `relevant_ids` is a convenience copy; **`qrels.jsonl` is authoritative**.
- `_note` / `_jaccard` are provenance fields (underscore-prefixed); ignore them in
  scoring.

**`qrels.jsonl`** — binary relevance judgments (authoritative):
```json
{"query_id": "multi_006", "relevant_ids": [263, 423, 637]}
```

### Strata (what each one tests)

| stratum | construction | who should win |
|---|---|---|
| **exact** | query = a salient phrase lifted from ONE memory; that memory is relevant (verified as the top FTS hit at build time) | lexical already strong; floor check |
| **paraphrase** | query restates ONE memory's meaning in DIFFERENT words (low lexical overlap, validated Jaccard ≤ ~0.18 vs content+keywords) | **dense embeddings** |
| **multihop** | query needs 2+ DISTINCT memories sharing an entity/concept (e.g. project + decision, or a multi-part runbook); ALL are relevant | **concept graph** |

Where a near-duplicate memory equally satisfies a single-target query, qrels was
augmented to include the twin (so a good retriever isn't penalised); deliberate
discriminator queries are kept single-target on purpose.

## Pluggable retriever interface

A retriever is any object implementing **one** method:

```python
def retrieve(self, query: str, k: int) -> list[int]:
    """Return up to k memory ids (corpus `id`s), ranked best-first."""
```

Optional lifecycle hooks the runner uses if present (duck-typed):

```python
def build_index(self, corpus: list[Memory]) -> None: ...   # timed separately
def index_size_bytes(self) -> int: ...                     # reported
name: str                                                  # label in reports
```

A bare callable `retrieve(query, k) -> list[int]` also works.

## Run it

```bash
.venv/bin/python scripts/export_corpus.py        # (re)build data/corpus.jsonl
.venv/bin/python scripts/build_eval_set.py        # (re)build queries+qrels (local)
.venv/bin/python scripts/dataset_stats.py         # validate + aggregate stats
.venv/bin/python -m pytest harness/test_harness.py -q

# evaluate a retriever (built-in alias or module:Class)
.venv/bin/python scripts/run_eval.py --retriever fts5
.venv/bin/python scripts/run_eval.py --retriever your_pkg.mod:YourRetriever --json results/yours.json
```

Programmatic use:

```python
from harness import load_dataset, run_benchmark
ds = load_dataset()
result = run_benchmark(MyRetriever(), ds)   # builds index, times queries
print(result.summary())                     # overall + per-stratum table
result.to_dict()                            # full machine-readable result
```

`run_benchmark` requests `retrieve_k=20` per query by default (≥ the max metric
cutoff of 10), macro-averages metrics over queries (overall + per stratum), and
reports per-query latency p50/p95 plus index build time/size when the hooks exist.

## Reference baseline

`harness.baselines.SqliteFtsRetriever` mirrors the production local-store search
(README "Search Algorithm"): FTS5 over content/category/tags/expanded_keywords,
`'"w1" OR "w2" ...'` MATCH, `ORDER BY bm25(), importance`. This is the lexical
"current system" any hybrid retriever must beat. (The Postgres `tsvector` path
uses weighted A/B/C/D ranking and an importance-first default; FTS5/BM25 is the
faithful, dependency-free relevance reference for the quality comparison.)
