# Production vector storage: pgvector HNSW + halfvec(1024); 1024-d embeddings (Voyage-3.5 / bge-large)

Phase 1 of the hybrid ([ADR-0004](0004-phase-the-hybrid-lexical-dense-first-graph-gated.md)) needs a
production home for the dense embeddings. Per [ADR-0002](0002-api-postgres-first-sqlite-stays-lexical.md)
that is **pgvector on the shared CNPG Postgres**, where claude-memory is already a database tenant — no new
datastore.

> ⚠️ **Correction (verified against live infra by a design challenger):** an earlier draft justified this
> with "Immich already runs pgvector on the same cluster." That is **false** — Immich runs its **own**
> Postgres, not the shared CNPG — so it is NOT evidence the shared cluster has the extension. **pgvector
> must be explicitly enabled on CNPG** (extension install, and possibly a CNPG operand-image change) via
> Terraform **before this can land**; do not assume it is already available.

Decisions:

- **Index: HNSW** (`USING hnsw (embedding halfvec_cosine_ops) WITH (m=16, ef_construction=64)`,
  query knob `hnsw.ef_search` set via `SET LOCAL` inside the recall txn under PgBouncer). Best
  speed-recall tradeoff, buildable on an empty table. **IVFFlat rejected** — it must be built *after*
  data exists (empty-table footgun) and has a lower recall ceiling.
- **Type: `halfvec(1024)`** (fp16) — halves index size at ~no recall loss; 1024-d halfvec = 2048
  bytes/row → single-digit MB for the whole corpus.
- **Dimension fixed at 1024**, chosen **once** (changing it later forces a full re-embed + HNSW
  rebuild). 1024 matches both the production model (Voyage-3.5) and the prototype model
  (bge-large-en-v1.5), so the column and all fusion code are identical regardless of model.
- **Model: Voyage-3.5** (1024-d, hosted) for **non-sensitive** memories (highest measured retrieval
  quality of the hosted options, free tier covers the corpus); **bge-large-en-v1.5** (1024-d, local,
  MIT) for **sensitive memories and the no-API-key fallback** ([ADR-0003](0003-external-embedding-apis-allowed-for-non-sensitive-memories.md)).
  `is_sensitive=1` rows are never embedded externally — `embedding=NULL`, lexical-only.
- **pgvectorscale / StreamingDiskANN deferred** — Rust/pgrx must be compiled into the CNPG operand
  image, and it only earns its keep above ~1–5M vectors; our corpus is orders of magnitude below that.

## Migration shape

A single **additive** Alembic migration: `ALTER TABLE memories ADD COLUMN embedding halfvec(1024)`
(NULL for sensitive) + `CREATE INDEX CONCURRENTLY … USING hnsw …`. The existing generated
`search_vector tsvector` + GIN index (`migrations/001`) are **untouched**, so lexical behaviour and
the SQLite-only degrade path are unchanged. pgvector enablement on CNPG and any extension/operand
change land as **Terraform/Terragrunt** in `infra/stacks/…` (GitOps, never kubectl) and trigger a
rolling restart of the shared cluster — coordinate accordingly.

## Consequences

- The prototype's in-process numpy matrix maps directly to this column; only the substrate changes,
  not the retrieval math.
- The prototype measured **bge-large** quality; a cheap follow-up should re-run the dense leg with
  **Voyage-3.5** on the non-sensitive corpus to confirm the hosted ceiling holds on our content
  before locking the production default.
- Production latency/ANN-approximation/filtered-top-k behaviour are unmeasured in the prototype and
  must be validated post-migration (a stated benchmark limitation).
