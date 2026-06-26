# Runbook: Promote the hybrid-recall dense (semantic) leg to production

Last updated: 2026-06-26

Take the dense-vector recall leg from **staged code** to **live**. The code, the
Alembic migration (005), and the embedding backends are already on the branch and
flag-gated OFF; this runbook is the ordered, reversible sequence that flips on the
production substrate. Until you run it, claude-memory keeps serving **lexical-only**
recall exactly as before — the dense leg is dormant.

> Scope: only the **dense** leg (`MEMORY_EMBEDDINGS_ENABLED`). The concept-graph leg
> (`MEMORY_GRAPH_ENABLED`) is a separate, later promotion and is out of scope here.

## TL;DR of why this is needed

The shared CNPG cluster (`dbaas/pg-cluster`) runs the operand image
`ghcr.io/cloudnative-pg/postgis:16`. **That image bundles PostGIS but NOT pgvector.**
Empirically, on that exact image:

```
postgres=# SELECT name FROM pg_available_extensions WHERE name = 'vector';
 name
------
(0 rows)
postgres=# CREATE EXTENSION vector;
ERROR:  extension "vector" is not available
```

Migration 005 is *availability-gated*: it checks `pg_available_extensions` and, finding
no `vector`, **correctly no-ops** the `memories.embedding halfvec(1024)` column, the
HNSW index, and `concepts.embedding`, landing only the (vector-free) graph tables. So
the dense leg literally cannot work until the operand image is swapped to one that
makes `vector` available. That swap is the first half of this runbook.

## ⚠️ Two warnings — read before doing anything

1. **`postgis:16` is not enough.** The CNPG PostGIS operand image does not contain
   pgvector. You cannot enable the dense leg by toggling a flag or re-running the
   migration alone — the operand image MUST change first.

2. **VectorChord / pgvecto.rs is NOT a substitute for pgvector.** The `dbaas` module
   ships a `postgres/postgres_Dockerfile` whose `CMD` sets
   `shared_preload_libraries=vectors.so` and which installs the pgvecto.rs binary —
   that provides the `vectors` / `vchord` extensions, a *different* vector engine. It
   does **not** provide `CREATE EXTENSION vector`, the `halfvec` type,
   `halfvec_cosine_ops`, or the `hnsw` access method. claude-memory's dense leg issues
   `embedding <=> $1::halfvec` over an HNSW index built with `halfvec_cosine_ops`
   (`src/claude_memory/api/recall.py`, `migrations/versions/005_add_embeddings_and_graph.py`),
   all of which are pgvector-only. **If you swap the cluster to a `vectors.so`/VectorChord
   image, migration 005 stays silently in its no-op (lexical-only) state and recall
   never gains a dense leg — with no error to tell you.** Use ONLY the genuine-pgvector
   operand image built from `deploy/infra/Dockerfile.pgvector` (PostGIS base +
   `postgresql-16-pgvector`).

## Blast radius (this is a SHARED, multi-tenant cluster)

`dbaas/pg-cluster` hosts many tenants (authentik, matrix, tripit, dawarich,
job_hunter, claude_memory, …). The operand-image swap triggers a **rolling restart of
all 3 instances**. Therefore:

- **Claim presence first:** `~/code/scripts/presence claim db:pg-cluster --purpose
  "pgvector operand swap for claude-memory dense recall"`. Coordinate with anyone
  using PG; expect brief connection blips at the switchover.
- **PostGIS must be preserved.** At least one tenant (`dawarich`) has the `postgis`
  extension installed. The replacement image is built **on top of** the CNPG PostGIS
  operand image, so PostGIS survives — the change is strictly additive (PostGIS +
  pgvector). Do not use a pure-pgvector image that lacks PostGIS.
- All infra changes go through Terraform/Terragrunt (GitOps) — **never** `kubectl
  edit`/`apply` the Cluster by hand.

## Prerequisites

- `kubectl` read access to `dbaas`; ability to open a PR / land a change in the infra
  repo (the operand swap is infra, not this repo).
- The genuine-pgvector operand image built and pushed (see Phase 0).
- A maintenance-ish window for the rolling restart (minutes, but it is multi-tenant).
- The claude-memory API deployment uses Postgres (the dense leg only exists in the
  API/Postgres deployment; the SQLite-only mode stays lexical by design — ADR-0002).

---

## Phase 0 — Build & publish the operand image

Per infra ADR-0002, first-party images build on GitHub Actions and publish to ghcr.

1. Build `deploy/infra/Dockerfile.pgvector` for `linux/amd64` and push, e.g.
   `ghcr.io/viktorbarzin/cnpg-postgis-pgvector:16-pgvector0.8.0`. The Dockerfile
   self-gates the build (`test -f .../extension/vector.control`), so a build that
   somehow lacked pgvector fails loudly rather than producing a no-op image.
2. **Verify the pushed image before touching the cluster** — run it standalone and
   confirm BOTH extensions are present and creatable:

   ```bash
   docker run --rm -e POSTGRES_PASSWORD=x -d --name pgv-check \
     ghcr.io/viktorbarzin/cnpg-postgis-pgvector:16-pgvector0.8.0
   sleep 5
   docker exec pgv-check psql -U postgres -c "CREATE EXTENSION vector;  SELECT extname FROM pg_extension WHERE extname='vector';"
   docker exec pgv-check psql -U postgres -c "CREATE EXTENSION postgis; SELECT postgis_version();"
   # halfvec + hnsw smoke test (the exact surface migration 005 / recall.py use):
   docker exec pgv-check psql -U postgres -c \
     "CREATE TABLE t(e halfvec(3)); CREATE INDEX ON t USING hnsw (e halfvec_cosine_ops); DROP TABLE t;"
   docker rm -f pgv-check
   ```

   All three must succeed. If `CREATE EXTENSION vector` errors, you built the wrong
   image — STOP. Do not proceed.

## Phase 1 — Stage & land the Terraform operand swap (infra repo)

The reviewable source is in this repo at `deploy/infra/dbaas-pg-cluster-pgvector.tf`.
Apply the equivalent change in the infra repo:

1. Add the `pg_cluster_image` variable to
   `infra/stacks/dbaas/modules/dbaas/main.tf` (default = the current image, so the
   change is **inert until you set the variable** — an automatic `terragrunt apply`
   stays a no-op).
2. Replace the two hard-coded `ghcr.io/cloudnative-pg/postgis:16` occurrences in
   `null_resource.pg_cluster` with `var.pg_cluster_image` (the `triggers.image` line
   AND the `spec.imageName` line in the embedded Cluster manifest). The trigger is what
   makes Terraform re-run the `kubectl apply` when the image changes.
3. Land that change first **with the default** (no-op) so the diff is reviewed and the
   variable plumbing is in place without restarting anything.

## Phase 2 — Flip the image (the rolling restart)

1. Claim presence on `db:pg-cluster` (see Blast radius) and announce the window.
2. Set `pg_cluster_image` to the pgvector tag from Phase 0 (via tfvars / the stack's
   variable wiring) and `terragrunt apply` the `dbaas` stack (or let GitOps apply the
   committed value). CNPG performs a **rolling update**: replicas re-created on the new
   image first, then a controlled switchover, then the former primary.
3. Watch it to completion:

   ```bash
   kubectl get cluster -n dbaas pg-cluster -o jsonpath='{.status.phase}{"\n"}'   # -> "Cluster in healthy state"
   kubectl get pods -n dbaas -l cnpg.io/cluster=pg-cluster -o wide               # all 3 Running, new image
   kubectl get cluster -n dbaas pg-cluster -o jsonpath='{.spec.imageName}{"\n"}' # the new pgvector tag
   ```
4. Confirm `vector` is now available on the live primary:

   ```bash
   PRIMARY=$(kubectl get cluster -n dbaas pg-cluster -o jsonpath='{.status.currentPrimary}')
   kubectl exec -n dbaas "$PRIMARY" -c postgres -- \
     psql -U postgres -tAc "SELECT 1 FROM pg_available_extensions WHERE name='vector'"   # -> 1
   ```

   Also re-confirm PostGIS still works for the existing tenant:
   `kubectl exec -n dbaas "$PRIMARY" -c postgres -- psql -U postgres -d dawarich -tAc "SELECT postgis_version()"`.

## Phase 3 — Re-run migration 005 (picks up the now-available vector objects)

With `vector` available, re-running migration 005 is what actually creates the
`embedding halfvec(1024)` column, the HNSW index, and `concepts.embedding`. The
migration is **idempotent**: the graph tables already created during the gated run are
left untouched; only the previously-skipped vector steps now execute.

Run the migration the same way the claude-memory API normally applies it (its Alembic
config against the `claude_memory` database). Then verify on the primary:

```bash
PRIMARY=$(kubectl get cluster -n dbaas pg-cluster -o jsonpath='{.status.currentPrimary}')
# embedding column is halfvec(1024), NULL-able:
kubectl exec -n dbaas "$PRIMARY" -c postgres -- psql -U postgres -d claude_memory -tAc \
  "SELECT udt_name, is_nullable FROM information_schema.columns WHERE table_name='memories' AND column_name='embedding'"
# HNSW index exists with halfvec_cosine_ops:
kubectl exec -n dbaas "$PRIMARY" -c postgres -- psql -U postgres -d claude_memory -tAc \
  "SELECT indexdef FROM pg_indexes WHERE indexname='idx_memories_embedding_hnsw'"
```

`udt_name=halfvec`, `is_nullable=YES`, and an index def containing `USING hnsw` +
`halfvec_cosine_ops` mean the substrate is ready.

## Phase 4 — Backfill embeddings for pre-existing rows

`schedule_embedding` (`src/claude_memory/api/recall.py`) only embeds on write **while
the flag is on**, so every memory stored before promotion has `embedding = NULL` and
would be invisible to the dense leg (`_dense_recall` filters `embedding IS NOT NULL`).
Backfill them before (or right around) flipping the flag.

Backfill rules (mirror the write path exactly so vectors are consistent):

- Embed only rows where `embedding IS NULL AND is_sensitive = 0 AND deleted_at IS NULL`.
- **Never embed sensitive rows** (`is_sensitive = 1`) — they stay NULL (lexical-only)
  forever (ADR-0003). For the hosted backend this is also a hard egress gate.
- Use the same backend selection as production: `select_embedder()` —
  `voyage-3.5` iff `VOYAGE_API_KEY` is set, else local `bge-large-en-v1.5`.
- Embedding is L2-normalised 1024-d; write it with the same `halfvec` text literal the
  write path uses (`_vector_literal`). Run in batches; it is idempotent (re-running
  only fills rows still NULL).

A one-shot backfill driver (uses the project's own embedder + pool) is the
straightforward path — iterate the NULL rows and call `embedder.embed_document(content,
is_sensitive=False)`, `UPDATE memories SET embedding = $1 WHERE id = $2`. Track
progress:

```bash
PRIMARY=$(kubectl get cluster -n dbaas pg-cluster -o jsonpath='{.status.currentPrimary}')
kubectl exec -n dbaas "$PRIMARY" -c postgres -- psql -U postgres -d claude_memory -tAc \
  "SELECT count(*) FILTER (WHERE embedding IS NOT NULL) AS embedded,
          count(*) FILTER (WHERE embedding IS NULL AND is_sensitive=0 AND deleted_at IS NULL) AS pending
   FROM memories"
```

Drive `pending` to 0 (sensitive rows are intentionally excluded and remain NULL).

## Phase 5 — Flip the flag

Set `MEMORY_EMBEDDINGS_ENABLED=on` (truthy: `1`/`true`/`yes`/`on`) on the claude-memory
**API** deployment (env var; via its Terraform/Helm values, not a manual `kubectl set
env`). The flag is read live by `embeddings_enabled()`, so the dense leg and
embed-on-write engage on rollout.

Because flipping the flag before/without a backfill is **safe** (the dense leg simply
finds fewer rows; `_fused_recall` degrades to lexical on any dense-leg issue), order is
forgiving — but recall quality is only fully realised once Phase 4 has populated the
back catalogue. Recommended order: Phase 4 backfill to ~0 pending, then Phase 5 flip.

### Verify the dense leg is actually contributing

1. Embed-on-write: store a fresh non-sensitive memory, confirm its row gets a non-NULL
   `embedding` shortly after (the background task in `schedule_embedding`).
2. Semantic hit: issue a recall whose query shares **no lexical tokens** with a known
   memory but is semantically close (e.g. query "what UI framework?" against a stored
   "prefers Svelte"). The lexical-only baseline would miss it; with the dense leg it
   should surface. (The fused path is in `_fused_recall` / `_fuse`.)

## Rollback

The change is reversible at every layer; peel back in reverse order.

| Symptom | Action |
|---|---|
| Dense recall misbehaving (bad results, latency) but DB healthy | Set `MEMORY_EMBEDDINGS_ENABLED=off` and roll out. Instantly back to lexical-only; the column/index stay (harmless, NULL-tolerant). No DB change. |
| Need to remove the schema | Run migration 005 **downgrade** (`alembic downgrade 004`). It drops the embedding column, the HNSW index, and the graph tables `IF EXISTS`; it does **not** drop the `vector` extension (other objects/tenants may use it) and leaves the 004 lexical schema byte-identical. |
| Operand image bad (cluster unhealthy, tenant broke, PostGIS regressed) | Set `pg_cluster_image` back to `ghcr.io/cloudnative-pg/postgis:16` and `terragrunt apply`. CNPG rolls back to the old operand image. `vector` becomes unavailable again and migration 005 reverts to its no-op gate automatically — lexical recall is unaffected throughout. |

Full-cluster restore (last resort) is the standard CNPG path:
`docs/runbooks/restore-postgresql.md` in the infra repo.

## Related

- `migrations/versions/005_add_embeddings_and_graph.py` — the availability-gated migration.
- `deploy/infra/Dockerfile.pgvector` — the genuine-pgvector operand image (NOT VectorChord).
- `deploy/infra/dbaas-pg-cluster-pgvector.tf` — the staged, inert-by-default image swap.
- `src/claude_memory/api/recall.py` — `_fused_recall` / `_dense_recall` / `schedule_embedding`.
- `src/claude_memory/embeddings.py` — `select_embedder` (Voyage-3.5 / bge-large).
- `docs/adr/0002`, `0003`, `0004`, `0006` — the decisions this promotion realises.
- infra `docs/architecture/databases.md` / `docs/runbooks/restore-postgresql.md` — the
  shared CNPG cluster and its restore path.
