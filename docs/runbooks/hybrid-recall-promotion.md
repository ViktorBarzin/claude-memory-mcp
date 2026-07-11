# Runbook: Promote hybrid recall (dense + concept-graph legs) to production

Last updated: 2026-06-26

Take the hybrid-recall upgrade from **staged code on the branch** to **live**, in the
correct gated order, with rollback at every layer. The code, the Alembic migration (005),
the embedding backends, and the concept-graph tables are all already on the branch and
**flag-gated OFF**. Until you run this runbook, claude-memory keeps serving **lexical-only**
recall exactly as it does today — both new legs are dormant.

> **This runbook supersedes nothing.** The dense-only promotion sequence lives in the
> sibling [`promote-pgvector-dense-recall.md`](promote-pgvector-dense-recall.md); this
> document is the **full hybrid** view — it covers both legs and the explicit, evidence-based
> decision about each one. Phases 0–6 here are the dense path (and reference the sibling for
> per-command detail); Phase 7 covers the concept-graph leg.

## The recommendation up front (read before doing anything)

The offline fair comparison ([`docs/research/hybrid-build-report.md`](../research/hybrid-build-report.md))
settled both legs definitively against a held-fixed `bge-large-en-v1.5` embedding model on
the preserved 5,452-memory / 119-query eval set:

| Leg | Verdict | Default | This runbook |
|---|---|---|---|
| **Dense** (`MEMORY_EMBEDDINGS_ENABLED`) | **Significant, robust win** — paraphrase recall@10 **+0.350**, overall nDCG@10 **+0.078** (paired-bootstrap CI clears zero) | ship **on** in production once the substrate is live | **Phases 0–6 — promote it** |
| **Concept graph** (`MEMORY_GRAPH_ENABLED`) | **graph-null on a VALID test** — graph candidates genuinely competed in the shared fused pool (up to 737 graph-only ids reached the top-10) yet the decisive +both-vs-+dense multi-hop CI never cleared zero at any weight | keep **off** | **Phase 7 — stage tables only, do NOT enable** |

So the operational goal of this runbook is: **enable the dense leg in production; build and
land the concept-graph schema additively but leave the graph flag off** (its production
read-path module is deliberately unbuilt — see Phase 7). The graph substrate ships so a future
revisit (precision-filtered graph leg, or graph-as-reranker rather than graph-as-RRF-leg) does
not need a fresh migration; it is **not** ready to serve traffic and the evidence says it would
not help if it did.

## Hard preconditions

- **All infra changes go through Terraform/Terragrunt (GitOps).** Never `kubectl
  edit`/`apply`/`patch` the Cluster or the claude-memory Deployment by hand.
- **Claim presence before any shared-infra mutation:** `~/code/scripts/presence claim
  db:pg-cluster --purpose "pgvector operand swap for claude-memory hybrid recall"`. The
  operand-image swap rolls a **shared, multi-tenant** cluster (Phase 2). If presence is already
  held by another session, defer.
- **The dense leg only exists in the API/Postgres deployment.** SQLite-only mode stays lexical
  by design (ADR-0002) and needs none of this.

---

## Phase 0 — Build & publish the genuine-pgvector operand image

Per infra ADR-0002, first-party images build on GitHub Actions and publish to ghcr — never
in-cluster.

1. Build [`deploy/infra/Dockerfile.pgvector`](../../deploy/infra/Dockerfile.pgvector) for
   `linux/amd64` and push, e.g. `ghcr.io/viktorbarzin/cnpg-postgis-pgvector:16-pgvector0.8.0`.
   The Dockerfile is `FROM ghcr.io/cloudnative-pg/postgis:16` + `postgresql-16-pgvector`, so
   the swap is **additive** — PostGIS survives for the tenant that uses it (`dawarich`). The
   build self-gates on `vector.control` being present, so a build that somehow lacked pgvector
   fails loudly rather than shipping a silent no-op image.
2. **Confirm the bundled pgvector version is ≥ 0.7.0** — `halfvec` and HNSW-on-`halfvec`
   landed in 0.7.0; an older pgvector cannot serve this schema. Verify the pushed image
   standalone before touching the cluster (full command block in the dense-only runbook,
   Phase 0):

   ```bash
   docker run --rm -e POSTGRES_PASSWORD=x -d --name pgv-check <image-tag>; sleep 5
   docker exec pgv-check psql -U postgres -c "CREATE EXTENSION vector; SELECT extversion FROM pg_extension WHERE extname='vector';"  # >= 0.7.0
   docker exec pgv-check psql -U postgres -c "CREATE EXTENSION postgis; SELECT postgis_version();"
   docker exec pgv-check psql -U postgres -c "CREATE TABLE t(e halfvec(3)); CREATE INDEX ON t USING hnsw (e halfvec_cosine_ops); DROP TABLE t;"
   docker rm -f pgv-check
   ```

   **⚠️ pgvecto.rs / VectorChord is NOT a substitute.** The `dbaas` module ships a legacy
   `postgres_Dockerfile` that bundles `vectors.so` — a *different* engine that does **not**
   provide `CREATE EXTENSION vector`, `halfvec`, `halfvec_cosine_ops`, or the `hnsw` access
   method. Using it leaves migration 005 in its silent no-op (lexical-only) state with no error.
   Use ONLY the genuine-pgvector image above.

## Phase 1 — Stage & land the Terraform operand swap (infra repo, inert)

The reviewable source is staged in this repo at
[`deploy/infra/dbaas-pg-cluster-pgvector.tf`](../../deploy/infra/dbaas-pg-cluster-pgvector.tf).
Apply the equivalent edit in the infra repo (`infra/stacks/dbaas/modules/dbaas/main.tf`):

1. Add the `pg_cluster_image` variable, **defaulting to the current image**
   (`ghcr.io/cloudnative-pg/postgis:16`).
2. Replace the two hard-coded image strings in `null_resource.pg_cluster` — the
   `triggers.image` line **and** the `spec.imageName` line in the embedded Cluster heredoc —
   with `var.pg_cluster_image`. The trigger is what makes Terraform re-run the `kubectl apply`
   when the image changes.
3. **Do NOT also change any Postgres parameter in the same apply.** CNPG rejects an
   `imageName` change combined with a `spec.postgresql.parameters` change under a switchover
   (CNPG issue #2530). pgvector needs **no** `shared_preload_libraries` and no GUC, so there is
   nothing to change anyway — leave `pg_params` untouched.
4. Land this **with the default** first, so the diff is reviewed and the variable plumbing is
   in place while the rendered manifest stays byte-identical to the live one (a GitOps
   auto-apply is a true no-op). Nothing restarts.

## Phase 2 — Flip the image (the rolling restart — coordinate)

1. **Claim presence** on `db:pg-cluster` and announce a window. This restarts a multi-tenant
   cluster.
2. **Back up first** — there is no PITR on this cluster; take a logical dump of `claude_memory`
   (and any tenant you are nervous about) before the swap.
3. **Pre-check PostGIS tenants.** At least `dawarich` has the `postgis` extension installed.
   The chosen image is built on the PostGIS base specifically so PostGIS survives; if you ever
   substitute a `standard` CNPG image instead, it would **drop PostGIS** and (for the `trixie`
   tag) jump the Debian base from bookworm → trixie, a glibc/ICU collation-provider change that
   can force a cluster-wide `REINDEX`. Stay on the bookworm-based PostGIS+pgvector image for
   collation parity.
4. Set `pg_cluster_image` to the Phase-0 tag and let GitOps apply the `dbaas` stack. **Expect a
   brief write outage, not a seamless HA switchover:** the cluster sets no
   `primaryUpdateMethod`/`Strategy`, so CNPG's defaults (`restart` + `unsupervised`) apply — it
   re-creates replicas on the new image, then **restarts the primary unsupervised**. All
   tenants see a short connection blip.
5. Watch to completion and confirm `vector` is now available on the live primary, and that
   PostGIS still answers (commands in the dense-only runbook, Phases 2–3).

## Phase 3 — Re-run migration 005 (picks up the now-available vector objects)

Migration 005 is **availability-gated**: on the old PostGIS-only image it correctly no-ops the
`memories.embedding halfvec(1024)` column, the HNSW index, and `concepts.embedding`, landing
only the (vector-free) graph tables. With `vector` now available, re-running it creates those
vector objects. It is **idempotent** — the graph tables already created during the gated run
are left untouched; only the previously-skipped vector steps execute, and the
`search_vector`/GIN lexical schema from migration 001 is byte-identical throughout.

Apply it the way the claude-memory API normally runs Alembic, then verify on the primary that
`memories.embedding` is `halfvec`, NULL-able, and that `idx_memories_embedding_hnsw` exists with
`halfvec_cosine_ops` (commands in the dense-only runbook, Phase 3).

## Phase 4 — Backfill dense embeddings for pre-existing rows

`schedule_embedding` only embeds on write **while the flag is on**, so every memory stored
before promotion has `embedding = NULL` and is invisible to the dense leg (`_dense_recall`
filters `embedding IS NOT NULL`). Backfill them before flipping the flag.

Mirror the write path exactly so vectors are consistent:

- Embed only `embedding IS NULL AND is_sensitive = 0 AND deleted_at IS NULL`.
- **Never embed sensitive rows** (`is_sensitive = 1`) — they stay NULL (lexical-only) forever
  (ADR-0003); for the hosted backend this is also a hard egress gate.
- Use production backend selection: `select_embedder()` — `voyage-3.5` iff `VOYAGE_API_KEY`
  is set, else local `bge-large-en-v1.5`. Output is L2-normalised 1024-d; write it with the same
  `halfvec` literal the write path uses (`_vector_literal`). Idempotent and batchable: re-running
  fills only rows still NULL. Drive `pending` to 0 (sensitive rows excluded by design).

## Phase 5 — Flip the dense flag

Set `MEMORY_EMBEDDINGS_ENABLED` truthy (`1`/`true`/`yes`/`on`) on the claude-memory **API**
deployment via its Terraform/Helm values — **not** a manual `kubectl set env`. The flag is read
live by `embeddings_enabled()`, so the dense leg and embed-on-write engage on rollout.

Flipping before a full backfill is **safe** (the dense leg simply finds fewer rows;
`_fused_recall` degrades to lexical on any dense-leg issue) — but recall quality is only fully
realised once Phase 4 is done. Recommended order: Phase 4 to ~0 pending, then Phase 5.

## Phase 6 — Verify the dense leg is actually contributing

1. **Embed-on-write:** store a fresh non-sensitive memory; confirm its row gets a non-NULL
   `embedding` shortly after (the background task in `schedule_embedding`).
2. **Semantic hit:** issue a recall whose query shares **no lexical tokens** with a known memory
   but is semantically close (e.g. query "what UI framework?" against a stored "prefers Svelte").
   Lexical-only would miss it; the dense leg should surface it. The fused path is `_fused_recall`
   / `_fuse`.

### (Optional) Phase 6b — Production A/B compare against the offline numbers

If you want a production confirmation of the offline +dense win before declaring done, mirror
the offline metric on a sample of real recalls with `MEMORY_EMBEDDINGS_ENABLED` toggled off vs
on (same queries, same corpus snapshot) and check the paraphrase/overall lift is in the offline
ballpark (paraphrase recall@10 ≈ +0.35, overall nDCG@10 ≈ +0.08). **Privacy:** keep any
production eval artifacts (queries/results derived from real memories) local and gitignored,
exactly as the offline harness does; publish only aggregate deltas.

---

## Phase 7 — Concept-graph leg: ship the schema, leave the flag OFF

**Decision: do NOT enable the concept-graph leg.** The fair comparison shows it is **graph-null
over dense** on a valid test — see [`hybrid-build-report.md`](../research/hybrid-build-report.md).
What *is* promoted here is only the additive schema, so a future revisit needs no new migration.

What lands (already in migration 005, Phase 3 created them unconditionally):

- Tables `concepts` / `concept_edges` / `memory_concepts` (the production mirror of the offline
  `benchmarks/retrievers/graph_build.py` dataclasses). `concept_edges` and `memory_concepts`
  carry no vector column; `concepts.embedding halfvec(1024)` is the one graph vector column and is
  pgvector-gated like `memories.embedding`.

What does **NOT** land / stays off:

- `MEMORY_GRAPH_ENABLED` stays **off** (default). With it off, fusion behaves as the dense path
  above; turning ONLY the graph flag on engages lexical-only RRF (harmless, no graph leg).
- **There is no production graph read-path module.** `_fused_recall` marks the graph leg
  `phase-2` and does not call one. The PPR/typed-1-hop graph leg exists only in the offline
  benchmark harness (`benchmarks/retrievers/hybrid.py`). **Do not wire it into the API** without a
  fresh decision instrument — the evidence says it would not improve recall and, at any weight
  where its candidates actually reach the top-k, it *degrades* precision (overall nDCG@10 collapses
  to 0.193 at w_graph=2.0). The single weight where +both is statistically indistinguishable from
  +dense (w_graph=0.25) is exactly the weight where the graph contributes **zero** ids to the top-10
  — i.e. "no harm" only when the graph is silent.

If a future run revisits the graph (a precision-filtered leg, or graph-as-reranker rather than an
RRF leg), the substrate is ready: write-path extraction + canonicalisation would populate these
tables off the hot path (sensitive rows excluded; route extraction through an in-cluster model, not
a hosted one), and a read-path module would seed PPR from the fused base hits. Cache the PPR
transition matrix once per graph — the offline run proved the un-cached rebuild on every call is a
hot-path killer (a full sweep never finished; cached it is ~14 ms p50 at 48k edges). None of that is
in scope until a fresh comparison clears zero.

---

## Rollback (peel back in reverse; reversible at every layer)

| Symptom | Action |
|---|---|
| Dense recall misbehaving (bad results, latency) but DB healthy | `MEMORY_EMBEDDINGS_ENABLED=off` and roll out. Instantly back to lexical-only; the column/index stay (harmless, NULL-tolerant). No DB change. |
| Graph flag was turned on by mistake | `MEMORY_GRAPH_ENABLED=off` and roll out. With no production graph module, this only ever engaged lexical-only RRF; reverting restores the dense/verbatim behaviour. |
| Need to remove the schema | `alembic downgrade 004` — drops the embedding column, HNSW index, and the three graph tables `IF EXISTS`; does **not** drop the `vector` extension (other tenants may use it); leaves the 004 lexical schema byte-identical. |
| Operand image bad (cluster unhealthy, tenant broke, PostGIS regressed) | Set `pg_cluster_image` back to `ghcr.io/cloudnative-pg/postgis:16` and apply. CNPG rolls back; `vector` becomes unavailable and migration 005 reverts to its no-op gate automatically — lexical recall is unaffected throughout. Restore from the Phase-2 dump only as a last resort. |

Full-cluster restore (last resort) is the standard CNPG path:
`docs/runbooks/restore-postgresql.md` in the infra repo.

## Related

- [`promote-pgvector-dense-recall.md`](promote-pgvector-dense-recall.md) — the dense-only
  promotion with per-command detail (Phases 0–6 here reference it).
- [`docs/research/hybrid-build-report.md`](../research/hybrid-build-report.md) — what was built
  and the fair-comparison verdict that drives the flag defaults above.
- `migrations/versions/005_add_embeddings_and_graph.py` — the availability-gated, additive migration.
- `deploy/infra/Dockerfile.pgvector` / `deploy/infra/dbaas-pg-cluster-pgvector.tf` — the genuine-pgvector
  operand image and the inert-by-default image swap.
- `src/claude_memory/api/recall.py` — `_fused_recall` / `_dense_recall` / `schedule_embedding`.
- `src/claude_memory/embeddings.py` — `select_embedder` (Voyage-3.5 / bge-large).
- `docs/adr/0002`–`0006` — the decisions this promotion realises.

---

## As executed (2026-07-10 → 2026-07-11)

Phases 0–6 ran to completion; dense leg LIVE and verified (embed-on-write; a
zero-lexical-overlap probe served the target at rank 1). Corrections future
re-runs need, discovered during execution:

- **Phase 3 mechanics:** `alembic upgrade head` can NOT re-run an
  already-stamped availability-gated migration. The working replay is
  `alembic stamp 004 && alembic upgrade head` — safe because 005 AND 006 are
  idempotent (existence-guarded); verified it preserves live `memory_links`
  rows. Also `CREATE EXTENSION vector` was pre-created as postgres.
- **Phase 5 runtime traps (three, stacked — the embed path had never actually
  run in prod):** (1) the API image shipped only the `[api]` extra, so the
  lazy `sentence_transformers` import failed and both embed-on-write and
  query-side dense embedding silently degraded to lexical; (2) with the extra
  installed, the 128Mi pod limit OOM-killed the torch import (exit 137) — now
  512Mi/2560Mi burstable (bge holds ~1.8Gi resident); (3) the runtime system
  user has no writable HOME, so the HuggingFace cache mkdir PermissionErrors
  and the model can never download at runtime — the model is now BAKED into
  the image at build time (`ENV HF_HOME=/opt/hf-cache` + a build-time
  `SentenceTransformer()` call; `chmod a+rwX` for the read-path `.locks`).
- **Phase 4 at scale:** `embed_document` encodes one text per call
  (batch_size=1) — a bulk backfill must batch via
  `embedder.model.encode(texts, batch_size=64)` + the module's
  `_l2_normalise` (identical math, ~25× faster). Piping the whole result
  through one `kubectl exec -i` stream dies with an apiserver i/o timeout on
  ~60MB — chunk stdin (500 rows/exec) with an idempotent
  `UPDATE … WHERE embedding IS NULL` writer.
- **Observability landed with the flip:** `/metrics` on the API (recall
  rate/latency/errors, dense-only-top5 contribution, link redirects/attaches,
  embed-write status, pending gauge) + Prometheus alerts + a weekly offline
  regression timer on the devvm (`memory-regression-weekly`, Sun 03:07 UTC).
  NOTE the infra Prometheus endpoints-job keep-whitelist: an app's metric
  prefix must be admitted or every series is silently dropped at ingestion.
