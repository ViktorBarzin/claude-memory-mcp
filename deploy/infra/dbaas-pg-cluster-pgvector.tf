# =============================================================================
# STAGED Terraform: swap the shared CNPG cluster's operand image to one that
# bundles GENUINE pgvector — for the hybrid-recall dense leg.
# =============================================================================
#
# STATUS: STAGED ARTIFACT — NOT APPLIED, and INERT BY DEFAULT.
#   This file lives in the claude-memory-mcp repo as the reviewable source of the
#   infra change. The CANONICAL home is the infra repo at
#   `infra/stacks/dbaas/modules/dbaas/main.tf` (the `null_resource.pg_cluster`
#   resource) — that repo is GitOps and auto-applies on push. The promotion runbook
#   (docs/runbooks/promote-pgvector-dense-recall.md) covers landing this there.
#
#   The variable below DEFAULTS to the image the cluster runs TODAY
#   (`ghcr.io/cloudnative-pg/postgis:16`). So merging this change is a NO-OP: the
#   rendered manifest is byte-identical to the live one until an operator flips
#   `pg_cluster_image` to the pgvector-bundled tag per the runbook. This is the
#   safety property that lets the change sit in a continuously-applied GitOps repo
#   without swapping a multi-tenant cluster's image out from under the other tenants
#   (authentik, matrix, tripit, dawarich, claude_memory, …).
#
# WHY AN IMAGE SWAP AT ALL
#   Migration 005 (migrations/versions/005_add_embeddings_and_graph.py) is
#   availability-gated: on `ghcr.io/cloudnative-pg/postgis:16`, `vector` is NOT in
#   `pg_available_extensions`, so the `halfvec(1024)` column + HNSW index steps no-op
#   and the dense recall leg can never function. The operand image must make `vector`
#   AVAILABLE; then a re-run of migration 005 picks up the column + index.
#
# THE TRAP (why a specific image, not "any vector image")
#   The dbaas module historically carries a `postgres/postgres_Dockerfile` that bundles
#   pgvecto.rs / VectorChord (`shared_preload_libraries=vectors.so`, extensions
#   `vectors` / `vchord`). That is a DIFFERENT engine and does NOT provide
#   `CREATE EXTENSION vector`, `halfvec`, `halfvec_cosine_ops`, or the `hnsw` access
#   method this codebase uses (`embedding <=> $1::halfvec` over an HNSW index in
#   src/claude_memory/api/recall.py). Pointing the cluster at that image would leave
#   migration 005 silently in its no-op (lexical-only) state with NO error. The
#   replacement image MUST be the genuine-pgvector build in
#   `deploy/infra/Dockerfile.pgvector` (PostGIS base + `postgresql-16-pgvector`,
#   explicitly NOT the vectors.so/VectorChord variant).

# -----------------------------------------------------------------------------
# Operand-image selector — add to infra/stacks/dbaas/modules/dbaas/main.tf
# -----------------------------------------------------------------------------
variable "pg_cluster_image" {
  type        = string
  description = <<-EOT
    CNPG operand image for the shared dbaas/pg-cluster.

    DEFAULT = the image the cluster runs today (PostGIS, NO pgvector) so this is inert
    until an operator promotes. To enable the hybrid-recall dense leg, set this to the
    genuine-pgvector operand image built from deploy/infra/Dockerfile.pgvector
    (e.g. "ghcr.io/viktorbarzin/cnpg-postgis-pgvector:16-pgvector0.8.0") and follow
    docs/runbooks/promote-pgvector-dense-recall.md.

    MUST be a real-pgvector image — NOT a pgvecto.rs/VectorChord (vectors.so) image,
    which does not provide CREATE EXTENSION vector / halfvec / hnsw.
  EOT
  default     = "ghcr.io/cloudnative-pg/postgis:16"
}

# -----------------------------------------------------------------------------
# The change to null_resource.pg_cluster
# -----------------------------------------------------------------------------
# In infra/stacks/dbaas/modules/dbaas/main.tf, the cluster is reconciled by an
# idempotent `kubectl apply` inside `null_resource.pg_cluster`. Two edits, both
# replacing the hard-coded "ghcr.io/cloudnative-pg/postgis:16" with var.pg_cluster_image:
#
#   1. triggers.image  — so the null_resource re-runs the apply when the image changes
#      (the trigger is what makes Terraform notice the swap and re-reconcile):
#
#        triggers = {
#          instances    = "3"
#      -   image        = "ghcr.io/cloudnative-pg/postgis:16"
#      +   image        = var.pg_cluster_image
#          storage_size = "20Gi"
#          ...
#        }
#
#   2. spec.imageName in the embedded Cluster manifest — the actual operand image CNPG
#      runs. Because the manifest is a heredoc, interpolate the variable in place:
#
#      -       imageName: ghcr.io/cloudnative-pg/postgis:16
#      +       imageName: ${var.pg_cluster_image}
#
# NOTE on the rolling restart: changing `spec.imageName` makes the CNPG operator perform
# a rolling update of the 3 instances (replicas first, then a switchover, then the old
# primary) — a brief, automatic, multi-tenant-affecting event. Coordinate it via the
# runbook (claim presence on db:pg-cluster); do NOT let it ride along silently with an
# unrelated apply. Everything else in the Cluster spec (instances, anti-affinity,
# postgresql.parameters, storage, resources) is UNCHANGED — this swap is image-only.
#
# After the swap reconciles, `vector` becomes available; re-run migration 005 to create
# the `embedding halfvec(1024)` column + HNSW index, backfill, then flip
# MEMORY_EMBEDDINGS_ENABLED. Full ordered+reversible sequence in the runbook.
