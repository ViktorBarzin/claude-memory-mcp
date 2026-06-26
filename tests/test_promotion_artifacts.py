"""The dense leg's PRODUCTION promotion artifacts exist and are internally consistent.

Migration ``005_add_embeddings_and_graph`` is *availability-gated*: on the live
CNPG cluster (operand image ``ghcr.io/cloudnative-pg/postgis:16``) ``pg_available_extensions``
has only ``postgis`` — no ``vector`` — so every vector step (the ``halfvec(1024)``
column, the HNSW index, ``concepts.embedding``) correctly **no-ops** and the dense
recall leg can never function there. The migration's own docstring says the fix
"lands separately" as a Terraform operand-image swap. **This module is the guard
that the artifact that fix points at actually exists on the branch** — without it
the promotion path is undefined (the gap a reviewer flagged: the migration narrates
a degradation whose remedy is absent).

These are deliberately **pure filesystem + content** assertions (no Docker, no DB),
so unlike ``tests/test_migration_005.py`` they run on every CI lane and cannot be
silently skipped. They assert three things, and one trap:

* a custom CNPG-compatible PG16 operand **Dockerfile** that bundles the GENUINE
  pgvector ``vector`` extension (Debian package ``postgresql-16-pgvector``) on a
  base that keeps the CNPG instance manager **and** PostGIS (the live cluster has a
  ``postgis``-using tenant — ``dawarich`` — so the swap must not drop PostGIS);
* a **staged Terraform** change for the shared cluster's operand ``imageName`` that
  is *inert until promoted* (variable-gated, default = the current ``postgis:16``)
  so an automatic ``terragrunt apply`` does not swap the multi-tenant cluster's
  image out from under the other tenants;
* a **promotion runbook** documenting the ordered, reversible sequence and the
  rollback path;
* THE TRAP: the operand Dockerfile must NOT use pgvecto.rs / VectorChord
  (``vectors.so`` preload, the ``vectors`` / ``vchord`` extensions). That is a
  DIFFERENT extension that does NOT provide ``CREATE EXTENSION vector`` /
  ``halfvec`` / ``halfvec_cosine_ops`` / ``hnsw``; swapping to it would silently
  leave migration 005 in its no-op (lexical-only) state with no error. The recall
  path's dense leg issues ``embedding <=> $2::halfvec`` over an HNSW index, which
  ONLY real pgvector provides.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Custom CNPG operand image Dockerfile (genuine pgvector on the PostGIS base).
DOCKERFILE = REPO_ROOT / "deploy" / "infra" / "Dockerfile.pgvector"
#: Staged Terraform change for the shared cluster's operand imageName.
TF_PATCH = REPO_ROOT / "deploy" / "infra" / "dbaas-pg-cluster-pgvector.tf"
#: Promotion runbook.
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "promote-pgvector-dense-recall.md"

#: Migration 005 — its docstring points at the "staged-only" operand-image swap.
MIGRATION_005 = REPO_ROOT / "migrations" / "versions" / "005_add_embeddings_and_graph.py"


# ── presence ──────────────────────────────────────────────────────────────────


def test_operand_dockerfile_exists() -> None:
    assert DOCKERFILE.is_file(), (
        f"{DOCKERFILE} missing: migration 005 no-ops the vector steps on the live "
        "postgis:16 cluster; the operand-image swap it points at must be staged."
    )


def test_terraform_image_swap_exists() -> None:
    assert TF_PATCH.is_file(), (
        f"{TF_PATCH} missing: the task requires the pgvector-on-CNPG Terraform staged "
        "on the branch (code/artifact only, not applied)."
    )


def test_promotion_runbook_exists() -> None:
    assert RUNBOOK.is_file(), f"{RUNBOOK} missing: the promotion runbook is a required artifact."


# ── the Dockerfile bundles GENUINE pgvector, not the VectorChord trap ───────────


def test_dockerfile_bundles_genuine_pgvector() -> None:
    """Base preserves PostGIS + CNPG instance manager; installs ``postgresql-16-pgvector``."""
    text = DOCKERFILE.read_text()
    # Built on the CNPG PostGIS operand image (keeps PostGIS for the dawarich tenant
    # AND the CNPG instance manager / barman-cloud the operator requires).
    assert "ghcr.io/cloudnative-pg/postgis:16" in text, (
        "operand image must extend the CNPG PostGIS operand image so PostGIS and the "
        "CNPG instance manager survive the swap"
    )
    # The genuine pgvector extension comes from this Debian package.
    assert "postgresql-16-pgvector" in text, (
        "must install the real pgvector Debian package (provides CREATE EXTENSION vector / halfvec / hnsw)"
    )


def _dockerfile_instructions(text: str) -> str:
    """The Dockerfile's ACTIVE instructions, lowercased — comment lines stripped.

    The Dockerfile deliberately *names* pgvecto.rs/VectorChord in its prose to warn
    future editors away from them; that documentation must not trip the trap check. We
    therefore assert against the executable lines only (everything that is not a
    full-line ``#`` comment). A trailing inline ``# ...`` after a real instruction is
    rare here and still scanned, which is the conservative choice.
    """
    lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    return "\n".join(lines).lower()


def test_dockerfile_does_not_use_vectorchord_trap() -> None:
    """The compounding trap: pgvecto.rs / VectorChord is NOT a pgvector substitute.

    ``vectors.so`` (shared_preload_libraries), the ``vectors`` / ``vchord`` extensions,
    and the ``pgvecto-rs`` binary deb provide a *different* vector engine that does NOT
    expose ``halfvec`` / ``halfvec_cosine_ops`` / the ``hnsw`` access method migration
    005 and the recall path depend on. If the operand image preloaded ``vectors.so`` the
    migration would silently stay in its no-op state with no error. We check the ACTIVE
    instructions (comments may name these to warn against them).
    """
    instructions = _dockerfile_instructions(DOCKERFILE.read_text())
    for forbidden in ("vectors.so", "pgvecto-rs", "pgvecto.rs", "vchord", "vectorchord"):
        assert forbidden not in instructions, (
            f"operand Dockerfile must NOT use the pgvecto.rs/VectorChord engine ({forbidden!r}); "
            "it does not provide pgvector's halfvec/hnsw/halfvec_cosine_ops"
        )


# ── the Terraform swap is inert until promoted ─────────────────────────────────


def test_terraform_swap_targets_the_shared_cluster_image() -> None:
    text = TF_PATCH.read_text()
    # Names the resource it patches and both the live image and the new pgvector image.
    assert "pg_cluster" in text, "the TF change must target the shared CNPG cluster (null_resource.pg_cluster)"
    assert "imageName" in text or "image" in text
    assert "ghcr.io/cloudnative-pg/postgis:16" in text, "must reference the current (default) operand image"
    assert "pgvector" in text.lower(), "must reference the pgvector-bundled replacement image"


def test_terraform_swap_is_variable_gated_and_inert_by_default() -> None:
    """An accidental/automatic apply must be a no-op until an operator opts in.

    The shared CNPG cluster is multi-tenant and ``terragrunt apply`` runs
    automatically on push (GitOps). The staged change therefore selects the operand
    image via a variable that DEFAULTS to the current ``postgis:16`` image, so the
    swap only happens when the runbook flips the variable. We assert a ``variable``
    block whose default is the current image.
    """
    text = TF_PATCH.read_text()
    assert "variable" in text, "image must be selected via a variable so the change is inert until promoted"
    assert 'default' in text and "ghcr.io/cloudnative-pg/postgis:16" in text, (
        "the image-selecting variable must DEFAULT to the current postgis:16 image (inert until promoted)"
    )


# ── the runbook documents the reversible sequence + the two warnings ────────────


def test_runbook_documents_ordered_reversible_promotion() -> None:
    """The runbook covers the ordered sequence and an explicit rollback path."""
    text = RUNBOOK.read_text().lower()
    # ordered promotion steps
    assert "rolling restart" in text or "rollout restart" in text or "rolling update" in text, (
        "runbook must cover the rolling restart of the shared CNPG cluster"
    )
    assert "migration 005" in text or "005" in text, "runbook must cover re-running migration 005"
    assert "backfill" in text, "runbook must cover backfilling embeddings for pre-existing rows"
    assert "memory_embeddings_enabled" in text, "runbook must cover flipping the MEMORY_EMBEDDINGS_ENABLED flag"
    # reversibility
    assert "rollback" in text or "roll back" in text, "runbook must document the rollback path"


def test_runbook_calls_out_postgis_lacks_pgvector_and_vectorchord_trap() -> None:
    """The two warnings the reviewer required, in plain words."""
    text = RUNBOOK.read_text().lower()
    # postgis:16 does not provide pgvector
    assert "postgis" in text and "pgvector" in text, "runbook must explain postgis:16 lacks pgvector"
    # VectorChord / pgvecto.rs is not a substitute
    assert "vectorchord" in text or "pgvecto" in text or "vectors.so" in text, (
        "runbook must warn that VectorChord / pgvecto.rs is NOT a substitute for pgvector"
    )


def test_runbook_is_cross_linked_from_migration_005() -> None:
    """Close the loop: migration 005's docstring must point at the runbook by name.

    The reviewer's core complaint was a *dangling* narrative — the migration says the
    operand-image swap "lands separately" but named no artifact. The migration must
    reference the runbook so the promotion path is discoverable from the gated code.
    """
    mig = MIGRATION_005.read_text()
    assert RUNBOOK.name in mig, (
        f"migration 005 must reference the promotion runbook ({RUNBOOK.name}) so the staged-only "
        "degradation narrative points at a real, discoverable artifact"
    )
