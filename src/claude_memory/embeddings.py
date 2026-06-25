"""Production dense-embedding backends for the hybrid-recall WRITE/READ paths.

This is the production counterpart to the offline ``benchmarks/retrievers/hybrid.py``
dense leg: it embeds a memory's content on write and a query on recall, producing an
L2-normalised 1024-d vector that maps onto Postgres ``halfvec(1024)`` and the cosine
``<=>`` operator over an HNSW index.

Design invariants (ADR-0002/0003/0006), each test-covered in ``tests/test_embeddings.py``:

* **No required deps / lazy imports.** This module imports with NO optional extras
  installed — ``sentence-transformers`` / ``numpy`` (``embeddings`` extra) and
  ``voyageai`` (``voyage`` extra) are imported *inside* the backend methods, on first
  use, never at module top level. A bare ``uv sync`` (and the shipped SQLite-only
  image) pulls none of them; the package still imports and SQLite-only mode stays
  purely lexical. The whole dense path is additionally flag-gated by
  ``MEMORY_EMBEDDINGS_ENABLED`` (default off) at the call sites.
* **Backend selection mirrors the offline rule.** ``VOYAGE_API_KEY`` present →
  hosted ``voyage-3.5`` (non-sensitive rows only); otherwise the local
  ``BAAI/bge-large-en-v1.5`` (1024-d, MIT) — the sensitive-safe / no-key fallback.
* **Sensitive rows are NEVER embedded.** ``embed_document(..., is_sensitive=True)``
  returns ``None`` for *every* backend (the embedding column stays NULL → lexical
  only). For the hosted backend this is also a hard egress gate: the content never
  reaches the API.
* **Output is L2-normalised, dim 1024.** Cosine similarity is then a dot product,
  matching the L2-normalised ``halfvec`` and the ``<=>`` operator. Vectors are
  returned as plain ``list[float]`` so callers (asyncpg) need no numpy.

The module is intentionally synchronous and CPU-bound; call sites run it OFF the hot
path (a threadpool / ``asyncio`` task) so the synchronous store response is never
blocked (the CLAUDE.md non-blocking rule).
"""

from __future__ import annotations

import math
import os
from typing import Protocol, cast, runtime_checkable

#: The hosted (Voyage) model. voyage-3.5 defaults to 1024-d; we pin it explicitly so a
#: future default change cannot silently break the ``halfvec(1024)`` contract.
VOYAGE_MODEL = "voyage-3.5"

#: The local default model — 1024-d, MIT-licensed, sensitive-safe + no-key fallback.
#: Identical to the offline harness's ``_LOCAL_MODEL`` so dense behaviour does not drift
#: between the benchmark and production.
LOCAL_MODEL = "BAAI/bge-large-en-v1.5"

#: BGE retrieval convention (BAAI model card): the QUERY carries this instruction
#: prefix; passages/documents are embedded raw. Applied to queries only.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

#: The fixed embedding dimensionality. Both backends are pinned to it; it equals the
#: ``halfvec(1024)`` column width and the HNSW index dimension.
EMBEDDING_DIM = 1024

#: Env var that selects the hosted backend when set (and non-empty).
VOYAGE_API_KEY_ENV = "VOYAGE_API_KEY"


def _l2_normalise(vec: list[float]) -> list[float]:
    """Return ``vec`` scaled to unit L2 norm (a zero vector is returned unchanged).

    Pure-Python so the module needs no numpy at import time; the per-call cost is
    negligible against the embedding compute itself.
    """
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


@runtime_checkable
class Embedder(Protocol):
    """The contract both backends satisfy.

    ``backend_label`` is the human label surfaced in logs/metrics (e.g.
    ``"hosted:voyage:voyage-3.5"`` / ``"local:BAAI/bge-large-en-v1.5"``); ``dim`` is the
    fixed output dimensionality (``EMBEDDING_DIM``).
    """

    backend_label: str
    dim: int

    def embed_document(self, content: str, *, is_sensitive: bool) -> list[float] | None:
        """Embed a stored memory's content for the WRITE path.

        Returns an L2-normalised ``dim``-vector, or ``None`` when ``is_sensitive`` is
        ``True`` (the row is never embedded — the column stays NULL → lexical only).
        """
        ...

    def embed_query(self, query: str) -> list[float]:
        """Embed a recall query for the READ path. Always returns a vector (queries are
        not subject to the sensitive gate)."""
        ...


class _Encoded(Protocol):
    """The slice of a numpy array we read back from ``SentenceTransformer.encode`` — a
    sequence of per-text rows, each row supporting ``.tolist()``."""

    def __getitem__(self, index: int) -> _EncodedRow: ...


class _EncodedRow(Protocol):
    def tolist(self) -> list[float]: ...


class _STModel(Protocol):
    """The minimal ``SentenceTransformer`` surface this module uses — typed locally so
    calls are statically checked even though the library ships no stubs."""

    max_seq_length: int

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> _Encoded: ...


class BgeEmbedder:
    """LOCAL backend: ``BAAI/bge-large-en-v1.5`` via ``sentence-transformers``.

    The sensitive-safe / no-key fallback — runs entirely on-box, so it is the only
    backend allowed to touch (non-sensitive) content when no hosted key is configured.
    The heavy ``sentence_transformers`` dep is imported lazily on first embed; the model
    is loaded once and reused.
    """

    def __init__(self) -> None:
        self.backend_label = f"local:{LOCAL_MODEL}"
        self.dim = EMBEDDING_DIM
        self._model: _STModel | None = None

    @property
    def model(self) -> _STModel:
        """The lazily-loaded ``SentenceTransformer`` (loaded on first access)."""
        return self._load_model()

    def _load_model(self) -> _STModel:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # CPU is fine for short memories; force CPU to avoid CUDA init noise. Cap
            # the window at 384 (median memory ~120 tokens) so a rare long memory does
            # not pad an entire batch — mirrors the offline harness.
            model = cast(_STModel, SentenceTransformer(LOCAL_MODEL, device="cpu"))
            model.max_seq_length = min(model.max_seq_length, 384)
            self._model = model
        return self._model

    def _encode(self, text: str, *, normalize: bool) -> list[float]:
        model = self._load_model()
        # SentenceTransformer.encode returns a numpy array; row.tolist() yields a
        # list[float].
        out = model.encode(
            [text],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vec: list[float] = [float(x) for x in out[0].tolist()]
        return _l2_normalise(vec) if normalize else vec

    def embed_document(self, content: str, *, is_sensitive: bool) -> list[float] | None:
        if is_sensitive:
            return None  # sensitive rows are never embedded (column stays NULL)
        return self._encode(content, normalize=True)

    def embed_query(self, query: str) -> list[float]:
        # BGE: the query carries the instruction prefix; documents are raw.
        return self._encode(BGE_QUERY_INSTRUCTION + query, normalize=True)


class _VoyageResult(Protocol):
    """The slice of ``voyageai`` embed results this module reads — a list of vectors,
    one per input text (each vector a sequence of floats)."""

    embeddings: list[list[float]]


class _VoyageClient(Protocol):
    """The minimal ``voyageai.Client`` surface this module uses — typed locally since
    the library ships no stubs."""

    def embed(
        self,
        texts: list[str],
        *,
        model: str,
        input_type: str,
        output_dimension: int,
    ) -> _VoyageResult: ...


class VoyageEmbedder:
    """HOSTED backend: ``voyage-3.5`` via the ``voyageai`` client.

    Used only when ``VOYAGE_API_KEY`` is set AND the row is non-sensitive (ADR-0003/
    0006) — sensitive content NEVER leaves the box, so ``embed_document`` returns
    ``None`` for it *before* any API call. ``output_dimension`` is pinned to 1024 to
    match the ``halfvec(1024)`` column. The client is imported lazily on first use.
    """

    def __init__(self) -> None:
        self.backend_label = f"hosted:voyage:{VOYAGE_MODEL}"
        self.dim = EMBEDDING_DIM
        self._client: _VoyageClient | None = None

    def _get_client(self) -> _VoyageClient:
        if self._client is None:
            import voyageai

            self._client = cast(_VoyageClient, voyageai.Client())
        return self._client

    def _embed(self, text: str, *, input_type: str) -> list[float]:
        client = self._get_client()
        result = client.embed(
            [text],
            model=VOYAGE_MODEL,
            input_type=input_type,
            output_dimension=EMBEDDING_DIM,
        )
        vec: list[float] = [float(x) for x in result.embeddings[0]]
        return _l2_normalise(vec)

    def embed_document(self, content: str, *, is_sensitive: bool) -> list[float] | None:
        if is_sensitive:
            return None  # hard egress gate: sensitive content never reaches the API
        return self._embed(content, input_type="document")

    def embed_query(self, query: str) -> list[float]:
        return self._embed(query, input_type="query")


def select_embedder() -> Embedder:
    """Choose the production embedding backend per the FINAL DESIGN rule.

    Hosted ``voyage-3.5`` iff ``VOYAGE_API_KEY`` is set and non-empty; otherwise the
    local ``bge-large`` fallback. Selection is cheap and imports NO heavy deps — the
    backend's dependency is imported only on its first ``embed_*`` call.
    """
    if os.environ.get(VOYAGE_API_KEY_ENV):
        return VoyageEmbedder()
    return BgeEmbedder()
