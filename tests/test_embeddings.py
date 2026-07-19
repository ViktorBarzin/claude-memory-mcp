"""Tests for the PRODUCTION dense-embedding module (``claude_memory.embeddings``).

Slice S6 contract (mypy strict, no ``Any``):

* an ``Embedder`` Protocol — ``embed_document`` / ``embed_query`` / ``dim`` /
  ``backend_label``;
* backend SELECTION mirroring the offline ``hybrid.py`` rule — Voyage when
  ``VOYAGE_API_KEY`` is set, bge-large otherwise (sensitive-safe / no-key fallback);
* an ``is_sensitive`` document NEVER produces a vector (returns ``None``) — never
  embedded, never egressed (ADR-0003 hard gate);
* LAZY optional imports — the package and module import cleanly with NO embedding
  extras installed (ADR-0002: base / SQLite-only mode pulls none); the heavy deps
  are imported only inside the selected backend, on first use;
* output is L2-normalised and dim 1024 (matches ``halfvec(1024)`` + cosine ``<=>``).

These tests must NOT hit the network or load a multi-GB model: the local backend's
``sentence-transformers`` model load and the Voyage HTTP client are monkeypatched
with deterministic fakes. Selection / sensitivity / lazy-import / shape / norm are all
exercised against those fakes.
"""

from __future__ import annotations

import importlib
import math
import sys
from types import ModuleType
from typing import TYPE_CHECKING

import pytest

import claude_memory.embeddings as emb

if TYPE_CHECKING:
    from collections.abc import Iterator


EXPECTED_DIM = 1024


# ── fakes: stand in for sentence-transformers and voyageai so tests stay offline ──


class _FakeSentenceTransformer:
    """Deterministic stand-in for ``sentence_transformers.SentenceTransformer``.

    ``encode`` returns an UN-normalised, fixed-dim vector per text so the module's
    own L2-normalisation is what the tests verify. Records the texts it saw so the
    BGE query-instruction prefix can be asserted.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self.max_seq_length = 512
        self.encoded: list[str] = []

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int = 32,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> object:
        import numpy as np

        self.encoded.extend(texts)
        # An un-normalised vector whose first two coords vary with text length so
        # different texts get different (but deterministic) directions; the rest are
        # a constant so the magnitude is clearly != 1 before normalisation.
        rows = []
        for t in texts:
            v = np.full(EXPECTED_DIM, 0.5, dtype=np.float32)
            v[0] = float(len(t) + 1)
            v[1] = float((len(t) % 7) + 2)
            rows.append(v)
        return np.asarray(rows, dtype=np.float32)


class _FakeVoyageEmbeddingResult:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.embeddings = embeddings


class _FakeVoyageClient:
    """Stand-in for ``voyageai.Client``; echoes a fixed-dim un-normalised vector and
    records the ``input_type`` so document-vs-query routing can be asserted. Also records
    constructor kwargs so the bounded-timeout hardening (a hung hosted API must not stall
    recall) can be asserted."""

    calls: list[tuple[str, str]] = []  # (text, input_type)
    init_kwargs: dict[str, object] = {}  # kwargs the real Client() was constructed with

    def __init__(self, **kwargs: object) -> None:
        _FakeVoyageClient.init_kwargs = dict(kwargs)

    def embed(self, texts: list[str], *, model: str, input_type: str, output_dimension: int) -> object:
        out: list[list[float]] = []
        for t in texts:
            _FakeVoyageClient.calls.append((t, input_type))
            vec = [0.25] * output_dimension
            vec[0] = float(len(t) + 1)
            out.append(vec)
        return _FakeVoyageEmbeddingResult(out)


@pytest.fixture(autouse=True)
def _reset_voyage_calls() -> Iterator[None]:
    _FakeVoyageClient.calls = []
    _FakeVoyageClient.init_kwargs = {}
    yield


@pytest.fixture(autouse=True)
def _reset_embedder_singletons() -> Iterator[None]:
    """select_embedder() now caches one backend instance per process; clear it around
    every test so each starts with a fresh, un-loaded backend (test isolation)."""
    emb.reset_embedder_cache()
    yield
    emb.reset_embedder_cache()


@pytest.fixture
def fake_local(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    """Inject a fake ``sentence_transformers`` module so the local backend never loads
    the real multi-GB model."""
    fake_mod = ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    return _FakeSentenceTransformer


@pytest.fixture
def fake_voyage(monkeypatch: pytest.MonkeyPatch) -> type[_FakeVoyageClient]:
    fake_mod = ModuleType("voyageai")
    fake_mod.Client = _FakeVoyageClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voyageai", fake_mod)
    return _FakeVoyageClient


def _l2(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


# ── 1. lazy imports: base install (no extras) still imports the package ──────────


def test_module_imports_without_any_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate a bare install: make the heavy optional deps un-importable, then
    re-import the module. It MUST import cleanly (ADR-0002) because the deps are
    imported lazily inside the backends, never at module top level."""
    real_import = importlib.import_module

    def _blocked_import(name: str, package: str | None = None) -> ModuleType:
        if name in {"sentence_transformers", "voyageai", "numpy"} or name.startswith(
            ("sentence_transformers.", "voyageai.", "numpy.")
        ):
            raise ModuleNotFoundError(f"blocked optional dep: {name}")
        return real_import(name, package)

    for mod in ("sentence_transformers", "voyageai"):
        monkeypatch.setitem(sys.modules, mod, None)  # force ModuleNotFoundError on import
    monkeypatch.setattr(importlib, "import_module", _blocked_import)

    reloaded = importlib.reload(emb)
    assert reloaded is not None
    # The Protocol and the selector are importable with no extras present.
    assert hasattr(reloaded, "Embedder")
    assert hasattr(reloaded, "select_embedder")


def test_select_embedder_does_not_eagerly_load_backend_deps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting a backend must not import the heavy deps; that happens on first
    embed() call. Proven by selecting while the deps are blocked from import."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    embedder = emb.select_embedder()  # must NOT raise despite blocked sentence_transformers
    assert embedder.backend_label.startswith("local:")
    assert embedder.dim == EXPECTED_DIM


# ── 2. the Embedder Protocol ─────────────────────────────────────────────────────


def test_backends_satisfy_embedder_protocol(
    monkeypatch: pytest.MonkeyPatch,
    fake_local: type[_FakeSentenceTransformer],
    fake_voyage: type[_FakeVoyageClient],
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    voyage = emb.select_embedder()
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    local = emb.select_embedder()
    # runtime_checkable Protocol membership
    assert isinstance(voyage, emb.Embedder)
    assert isinstance(local, emb.Embedder)


# ── 3. backend selection ─────────────────────────────────────────────────────────


def test_selects_voyage_when_key_present(
    monkeypatch: pytest.MonkeyPatch, fake_voyage: type[_FakeVoyageClient]
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = emb.select_embedder()
    assert embedder.backend_label == "hosted:voyage:voyage-3.5"


def test_selects_bge_when_no_key(monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = emb.select_embedder()
    assert embedder.backend_label == "local:BAAI/bge-large-en-v1.5"


def test_voyage_client_constructed_with_bounded_timeout(
    monkeypatch: pytest.MonkeyPatch, fake_voyage: type[_FakeVoyageClient]
) -> None:
    """The hosted client MUST carry a bounded request timeout. The voyageai default is
    timeout=None (no timeout), so a hung/slow API would otherwise stall the recall thread
    indefinitely — the embed call runs in a threadpool, but an unbounded hang still pins a
    worker and never returns a result. A finite timeout bounds the worst case."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = emb.select_embedder()
    # Force lazy client construction.
    embedder.embed_query("q")
    timeout = _FakeVoyageClient.init_kwargs.get("timeout")
    assert isinstance(timeout, (int, float)) and timeout > 0, (
        "voyageai.Client must be constructed with a positive finite timeout"
    )
    assert timeout == emb.VOYAGE_TIMEOUT_SECONDS


# ── 4. sensitive documents are NEVER embedded (ADR-0003 hard gate) ───────────────


def test_sensitive_document_returns_none_voyage(
    monkeypatch: pytest.MonkeyPatch, fake_voyage: type[_FakeVoyageClient]
) -> None:
    """A sensitive row must NEVER reach the hosted backend — None, and zero API calls."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = emb.select_embedder()
    out = embedder.embed_document("postgres://user:pw@host/db", is_sensitive=True)
    assert out is None
    assert _FakeVoyageClient.calls == []  # never egressed


def test_sensitive_document_returns_none_local(
    monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]
) -> None:
    """Even the LOCAL backend yields None for a sensitive row (column stays NULL →
    lexical-only). Sensitivity is enforced uniformly, not just for egress."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = emb.select_embedder()
    out = embedder.embed_document("api_key = SECRETSECRETSECRET", is_sensitive=True)
    assert out is None


def test_nonsensitive_document_is_embedded(
    monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = emb.select_embedder()
    out = embedder.embed_document("the user prefers dark mode", is_sensitive=False)
    assert out is not None
    assert len(out) == EXPECTED_DIM


# ── 5. output is L2-normalised, dim 1024 ─────────────────────────────────────────


def test_local_document_vector_is_l2_normalised_dim_1024(
    monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = emb.select_embedder()
    out = embedder.embed_document("hello world", is_sensitive=False)
    assert out is not None
    assert embedder.dim == EXPECTED_DIM
    assert len(out) == EXPECTED_DIM
    assert _l2(out) == pytest.approx(1.0, abs=1e-5)


def test_local_query_vector_is_l2_normalised_dim_1024(
    monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = emb.select_embedder()
    qvec = embedder.embed_query("what does the user prefer")
    assert len(qvec) == EXPECTED_DIM
    assert _l2(qvec) == pytest.approx(1.0, abs=1e-5)


def test_voyage_vectors_are_l2_normalised_dim_1024(
    monkeypatch: pytest.MonkeyPatch, fake_voyage: type[_FakeVoyageClient]
) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = emb.select_embedder()
    doc = embedder.embed_document("non sensitive content", is_sensitive=False)
    qry = embedder.embed_query("a query")
    assert doc is not None
    assert len(doc) == EXPECTED_DIM
    assert len(qry) == EXPECTED_DIM
    assert _l2(doc) == pytest.approx(1.0, abs=1e-5)
    assert _l2(qry) == pytest.approx(1.0, abs=1e-5)


# ── 6. document vs query routing (BGE prefix; Voyage input_type) ─────────────────


def test_bge_query_instruction_applied_to_query_only(
    monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]
) -> None:
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    embedder = emb.select_embedder()
    embedder.embed_document("plain passage", is_sensitive=False)
    embedder.embed_query("a question")
    model = embedder.model  # type: ignore[attr-defined]
    # the passage went in raw; the query carries the retrieval instruction prefix
    assert "plain passage" in model.encoded
    assert any(t.endswith("a question") and t != "a question" for t in model.encoded)


def test_voyage_input_type_routing(monkeypatch: pytest.MonkeyPatch, fake_voyage: type[_FakeVoyageClient]) -> None:
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    embedder = emb.select_embedder()
    embedder.embed_document("doc text", is_sensitive=False)
    embedder.embed_query("query text")
    kinds = {text: itype for text, itype in _FakeVoyageClient.calls}
    assert kinds["doc text"] == "document"
    assert kinds["query text"] == "query"


# ── 7. process singleton: the heavy model loads ONCE, not per recall ─────────────
# Regression for the 2026-07 recall-latency incident. select_embedder() returned a
# FRESH backend on every call, and the recall hot path (api/recall.py:_fused_recall)
# calls it per request — so the ~1.3GB bge-large model was re-instantiated on EVERY
# recall (prod evidence: 87 model-loads for 91 recalls, avg recall 5.1s, OOMKill when
# two loads overlapped). The contract locked in here: one process → one model load,
# reused across every recall and every embed-on-write.


def test_local_model_loads_once_across_recalls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five recalls (select_embedder() + embed_query(), exactly as _fused_recall does
    per request) must construct the bge model ONCE — not once per recall."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    constructions: list[str] = []

    class _CountingSentenceTransformer(_FakeSentenceTransformer):
        def __init__(self, model_name: str, device: str = "cpu") -> None:
            constructions.append(model_name)
            super().__init__(model_name, device=device)

    fake_mod = ModuleType("sentence_transformers")
    fake_mod.SentenceTransformer = _CountingSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)

    for i in range(5):
        emb.select_embedder().embed_query(f"recall query {i}")

    assert len(constructions) == 1, (
        f"bge-large re-instantiated {len(constructions)}x across 5 recalls — the "
        "per-call model-reload regression is back (recall latency ~5s, OOM risk)"
    )


def test_select_embedder_returns_process_singleton(
    monkeypatch: pytest.MonkeyPatch, fake_local: type[_FakeSentenceTransformer]
) -> None:
    """Repeated select_embedder() calls return the SAME instance for a given backend
    choice, so the lazily-loaded model is shared rather than reloaded."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert emb.select_embedder() is emb.select_embedder()
