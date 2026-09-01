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
import time
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

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

#: The ONNX-served local model (ADR-0016 GPU plan, 2026-08-31). 509M params, native
#: 1024-d so it reuses the existing ``halfvec(1024)`` column and HNSW index unchanged,
#: Apache-2.0, and multilingual — 9.2% of the corpus carries Cyrillic that the
#: English-only bge-large cannot represent.
ONNX_MODEL = "Qwen/Qwen3-Embedding-0.6B"

#: Qwen3-Embedding's OWN retrieval convention, taken verbatim from the model's
#: ``config_sentence_transformers.json`` prompts["query"]. It is NOT bge's prefix, and
#: documents take NO instruction. Omitting it costs ~1-5% retrieval per the model card.
ONNX_QUERY_INSTRUCTION = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

#: Directory holding the exported ONNX graphs + ``tokenizer.json``, baked into the image.
ONNX_MODEL_DIR_ENV = "MEMORY_ONNX_MODEL_DIR"
ONNX_MODEL_DIR_DEFAULT = "/opt/onnx-embed"

#: Ordered onnxruntime execution providers. Default is CPU-only so a deploy of this code
#: changes nothing about scheduling; the GPU flip is this one env var.
ONNX_PROVIDERS_ENV = "MEMORY_ONNX_PROVIDERS"
ONNX_PROVIDERS_DEFAULT = "CPUExecutionProvider"

#: Per-provider graph file. One fp16 graph serves both providers.
#:
#: int8 was tried first and rejected on measurement (2026-09-01): dynamic int8
#: quantisation of this 0.6B decoder-only model produced vectors scoring cosine 0.16-0.37
#: against the sentence-transformers reference, with the whole embedding space collapsed
#: into a 0.89-0.95 band. It passed every cheap check — 1024-d, unit norm, correct
#: conventions, plausible latency — and would have ranked badly in production while
#: looking healthy. The exporter now gates fidelity at >=0.99 before a graph can be
#: baked, so a repeat cannot reach a registry.
#:
#: fp16 was then attempted and abandoned the same day: converting a graph this size with
#: onnxconverter_common fails whichever way round it is run — with shape inference on it
#: cannot serialise (the fp32 model is ~2.4 GB, past protobuf's 2 GB ceiling), and with
#: inference off the converter cannot place casts and emits a graph onnxruntime rejects at
#: load. So fp32 ships. A T4 runs fp32 on a 0.6B model comfortably, and it is the artifact
#: the fidelity gate has actually accepted, at cosine 1.00000 on every probe.
#:
#: The filename is the exporter's own, unrenamed: a large export splits into
#: ``model.onnx`` plus ``model.onnx_data`` and stores that reference inside the graph by
#: filename, so renaming it anywhere breaks the pair.
ONNX_FILE_BY_PROVIDER = {
    "CUDAExecutionProvider": "model.onnx",
    "CPUExecutionProvider": "model.onnx",
}

#: Selects the backend explicitly. Unset keeps the historical rule (Voyage when keyed,
#: else bge-large), so landing this module changes no running behaviour.
BACKEND_ENV = "MEMORY_EMBEDDING_BACKEND"

#: Env var that selects the hosted backend when set (and non-empty).
VOYAGE_API_KEY_ENV = "VOYAGE_API_KEY"

#: Bounded request timeout (seconds) for the hosted client. voyageai.Client defaults to
#: ``timeout=None`` (no timeout); on the recall READ path embed_query runs in a threadpool,
#: so an unbounded hang would pin a worker and never return — a hung/slow hosted API must
#: not stall recall indefinitely. A finite timeout caps the worst case (the recall caller
#: then degrades to lexical-only on the raised timeout).
VOYAGE_TIMEOUT_SECONDS = 10.0


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


class _OrtNode(Protocol):
    """One input node of an onnxruntime graph — we only read its name."""

    name: str


class _OrtSession(Protocol):
    """The minimal ``onnxruntime.InferenceSession`` surface this module uses."""

    def run(self, output_names: None, input_feed: dict[str, object]) -> list[object]: ...
    def get_inputs(self) -> list[_OrtNode]: ...
    def get_providers(self) -> list[str]: ...


class _NDArray(Protocol):
    """The slice of a numpy array the ONNX path touches."""

    shape: tuple[int, ...]

    def __getitem__(self, index: object) -> _NDArray: ...
    def tolist(self) -> object: ...


class _NumpyModule(Protocol):
    """The two numpy constructors the ONNX path needs.

    Reached through ``importlib`` rather than ``import numpy`` on purpose: a direct import
    makes mypy resolve numpy's bundled stubs, which use Python-3.12-only ``type``
    statements and fail under this project's pinned ``python_version = "3.11"``. Every
    other optional dep in this module is kept behind a local Protocol for the same
    reason — we rely on none of their type information.
    """

    int64: object

    def array(self, obj: object, dtype: object = ...) -> _NDArray: ...
    def asarray(self, obj: object) -> _NDArray: ...


class _Encoding(Protocol):
    """The slice of a ``tokenizers.Encoding`` this module reads."""

    ids: list[int]
    attention_mask: list[int]


class _Tokenizer(Protocol):
    """The minimal ``tokenizers.Tokenizer`` surface this module uses."""

    def encode(self, sequence: str) -> _Encoding: ...


def short_provider(provider: str) -> str:
    """``CUDAExecutionProvider`` -> ``cuda``. The metric label form."""
    stripped = provider.lower().removesuffix("executionprovider")
    return stripped or provider.lower()


class OnnxEmbedder:
    """LOCAL backend: ``Qwen/Qwen3-Embedding-0.6B`` served by onnxruntime.

    One exported graph per execution provider (fp16 for CUDA, int8 for CPU), both from
    the same weights. Providers are tried in order: the first one whose session builds
    becomes the primary, and if a primary embed raises at runtime the next provider in
    the list serves that call instead. That is the plan's "CPU fallback" — it keeps dense
    recall available through a GPU outage rather than dropping to lexical.

    The model's conventions differ from bge's and both are load-bearing: last-token (EOS)
    pooling rather than CLS, and the query instruction in
    :data:`ONNX_QUERY_INSTRUCTION` applied to queries only. Neither mistake raises; both
    produce vectors that look valid and rank badly, so both are test-covered.
    """

    def __init__(
        self,
        *,
        model_dir: str | None = None,
        providers: list[str] | None = None,
        observer: object | None = None,
    ) -> None:
        self.backend_label = f"onnx:{ONNX_MODEL}"
        self.dim = EMBEDDING_DIM
        self.model_dir = model_dir or os.environ.get(ONNX_MODEL_DIR_ENV, ONNX_MODEL_DIR_DEFAULT)
        raw = os.environ.get(ONNX_PROVIDERS_ENV, ONNX_PROVIDERS_DEFAULT) if providers is None else ",".join(providers)
        self.providers = [p.strip() for p in raw.split(",") if p.strip()]
        self._observer = observer
        self._sessions: dict[str, _OrtSession] = {}
        self._tokenizer: _Tokenizer | None = None

    # ── lazy resources ────────────────────────────────────────────────────────
    def _tok(self) -> _Tokenizer:
        if self._tokenizer is None:
            from tokenizers import Tokenizer  # lazy by design (ADR-0002)

            self._tokenizer = cast(_Tokenizer, Tokenizer.from_file(os.path.join(self.model_dir, "tokenizer.json")))
        return self._tokenizer

    def _session(self, provider: str) -> _OrtSession:
        """Build (once) the session for ``provider``, or raise if ``provider`` is not
        the one actually serving it.

        onnxruntime does NOT raise when a requested execution provider is unavailable.
        It logs a warning to stderr, silently falls back to CPU, and hands back a working
        session. Measured 2026-09-01: a pod whose image was missing ``libcublasLt`` built
        a "CUDA" session that ran every embed on the CPU, and construction success alone
        reported it as GPU-served — so the fallback counter stayed at zero and the alert
        for exactly that condition could never fire.

        ``get_providers()`` reports what actually registered, so that is what decides.
        """
        cached = self._sessions.get(provider)
        if cached is not None:
            return cached
        import onnxruntime as ort  # lazy by design (ADR-0002)

        # Put the CUDA and cuDNN shared libraries on the loader path before building a
        # session. The nvidia pip wheels install their .so files under
        # site-packages/nvidia/<lib>/lib/, which is NOT on the default loader path, so
        # without this onnxruntime cannot dlopen its own CUDA provider and reports
        # "libcublasLt.so.12: cannot open shared object file" — then silently serves the
        # request on the CPU. Measured 2026-09-01: with the libraries installed but
        # unreachable the CUDA provider never registered; calling this made the same pod
        # serve at 21 ms instead of 175 ms. onnxruntime's own helper is used rather than
        # a hand-set LD_LIBRARY_PATH so the paths stay the library's business.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        path = os.path.join(self.model_dir, ONNX_FILE_BY_PROVIDER[provider])
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = cast(_OrtSession, ort.InferenceSession(path, opts, providers=[provider]))
        registered = session.get_providers()
        if provider not in registered:
            raise RuntimeError(
                f"{provider} did not register (onnxruntime fell back to {registered}); "
                "the provider's libraries are missing from the image"
            )
        self._sessions[provider] = session
        return session

    def warm(self) -> str:
        """Build the primary session and run one embed. Returns the serving provider.

        Called from the readiness probe so a restarting pod never serves a cold model.
        """
        self.embed_query("warmup")
        return short_provider(self._primary())

    def _primary(self) -> str:
        """The first provider whose session builds. Why each earlier one did not build is
        kept and reported if none do, so a misconfigured image says which library was
        missing rather than just "no provider"."""
        reasons: list[str] = []
        for provider in self.providers:
            try:
                self._session(provider)
            except Exception as exc:  # noqa: BLE001 — an unavailable provider is expected; try the next
                reasons.append(f"{provider}: {exc}")
                continue
            return provider
        detail = "; ".join(reasons) or "none configured"
        raise RuntimeError(
            f"no usable onnxruntime provider among {self.providers} in {self.model_dir} ({detail})"
        )

    # ── inference ─────────────────────────────────────────────────────────────
    def _numpy(self) -> _NumpyModule:
        import importlib  # lazy by design (ADR-0002)

        return cast(_NumpyModule, importlib.import_module("numpy"))

    def _forward(self, session: _OrtSession, text: str) -> list[float]:
        np = self._numpy()
        enc = self._tok().encode(text)
        token_count = len(enc.ids)
        feed: dict[str, object] = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        if any(node.name == "position_ids" for node in session.get_inputs()):
            feed["position_ids"] = np.array([list(range(token_count))], dtype=np.int64)
        hidden = np.asarray(session.run(None, feed)[0])
        # Last-token (EOS) pooling: batch size is 1 and nothing is padded, so the final
        # ATTENDED position is the pooled one. Pooling index 0 here would be CLS pooling,
        # which this model was not trained for — it does not raise, it just ranks badly.
        last = sum(enc.attention_mask) - 1
        pooled = cast("list[float]", hidden[0][last].tolist())
        return [float(x) for x in pooled]

    def _embed(self, text: str) -> list[float]:
        primary = self._primary()
        order = [primary, *(p for p in self.providers if p != primary)]
        last_error: Exception | None = None
        for index, provider in enumerate(order):
            started = time.perf_counter()
            try:
                vec = self._forward(self._session(provider), text)
            except Exception as exc:  # noqa: BLE001 — fall through to the next provider
                last_error = exc
                continue
            self._observe(provider, time.perf_counter() - started, order[0] if index else None)
            return _l2_normalise(vec)
        raise RuntimeError(f"all onnx providers failed for {self.backend_label}") from last_error

    def _observe(self, provider: str, seconds: float, fell_back_from: str | None) -> None:
        target = self._observer if self._observer is not None else _embed_observer
        if target is None:
            return
        observer = cast("Callable[[str, float, str | None], None]", target)
        observer(
            short_provider(provider),
            seconds,
            short_provider(fell_back_from) if fell_back_from else None,
        )

    def embed_document(self, content: str, *, is_sensitive: bool) -> list[float] | None:
        if is_sensitive:
            return None  # sensitive rows are never embedded (column stays NULL)
        return self._embed(content)  # documents take NO instruction

    def embed_query(self, query: str) -> list[float]:
        return self._embed(ONNX_QUERY_INSTRUCTION + query)


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

            # Bounded timeout: the library default is None (unbounded). embed runs in a
            # threadpool on the recall path, so an unbounded hang would pin a worker; the
            # finite timeout lets the recall caller degrade to lexical on a slow/hung API.
            self._client = cast(_VoyageClient, voyageai.Client(timeout=VOYAGE_TIMEOUT_SECONDS))
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


#: Process-wide embedder singletons, keyed on the selection input (Voyage key present
#: → hosted, else local). select_embedder() is called on EVERY recall AND every
#: embed-on-write; the heavy part of a backend is the model/client it lazily loads on
#: first embed (bge-large is ~1.3GB). Returning a FRESH backend per call therefore
#: re-instantiated the model on every recall — the 2026-07 latency regression (~5s
#: recalls, OOM when two loads overlapped). One cached instance per backend choice
#: makes the model load exactly once per process and be reused everywhere.
_embedder_cache: dict[str, Embedder] = {}

#: Process-wide embed observer, installed by the API layer via :func:`set_embed_observer`.
_embed_observer: object | None = None


def set_embed_observer(observer: object | None) -> None:
    """Register the process-wide embed observer (provider, seconds, fell_back_from).

    Set once at API startup so :class:`OnnxEmbedder` can report which execution provider
    served each embed WITHOUT this module importing ``api.metrics`` — embeddings.py has
    to keep importing with no optional extras installed (ADR-0002).
    """
    global _embed_observer
    _embed_observer = observer


def select_embedder() -> Embedder:
    """Choose the production embedding backend, returning a process-wide singleton so the
    heavy model loads once (not once per call).

    ``MEMORY_EMBEDDING_BACKEND`` selects explicitly (``onnx`` | ``voyage`` | ``bge``).
    Unset keeps the historical rule — hosted ``voyage-3.5`` iff ``VOYAGE_API_KEY`` is set
    and non-empty, else local ``bge-large`` — so deploying the ONNX backend changes no
    running behaviour until the env says so. That is the flag the GPU cutover flips, and
    flipping it back is the model-side rollback.

    Selection is cheap and imports NO heavy deps: a backend's dependency is imported only
    on its first ``embed_*`` call. The chosen backend is CACHED per selection, so recall
    and embed-on-write reuse one loaded model rather than re-instantiating it per call.
    """
    explicit = os.environ.get(BACKEND_ENV, "").strip().lower()
    if explicit:
        key = explicit
    else:
        key = "voyage" if os.environ.get(VOYAGE_API_KEY_ENV) else "bge"

    embedder = _embedder_cache.get(key)
    if embedder is None:
        if key == "onnx":
            embedder = OnnxEmbedder()
        elif key == "voyage":
            embedder = VoyageEmbedder()
        elif key == "bge":
            embedder = BgeEmbedder()
        else:
            raise ValueError(f"{BACKEND_ENV}={explicit!r} is not one of: onnx, voyage, bge")
        _embedder_cache[key] = embedder
    return embedder


def reset_embedder_cache() -> None:
    """Drop the cached embedder singletons; the next :func:`select_embedder` rebuilds
    lazily. Used by tests for isolation — the loaded model is otherwise process-global."""
    _embedder_cache.clear()
