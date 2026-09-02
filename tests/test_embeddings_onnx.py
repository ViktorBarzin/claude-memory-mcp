"""Tests for the ONNX-served ``Qwen/Qwen3-Embedding-0.6B`` backend.

Two of this model's conventions differ from bge's, and BOTH fail silently — they
produce vectors that look valid (right dim, unit norm) and rank badly:

* **last-token (EOS) pooling**, where bge uses CLS. Pooling position 0 instead of the
  final attended position is a one-character mistake that no assertion on shape or norm
  would catch, so it gets a direct test.
* **its own query instruction**, ``Instruct: ...\\nQuery:``, applied to queries only.
  Reusing bge's prefix, or applying it to documents, costs retrieval quality per the
  model card without erroring.

Also covered: the CUDA-then-CPU provider fallback that keeps dense recall alive through
a GPU outage, and the ADR-0003 sensitive gate.

Like ``test_embeddings.py`` these stay offline — ``onnxruntime`` and ``tokenizers`` are
monkeypatched with deterministic fakes, so no graph is loaded and no model is downloaded.
"""

from __future__ import annotations

import math
import sys
from typing import cast
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, ClassVar

import pytest

import claude_memory.embeddings as emb

if TYPE_CHECKING:
    from collections.abc import Iterator

EXPECTED_DIM = 1024

#: Providers whose LIBRARIES are missing — onnxruntime silently serves these on CPU.
UNAVAILABLE: set[str] = set()
#: Providers whose session cannot be built at all (unreadable graph) — these do raise.
UNSESSIONABLE: set[str] = set()
#: Providers whose session builds but whose run() raises, e.g. VRAM pressure mid-request.
BROKEN: set[str] = set()


class _FakeEncoding:
    def __init__(self, text: str) -> None:
        # One "token" per 4 characters, minimum 2, so sequence length varies with input
        # and the pooling index is not trivially 0 or 1.
        n = max(2, len(text) // 4)
        self.ids = list(range(n))
        self.attention_mask = [1] * n


class _FakeTokenizer:
    encoded: ClassVar[list[str]] = []

    @staticmethod
    def from_file(path: str) -> _FakeTokenizer:
        return _FakeTokenizer()

    def encode(self, sequence: str) -> _FakeEncoding:
        _FakeTokenizer.encoded.append(sequence)
        return _FakeEncoding(sequence)


class _FakeSession:
    """Returns a hidden state where position i is distinguishable from every other.

    ``hidden[0, i, 0] = i + 1`` and ``hidden[0, i, 1] = 1.0``, everything else 0. So the
    ratio ``vec[0] / vec[1]`` of the POOLED vector survives L2 normalisation and equals
    (pooled index + 1) — which is exactly the assertion that separates last-token pooling
    from CLS pooling.
    """

    def __init__(self, requested: str) -> None:
        # Model onnxruntime's REAL behaviour: an unavailable provider does not raise,
        # it silently falls back to CPU and the session still constructs. Measured
        # 2026-09-01 — a pod missing libcublasLt built a "CUDA" session that ran on the
        # CPU. An earlier version of this fake raised instead, which is why the original
        # provider-detection bug shipped: the fake encoded the assumption under test.
        self.requested = requested
        self.provider = "CPUExecutionProvider" if requested in UNAVAILABLE else requested
        self.runs = 0

    def get_inputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def get_providers(self) -> list[str]:
        # Always includes the CPU fallback, as onnxruntime does.
        return [self.provider] if self.provider == "CPUExecutionProvider" else [self.provider, "CPUExecutionProvider"]

    def run(self, output_names: None, input_feed: dict[str, object]) -> list[object]:
        import numpy as np

        if self.provider in BROKEN:
            raise RuntimeError(f"{self.provider} failed at inference")
        self.runs += 1
        ids = np.asarray(input_feed["input_ids"])
        seq = int(ids.shape[1])
        hidden = np.zeros((1, seq, EXPECTED_DIM), dtype=np.float32)
        for i in range(seq):
            hidden[0, i, 0] = float(i + 1)
            hidden[0, i, 1] = 1.0
        return [hidden]


def _fake_ort_module() -> ModuleType:
    mod = ModuleType("onnxruntime")

    class _SessionOptions:
        graph_optimization_level: object = None

    def _inference_session(path: str, opts: object, providers: list[object]) -> _FakeSession:
        # onnxruntime accepts a bare provider name or a (name, options) pair.
        # Normalise so the fakes below only ever see the name.
        providers = [p[0] if isinstance(p, tuple) else p for p in providers]
        # Two distinct real behaviours. A missing GPU *library* does NOT raise: the
        # session builds and quietly runs on CPU (see _FakeSession). A session that
        # cannot be built at all — unreadable graph file, unusable CPU provider — does
        # raise. UNSESSIONABLE models the second; UNAVAILABLE the first.
        if providers[0] in UNSESSIONABLE:
            raise RuntimeError(f"cannot build a session for {providers[0]}: {path} unreadable")
        return _FakeSession(providers[0])

    mod.SessionOptions = _SessionOptions  # type: ignore[attr-defined]
    mod.InferenceSession = _inference_session  # type: ignore[attr-defined]
    mod.GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="all")  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def _fakes(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    UNAVAILABLE.clear()
    UNSESSIONABLE.clear()
    BROKEN.clear()
    _FakeTokenizer.encoded = []
    tok_mod = ModuleType("tokenizers")
    tok_mod.Tokenizer = _FakeTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tokenizers", tok_mod)
    monkeypatch.setitem(sys.modules, "onnxruntime", _fake_ort_module())
    emb.reset_embedder_cache()
    emb.set_embed_observer(None)
    yield
    emb.reset_embedder_cache()
    emb.set_embed_observer(None)


def _l2(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


def _pooled_index(vec: list[float]) -> int:
    """Recover which sequence position the fake hidden state was pooled from."""
    return round(vec[0] / vec[1])


def _both() -> emb.OnnxEmbedder:
    return emb.OnnxEmbedder(
        model_dir="/nonexistent",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


# ── pooling: the silent-failure case ─────────────────────────────────────────────


def test_pools_the_last_token_not_the_first() -> None:
    """bge pools CLS (position 0); Qwen3-Embedding pools the final EOS position.

    Pooling position 0 here would give a ratio of 1; the correct last-token pooling gives
    the sequence length.
    """
    text = "a" * 40  # 40 chars -> 10 tokens
    vec = _both().embed_document(text, is_sensitive=False)
    assert vec is not None
    assert _pooled_index(vec) == 10
    assert _pooled_index(vec) != 1, "pooled position 0 — that is CLS pooling, not last-token"


def test_pooled_position_tracks_sequence_length() -> None:
    """A longer input must pool from a later position, which a fixed index would not do."""
    e = _both()
    short = e.embed_document("b" * 20, is_sensitive=False)
    long = e.embed_document("b" * 80, is_sensitive=False)
    assert short is not None and long is not None
    assert _pooled_index(short) == 5
    assert _pooled_index(long) == 20


# ── the query instruction ────────────────────────────────────────────────────────


def test_query_carries_the_qwen_instruction_and_documents_do_not() -> None:
    e = _both()
    e.embed_query("which DNS server do we use")
    e.embed_document("Technitium DNS runs on 10.0.20.201", is_sensitive=False)
    query_text, doc_text = _FakeTokenizer.encoded
    assert query_text.startswith(emb.ONNX_QUERY_INSTRUCTION)
    assert query_text.endswith("which DNS server do we use")
    assert doc_text == "Technitium DNS runs on 10.0.20.201"


def test_query_instruction_is_qwens_not_bges() -> None:
    """Carrying bge's prefix into this model degrades retrieval without erroring."""
    assert emb.ONNX_QUERY_INSTRUCTION.startswith("Instruct:")
    assert "Query:" in emb.ONNX_QUERY_INSTRUCTION
    assert emb.BGE_QUERY_INSTRUCTION not in emb.ONNX_QUERY_INSTRUCTION


# ── output contract ──────────────────────────────────────────────────────────────


def test_vectors_are_l2_normalised_dim_1024() -> None:
    e = _both()
    for vec in (e.embed_query("q"), e.embed_document("d", is_sensitive=False)):
        assert vec is not None
        assert len(vec) == EXPECTED_DIM
        assert _l2(vec) == pytest.approx(1.0, abs=1e-9)


def test_sensitive_document_is_never_embedded() -> None:
    """ADR-0003 hard gate: a sensitive row keeps a NULL embedding."""
    assert _both().embed_document("bank details", is_sensitive=True) is None


def test_dim_and_label_match_the_halfvec_contract() -> None:
    e = _both()
    assert e.dim == EXPECTED_DIM == emb.EMBEDDING_DIM
    assert e.backend_label == f"onnx:{emb.ONNX_MODEL}"


# ── provider selection and fallback ──────────────────────────────────────────────


def test_prefers_the_first_available_provider() -> None:
    assert _both().warm() == "cuda"


def test_falls_back_when_the_gpu_provider_is_unavailable() -> None:
    """A pod scheduled without a usable GPU still serves dense recall on CPU."""
    UNAVAILABLE.add("CUDAExecutionProvider")
    assert _both().warm() == "cpu"


def test_a_silently_downgraded_session_is_not_reported_as_gpu() -> None:
    """The 2026-09-01 production bug, as a test.

    onnxruntime does not raise when a provider is unavailable — it falls back to CPU and
    still returns a usable session. Reporting the REQUESTED provider rather than the
    registered one made a CPU-served deployment look GPU-served, which in turn kept
    memory_embed_fallbacks_total at zero and made the alert for that exact condition
    unfireable. The embedder must believe get_providers(), not the request.
    """
    UNAVAILABLE.add("CUDAExecutionProvider")
    e = _both()
    assert e.warm() == "cpu", "a silently-downgraded CUDA session must not report as cuda"

    seen: list[tuple[str, str | None]] = []
    e2 = emb.OnnxEmbedder(
        model_dir="/nonexistent",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        observer=lambda provider, seconds, fell_back_from: seen.append((provider, fell_back_from)),
    )
    e2.embed_query("served by whom?")
    assert seen == [("cpu", None)], f"metrics must attribute this embed to cpu, got {seen}"


def test_falls_back_when_the_gpu_provider_raises_at_inference() -> None:
    """The GPU session builds but inference fails, e.g. VRAM pressure mid-request."""
    BROKEN.add("CUDAExecutionProvider")
    seen: list[tuple[str, float, str | None]] = []
    e = emb.OnnxEmbedder(
        model_dir="/nonexistent",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        observer=lambda provider, seconds, fell_back_from: seen.append((provider, seconds, fell_back_from)),
    )
    vec = e.embed_query("still works")
    assert vec is not None
    assert _l2(vec) == pytest.approx(1.0, abs=1e-9)
    assert [(p, f) for p, _, f in seen] == [("cpu", "cuda")]


def test_an_unusable_provider_is_attempted_once_not_per_call() -> None:
    """Caching the FAILURE matters as much as caching the session.

    Building a session reads the whole graph (~2.4 GB for this model), so retrying a
    doomed provider on every embed costs a full model load per call. Measured before this
    was fixed: with CUDA unavailable the re-embed ran at 0.1 memories/s (12.4 s each)
    against the CPU path's ~0.2 s, because each call rebuilt the CUDA session just to
    raise again.
    """
    builds: list[str] = []
    UNAVAILABLE.add("CUDAExecutionProvider")

    def _counting_session(path: str, opts: object, providers: list[object]) -> _FakeSession:
        # CUDA is passed as (name, options) so its arena strategy can be set;
        # CPU stays a bare name. Count the name either way.
        name = providers[0][0] if isinstance(providers[0], tuple) else providers[0]
        builds.append(cast(str, name))
        return _FakeSession(cast(str, name))

    mod = sys.modules["onnxruntime"]
    mod.InferenceSession = _counting_session  # type: ignore[attr-defined]

    e = _both()
    for _ in range(5):
        e.embed_query("repeatedly")

    cuda_builds = builds.count("CUDAExecutionProvider")
    assert cuda_builds == 1, f"the unusable provider was rebuilt {cuda_builds} times, expected 1"
    assert builds.count("CPUExecutionProvider") == 1, "the serving session must be cached too"


def test_raises_when_every_provider_fails() -> None:
    UNSESSIONABLE.update({"CUDAExecutionProvider", "CPUExecutionProvider"})
    with pytest.raises(RuntimeError, match="no usable onnxruntime provider"):
        _both().embed_query("nothing can serve this")


def test_observer_records_the_serving_provider_without_fallback() -> None:
    seen: list[tuple[str, str | None]] = []
    e = emb.OnnxEmbedder(
        model_dir="/nonexistent",
        providers=["CUDAExecutionProvider"],
        observer=lambda provider, seconds, fell_back_from: seen.append((provider, fell_back_from)),
    )
    e.embed_query("hello")
    assert seen == [("cuda", None)]


def test_module_level_observer_is_used_when_no_instance_observer() -> None:
    seen: list[str] = []
    emb.set_embed_observer(lambda provider, seconds, fell_back_from: seen.append(provider))
    emb.OnnxEmbedder(model_dir="/nonexistent", providers=["CPUExecutionProvider"]).embed_query("hi")
    assert seen == ["cpu"]


def test_short_provider_naming() -> None:
    assert emb.short_provider("CUDAExecutionProvider") == "cuda"
    assert emb.short_provider("CPUExecutionProvider") == "cpu"


# ── backend selection ────────────────────────────────────────────────────────────


def test_backend_env_selects_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(emb.BACKEND_ENV, "onnx")
    assert isinstance(emb.select_embedder(), emb.OnnxEmbedder)


def test_unset_backend_env_keeps_the_historical_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Landing this module must not change what a running deployment serves."""
    monkeypatch.delenv(emb.BACKEND_ENV, raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    assert isinstance(emb.select_embedder(), emb.BgeEmbedder)


def test_backend_env_rejects_an_unknown_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(emb.BACKEND_ENV, "gpt5")
    with pytest.raises(ValueError, match="not one of"):
        emb.select_embedder()


def test_onnx_backend_is_a_process_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(emb.BACKEND_ENV, "onnx")
    assert emb.select_embedder() is emb.select_embedder()


def test_onnx_satisfies_the_embedder_protocol() -> None:
    assert isinstance(_both(), emb.Embedder)


# ── packaging invariant ──────────────────────────────────────────────────────────


def test_every_provider_maps_to_a_baked_graph() -> None:
    """The image bakes one gated graph; if a provider is added to the default list its
    graph must be baked too, or the pod fails at first embed rather than at build."""
    for provider in emb.ONNX_PROVIDERS_DEFAULT.split(","):
        assert provider.strip() in emb.ONNX_FILE_BY_PROVIDER
    assert set(emb.ONNX_FILE_BY_PROVIDER.values()) == {"model.onnx"}


# --- CUDA arena growth (bead code-n3xl) --------------------------------------
# onnxruntime's CUDA BFC arena defaults to arena_extend_strategy=kNextPowerOfTwo,
# so it DOUBLES when it needs more and never returns it. Measured on
# claude-memory 2026-09-02: resident VRAM sat at 3218 MiB for 18h, then stepped
# to 7314 MiB in one hour and held flat there for six with no pod restart.
# +4096 exactly is the arena doubling, not the model growing, and a flat line is
# not a leak. kSameAsRequested makes the arena track what inference actually
# asks for.
def _recording_ort_module() -> tuple[ModuleType, list[object]]:
    """The fake ort module, plus a list that captures each `providers` argument."""
    mod = _fake_ort_module()
    seen: list[object] = []
    inner = mod.InferenceSession  # type: ignore[attr-defined]

    def _recorder(path: str, opts: object, providers: list[object]) -> object:
        seen.append(providers[0])
        return inner(path, opts, providers)

    mod.InferenceSession = _recorder  # type: ignore[attr-defined]
    return mod, seen


def test_cuda_provider_pins_the_arena_extend_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod, seen = _recording_ort_module()
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    emb.OnnxEmbedder(model_dir="/m", providers=["CUDAExecutionProvider"]).embed_query("hi")

    assert seen, "no session was built"
    entry = seen[0]
    assert isinstance(entry, tuple), (
        "CUDAExecutionProvider must be passed as (name, options) so the arena "
        f"strategy can be set; got a bare {type(entry).__name__}"
    )
    name, options = entry
    assert name == "CUDAExecutionProvider"
    assert options["arena_extend_strategy"] == "kSameAsRequested"


def test_cpu_provider_is_still_passed_bare(monkeypatch: pytest.MonkeyPatch) -> None:
    # The arena strategy is a CUDA-EP option. Attaching it to the CPU provider
    # would be a silent no-op at best and a registration error at worst.
    mod, seen = _recording_ort_module()
    monkeypatch.setitem(sys.modules, "onnxruntime", mod)
    emb.OnnxEmbedder(model_dir="/m", providers=["CPUExecutionProvider"]).embed_query("hi")

    assert seen == ["CPUExecutionProvider"]
