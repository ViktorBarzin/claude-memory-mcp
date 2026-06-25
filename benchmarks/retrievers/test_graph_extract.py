"""Unit tests for BATCHED + CACHED LLM triple extraction (slice S2).

These tests are STRICTLY LLM-FREE: the real extractor shells out to
``claude -p --model haiku`` (non-sensitive) or a local model (sensitive), but CI
must NEVER call a live LLM. So the extractor takes an INJECTABLE callable — a
``BatchExtractFn`` — and every test passes a deterministic stub. A
``CallCountingStub`` wraps the stub and records how many times it was invoked, so
the tests can assert the cache short-circuits ALL LLM calls on a rerun (stub
called 0 times the second time).

What slice S2 guarantees (acceptance):
  * BATCHING: memories are grouped ~15-25 id-tagged per call (default 25), so the
    full 5452-memory corpus costs ~220-360 calls, NOT 5452 sequential calls.
  * CACHE: every batch's triples are written to ``triples_<corpusfp>.jsonl``
    (gitignored), keyed by a (id, content) hash; a populated cache makes a rerun
    cost 0 LLM calls (the stub is invoked 0 times).
  * CACHE KEY = (id, content): editing a memory's content invalidates ONLY that
    memory's cache row; an unchanged-content rerun reuses everything.
  * SENSITIVE FILTER: ``is_sensitive=1`` records are never sent to the extractor.
    The real corpus has NO ``is_sensitive`` field (verified: rows carry only
    id/content/category/tags/expanded_keywords/importance), so on real data the
    guard is a no-op — the test therefore constructs a SYNTHETIC record carrying
    the field (challenger-corrected).

Run:  .venv/bin/python -m pytest retrievers/test_graph_extract.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from retrievers.graph_extract import (
    BatchExtractFn,
    Triple,
    TripleExtractor,
    _content_hash,
    extract_triples,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _records(n: int, *, start: int = 1) -> list[dict[str, object]]:
    """n synthetic non-sensitive corpus records (dicts, mirroring corpus.jsonl)."""
    return [
        {"id": i, "content": f"memory number {i} talks about topic {i % 7}"}
        for i in range(start, start + n)
    ]


class CallCountingStub:
    """Wraps a batch-extract stub and counts how many BATCHES (LLM calls) it saw.

    The stub returns, per memory id in the batch, a single deterministic triple
    ``[content, "mentions", "topic-<id>"]`` — enough to assert structure and that
    every requested id is covered, without any real model.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.batch_sizes: list[int] = []

    def __call__(self, batch: list[dict[str, object]]) -> dict[int, list[Triple]]:
        self.calls += 1
        self.batch_sizes.append(len(batch))
        out: dict[int, list[Triple]] = {}
        for rec in batch:
            mid = int(rec["id"])  # type: ignore[call-overload]
            out[mid] = [(str(rec["content"]), "mentions", f"topic-{mid}")]
        return out


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def test_batching_groups_15_to_25_per_call(tmp_path: Path) -> None:
    """60 memories at the default batch size (25) → 3 calls (25, 25, 10); every
    batch is within the design's ~15-25 band except the final remainder, and no
    batch exceeds the cap."""
    stub = CallCountingStub()
    cache = tmp_path / "triples_x.jsonl"
    out = extract_triples(_records(60), stub, cache_path=cache, batch_size=25)

    assert stub.calls == 3, "60 / 25 → ceil = 3 batches"
    assert stub.batch_sizes == [25, 25, 10]
    assert all(sz <= 25 for sz in stub.batch_sizes), "no batch exceeds the cap"
    # every memory id got at least one triple
    assert set(out) == {r["id"] for r in _records(60)}
    assert all(len(v) >= 1 for v in out.values())


def test_batch_size_in_design_band() -> None:
    """The default batch size lands inside the design's documented 15-25 band so
    the full 5452 corpus costs ~220-360 calls (5452/25≈218 .. 5452/15≈363)."""
    extractor = TripleExtractor(lambda batch: {})
    assert 15 <= extractor.batch_size <= 25


def test_stub_satisfies_the_injectable_extractor_protocol() -> None:
    """The injection point is the ``BatchExtractFn`` Protocol; a stub that matches
    its call shape must be accepted (so any in-cluster / hosted extractor can be
    swapped in without touching the batching/caching machinery)."""
    stub = CallCountingStub()
    assert isinstance(stub, BatchExtractFn)


# ---------------------------------------------------------------------------
# caching — the core acceptance criterion
# ---------------------------------------------------------------------------

def test_populated_cache_short_circuits_all_llm_calls(tmp_path: Path) -> None:
    """First run populates the cache (3 calls for 60 memories); the SECOND run over
    the SAME corpus invokes the stub 0 times — the acceptance criterion that reruns
    cost 0 LLM calls."""
    cache = tmp_path / "triples_run.jsonl"
    recs = _records(60)

    stub1 = CallCountingStub()
    out1 = extract_triples(recs, stub1, cache_path=cache, batch_size=25)
    assert stub1.calls == 3
    assert cache.exists(), "first run must persist triples to the cache file"

    # second run: same corpus, fresh stub → must be served entirely from cache.
    stub2 = CallCountingStub()
    out2 = extract_triples(recs, stub2, cache_path=cache, batch_size=25)
    assert stub2.calls == 0, "a populated cache must short-circuit ALL LLM calls"
    assert out2 == out1, "cache-served triples must equal the first run's output"


def test_partial_cache_only_extracts_the_misses(tmp_path: Path) -> None:
    """If the cache covers some ids, only the UNCACHED ids are sent to the LLM, and
    they are batched among themselves — so incremental corpus growth is cheap."""
    cache = tmp_path / "triples_partial.jsonl"
    first = _records(25)  # ids 1..25
    stub1 = CallCountingStub()
    extract_triples(first, stub1, cache_path=cache, batch_size=25)
    assert stub1.calls == 1

    # now ask for 1..40: 1..25 are cached, 26..40 (15 new) need exactly one batch.
    grown = _records(40)
    stub2 = CallCountingStub()
    out = extract_triples(grown, stub2, cache_path=cache, batch_size=25)
    assert stub2.calls == 1, "only the 15 new ids should hit the LLM"
    assert stub2.batch_sizes == [15]
    assert set(out) == {r["id"] for r in grown}  # but the result covers all 40


def test_cache_key_is_id_and_content(tmp_path: Path) -> None:
    """The cache key is a (id, content) hash: changing a memory's CONTENT
    invalidates only that row, forcing re-extraction of just that memory."""
    cache = tmp_path / "triples_key.jsonl"
    recs = _records(20)
    stub1 = CallCountingStub()
    extract_triples(recs, stub1, cache_path=cache, batch_size=25)
    assert stub1.calls == 1

    # edit ONE memory's content → its (id, content) hash changes → it misses.
    edited = [dict(r) for r in recs]
    edited[5]["content"] = "completely different content now"
    stub2 = CallCountingStub()
    extract_triples(edited, stub2, cache_path=cache, batch_size=25)
    assert stub2.calls == 1, "only the one edited memory should re-extract"
    assert stub2.batch_sizes == [1]


def test_content_hash_depends_on_both_id_and_content() -> None:
    """_content_hash is sensitive to BOTH fields (so two memories with identical
    content but different ids don't collide, and an edit changes the key)."""
    h = _content_hash
    assert h(1, "abc") == h(1, "abc")  # stable
    assert h(1, "abc") != h(2, "abc")  # id matters
    assert h(1, "abc") != h(1, "abd")  # content matters


# ---------------------------------------------------------------------------
# sensitive filter — SYNTHETIC fixture (real corpus has no is_sensitive field)
# ---------------------------------------------------------------------------

def test_sensitive_records_are_never_sent_to_the_extractor(tmp_path: Path) -> None:
    """is_sensitive=1 records must never reach the (external) extractor. Because the
    real corpus has NO is_sensitive field, this test constructs a SYNTHETIC fixture
    carrying it (challenger-corrected). The sensitive row is filtered out: it is not
    in any batch, and it gets NO triples in the result (lexical-only downstream)."""
    cache = tmp_path / "triples_sensitive.jsonl"
    recs: list[dict[str, object]] = [
        {"id": 1, "content": "public memory about kubernetes", "is_sensitive": 0},
        {"id": 2, "content": "SECRET personal memory", "is_sensitive": 1},
        {"id": 3, "content": "public memory about postgres"},  # field absent → non-sensitive
    ]
    seen_ids: list[int] = []

    def recording_stub(batch: list[dict[str, object]]) -> dict[int, list[Triple]]:
        for rec in batch:
            seen_ids.append(int(rec["id"]))  # type: ignore[call-overload]
        return {int(r["id"]): [("s", "rel", "o")] for r in batch}  # type: ignore[call-overload]

    out = extract_triples(recs, recording_stub, cache_path=cache, batch_size=25)

    assert 2 not in seen_ids, "the is_sensitive=1 memory must never be sent to the LLM"
    assert seen_ids == [1, 3], "only non-sensitive ids reach the extractor"
    assert 2 not in out, "sensitive memory gets no extracted triples (lexical-only)"
    assert set(out) == {1, 3}


def test_all_sensitive_corpus_makes_zero_calls(tmp_path: Path) -> None:
    """A corpus that is entirely sensitive must make ZERO LLM calls and produce no
    triples (the privacy hard-gate, ADR-0003)."""
    cache = tmp_path / "triples_allsensitive.jsonl"
    recs: list[dict[str, object]] = [
        {"id": i, "content": f"secret {i}", "is_sensitive": 1} for i in range(1, 6)
    ]
    stub = CallCountingStub()
    out = extract_triples(recs, stub, cache_path=cache, batch_size=25)
    assert stub.calls == 0
    assert out == {}


# ---------------------------------------------------------------------------
# triple shape / extractor parsing contract
# ---------------------------------------------------------------------------

def test_extractor_returns_subject_relation_object_triples(tmp_path: Path) -> None:
    """Extracted triples are (subject, relation, object) 3-tuples of strings — the
    EDC open-triple shape the canonicalization slice (S3) consumes."""
    cache = tmp_path / "triples_shape.jsonl"
    out = extract_triples(_records(3), CallCountingStub(), cache_path=cache, batch_size=25)
    for triples in out.values():
        for t in triples:
            assert isinstance(t, tuple) and len(t) == 3
            assert all(isinstance(part, str) for part in t)


def test_cache_roundtrips_triples_unchanged(tmp_path: Path) -> None:
    """Triples written to the JSONL cache and read back are identical (the cache is
    the source of truth on rerun, so a lossy serialization would corrupt S3's
    input)."""
    cache = tmp_path / "triples_roundtrip.jsonl"
    recs = [
        {"id": 1, "content": "Viktor prefers SvelteKit for the frontend"},
        {"id": 2, "content": "the cluster runs on Proxmox"},
    ]

    def stub(batch: list[dict[str, object]]) -> dict[int, list[Triple]]:
        return {
            1: [("Viktor", "prefers", "SvelteKit"), ("SvelteKit", "used-in", "frontend")],
            2: [("cluster", "runs-on", "Proxmox")],
        }

    out1 = extract_triples(recs, stub, cache_path=cache, batch_size=25)
    # rerun served purely from cache → identical structure incl. relation strings.
    out2 = extract_triples(recs, lambda b: pytest.fail("must not call LLM"), cache_path=cache)
    assert out2 == out1
    assert out1[1] == [("Viktor", "prefers", "SvelteKit"), ("SvelteKit", "used-in", "frontend")]


# ---------------------------------------------------------------------------
# default LLM extractor wiring (still LLM-FREE: we only assert the prompt builder
# and parser, never shell out)
# ---------------------------------------------------------------------------

def test_prompt_tags_each_memory_with_its_id(tmp_path: Path) -> None:
    """The batch prompt must id-tag each memory so the model's per-memory triples
    can be routed back. We assert the prompt builder emits each id; this is what
    makes batching safe (no cross-memory attribution)."""
    from retrievers.graph_extract import build_batch_prompt

    batch = _records(3, start=10)  # ids 10,11,12
    prompt = build_batch_prompt(batch)
    for rec in batch:
        assert f"[{rec['id']}]" in prompt, "each memory must be id-tagged in the prompt"
    # the instruction must ask for triples (subject/relation/object) as JSON.
    low = prompt.lower()
    assert "triple" in low and "json" in low


def test_parse_batch_response_routes_triples_by_id() -> None:
    """The response parser maps the model's JSON back to per-id triple lists and
    tolerates ids the model omitted (→ empty list, never a KeyError)."""
    from retrievers.graph_extract import parse_batch_response

    raw = (
        '{"10": [["a", "rel", "b"]], '
        '"11": [["c", "is-a", "d"], ["c", "part-of", "e"]]}'
    )
    parsed = parse_batch_response(raw, expected_ids=[10, 11, 12])
    assert parsed[10] == [("a", "rel", "b")]
    assert parsed[11] == [("c", "is-a", "d"), ("c", "part-of", "e")]
    assert parsed[12] == [], "an omitted id yields an empty list, not an error"


def test_parse_batch_response_tolerates_fenced_json() -> None:
    """`claude -p` often wraps JSON in ```json fences or adds prose; the parser must
    extract the JSON object regardless (robustness for the real haiku path)."""
    from retrievers.graph_extract import parse_batch_response

    raw = 'Here are the triples:\n```json\n{"5": [["x", "y", "z"]]}\n```\nDone.'
    parsed = parse_batch_response(raw, expected_ids=[5])
    assert parsed[5] == [("x", "y", "z")]
