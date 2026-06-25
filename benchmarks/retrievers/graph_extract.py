"""BATCHED + CACHED LLM triple extraction (EDC open triples) — slice S2.

This is the FIRST stage of "the graph done right": turn each memory into a set of
open ``(subject, relation, object)`` triples via an LLM, so the next slice (S3)
can CANONICALISE those surface forms into a typed concept graph. It exists to make
extraction over the 5452-memory corpus TRACTABLE and REPRODUCIBLE under the task's
hard constraints:

* NOT 5452 sequential LLM calls. Memories are BATCHED ~15-25 id-tagged per call
  (default 25) → ~220-360 calls for the full corpus (5452/25≈218 .. 5452/15≈363).
* CACHED to disk. Every memory's triples are written to a gitignored
  ``triples_<corpusfp>.jsonl`` keyed by a ``(id, content)`` hash, so a rerun over
  the same corpus costs ZERO LLM calls and an incremental rerun only re-extracts
  changed/new memories.
* SENSITIVE rows are never sent externally. A record with ``is_sensitive == 1`` is
  filtered out before batching (ADR-0003 privacy gate). NOTE: the real benchmark
  corpus has NO ``is_sensitive`` field (rows carry only
  id/content/category/tags/expanded_keywords/importance), so on real data this
  guard is a NO-OP; production sensitive memories route through an in-cluster local
  model instead of ``claude``. The guard is exercised by a synthetic test fixture.
* CI never calls a live LLM. The extractor is an INJECTABLE callable
  (``BatchExtractFn``); tests pass a deterministic stub. The real default
  (``default_haiku_extractor``) shells out to ``claude -p --model haiku`` and is
  only used by the offline extraction script, never imported into the CI test path.

The cache file is line-delimited JSON, one row per memory::

    {"id": 1, "h": "<sha256-16 of (id, content)>", "triples": [["s", "rel", "o"], ...]}

Reading it back yields ``dict[int, list[Triple]]``; the ``h`` lets a rerun decide,
per memory, whether the cached triples still apply (content unchanged) or must be
re-extracted (content edited).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess  # used only by the offline default extractor, never in CI
from collections.abc import Iterable, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Protocol, runtime_checkable

# An EDC "open triple": (subject, relation, object), all free-text surface forms.
# Canonicalisation into typed concept nodes/relations happens in S3 (graph_build).
Triple = tuple[str, str, str]

# Memory records are plain dicts (mirroring corpus.jsonl rows) rather than the
# frozen ``harness.types.Memory`` dataclass — DELIBERATELY: ``Memory`` has no
# ``is_sensitive`` field, but the privacy guard must read one when present. Dicts
# also keep this stage decoupled from the harness type and trivially serialisable.
Record = Mapping[str, object]

# Default batch size. Inside the design's documented 15-25 band; 25 minimises call
# count (~218 for the full corpus) while staying small enough that a single haiku
# response stays well within output limits and per-memory attribution is reliable.
_DEFAULT_BATCH_SIZE = 25


@runtime_checkable
class BatchExtractFn(Protocol):
    """An injectable batch extractor.

    Given a batch of memory records (each a mapping with at least ``id`` and
    ``content``), return a mapping from memory id to its list of open triples. An
    id may map to an empty list (no extractable relations). Implementations MUST
    NOT be called for sensitive memories — the caller filters those out first.
    """

    def __call__(self, batch: list[dict[str, object]]) -> dict[int, list[Triple]]:
        ...


def _content_hash(memory_id: int, content: str) -> str:
    """Cache key for one memory = sha256(id ‖ content), truncated to 16 hex chars.

    Sensitive to BOTH id and content, so two memories with identical content but
    different ids never collide and any content edit invalidates exactly that row.
    """
    h = sha256()
    h.update(str(memory_id).encode())
    h.update(b"\x00")
    h.update(content.encode("utf-8", "replace"))
    return h.hexdigest()[:16]


def _is_sensitive(record: Record) -> bool:
    """True iff the record is flagged sensitive. Absent field → non-sensitive
    (the real corpus case). Accepts 1/True/"1"/"true" defensively."""
    val = record.get("is_sensitive", 0)
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes"}
    return bool(val)


def _load_cache(cache_path: Path) -> dict[str, list[Triple]]:
    """Load the triple cache keyed by ``(id, content)`` hash. Tolerant of a missing
    file (first run) and of malformed lines (skipped, so a partially-written cache
    from an interrupted run is still usable)."""
    cache: dict[str, list[Triple]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                key = str(row["h"])
                triples = [tuple(t) for t in row["triples"]]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            # keep only well-formed 3-tuples of strings
            cache[key] = [
                (str(a), str(b), str(c))
                for t in triples
                if isinstance(t, tuple) and len(t) == 3
                for (a, b, c) in [t]
            ]
    return cache


def _append_cache(cache_path: Path, rows: Iterable[tuple[int, str, list[Triple]]]) -> None:
    """Append freshly-extracted (id, hash, triples) rows to the cache file. Append
    (not rewrite) so an interrupted multi-batch run keeps every batch it finished —
    reruns then resume from where they stopped."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for mid, h, triples in rows:
            f.write(
                json.dumps(
                    {"id": mid, "h": h, "triples": [list(t) for t in triples]},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _batched(items: Sequence[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


class TripleExtractor:
    """Batched + cached open-triple extractor.

    Holds the injectable :class:`BatchExtractFn` and the batch size; the heavy
    lifting is in :func:`extract`. Construct with a stub in tests, or
    :func:`default_haiku_extractor` for the offline run.
    """

    def __init__(self, extract_fn: BatchExtractFn, *, batch_size: int = _DEFAULT_BATCH_SIZE) -> None:
        if not 1 <= batch_size <= 25:
            raise ValueError(f"batch_size must be in 1..25 (design band ~15-25), got {batch_size}")
        self.extract_fn = extract_fn
        self.batch_size = batch_size
        self.calls = 0  # number of LLM (batch) calls actually made this run

    def extract(self, records: Sequence[Record], *, cache_path: Path) -> dict[int, list[Triple]]:
        """Return ``{memory_id: [triple, ...]}`` for every NON-SENSITIVE record,
        served from cache where possible and from batched LLM calls otherwise.

        Sensitive records (``is_sensitive == 1``) are dropped before anything else
        and never appear in the result (lexical-only downstream). The result
        preserves input order of the surviving ids.
        """
        # 1) privacy gate: drop sensitive rows BEFORE any extraction or caching.
        eligible: list[dict[str, object]] = [
            dict(r) for r in records if not _is_sensitive(r)
        ]

        # 2) cache lookup, per memory, by (id, content) hash.
        cache = _load_cache(cache_path)
        result: dict[int, list[Triple]] = {}
        misses: list[dict[str, object]] = []
        miss_hashes: dict[int, str] = {}
        for rec in eligible:
            mid = int(rec["id"])  # type: ignore[call-overload]
            content = str(rec["content"])
            h = _content_hash(mid, content)
            cached = cache.get(h)
            if cached is not None:
                result[mid] = cached
            else:
                misses.append(rec)
                miss_hashes[mid] = h

        # 3) batch the misses; a fully-cached corpus makes ZERO calls.
        for batch in _batched(misses, self.batch_size):
            self.calls += 1
            extracted = self.extract_fn(batch)
            new_rows: list[tuple[int, str, list[Triple]]] = []
            for rec in batch:
                mid = int(rec["id"])  # type: ignore[call-overload]
                triples = extracted.get(mid, [])
                # normalise to 3-tuples of str (defensive against list-shaped triples)
                norm: list[Triple] = [
                    (str(t[0]), str(t[1]), str(t[2]))
                    for t in triples
                    if len(tuple(t)) == 3
                ]
                result[mid] = norm
                new_rows.append((mid, miss_hashes[mid], norm))
            _append_cache(cache_path, new_rows)

        return result


def extract_triples(
    records: Sequence[Record],
    extract_fn: BatchExtractFn,
    *,
    cache_path: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> dict[int, list[Triple]]:
    """Functional convenience wrapper around :class:`TripleExtractor` — the entry
    point the offline extraction script and the tests call.

    See :meth:`TripleExtractor.extract` for the contract.
    """
    return TripleExtractor(extract_fn, batch_size=batch_size).extract(
        records, cache_path=cache_path
    )


# ---------------------------------------------------------------------------
# Default extractor: claude -p --model haiku (NON-SENSITIVE memories only).
#
# This is the ONLY code path that touches a live model. It is never imported by
# the CI test module — tests inject a stub. The prompt builder and the response
# parser ARE unit-tested (LLM-free) because they decide attribution and robustness.
# ---------------------------------------------------------------------------

_PROMPT_HEADER = (
    "You extract knowledge-graph triples from short personal-memory notes.\n"
    "For EACH memory below (tagged with its id like [42]), output the salient open "
    "triples as [subject, relation, object]. Use concise lowercase relation phrases "
    "(e.g. prefers, is-a, used-in, part-of, depends-on, runs-on, located-in, "
    "mentions). Keep subjects/objects as the surface entities mentioned; do not "
    "invent facts. A memory with no clear relation may have an empty list.\n\n"
    "Return ONLY a single JSON object mapping each memory id (as a string) to its "
    "list of triples, e.g. "
    '{"42": [["viktor", "prefers", "sveltekit"]], "43": []}. No prose, no code '
    "fences.\n\nMEMORIES:\n"
)


def build_batch_prompt(batch: list[dict[str, object]]) -> str:
    """Build the id-tagged batch prompt. Each memory is prefixed with ``[<id>]`` so
    the model's per-memory triples route back unambiguously — the safety property
    that makes batching correct."""
    lines = [_PROMPT_HEADER]
    for rec in batch:
        mid = int(rec["id"])  # type: ignore[call-overload]
        content = str(rec["content"]).replace("\n", " ").strip()
        lines.append(f"[{mid}] {content}")
    return "\n".join(lines)


_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_batch_response(raw: str, expected_ids: Sequence[int]) -> dict[int, list[Triple]]:
    """Parse the model's JSON-object response into ``{id: [triple, ...]}``.

    Robust to the common ``claude -p`` habits of wrapping JSON in ``` fences or
    adding surrounding prose: the first ``{...}`` span is extracted. Any expected id
    the model omitted maps to an empty list (never a KeyError); malformed triples
    are skipped.
    """
    obj: dict[str, object] = {}
    match = _JSON_OBJ_RE.search(raw)
    if match:
        try:
            loaded = json.loads(match.group(0))
            if isinstance(loaded, dict):
                obj = loaded
        except json.JSONDecodeError:
            obj = {}

    out: dict[int, list[Triple]] = {}
    for mid in expected_ids:
        # JSON object keys are always strings, so the id is looked up as str(mid);
        # an omitted id falls through to an empty list (never a KeyError).
        raw_triples = obj.get(str(mid), [])
        triples: list[Triple] = []
        if isinstance(raw_triples, list):
            for t in raw_triples:
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    triples.append((str(t[0]), str(t[1]), str(t[2])))
        out[mid] = triples
    return out


def default_haiku_extractor(
    batch: list[dict[str, object]],
    *,
    model: str = "haiku",
    timeout_s: float = 120.0,
) -> dict[int, list[Triple]]:
    """The production-offline batch extractor: shell out to ``claude -p --model
    <model>`` with the id-tagged batch prompt and parse the JSON response.

    NEVER reached in CI (tests inject a stub). Raises ``RuntimeError`` if the
    ``claude`` CLI is unavailable or the call fails, so a broken extraction surfaces
    loudly rather than silently producing an empty graph.
    """
    claude = shutil.which("claude")
    if claude is None:  # pragma: no cover - environment-dependent, never in CI
        raise RuntimeError("`claude` CLI not found on PATH; cannot run haiku extraction")
    prompt = build_batch_prompt(batch)
    expected_ids = [int(r["id"]) for r in batch]  # type: ignore[call-overload]
    try:  # pragma: no cover - exercises the live CLI, never in CI
        proc = subprocess.run(  # fixed argv, no shell, trusted binary
            [claude, "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:  # pragma: no cover
        raise RuntimeError(f"claude haiku extraction failed: {exc}") from exc
    return parse_batch_response(proc.stdout, expected_ids)
