#!/usr/bin/env python3
"""One-shot store cleanup for the claude-memory production API (ADR-0007).

Operator tool, safety first:
  - DRY-RUN by default: prints the full change plan + counts; nothing is written
    until --execute is passed.
  - NEVER deletes anything. Obsolete entries are tombstoned via importance plus a
    `supersedes` link (successor -> old), so recall redirects to current truth.
  - Every write is verified by an immediate GET; a failed verify marks the unit
    failed and the run continues (failures are logged and skipped, never fatal).
  - Rate-limit friendly: a small sleep after every write; GETs retry, writes do
    not auto-retry on ambiguous errors (only on a clean 429).
  - Resumable: --checkpoint FILE records finished units; re-runs skip them. The
    LLM step is batched across --workers and only re-composes unfinished units.

Phases (--phase all runs them in this order):
  series      reassemble part-N-of-M fragment series into ONE self-contained
              memory (<=1,400 chars joined) or an LLM-composed hub + as-few-as-
              possible detail parts (part-of links), then supersede + downgrade
              every old fragment to importance 0.3.
  importance  deflate inflated importance: part-N-of-M fragments -> min(cur, 0.4);
              session summaries (or dated decisions/projects) older than 30 days
              -> cap 0.6. Nothing else changes.
  tombstones  freetext "[SUPERSEDED...]" entries -> importance 0.3; when the text
              names a successor ("see #N" / "superseded by #N") also create the
              supersedes link successor -> old.
  dupes       merge the known duplicate clusters (fixed id list) into one
              LLM-consolidated memory; supersede + downgrade members.
  categories  fold legacy category twins onto their canonical spelling
              (existing rows; the API folds on write after the server change).
  corrupted   #6144 (raw XML blob): LLM-extract any real fact into a fresh
              memory, then floor the blob (importance 0.1, "[CORRUPTED -
              superseded]" prefix, content bounded); #5982: strip the trailing
              tool-call XML residue in place (LLM hub+parts split if the cleaned
              content exceeds the bound).

Usage:
    MEMORY_API_KEY=... python scripts/store_cleanup.py \
        --api-url https://claude-memory.viktorbarzin.me --phase all [--execute] \
        [--checkpoint cleanup.ckpt.json] [--workers 4] [--report report.json]

Auth: MEMORY_API_KEY env var. Wizard's key lives in the K8s secret
`claude-memory-secrets`, field `api_keys`.

Endpoint contract (ADR-0007): links via POST /api/memories/{id}/links
{"target_id": ..., "link_type": ...}; single-entry reads via GET
/api/memories/{id}. Both land with the server stream this branch deploys.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("store_cleanup")

# --- ADR-0007 constants -------------------------------------------------------

MAX_CONTENT_CHARS = 1400  # hard Memory content bound — unicode characters, not bytes
LEGACY_CHOP_CHARS = 500  # chunk size of the retired _split_content chopper
MAX_TAGS_CHARS = 500  # server-side Field(max_length=500) on tags

LINK_SUPERSEDES = "supersedes"
LINK_PART_OF = "part-of"
LINK_SEE_ALSO = "see-also"  # closed enum per ADR-0007; unused by the cleanup itself
LINK_RESOLVED_BY = "resolved-by"  # closed enum per ADR-0007; unused by the cleanup itself

PART_IMPORTANCE_CAP = 0.4  # importance phase: part-N-of-M fragments
SUMMARY_IMPORTANCE_CAP = 0.6  # importance phase: aged session summaries / dated decisions
TOMBSTONE_IMPORTANCE = 0.3  # superseded entries (series fragments, dupes, freetext tombstones)
CORRUPTED_IMPORTANCE = 0.1  # the unrecoverable XML blob
DETAIL_PART_IMPORTANCE = 0.4  # newly stored LLM detail parts (hubs are the entry point)
EXTRACTED_FACT_IMPORTANCE = 0.5  # fresh fact extracted from the corrupted blob

SUMMARY_AGE_DAYS = 30
WRITE_SLEEP_SECONDS = 0.2

TOMBSTONE_PREFIX = "[SUPERSEDED"
CORRUPTED_PREFIX = "[CORRUPTED - superseded]"
RESIDUE_MARKER = "</content>"

CORRUPTED_XML_BLOB_ID = 6144
CORRUPTED_RESIDUE_ID = 5982

# Exact known duplicate clusters — do not generalize.
DUPE_CLUSTERS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("xiaomi-email", (5989, 6068)),
    ("nfs-migration", (676, 677)),
    ("zsh-word-split-gotcha", (3547, 3548, 6063, 6084, 6215)),
    ("cnpg", (1431, 1173)),
)

CATEGORY_FOLD_MAP = {
    "gotcha": "gotchas",
    "project": "projects",
    "reference": "references",
    "infra": "infrastructure",
    "bug": "gotchas",
    "incident": "incidents",
    "procedures": "runbook",
}

PHASE_ORDER = ("series", "importance", "tombstones", "dupes", "categories", "corrupted")


class ClientError(RuntimeError):
    """An API call failed."""


class VerifyError(RuntimeError):
    """A write's immediate GET verification did not match."""


class ComposeError(RuntimeError):
    """The LLM compose step failed or returned unusable output."""


# --- mechanical logic (pure, unit-tested) --------------------------------------

_PART_TAG_RE = re.compile(r"^part-(\d+)-of-(\d+)$")
_DATE_TAG_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")
_SUCCESSOR_RE = re.compile(r"(?:\bsee|\bsuperseded\s+by)\s+#(\d+)", re.IGNORECASE)


def split_tags(tags: str) -> list[str]:
    return [t.strip() for t in tags.split(",") if t.strip()]


def parse_part_tag(tags: str) -> tuple[int, int] | None:
    """Return (n, m) from the first exact part-N-of-M tag in a comma-separated tag string."""
    for tag in split_tags(tags):
        match = _PART_TAG_RE.match(tag)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def base_tags(tags: str) -> list[str]:
    """The tag list with every part-N-of-M tag removed, original order preserved."""
    return [t for t in split_tags(tags) if not _PART_TAG_RE.match(t)]


def series_key(memory: dict[str, Any]) -> tuple[str, str, tuple[str, ...], int] | None:
    """Grouping key for fragment series: same owner + category + base tags + M."""
    part = parse_part_tag(memory.get("tags") or "")
    if part is None:
        return None
    _, m = part
    return (
        str(memory.get("owner") or ""),
        str(memory.get("category") or ""),
        tuple(sorted(base_tags(memory.get("tags") or ""))),
        m,
    )


def group_series(memories: list[dict[str, Any]]) -> dict[tuple[str, str, tuple[str, ...], int], list[tuple[int, dict[str, Any]]]]:
    """Group part-N-of-M fragments by series key. Values are unordered (n, memory) pairs."""
    groups: dict[tuple[str, str, tuple[str, ...], int], list[tuple[int, dict[str, Any]]]] = {}
    for memory in memories:
        key = series_key(memory)
        if key is None:
            continue
        n, _ = parse_part_tag(memory["tags"])  # type: ignore[misc]  # key is not None => part tag exists
        groups.setdefault(key, []).append((n, memory))
    return groups


def split_group_into_runs(
    entries: list[tuple[int, dict[str, Any]]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    """Split a tag-colliding group into candidate series runs.

    Distinct series can share identical base tags and M (e.g. two 3-part
    "session-summary" summaries). The legacy chopper inserted each series'
    parts in order 1..M with ascending ids, so: sort by id and cut a new run
    at every n == 1. Runs that do not validate as exactly 1..M are skipped by
    the caller — a colliding group is never wrongly merged.
    """
    runs: list[list[tuple[int, dict[str, Any]]]] = []
    current: list[tuple[int, dict[str, Any]]] = []
    for n, memory in sorted(entries, key=lambda e: e[1]["id"]):
        if n == 1 and current:
            runs.append(current)
            current = []
        current.append((n, memory))
    if current:
        runs.append(current)
    return runs


def order_series(entries: list[tuple[int, dict[str, Any]]], m: int) -> list[dict[str, Any]] | None:
    """Return the fragments ordered 1..M when the series is complete, else None."""
    if m < 2 or len(entries) != m:
        return None
    if sorted(n for n, _ in entries) != list(range(1, m + 1)):
        return None
    return [memory for _, memory in sorted(entries, key=lambda e: e[0])]


def reassemble(contents: list[str], chop_chars: int = LEGACY_CHOP_CHARS) -> str:
    """Invert the retired chopper: a fragment of exactly the chop size was a
    mid-paragraph hard split (its successor continues seamlessly); anything
    shorter ended on a paragraph boundary (rejoin with a blank line)."""
    pieces = [contents[0]]
    for prev, nxt in zip(contents, contents[1:]):
        pieces.append(nxt if len(prev) == chop_chars else f"\n\n{nxt}")
    return "".join(pieces)


def fold_category(category: str) -> str:
    return CATEGORY_FOLD_MAP.get(category, category)


def _parse_when(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        when = datetime.fromisoformat(value)
    except ValueError:
        return None
    return when if when.tzinfo else when.replace(tzinfo=timezone.utc)


def _date_tag(tag_list: list[str]) -> datetime | None:
    """First valid 20XX-XX-XX tag as an aware datetime (the session date)."""
    for tag in tag_list:
        if _DATE_TAG_RE.match(tag):
            try:
                return datetime.strptime(tag, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue  # e.g. 2026-99-99 — shaped like a date but not one
    return None


def _is_aged_summary(memory: dict[str, Any], now: datetime) -> bool:
    tag_list = split_tags(memory.get("tags") or "")
    dated = _date_tag(tag_list)
    is_summary = "session-summary" in tag_list
    is_dated_decision = memory.get("category") in ("decisions", "projects") and dated is not None
    if not (is_summary or is_dated_decision):
        return False
    reference = dated or _parse_when(memory.get("created_at"))
    if reference is None:
        return False
    return (now - reference) > timedelta(days=SUMMARY_AGE_DAYS)


def importance_new_value(memory: dict[str, Any], now: datetime) -> float | None:
    """New importance under the deflation rules, or None when nothing changes.

    Only ever lowers: part-N-of-M fragments cap at 0.4; session summaries (or
    dated decisions/projects) older than 30 days cap at 0.6. Nothing else.
    """
    if parse_part_tag(memory.get("tags") or "") is not None:
        cap = PART_IMPORTANCE_CAP
    elif _is_aged_summary(memory, now):
        cap = SUMMARY_IMPORTANCE_CAP
    else:
        return None
    current = float(memory.get("importance") or 0.0)
    return cap if current > cap + 1e-9 else None


def is_tombstone(content: str) -> bool:
    return content.startswith(TOMBSTONE_PREFIX)


def parse_successor_id(content: str) -> int | None:
    """Successor id named by a freetext tombstone: "see #N" / "superseded by #N"."""
    match = _SUCCESSOR_RE.search(content)
    return int(match.group(1)) if match else None


def strip_residue(content: str, marker: str = RESIDUE_MARKER) -> str | None:
    """Cut trailing tool-call XML residue (marker onwards). None when no marker."""
    index = content.find(marker)
    if index == -1:
        return None
    return content[:index].rstrip()


def build_corrupted_content(original: str, limit: int = MAX_CONTENT_CHARS) -> str:
    """Prefix with the corrupted marker and bound the row to the content limit."""
    prefix = f"{CORRUPTED_PREFIX}\n"
    return prefix + original[: limit - len(prefix)]


def _cap_tags(tag_list: list[str], limit: int = MAX_TAGS_CHARS) -> str:
    joined = ",".join(tag_list)
    while len(joined) > limit and tag_list:
        tag_list = tag_list[:-1]
        joined = ",".join(tag_list)
    return joined


# --- LLM output handling (pure, unit-tested) ------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
MAX_COMPOSED_PARTS = 25


def parse_json_block(text: str) -> Any:
    """Extract the JSON object from LLM output (fenced or naked)."""
    fenced = _FENCE_RE.search(text)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ComposeError(f"no JSON object in LLM output: {text[:200]!r}")
    try:
        return json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ComposeError(f"unparseable JSON in LLM output: {exc}") from exc


def validate_series_compose(obj: Any) -> tuple[str, list[str]]:
    """Validate {"hub": str, "parts": [str, ...]} against the content bound."""
    if not isinstance(obj, dict):
        raise ComposeError(f"expected a JSON object, got {type(obj).__name__}")
    hub = obj.get("hub")
    parts = obj.get("parts") or []
    if not isinstance(hub, str) or not hub.strip():
        raise ComposeError("compose result has no usable 'hub'")
    if len(hub) > MAX_CONTENT_CHARS:
        raise ComposeError(f"hub exceeds {MAX_CONTENT_CHARS} chars ({len(hub)})")
    if not isinstance(parts, list) or not all(isinstance(p, str) for p in parts):
        raise ComposeError("'parts' must be a list of strings")
    if len(parts) > MAX_COMPOSED_PARTS:
        raise ComposeError(f"too many parts ({len(parts)}) — expected as few as possible")
    for part in parts:
        if not part.strip():
            raise ComposeError("empty part in compose result")
        if len(part) > MAX_CONTENT_CHARS:
            raise ComposeError(f"part exceeds {MAX_CONTENT_CHARS} chars ({len(part)})")
    return hub, list(parts)


def validate_single_compose(obj: Any, allow_empty: bool = False) -> str:
    """Validate {"content": str} against the content bound."""
    if not isinstance(obj, dict):
        raise ComposeError(f"expected a JSON object, got {type(obj).__name__}")
    content = obj.get("content")
    if not isinstance(content, str):
        raise ComposeError("compose result has no 'content' string")
    if not content.strip() and not allow_empty:
        raise ComposeError("compose result content is empty")
    if len(content) > MAX_CONTENT_CHARS:
        raise ComposeError(f"content exceeds {MAX_CONTENT_CHARS} chars ({len(content)})")
    return content


def series_prompt(joined: str) -> str:
    return (
        "Rewrite this reassembled session-summary series as ONE hub memory of at most "
        f"{MAX_CONTENT_CHARS} characters (outcome + root cause + key facts) plus AS FEW AS "
        f"POSSIBLE self-contained detail memories of at most {MAX_CONTENT_CHARS} characters "
        "each. Every memory must be understandable entirely on its own — no 'part N' "
        "phrasing, no reliance on the other memories. Output ONLY JSON: "
        '{"hub": "...", "parts": ["...", ...]} (parts may be empty).\n\n'
        f"<series>\n{joined}\n</series>\n"
    )


def dupes_prompt(members: list[dict[str, Any]]) -> str:
    listing = "\n\n".join(f"--- memory #{m['id']} ---\n{m['content']}" for m in members)
    return (
        "Consolidate these near-duplicate memories into ONE self-contained memory of at "
        f"most {MAX_CONTENT_CHARS} characters, preserving every distinct fact. Output ONLY "
        'JSON: {"content": "..."}.\n\n'
        f"{listing}\n"
    )


def corrupted_prompt(content: str) -> str:
    return (
        "The following memory content is a corrupted raw XML blob. If it contains any real, "
        "durable fact worth remembering, rewrite it as ONE self-contained memory of at most "
        f"{MAX_CONTENT_CHARS} characters. If there is no real fact, return an empty string. "
        'Output ONLY JSON: {"content": "..."} ("" when nothing is worth keeping).\n\n'
        f"<blob>\n{content}\n</blob>\n"
    )


def strip_split_prompt(cleaned: str) -> str:
    return (
        "Rewrite this memory as ONE hub memory of at most "
        f"{MAX_CONTENT_CHARS} characters (outcome + root cause + key facts) plus AS FEW AS "
        f"POSSIBLE self-contained detail memories of at most {MAX_CONTENT_CHARS} characters "
        'each. Output ONLY JSON: {"hub": "...", "parts": ["...", ...]} (parts may be empty).\n\n'
        f"<memory>\n{cleaned}\n</memory>\n"
    )


# --- units (one unit = one resumable, checkpointable change) ----------------------


@dataclass
class SeriesUnit:
    key: str
    fragments: list[dict[str, Any]]  # ordered 1..M
    joined: str
    mode: str  # "rewrite" | "llm"
    new_category: str
    new_tags: str
    new_importance: float

    def describe(self) -> str:
        ids = ",".join(str(f["id"]) for f in self.fragments)
        if self.mode == "rewrite":
            return (
                f"reassemble {len(self.fragments)} fragments (ids {ids}) into ONE memory "
                f"({len(self.joined)} chars, category={self.new_category}); supersede + drop "
                f"fragments to {TOMBSTONE_IMPORTANCE}"
            )
        return (
            f"LLM-compose {len(self.fragments)} fragments (ids {ids}, joined {len(self.joined)} "
            f"chars) into hub + parts (category={self.new_category}); supersede + drop fragments "
            f"to {TOMBSTONE_IMPORTANCE}"
        )


@dataclass
class ImportanceUnit:
    key: str
    memory_id: int
    old: float
    new: float
    reason: str

    def describe(self) -> str:
        return f"id {self.memory_id}: importance {self.old} -> {self.new} ({self.reason})"


@dataclass
class TombstoneUnit:
    key: str
    memory_id: int
    old_importance: float
    successor_id: int | None

    def describe(self) -> str:
        parts = []
        if self.old_importance > TOMBSTONE_IMPORTANCE + 1e-9:
            parts.append(f"importance {self.old_importance} -> {TOMBSTONE_IMPORTANCE}")
        if self.successor_id is not None:
            parts.append(f"link supersedes #{self.successor_id} -> #{self.memory_id}")
        return f"id {self.memory_id}: {'; '.join(parts)}"


@dataclass
class DupeUnit:
    key: str
    label: str
    members: list[dict[str, Any]]
    missing: list[int]
    new_category: str
    new_tags: str
    new_importance: float

    def describe(self) -> str:
        ids = ",".join(str(m["id"]) for m in self.members)
        note = f" (missing from store: {self.missing})" if self.missing else ""
        return (
            f"merge cluster '{self.label}' (ids {ids}){note}: LLM-consolidate into one memory "
            f"(category={self.new_category}); supersede + drop members to {TOMBSTONE_IMPORTANCE}"
        )


@dataclass
class CategoryUnit:
    key: str
    memory_id: int
    old: str
    new: str

    def describe(self) -> str:
        return f"id {self.memory_id}: category '{self.old}' -> '{self.new}'"


@dataclass
class CorruptedUnit:
    key: str
    memory: dict[str, Any]
    mode: str  # "blob" | "strip" | "strip-split"
    cleaned: str | None = None

    def describe(self) -> str:
        mid = self.memory["id"]
        if self.mode == "blob":
            return (
                f"id {mid}: LLM-extract any real fact into a fresh memory (supersedes link), then "
                f"importance -> {CORRUPTED_IMPORTANCE} + content prefixed '{CORRUPTED_PREFIX}' "
                f"(bounded to {MAX_CONTENT_CHARS} chars)"
            )
        if self.mode == "strip":
            return f"id {mid}: strip trailing tool-call XML residue in place ({len(self.cleaned or '')} chars remain)"
        return (
            f"id {mid}: strip residue leaves {len(self.cleaned or '')} chars (> {MAX_CONTENT_CHARS}) — "
            f"LLM-split into hub + parts, supersede + drop original to {TOMBSTONE_IMPORTANCE}"
        )


CleanupUnit = SeriesUnit | ImportanceUnit | TombstoneUnit | DupeUnit | CategoryUnit | CorruptedUnit


# --- phase planners (pure over a fetched memory list) ------------------------------


def plan_series(memories: list[dict[str, Any]]) -> tuple[list[SeriesUnit], list[str]]:
    units: list[SeriesUnit] = []
    notes: list[str] = []
    groups = group_series(memories)
    for key in sorted(groups, key=str):
        m = key[3]
        for run in split_group_into_runs(groups[key]):
            ids = ",".join(str(memory["id"]) for _, memory in sorted(run, key=lambda e: e[1]["id"]))
            ordered = order_series(run, m)
            if ordered is None:
                notes.append(f"series ids {ids}: incomplete or ambiguous part-1..{m} run - skipped")
                continue
            if any(f.get("is_sensitive") for f in ordered):
                notes.append(f"series ids {ids}: contains sensitive fragment(s) - skipped")
                continue
            joined = reassemble([f["content"] for f in ordered])
            first = ordered[0]
            units.append(
                SeriesUnit(
                    key="series:" + "-".join(str(f["id"]) for f in ordered),
                    fragments=ordered,
                    joined=joined,
                    mode="rewrite" if len(joined) <= MAX_CONTENT_CHARS else "llm",
                    new_category=fold_category(str(first.get("category") or "facts")),
                    new_tags=_cap_tags(base_tags(first.get("tags") or "")),
                    new_importance=min(
                        max(float(f.get("importance") or 0.0) for f in ordered), SUMMARY_IMPORTANCE_CAP
                    ),
                )
            )
    return units, notes


def plan_importance(memories: list[dict[str, Any]], now: datetime) -> tuple[list[ImportanceUnit], list[str]]:
    units = []
    for memory in memories:
        new = importance_new_value(memory, now)
        if new is None:
            continue
        reason = (
            f"part-N-of-M fragment cap {PART_IMPORTANCE_CAP}"
            if parse_part_tag(memory.get("tags") or "")
            else f"aged summary cap {SUMMARY_IMPORTANCE_CAP}"
        )
        units.append(
            ImportanceUnit(
                key=f"importance:{memory['id']}",
                memory_id=memory["id"],
                old=float(memory.get("importance") or 0.0),
                new=new,
                reason=reason,
            )
        )
    return units, []


def plan_tombstones(memories: list[dict[str, Any]]) -> tuple[list[TombstoneUnit], list[str]]:
    units: list[TombstoneUnit] = []
    notes: list[str] = []
    for memory in memories:
        content = memory.get("content") or ""
        if not is_tombstone(content):
            continue
        old = float(memory.get("importance") or 0.0)
        successor = parse_successor_id(content)
        if successor == memory["id"]:
            successor = None
        if old <= TOMBSTONE_IMPORTANCE + 1e-9 and successor is None:
            notes.append(f"tombstone id {memory['id']}: already at importance <= {TOMBSTONE_IMPORTANCE}, no successor named - skipped")
            continue
        units.append(
            TombstoneUnit(
                key=f"tombstones:{memory['id']}",
                memory_id=memory["id"],
                old_importance=old,
                successor_id=successor,
            )
        )
    return units, notes


def plan_dupes(memories: list[dict[str, Any]]) -> tuple[list[DupeUnit], list[str]]:
    by_id = {m["id"]: m for m in memories}
    units: list[DupeUnit] = []
    notes: list[str] = []
    for label, ids in DUPE_CLUSTERS:
        members = [by_id[i] for i in ids if i in by_id]
        missing = [i for i in ids if i not in by_id]
        if len(members) < 2:
            notes.append(f"dupes '{label}': only {len(members)}/{len(ids)} members present - skipped")
            continue
        if any(m.get("is_sensitive") for m in members):
            notes.append(f"dupes '{label}': contains sensitive member(s) - skipped")
            continue
        folded = Counter(fold_category(str(m.get("category") or "facts")) for m in members)
        merged_tags: list[str] = []
        for member in members:
            for tag in base_tags(member.get("tags") or ""):
                if tag not in merged_tags:
                    merged_tags.append(tag)
        units.append(
            DupeUnit(
                key=f"dupes:{label}",
                label=label,
                members=members,
                missing=missing,
                new_category=folded.most_common(1)[0][0],
                new_tags=_cap_tags(merged_tags),
                new_importance=max(float(m.get("importance") or 0.0) for m in members),
            )
        )
    return units, notes


def plan_categories(memories: list[dict[str, Any]]) -> tuple[list[CategoryUnit], list[str]]:
    units = []
    for memory in memories:
        old = str(memory.get("category") or "")
        new = fold_category(old)
        if new != old:
            units.append(CategoryUnit(key=f"categories:{memory['id']}", memory_id=memory["id"], old=old, new=new))
    return units, []


def plan_corrupted(memories: list[dict[str, Any]]) -> tuple[list[CorruptedUnit], list[str]]:
    by_id = {m["id"]: m for m in memories}
    units: list[CorruptedUnit] = []
    notes: list[str] = []

    blob = by_id.get(CORRUPTED_XML_BLOB_ID)
    if blob is None:
        notes.append(f"corrupted id {CORRUPTED_XML_BLOB_ID}: not found in store - skipped")
    elif blob.get("is_sensitive"):
        notes.append(f"corrupted id {CORRUPTED_XML_BLOB_ID}: sensitive (content masked) - skipped")
    else:
        units.append(CorruptedUnit(key=f"corrupted:{CORRUPTED_XML_BLOB_ID}", memory=blob, mode="blob"))

    residue = by_id.get(CORRUPTED_RESIDUE_ID)
    if residue is None:
        notes.append(f"corrupted id {CORRUPTED_RESIDUE_ID}: not found in store - skipped")
    elif residue.get("is_sensitive"):
        notes.append(f"corrupted id {CORRUPTED_RESIDUE_ID}: sensitive (content masked) - skipped")
    else:
        cleaned = strip_residue(residue.get("content") or "")
        if cleaned is None:
            notes.append(f"corrupted id {CORRUPTED_RESIDUE_ID}: no '{RESIDUE_MARKER}' residue marker - skipped")
        elif not cleaned.strip():
            notes.append(f"corrupted id {CORRUPTED_RESIDUE_ID}: nothing left after stripping residue - skipped")
        else:
            mode = "strip" if len(cleaned) <= MAX_CONTENT_CHARS else "strip-split"
            units.append(
                CorruptedUnit(key=f"corrupted:{CORRUPTED_RESIDUE_ID}", memory=residue, mode=mode, cleaned=cleaned)
            )
    return units, notes


# --- API client ---------------------------------------------------------------------


class MemoryClient:
    """Thin urllib client for the memory API. GETs retry; writes retry only on 429
    (a refused request was definitely not applied — anything else is ambiguous)."""

    def __init__(self, api_url: str, api_key: str, timeout: float = 60.0) -> None:
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None,
                 query: dict[str, Any] | None = None) -> Any:
        url = self.api_url + path
        if query:
            url += "?" + urllib.parse.urlencode(query)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        attempts = 3
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = response.read().decode()
                    return json.loads(payload) if payload else None
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode()[:300]
                except OSError:
                    pass
                if exc.code == 429 and attempt < attempts:
                    time.sleep(2**attempt)
                    continue
                raise ClientError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if method == "GET" and attempt < attempts:
                    time.sleep(2**attempt)
                    continue
                raise ClientError(f"{method} {path} -> {exc}") from exc
        raise ClientError(f"{method} {path}: retries exhausted")

    def auth_user(self) -> str:
        response = self._request("GET", "/api/auth-check")
        return str(response["user_id"])

    def list_all(self, page_size: int = 500) -> list[dict[str, Any]]:
        memories: list[dict[str, Any]] = []
        offset = 0
        while True:
            response = self._request("GET", "/api/memories", query={"limit": page_size, "offset": offset})
            page = response.get("memories", [])
            memories.extend(page)
            total = int(response.get("total") or 0)
            offset += len(page)
            if not page or offset >= total:
                return memories

    def get(self, memory_id: int) -> dict[str, Any]:
        response = self._request("GET", f"/api/memories/{memory_id}")
        if isinstance(response, dict) and isinstance(response.get("memory"), dict):
            merged = dict(response["memory"])
            if "links" in response and "links" not in merged:
                merged["links"] = response["links"]
            return merged
        if not isinstance(response, dict):
            raise ClientError(f"GET /api/memories/{memory_id}: unexpected response shape")
        return response

    def store(self, content: str, category: str, tags: str, importance: float,
              expanded_keywords: str = "") -> int:
        response = self._request(
            "POST",
            "/api/memories",
            body={
                "content": content,
                "category": category,
                "tags": tags,
                "importance": importance,
                "expanded_keywords": expanded_keywords,
            },
        )
        return int(response["id"])

    def update(self, memory_id: int, **fields: Any) -> None:
        self._request("PUT", f"/api/memories/{memory_id}", body=fields)

    def create_link(self, src_id: int, target_id: int, link_type: str) -> None:
        # ADR-0007 endpoint shape; the links API lands with the server stream.
        self._request("POST", f"/api/memories/{src_id}/links",
                      body={"target_id": target_id, "link_type": link_type})


# --- LLM composer ---------------------------------------------------------------------


class ClaudeComposer:
    """Compose via a `claude -p --model haiku` subprocess; prompt on stdin, JSON out."""

    def __init__(self, model: str = "haiku", timeout_s: float = 240.0) -> None:
        self.model = model
        self.timeout_s = timeout_s

    def compose(self, prompt: str) -> Any:
        last_error: ComposeError | None = None
        for _ in range(2):  # one retry on bad output — haiku occasionally chats around the JSON
            try:
                process = subprocess.run(
                    ["claude", "-p", "--model", self.model],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                )
            except FileNotFoundError as exc:
                raise ComposeError("`claude` CLI not found on PATH") from exc
            except subprocess.TimeoutExpired as exc:
                raise ComposeError(f"claude -p timed out after {self.timeout_s}s") from exc
            if process.returncode != 0:
                last_error = ComposeError(f"claude -p exited {process.returncode}: {process.stderr[-300:]}")
                continue
            try:
                return parse_json_block(process.stdout)
            except ComposeError as exc:
                last_error = exc
        raise last_error if last_error else ComposeError("compose failed")


# --- checkpoint --------------------------------------------------------------------------


class Checkpoint:
    """Set of finished unit keys, persisted as JSON so interrupted runs resume."""

    def __init__(self, path: Path | str | None) -> None:
        self.path = Path(path) if path else None
        self._processed: set[str] = set()
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text())
            self._processed = set(data.get("processed", []))

    def seen(self, key: str) -> bool:
        return key in self._processed

    def mark(self, key: str) -> None:
        self._processed.add(key)
        if self.path:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps({"processed": sorted(self._processed)}, indent=1))
            tmp.replace(self.path)


# --- runner -------------------------------------------------------------------------------


class Runner:
    """Plans and (with execute=True) applies one phase at a time.

    Client and composer are injected so tests run with fakes — no network, no
    subprocess. Every write is verified by an immediate GET; per-unit failures
    are recorded and the run continues.
    """

    def __init__(self, client: Any, composer: Any, execute: bool, checkpoint: Checkpoint,
                 sleep_s: float = WRITE_SLEEP_SECONDS, workers: int = 4,
                 now: datetime | None = None) -> None:
        self.client = client
        self.composer = composer
        self.execute = execute
        self.checkpoint = checkpoint
        self.sleep_s = sleep_s
        self.workers = max(1, workers)
        self.now = now or datetime.now(timezone.utc)
        self._composed: dict[str, Any] = {}

    # -- write helpers: every write sleeps (rate-limit) and verifies by GET --

    def _sleep(self) -> None:
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)

    def _verify(self, memory_id: int, **expect: Any) -> None:
        got = self.client.get(memory_id)
        for key, want in expect.items():
            actual = got.get(key)
            if isinstance(want, float):
                matches = isinstance(actual, (int, float)) and abs(float(actual) - want) < 1e-6
            else:
                matches = actual == want
            if not matches:
                raise VerifyError(f"memory {memory_id}: expected {key}={want!r}, GET returned {actual!r}")

    def _store(self, content: str, category: str, tags: str, importance: float,
               writes: list[str], what: str) -> int:
        new_id = int(self.client.store(content=content, category=category, tags=tags, importance=importance))
        self._sleep()
        self._verify(new_id, content=content, category=category, importance=importance)
        writes.append(f"stored {what} as #{new_id} ({len(content)} chars)")
        return new_id

    def _update(self, memory_id: int, writes: list[str], **fields: Any) -> None:
        self.client.update(memory_id, **fields)
        self._sleep()
        self._verify(memory_id, **fields)
        summary = ", ".join(f"{k}={v!r}" if not isinstance(v, str) else f"{k}=<{len(v)} chars>"
                            for k, v in fields.items())
        writes.append(f"updated #{memory_id}: {summary}")

    def _link(self, src_id: int, dst_id: int, link_type: str, writes: list[str]) -> None:
        self.client.create_link(src_id, dst_id, link_type)
        self._sleep()
        self._verify_link(src_id, dst_id, link_type)
        writes.append(f"linked #{src_id} -{link_type}-> #{dst_id}")

    def _verify_link(self, src_id: int, dst_id: int, link_type: str) -> None:
        got = self.client.get(src_id)
        links = got.get("links")
        if links is None:
            return  # GET does not expose links — the POST already returned 2xx
        blob = json.dumps(links)
        if str(dst_id) not in blob or link_type not in blob:
            raise VerifyError(f"link #{src_id} -{link_type}-> #{dst_id} not visible in GET after create")

    def _memory_exists(self, memory_id: int, known_ids: set[int]) -> bool:
        if memory_id in known_ids:
            return True
        try:
            self.client.get(memory_id)
            return True
        except ClientError:
            return False

    # -- LLM batching --

    def _prompt_for(self, unit: CleanupUnit) -> str | None:
        if isinstance(unit, SeriesUnit) and unit.mode == "llm":
            return series_prompt(unit.joined)
        if isinstance(unit, DupeUnit):
            return dupes_prompt(unit.members)
        if isinstance(unit, CorruptedUnit) and unit.mode == "blob":
            return corrupted_prompt(unit.memory.get("content") or "")
        if isinstance(unit, CorruptedUnit) and unit.mode == "strip-split":
            return strip_split_prompt(unit.cleaned or "")
        return None

    def _precompose(self, units: list[CleanupUnit]) -> None:
        """Run the LLM step for all pending units, batched across --workers."""
        pending = [(u, self._prompt_for(u)) for u in units if not self.checkpoint.seen(u.key)]
        pending = [(u, p) for u, p in pending if p is not None]
        if not pending:
            return
        log.info("composing %d unit(s) via LLM (%d workers)", len(pending), self.workers)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.composer.compose, prompt): unit for unit, prompt in pending}
            for future in as_completed(futures):
                unit = futures[future]
                try:
                    self._composed[unit.key] = future.result()
                except Exception as exc:  # noqa: BLE001 — per-unit failures must never be fatal
                    self._composed[unit.key] = ComposeError(str(exc))

    def _take_composed(self, unit: CleanupUnit) -> Any:
        result = self._composed.get(unit.key)
        if result is None:
            raise ComposeError(f"no composed output for {unit.key}")
        if isinstance(result, Exception):
            raise result
        return result

    # -- unit apply --

    def _apply_series(self, unit: SeriesUnit, writes: list[str]) -> None:
        if unit.mode == "llm":
            hub_content, parts = validate_series_compose(self._take_composed(unit))
        else:
            hub_content, parts = unit.joined, []
        hub_id = self._store(hub_content, unit.new_category, unit.new_tags, unit.new_importance,
                             writes, "hub" if parts else "rewritten series")
        for part_content in parts:
            part_id = self._store(part_content, unit.new_category, unit.new_tags,
                                  DETAIL_PART_IMPORTANCE, writes, "detail part")
            self._link(part_id, hub_id, LINK_PART_OF, writes)
        for fragment in unit.fragments:
            # Link BEFORE downgrading: if the run dies between the two, the old
            # fragment still ranks (redirecting to the hub) rather than vanishing.
            self._link(hub_id, fragment["id"], LINK_SUPERSEDES, writes)
            if float(fragment.get("importance") or 0.0) > TOMBSTONE_IMPORTANCE + 1e-9:
                self._update(fragment["id"], writes, importance=TOMBSTONE_IMPORTANCE)

    def _apply_importance(self, unit: ImportanceUnit, writes: list[str]) -> None:
        self._update(unit.memory_id, writes, importance=unit.new)

    def _apply_tombstone(self, unit: TombstoneUnit, writes: list[str], known_ids: set[int]) -> None:
        if unit.successor_id is not None:
            if self._memory_exists(unit.successor_id, known_ids):
                self._link(unit.successor_id, unit.memory_id, LINK_SUPERSEDES, writes)
            else:
                writes.append(f"successor #{unit.successor_id} not found - link skipped")
        if unit.old_importance > TOMBSTONE_IMPORTANCE + 1e-9:
            self._update(unit.memory_id, writes, importance=TOMBSTONE_IMPORTANCE)

    def _apply_dupe(self, unit: DupeUnit, writes: list[str]) -> None:
        content = validate_single_compose(self._take_composed(unit))
        new_id = self._store(content, unit.new_category, unit.new_tags, unit.new_importance,
                             writes, f"consolidated '{unit.label}'")
        for member in unit.members:
            self._link(new_id, member["id"], LINK_SUPERSEDES, writes)
            if float(member.get("importance") or 0.0) > TOMBSTONE_IMPORTANCE + 1e-9:
                self._update(member["id"], writes, importance=TOMBSTONE_IMPORTANCE)

    def _apply_corrupted(self, unit: CorruptedUnit, writes: list[str]) -> None:
        memory = unit.memory
        if unit.mode == "strip":
            self._update(memory["id"], writes, content=unit.cleaned)
            return
        if unit.mode == "strip-split":
            hub_content, parts = validate_series_compose(self._take_composed(unit))
            tags = _cap_tags(base_tags(memory.get("tags") or ""))
            category = fold_category(str(memory.get("category") or "facts"))
            importance = min(float(memory.get("importance") or 0.0) or EXTRACTED_FACT_IMPORTANCE,
                             SUMMARY_IMPORTANCE_CAP)
            hub_id = self._store(hub_content, category, tags, importance, writes, "hub")
            for part_content in parts:
                part_id = self._store(part_content, category, tags, DETAIL_PART_IMPORTANCE, writes, "detail part")
                self._link(part_id, hub_id, LINK_PART_OF, writes)
            self._link(hub_id, memory["id"], LINK_SUPERSEDES, writes)
            self._update(memory["id"], writes, importance=TOMBSTONE_IMPORTANCE)
            return
        # mode == "blob" (#6144)
        fact = validate_single_compose(self._take_composed(unit), allow_empty=True)
        if fact.strip():
            tags = _cap_tags(base_tags(memory.get("tags") or ""))
            category = fold_category(str(memory.get("category") or "facts"))
            new_id = self._store(fact, category, tags, EXTRACTED_FACT_IMPORTANCE, writes, "extracted fact")
            self._link(new_id, memory["id"], LINK_SUPERSEDES, writes)
        else:
            writes.append("LLM found no real fact in the blob - nothing extracted")
        self._update(memory["id"], writes, importance=CORRUPTED_IMPORTANCE,
                     content=build_corrupted_content(memory.get("content") or ""))

    def _apply(self, phase: str, unit: CleanupUnit, known_ids: set[int]) -> list[str]:
        writes: list[str] = []
        if isinstance(unit, SeriesUnit):
            self._apply_series(unit, writes)
        elif isinstance(unit, ImportanceUnit):
            self._apply_importance(unit, writes)
        elif isinstance(unit, TombstoneUnit):
            self._apply_tombstone(unit, writes, known_ids)
        elif isinstance(unit, DupeUnit):
            self._apply_dupe(unit, writes)
        elif isinstance(unit, CategoryUnit):
            self._update(unit.memory_id, writes, category=unit.new)
        elif isinstance(unit, CorruptedUnit):
            self._apply_corrupted(unit, writes)
        else:  # pragma: no cover — closed union
            raise ClientError(f"unknown unit type for phase {phase}: {type(unit).__name__}")
        return writes

    # -- phase driver --

    def _plan(self, phase: str, memories: list[dict[str, Any]]) -> tuple[list[CleanupUnit], list[str]]:
        if phase == "series":
            return plan_series(memories)  # type: ignore[return-value]
        if phase == "importance":
            return plan_importance(memories, self.now)  # type: ignore[return-value]
        if phase == "tombstones":
            return plan_tombstones(memories)  # type: ignore[return-value]
        if phase == "dupes":
            return plan_dupes(memories)  # type: ignore[return-value]
        if phase == "categories":
            return plan_categories(memories)  # type: ignore[return-value]
        if phase == "corrupted":
            return plan_corrupted(memories)  # type: ignore[return-value]
        raise ValueError(f"unknown phase: {phase}")

    def run_phase(self, phase: str, memories: list[dict[str, Any]]) -> dict[str, Any]:
        units, notes = self._plan(phase, memories)
        known_ids = {m["id"] for m in memories}
        summary: dict[str, Any] = {
            "phase": phase,
            "planned": len(units),
            "applied": 0,
            "failed": 0,
            "skipped": len(notes),
            "notes": notes,
            "units": [],
        }
        for note in notes:
            print(f"[{phase}] SKIP {note}")
        if self.execute:
            self._precompose(units)
        for unit in units:
            record: dict[str, Any] = {"key": unit.key, "detail": unit.describe()}
            if self.checkpoint.seen(unit.key):
                summary["skipped"] += 1
                record["status"] = "skipped-checkpoint"
                print(f"[{phase}] SKIP {unit.key}: already processed (checkpoint)")
            elif not self.execute:
                record["status"] = "planned"
                print(f"[{phase}] PLAN {unit.key}: {unit.describe()}")
            else:
                try:
                    record["writes"] = self._apply(phase, unit, known_ids)
                    record["status"] = "applied"
                    summary["applied"] += 1
                    self.checkpoint.mark(unit.key)
                    print(f"[{phase}] DONE {unit.key}: {unit.describe()}")
                except Exception as exc:  # noqa: BLE001 — log and skip, never fatal
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    summary["failed"] += 1
                    log.error("[%s] FAILED %s: %s", phase, unit.key, exc)
            summary["units"].append(record)
        print(
            f"[{phase}] planned={summary['planned']} applied={summary['applied']} "
            f"failed={summary['failed']} skipped={summary['skipped']}"
        )
        return summary


# --- CLI ------------------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot claude-memory store cleanup (ADR-0007). DRY-RUN by default: "
        "prints the full change plan; pass --execute to apply. Never deletes — obsolete "
        "entries are tombstoned via importance + supersedes links, and every write is "
        "verified by an immediate GET.",
        epilog="Auth: set MEMORY_API_KEY. Wizard's key lives in the K8s secret "
        "`claude-memory-secrets`, field `api_keys`.",
    )
    parser.add_argument("--api-url", required=True, help="Memory API base URL, e.g. https://claude-memory.viktorbarzin.me")
    parser.add_argument("--phase", required=True, choices=("all", *PHASE_ORDER),
                        help="which cleanup phase to run ('all' runs every phase in order)")
    parser.add_argument("--execute", action="store_true",
                        help="apply the changes (default is a dry run that only prints the plan)")
    parser.add_argument("--checkpoint", metavar="FILE",
                        help="JSON checkpoint file; already-processed ids are skipped so interrupted runs resume")
    parser.add_argument("--workers", type=int, default=4, help="parallel workers for the LLM compose step (default 4)")
    parser.add_argument("--report", metavar="FILE", help="write a JSON summary of the run to FILE")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    api_key = os.environ.get("MEMORY_API_KEY", "")
    if not api_key:
        log.error("MEMORY_API_KEY is not set (wizard's key: K8s secret claude-memory-secrets, field api_keys)")
        return 2

    client = MemoryClient(args.api_url, api_key)
    runner = Runner(
        client=client,
        composer=ClaudeComposer(),
        execute=args.execute,
        checkpoint=Checkpoint(args.checkpoint),
        sleep_s=WRITE_SLEEP_SECONDS,
        workers=args.workers,
    )

    try:
        me = client.auth_user()
    except ClientError as exc:
        log.error("auth-check against %s failed: %s", args.api_url, exc)
        return 2
    mode = "EXECUTE" if args.execute else "DRY-RUN (no writes; pass --execute to apply)"
    print(f"store_cleanup: {mode} | api={args.api_url} | user={me} | phase={args.phase}")

    phases = PHASE_ORDER if args.phase == "all" else (args.phase,)
    report: dict[str, Any] = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "user": me,
        "execute": args.execute,
        "phases": {},
    }
    exit_code = 0
    for phase in phases:
        # Re-fetch per phase: earlier phases change importance/content the later ones read.
        try:
            memories = [m for m in client.list_all() if m.get("owner") == me]
        except ClientError as exc:
            log.error("listing memories failed before phase %s: %s", phase, exc)
            exit_code = 1
            break
        print(f"\n=== phase {phase} ({len(memories)} memories owned by {me}) ===")
        summary = runner.run_phase(phase, memories)
        report["phases"][phase] = summary
        if summary["failed"]:
            exit_code = 1

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}")

    totals = {k: sum(p[k] for p in report["phases"].values()) for k in ("planned", "applied", "failed", "skipped")}
    print(f"\nTOTAL planned={totals['planned']} applied={totals['applied']} "
          f"failed={totals['failed']} skipped={totals['skipped']}")
    if not args.execute:
        print("dry run only - nothing was written")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
