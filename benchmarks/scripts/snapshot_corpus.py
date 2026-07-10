#!/usr/bin/env python3
"""Snapshot the CURRENT remote memory store into a NEW dated corpus dir.

This is the API twin of ``export_corpus.py`` (which reads the local SQLite cache):
it pulls every live memory for the authenticated user via ``GET /api/memories/sync``
— the one endpoint that returns ``expanded_keywords`` and ``is_sensitive`` for all
rows — and writes the harness corpus format (README "Dataset schema") to a fresh
snapshot dir, e.g.::

    MEMORY_API_KEY=... .venv/bin/python scripts/snapshot_corpus.py
    # → benchmarks/snapshots/2026-07-10/corpus.jsonl + snapshot_meta.json

Privacy: exactly like the original export, rows with ``is_sensitive`` set are
EXCLUDED entirely (the SQLite export's ``WHERE is_sensitive=0``), and the output
lives under ``benchmarks/snapshots/`` which is gitignored — NEVER commit a snapshot.

The preserved 5,452-memory eval set (``benchmarks/data`` → the benchmark-artifacts
symlink) is the FIXED baseline reference and must never be modified; this script
refuses to write anywhere inside it (``assert_snapshot_dir_safe``).

Environment:
    MEMORY_API_URL - API base URL (alias CLAUDE_MEMORY_API_URL; default the
                     production endpoint https://claude-memory.viktorbarzin.me)
    MEMORY_API_KEY - API key (alias CLAUDE_MEMORY_API_KEY); sent as a Bearer token
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

_BENCH_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BENCH_ROOT))

DEFAULT_API_URL = "https://claude-memory.viktorbarzin.me"
DEFAULT_SNAPSHOT_ROOT = _BENCH_ROOT / "snapshots"
PRESERVED_DATA_DIR = _BENCH_ROOT / "data"

# The corpus.jsonl fields, in the schema order the preserved set uses.
_CORPUS_FIELDS = ("id", "content", "category", "tags", "expanded_keywords", "importance")


def fetch_all_memories(api_url: str, api_key: str, *, timeout: float = 120.0) -> list[dict[str, Any]]:
    """Pull every live (non-deleted) memory for the key's user via the sync endpoint."""
    req = urllib.request.Request(
        f"{api_url.rstrip('/')}/api/memories/sync",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    memories = payload.get("memories")
    if not isinstance(memories, list):
        raise ValueError(f"unexpected sync response shape: keys={sorted(payload)}")
    return memories


def to_corpus_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Map raw API rows to corpus.jsonl records, excluding sensitive/deleted rows.

    Mirrors export_corpus.py: sensitive rows (``is_sensitive`` truthy) are excluded
    entirely; deleted rows (``deleted_at`` set — belt-and-braces, the sync endpoint
    already filters them) are excluded too; records are sorted by id. ``None``
    optional fields normalise to the harness defaults ("" / "facts" / 0.5).
    """
    records: list[dict[str, Any]] = []
    sensitive = 0
    deleted = 0
    for row in rows:
        if row.get("is_sensitive"):
            sensitive += 1
            continue
        if row.get("deleted_at") is not None:
            deleted += 1
            continue
        records.append(
            {
                "id": int(row["id"]),
                "content": row["content"],
                "category": row.get("category") or "facts",
                "tags": row.get("tags") or "",
                "expanded_keywords": row.get("expanded_keywords") or "",
                "importance": 0.5 if row.get("importance") is None else row["importance"],
            }
        )
    records.sort(key=lambda r: r["id"])
    stats = {
        "total_rows": len(rows),
        "sensitive_excluded": sensitive,
        "deleted_excluded": deleted,
        "written": len(records),
    }
    return records, stats


def write_corpus_jsonl(records: list[dict[str, Any]], out_path: Path) -> int:
    """Write corpus records as JSONL (one object per line, unicode preserved)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({k: rec[k] for k in _CORPUS_FIELDS}, ensure_ascii=False) + "\n")
    return len(records)


def assert_snapshot_dir_safe(out_dir: Path, preserved_data_dir: Path = PRESERVED_DATA_DIR) -> None:
    """Refuse any snapshot destination inside the PRESERVED eval-set dir.

    ``benchmarks/data`` is a symlink to the preserved benchmark artifacts — the fixed
    baseline the regression gate compares against — so both paths are fully resolved
    before the containment check (a destination routed through the symlink is caught).
    """
    resolved_out = out_dir.resolve()
    resolved_preserved = preserved_data_dir.resolve()
    if resolved_out == resolved_preserved or resolved_preserved in resolved_out.parents:
        raise ValueError(
            f"refusing to write snapshot into the preserved eval set: {out_dir} "
            f"resolves inside {resolved_preserved}. Preserved data is the fixed "
            "baseline reference and must never be modified; use a fresh dir under "
            f"{DEFAULT_SNAPSHOT_ROOT} instead."
        )


def write_snapshot(
    records: list[dict[str, Any]],
    stats: dict[str, int],
    out_dir: Path,
    *,
    source: str,
    force: bool = False,
) -> dict[str, Any]:
    """Write corpus.jsonl + snapshot_meta.json into ``out_dir`` (a NEW snapshot dir).

    Refuses to overwrite an existing snapshot unless ``force`` — snapshots are cheap;
    the preserved baseline habit of never clobbering an eval artifact applies here too.
    Returns the metadata written to snapshot_meta.json (counts, fingerprint, source).
    """
    assert_snapshot_dir_safe(out_dir)
    corpus_path = out_dir / "corpus.jsonl"
    if corpus_path.exists() and not force:
        raise ValueError(f"snapshot already exists: {corpus_path} (pass --force to overwrite)")

    write_corpus_jsonl(records, corpus_path)

    # Fingerprint with the SAME (id, content) scheme the cached-embedding key uses,
    # so a snapshot is identifiable in cache filenames and regression outputs.
    from harness.dataset import load_corpus
    from retrievers.hybrid import _corpus_fingerprint

    meta = {
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": source,
        "corpus_fingerprint": _corpus_fingerprint(load_corpus(corpus_path)),
        "stats": stats,
    }
    (out_dir / "snapshot_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--api-url",
        default=os.environ.get("MEMORY_API_URL")
        or os.environ.get("CLAUDE_MEMORY_API_URL")
        or DEFAULT_API_URL,
        help="memory API base URL (default: $MEMORY_API_URL or the production endpoint)",
    )
    ap.add_argument(
        "--name",
        default=_dt.date.today().isoformat(),
        help="snapshot dir name under --out-root (default: today's date)",
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT)
    ap.add_argument("--force", action="store_true", help="overwrite an existing snapshot of the same name")
    args = ap.parse_args()

    api_key = os.environ.get("MEMORY_API_KEY") or os.environ.get("CLAUDE_MEMORY_API_KEY")
    if not api_key:
        raise SystemExit("MEMORY_API_KEY (or CLAUDE_MEMORY_API_KEY) must be set")

    out_dir = args.out_root / args.name
    assert_snapshot_dir_safe(out_dir)

    rows = fetch_all_memories(args.api_url, api_key)
    records, stats = to_corpus_records(rows)
    meta = write_snapshot(records, stats, out_dir, source=args.api_url, force=args.force)

    json.dump({**meta, "out_dir": str(out_dir)}, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
