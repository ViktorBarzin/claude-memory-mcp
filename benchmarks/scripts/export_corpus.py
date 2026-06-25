#!/usr/bin/env python3
"""Export the local SQLite memory cache to a LOCAL-ONLY corpus.jsonl.

Privacy: emits ONLY rows where is_sensitive=0. The output file lives under
benchmarks/data/ which is gitignored. NEVER commit corpus.jsonl.

Fields emitted per line: {id, content, category, tags, expanded_keywords, importance}
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path.home() / ".claude" / "claude-memory" / "memory" / "memory.db"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "corpus.jsonl"


def export(db_path: Path, out_path: Path) -> dict:
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    total = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    sensitive = cur.execute(
        "SELECT COUNT(*) FROM memories WHERE is_sensitive=1"
    ).fetchone()[0]

    rows = cur.execute(
        """
        SELECT id, content, category, tags, expanded_keywords, importance
        FROM memories
        WHERE is_sensitive=0
        ORDER BY id
        """
    ).fetchall()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            rec = {
                "id": r["id"],
                "content": r["content"],
                "category": r["category"],
                "tags": r["tags"],
                "expanded_keywords": r["expanded_keywords"],
                "importance": r["importance"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    con.close()

    return {
        "total_rows": total,
        "sensitive_excluded": sensitive,
        "non_sensitive_written": written,
        "out_path": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    stats = export(args.db, args.out)
    json.dump(stats, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
