"""Re-embed every live non-sensitive memory with the currently-selected backend.

Run inside the API pod after a backend change, while the dense leg is OFF:

    MEMORY_EMBEDDING_BACKEND=onnx MEMORY_ONNX_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
      python scripts/reembed.py --batch 64

Why the dense leg must be off for the duration: this rewrites the embedding column in
place, so mid-run the HNSW index holds a mix of old-model and new-model vectors and a
query embedded by either model ranks against both. Serving lexical-only results while it
runs is the agreed handling (see docs/plans/2026-08-31-gpu-query-embeddings.md), rather
than tolerating a half-migrated index.

Sensitive rows are skipped and keep their NULL embedding, per ADR-0003 — the same gate
the write path applies, asserted here rather than assumed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from claude_memory.api.database import close_pool, get_pool, init_pool
from claude_memory.embeddings import select_embedder

SELECT_BATCH = """
    SELECT id, content
    FROM memories
    WHERE deleted_at IS NULL AND NOT is_sensitive AND id > $1
    ORDER BY id
    LIMIT $2
"""


async def _run(batch: int, dry_run: bool, limit: int | None) -> int:
    embedder = select_embedder()
    print(f"backend: {embedder.backend_label} dim={embedder.dim}", flush=True)
    if os.environ.get("MEMORY_EMBEDDINGS_ENABLED", "").lower() in {"1", "true", "yes"}:
        print(
            "REFUSING: MEMORY_EMBEDDINGS_ENABLED is on. Disable the dense leg first, or"
            " recall will rank queries against a half-migrated index.",
            file=sys.stderr,
        )
        return 2

    await init_pool()
    pool = await get_pool()
    started, done, after = time.perf_counter(), 0, 0
    try:
        while True:
            async with pool.acquire() as conn:
                rows = await conn.fetch(SELECT_BATCH, after, batch)
            if not rows:
                break
            for row in rows:
                after = row["id"]
                vector = await asyncio.to_thread(embedder.embed_document, row["content"], is_sensitive=False)
                if vector is None:  # defensive: the query already excludes sensitive rows
                    continue
                if not dry_run:
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE memories SET embedding = $2::halfvec WHERE id = $1",
                            row["id"],
                            str(vector),
                        )
                done += 1
                if done % 100 == 0:
                    rate = done / (time.perf_counter() - started)
                    print(f"  {done} embedded ({rate:.1f}/s, last id {after})", flush=True)
                if limit is not None and done >= limit:
                    break
            if limit is not None and done >= limit:
                break
    finally:
        await close_pool()

    elapsed = time.perf_counter() - started
    verb = "would re-embed" if dry_run else "re-embedded"
    print(f"{verb} {done} memories in {elapsed:.1f}s ({done / max(elapsed, 1e-9):.1f}/s)", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=64, help="rows fetched per round trip")
    parser.add_argument("--limit", type=int, default=None, help="stop after N rows (for a timed sample)")
    parser.add_argument("--dry-run", action="store_true", help="embed but do not write")
    args = parser.parse_args()
    return asyncio.run(_run(args.batch, args.dry_run, args.limit))


if __name__ == "__main__":
    sys.exit(main())
