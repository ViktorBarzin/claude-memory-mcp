"""Export a Cyrillic-focused eval corpus from the live store.

Every non-sensitive memory carrying Cyrillic, plus a deterministic sample of English
ones as distractors. Smaller than the real store (which inflates absolute scores for
both models equally), so it is built for COMPARING two embedders on Bulgarian recall,
not for predicting live recall@k.

Writes corpus.jsonl in the harness's schema to $OUT.
"""
import asyncio
import json
import os
import random

import asyncpg

OUT = os.environ.get("OUT", "/tmp/cyr-eval/corpus.jsonl")
DISTRACTORS = int(os.environ.get("DISTRACTORS", "1400"))
RNG = random.Random(20260901)

CYR = r"[Ѐ-ӿ]"


async def main():
    c = await asyncpg.connect(os.environ["DATABASE_URL"])
    cyr = await c.fetch(
        f"""SELECT id, content, category, tags, expanded_keywords, importance
            FROM memories
            WHERE deleted_at IS NULL AND NOT is_sensitive AND content ~ '{CYR}'
            ORDER BY id"""
    )
    eng = await c.fetch(
        f"""SELECT id, content, category, tags, expanded_keywords, importance
            FROM memories
            WHERE deleted_at IS NULL AND NOT is_sensitive AND content !~ '{CYR}'
            ORDER BY id"""
    )
    await c.close()

    sample = RNG.sample(list(eng), min(DISTRACTORS, len(eng)))
    rows = list(cyr) + sample
    rows.sort(key=lambda r: r["id"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        for r in rows:
            fh.write(
                json.dumps(
                    {
                        "id": r["id"],
                        "content": r["content"],
                        "category": r["category"] or "",
                        "tags": r["tags"] or "",
                        "expanded_keywords": r["expanded_keywords"] or "",
                        "importance": float(r["importance"] or 0.5),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {OUT}: {len(cyr)} cyrillic + {len(sample)} english distractors = {len(rows)} docs")


asyncio.run(main())
