#!/usr/bin/env python3
"""Validate the eval set and print AGGREGATE stats (safe to share / commit-able
numbers only — prints NO raw memory content)."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load_dataset  # noqa: E402


def main() -> None:
    ds = load_dataset(validate=True)  # raises on any referential-integrity issue

    strata = Counter(q.stratum for q in ds.queries)
    rel_per_q = {s: [] for s in strata}
    for q in ds.queries:
        rel_per_q[q.stratum].append(len(ds.qrels[q.query_id]))

    # how many DISTINCT corpus memories are exercised as relevant
    relevant_union = set()
    for rels in ds.qrels.values():
        relevant_union |= rels

    out = {
        "corpus_count": len(ds.corpus),
        "query_count": len(ds.queries),
        "strata": dict(strata),
        "relevant_ids_per_query": {
            s: {
                "min": min(v),
                "median": statistics.median(v),
                "max": max(v),
                "mean": round(statistics.fmean(v), 2),
            }
            for s, v in rel_per_q.items()
        },
        "distinct_relevant_memories": len(relevant_union),
        "validation": "PASS (all qrels ids exist in corpus; every query has qrels)",
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
