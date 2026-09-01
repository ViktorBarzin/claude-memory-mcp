"""Score the LIVE recall path against the preserved eval set.

The offline harness scores a retriever built in-process. This scores what sessions
actually get: `homelab memory recall`, through the API, with whatever backend and
execution provider the deployment is currently running. That makes it the check that
matters for a serving change, and the one that catches a fusion or link-semantics
regression the offline dense leg alone would miss.

    python benchmarks/scripts/live_eval.py --out benchmarks/results/live-<label>.json

MEASURED 2026-09-01: this script's QUALITY numbers are not usable, and the reason is
worth keeping. Memory ids have been reassigned since the eval set was built, so the
preserved qrels no longer join to the live store — corpus id 197 is a Prometheus
compaction memory, live id 197 is an ingress-resilience one. Scored live, the whole set
returns recall@5 0.008 against an offline lexical baseline of 0.666, which measures the
id mismatch and nothing else. The qrels remain self-consistent with the preserved
corpus.jsonl, so the QUALITY gate belongs in the offline harness over that snapshot.

What this script is still good for is LATENCY through the real serving path: the CLI,
the API, fusion and link semantics, exactly as a session experiences it. Read
latency_seconds; ignore the metric block unless the eval set has been rebuilt against
current ids.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.metrics import ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def _load(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text().splitlines() if line.strip()]


def _recall(query: str, k: int) -> tuple[list[int], float]:
    started = time.perf_counter()
    proc = subprocess.run(
        ["homelab", "memory", "recall", query, "--limit", str(k), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        return [], elapsed
    payload = json.loads(proc.stdout)
    rows = payload if isinstance(payload, list) else (payload.get("memories") or payload.get("results") or [])
    return [int(r["id"]) for r in rows], elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--label", default="live")
    args = parser.parse_args()

    queries = _load("queries.jsonl")
    qrels: dict[str, set[int]] = {}
    for row in _load("qrels.jsonl"):
        qrels.setdefault(row["query_id"], set()).update(row.get("relevant_ids", [row["memory_id"]] if "memory_id" in row else []))

    per_stratum: dict[str, list[dict[str, float]]] = {}
    latencies: list[float] = []
    overall: list[dict[str, float]] = []

    for i, q in enumerate(queries, 1):
        relevant = qrels.get(q["query_id"]) or set(q.get("relevant_ids", []))
        if not relevant:
            continue
        ranked, elapsed = _recall(q["text"], args.k)
        latencies.append(elapsed)
        scores = {
            "recall@5": recall_at_k(ranked, relevant, 5),
            "recall@10": recall_at_k(ranked, relevant, 10),
            "ndcg@10": ndcg_at_k(ranked, relevant, 10),
            "mrr": reciprocal_rank(ranked, relevant),
        }
        overall.append(scores)
        stratum = q.get("stratum", "unknown")
        per_stratum.setdefault(stratum, []).append(scores)
        if CYRILLIC.search(q["text"]):
            per_stratum.setdefault("cyrillic", []).append(scores)
        if i % 20 == 0:
            print(f"  {i}/{len(queries)} queries", flush=True)

    def _mean(rows: list[dict[str, float]]) -> dict[str, float]:
        return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in rows[0]}

    latencies.sort()
    result = {
        "retriever_name": args.label,
        "n_queries": len(overall),
        "retrieve_k": args.k,
        "overall": _mean(overall),
        "per_stratum": {name: {"n_queries": len(rows), "metrics": _mean(rows)} for name, rows in per_stratum.items()},
        "latency_seconds": {
            "p50": round(latencies[len(latencies) // 2], 3),
            "p90": round(latencies[int(len(latencies) * 0.9)], 3),
            "max": round(latencies[-1], 3),
        },
        "caveat": "QUALITY METRICS INVALID: memory ids were reassigned after this eval set was built, so the qrels do not join to the live store. Read latency_seconds only.",
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
