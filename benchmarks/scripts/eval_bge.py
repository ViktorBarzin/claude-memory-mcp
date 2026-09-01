"""Score the OUTGOING bge-large backend on an eval set, for a like-for-like comparison.

The stored `results/hybrid.json` baseline was built against the June corpus snapshot, so
it cannot be compared to a run over a different corpus. When a new eval set is built —
the Cyrillic set, for instance — the bge side has to be re-run over that same corpus or
there is no baseline at all, only a number with nothing to sit against.

Runs on the workstation CPU with sentence-transformers, because bge-large is no longer in
the runtime image (the ONNX migration removed torch). Slow by design: roughly 0.6 s a
document, so a 2,000-document corpus is about twenty minutes. That is the price of a
real comparison rather than an assumed one.

    python benchmarks/scripts/eval_bge.py --data-dir <dir> --out <file>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.dataset import load_dataset  # noqa: E402
from harness.runner import run_benchmark  # noqa: E402
from retrievers.hybrid import HybridRetriever  # noqa: E402

CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    if args.data_dir:
        dataset = load_dataset(
            corpus_path=args.data_dir / "corpus.jsonl",
            queries_path=args.data_dir / "queries.jsonl",
            qrels_path=args.data_dir / "qrels.jsonl",
        )
    else:
        dataset = load_dataset()
    print(f"corpus {len(dataset.corpus)} docs, {len(dataset.queries)} queries", flush=True)

    retriever = HybridRetriever()  # default model_name is bge-large-en-v1.5
    started = time.perf_counter()
    result = run_benchmark(retriever, dataset, retrieve_k=args.k, retriever_name="hybrid-bge")
    payload = result.to_dict()
    payload["embedding_backend"] = retriever.embedding_backend
    payload["wall_seconds"] = round(time.perf_counter() - started, 1)
    payload["cyrillic_query_ids"] = [q.query_id for q in dataset.queries if CYRILLIC.search(q.text)]
    payload["data_dir"] = str(args.data_dir) if args.data_dir else "benchmarks/data"

    print(json.dumps(payload["overall"], indent=2), flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
