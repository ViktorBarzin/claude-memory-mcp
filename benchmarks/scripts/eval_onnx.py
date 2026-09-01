"""Score the ONNX backend on the preserved eval set, using whatever provider is live.

Runs INSIDE the API pod, because that is where the GPU and the baked ONNX graph are.
Embedding 5,452 documents with Qwen3-Embedding-0.6B takes hours on a CPU and minutes on
the T4, so this is not a job for the workstation.

    kubectl -n claude-memory cp benchmarks <pod>:/tmp/benchmarks
    kubectl -n claude-memory exec <pod> -- python /tmp/benchmarks/scripts/eval_onnx.py \
        --out /tmp/benchmarks/results/hybrid-onnx.json

It subclasses the same HybridRetriever the stored baselines came from, swapping only the
dense leg's embedder. Everything else — the FTS leg, the RRF fusion, the weights, the
metrics — is the code that produced results/hybrid.json, so the comparison is like for
like rather than two different pipelines scored against one number.

Why this and not the live API: memory ids were reassigned after the eval set was built,
so the preserved qrels no longer join to the live store (see live_eval.py). They are
still self-consistent with the preserved corpus.jsonl, which is what this scores.
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

from claude_memory.embeddings import ONNX_MODEL, OnnxEmbedder, short_provider  # noqa: E402

CYRILLIC = re.compile(r"[Ѐ-ӿ]")


class OnnxHybridRetriever(HybridRetriever):
    """HybridRetriever with the dense leg served by onnxruntime instead of torch."""

    def __init__(self) -> None:
        super().__init__(model_name=ONNX_MODEL)
        self._onnx = OnnxEmbedder()
        self._provider: str | None = None

    def _select_embedding_backend(self) -> str:
        self.model_name = ONNX_MODEL
        return f"onnx:{ONNX_MODEL}"

    def _embed_local(self, texts: list[str]):
        import numpy as np

        started = time.perf_counter()
        rows = []
        for i, text in enumerate(texts, 1):
            rows.append(self._onnx.embed_document(text, is_sensitive=False))
            if i % 500 == 0:
                rate = i / (time.perf_counter() - started)
                print(f"    embedded {i}/{len(texts)} docs ({rate:.1f}/s)", flush=True)
        self._provider = short_provider(self._onnx._primary())
        return np.asarray(rows, dtype=np.float32)

    def _embed_query_local(self, query: str):
        import numpy as np

        return np.asarray(self._onnx.embed_query(query), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="alternate eval set (corpus/queries/qrels .jsonl); default is benchmarks/data",
    )
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

    retriever = OnnxHybridRetriever()
    started = time.perf_counter()
    result = run_benchmark(retriever, dataset, retrieve_k=args.k, retriever_name="hybrid-onnx")
    elapsed = time.perf_counter() - started

    payload = result.to_dict()
    payload["embedding_backend"] = retriever.embedding_backend
    payload["execution_provider"] = retriever._provider
    payload["wall_seconds"] = round(elapsed, 1)

    # The reason for the model change: queries whose text carries Cyrillic. Reported
    # separately so the multilingual claim is measured rather than assumed.
    cyr = [q.query_id for q in dataset.queries if CYRILLIC.search(q.text)]
    payload["cyrillic_query_ids"] = cyr
    payload["data_dir"] = str(args.data_dir) if args.data_dir else "benchmarks/data"
    if not cyr:
        payload["cyrillic_note"] = (
            "The preserved eval set contains no Cyrillic queries (its corpus snapshot has 28 "
            "Cyrillic docs of 5,452). This run therefore does NOT measure the multilingual "
            "claim; that needs an eval set rebuilt against current ids."
        )

    print(json.dumps(payload, indent=2)[:3000], flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
