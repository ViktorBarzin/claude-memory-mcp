"""Benchmark harness for claude-memory recall evaluation.

Public API:
    from harness import Retriever, load_dataset, run_benchmark, BenchmarkResult
    from harness import metrics

A retriever is any object (or callable) implementing:
    retrieve(query: str, k: int) -> list[memory_id]   # ranked, best first

memory_id matches the `id` field in corpus.jsonl / qrels.jsonl (int).
"""
from .types import Retriever, Query, Memory, Qrels
from .dataset import load_dataset, Dataset
from .runner import run_benchmark, BenchmarkResult, StratumResult
from . import metrics

__all__ = [
    "Retriever",
    "Query",
    "Memory",
    "Qrels",
    "load_dataset",
    "Dataset",
    "run_benchmark",
    "BenchmarkResult",
    "StratumResult",
    "metrics",
]
