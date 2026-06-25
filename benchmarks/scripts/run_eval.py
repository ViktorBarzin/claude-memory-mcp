#!/usr/bin/env python3
"""Run the benchmark for a named retriever and print overall + per-stratum metrics.

Usage:
    .venv/bin/python scripts/run_eval.py --retriever fts5      # lexical baseline
    .venv/bin/python scripts/run_eval.py --retriever substring # demo
    .venv/bin/python scripts/run_eval.py --retriever mypkg.mymod:MyRetriever
    .venv/bin/python scripts/run_eval.py --retriever fts5 --json results/fts5.json

The --retriever value is either a built-in alias or a "module:Class" path. The
class is instantiated with no args; the runner calls build_index() if present.

Outputs are LOCAL-ONLY when written under results/ (gitignored): a results file
may echo retrieved ids (not content), but keep it local to be safe.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load_dataset, run_benchmark  # noqa: E402
from harness.baselines import SqliteFtsRetriever  # noqa: E402
from harness.example_retriever import SubstringRetriever  # noqa: E402

ALIASES = {
    "fts5": lambda: SqliteFtsRetriever(sort_by="relevance"),
    "fts5_importance": lambda: SqliteFtsRetriever(sort_by="importance"),
    "substring": SubstringRetriever,
}


def resolve(spec: str):
    if spec in ALIASES:
        return ALIASES[spec]()
    if ":" not in spec:
        raise SystemExit(f"unknown retriever alias '{spec}' (use module:Class or one of {list(ALIASES)})")
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="fts5")
    ap.add_argument("--k", type=int, default=20, help="depth requested from retriever")
    ap.add_argument("--json", type=Path, default=None, help="write full result JSON here")
    args = ap.parse_args()

    ds = load_dataset(validate=True)
    retr = resolve(args.retriever)
    res = run_benchmark(retr, ds, retrieve_k=args.k)
    print(res.summary())

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res.to_dict(), indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
