# Post-cleanup retrieval gate

The one-shot store cleanup (ADR-0007) rewrites real memories — splitting oversize blobs,
reassembling `part-N-of-M` fragments, encoding historical supersessions as `supersedes`
links. **This gate proves retrieval quality did not regress**: it re-runs the benchmark
retrievers against a snapshot of the post-cleanup store, scored with the **preserved
119-query eval set**, and compares against the stored baseline numbers
(`docs/research/hybrid-build-report.md` §2, cut on the preserved 5,452-memory corpus,
fingerprint `ca7b1d4ed22672e8`).

The preserved eval set (`benchmarks/data` → the benchmark-artifacts symlink) is the
**fixed baseline reference — never modify or regenerate it**. Snapshots always go to a
new dated dir; `snapshot_corpus.py` refuses to write into the preserved dir.

Everything below runs from `benchmarks/` with its local venv (`.venv/bin/python`);
all snapshot/result artifacts are gitignored (real personal memories — never commit).

## 1. Snapshot the post-cleanup store

```bash
MEMORY_API_KEY=... .venv/bin/python scripts/snapshot_corpus.py
# → snapshots/<today>/corpus.jsonl + snapshot_meta.json
```

Pulls every live memory via `GET /api/memories/sync` and writes the harness corpus
format, **excluding sensitive entries exactly like the original export** (the
`is_sensitive=0` rule). `--name post-cleanup` overrides the date-based dir name;
`MEMORY_API_URL` overrides the production endpoint.

## 2. Know your id spaces: the content bridge

**The preserved eval set carries LOCAL-SQLite ids; the API store carries REMOTE ids.**
Verified live (2026-07-10): 0/5,452 preserved (id, content) pairs match the remote
store, but 5,340 contents match verbatim under a different id — including 137/139
distinct gold ids. The cleanup rewrites the API store, so its report is in remote ids
too. Scoring an API snapshot against the preserved qrels therefore **requires the
bridge**: `--bridge data/corpus.jsonl` translates gold ids into the snapshot's id
space by **exact content match** (exact-only on purpose — a fuzzy bridge could attach
gold judgments to the wrong memory). Content shared by near-duplicate twins expands
the gold set to all twins (the qrels twin precedent); an unbridged gold id is dropped
and reported, never passed through raw where it could collide with an unrelated
remote id.

Two preserved gold ids (2 of 139) no longer content-match the live store (edited or
deleted since the set was frozen on 2026-06-25) — expect exactly one query
(`para_012`) to drop under `--drop-missing-gold`, independent of the cleanup.

## 3. Get the id-map from the cleanup report

Gold ids that the cleanup **superseded must be mapped to their successors** — otherwise
the gate penalises retrievers for correctly serving current truth. `store_cleanup.py
--report` emits this old→new mapping (in **remote** ids — the store the cleanup
rewrote); `regression_run.py --id-map FILE` accepts the report directly. Accepted
shapes: a flat JSON object `{"old": new, ...}`, a list of `[old, new]` pairs (or
`{"old":…, "new":…}` objects), either optionally nested under an `id_map` /
`superseded_by` / `supersessions` key, or plain `old new` text lines. Supersession
chains (a→b→c) collapse to the final successor; a cycle is treated as a corrupt
report and errors. The bridge runs first, then the id-map —
gold: local id → *(bridge)* → remote id → *(id-map)* → successor.

The map is applied on **both sides** of the eval:

- **gold side** — qrels/queries ids are remapped old→successor;
- **retrieval side** — retrieved rankings pass through the same map, emulating the
  production **supersedes-redirect** (ADR-0007: the successor is *served in place of*
  the superseded entry whenever it would rank). Disable with `--no-redirect` to measure
  raw pre-redirect rankings.

## 4. Run the regression gate

```bash
.venv/bin/python scripts/regression_run.py \
    --snapshot <today> \
    --bridge data/corpus.jsonl \
    --id-map path/to/cleanup-report.json \
    --drop-missing-gold \
    --json results/regression-<today>.json
echo $?   # 0 = PASS, 1 = regression beyond threshold, 2 = usage/data error
```

- Runs `fts` (the faithful production lexical path) always, and `dense` (the +dense
  config, fixed local `bge-large-en-v1.5` — hosted keys are deliberately ignored) when
  the model is already in the local HF cache; force with `--retrievers fts,dense`.
  A changed corpus re-embeds once on CPU (~minutes) and caches under `cache/` keyed by
  the new corpus fingerprint.
- Prints the comparison table — recall@10 + nDCG@10, overall and per stratum
  (exact / paraphrase / multihop) — against the stored baseline, with deltas.
- **Gate: FAIL if overall recall@10 drops more than 0.02 vs baseline** for any
  evaluated retriever (`--threshold` to adjust; a drop of exactly the threshold passes).
  Per-stratum deltas are reported but do not gate.
- A gold id still missing from the snapshot after bridge+map is a **hard error** — it
  means the id-map is incomplete or the cleanup deleted a gold memory. Fix the map, or
  consciously accept the loss with `--drop-missing-gold` (dropped queries are listed;
  they shrink the eval, so justify each one). **Expected drop set: exactly
  `para_012`** (pre-existing drift, §2); anything more means the id-map is missing
  supersessions — stop and fix it.
- `--baseline previous-run.json` chains gates off a prior `--json` output instead of
  the build-report constants.

## 5. Expected outcome

- **`fts` should sit near its baseline** (overall recall@10 0.6952 ± threshold). Small
  *gains* are expected — deduplicated twins and reassembled fragments concentrate rank
  mass; the redirect scores tombstone hits as their successors.
- **`dense` should hold its baseline** (overall recall@10 0.8338) once re-embedded.
  Rewritten entries re-embed under the same fixed model, so paraphrase recall should
  stay in the +dense band (0.7250 baseline).
- **FAIL** → diff the per-stratum deltas in the table: a paraphrase-only dense drop
  points at content rewrites (check the split/rewrite of the affected gold memories);
  an across-the-board fts drop points at lost entries or an incomplete id-map (check
  `gold.missing_after_map` and `gold.dropped_queries` in the JSON). **Do not land the
  cleanup while the gate fails.**

**Pre-cleanup dry-run (run it — it validates the tooling and prices in drift):**
snapshot the *current* store and run the gate with `--bridge` but no `--id-map`.
Store additions since the preserved set was frozen (2026-06-25) shift numbers even
with a perfect cleanup; the dry-run gives you that drift as its own number, and
`--baseline results/<dry-run>.json` lets the post-cleanup run gate against the
pre-cleanup run directly, isolating the cleanup's effect. Executed 2026-07-10
against the live store (6,186 non-sensitive memories, +~730 vs the preserved set;
118/119 queries): fts overall recall@10 **0.7025 vs 0.6952 baseline (Δ+0.0073) —
PASS**, all strata within +0.01/−0.003. Drift is currently small and slightly
positive; the 0.02 threshold has ample headroom.
