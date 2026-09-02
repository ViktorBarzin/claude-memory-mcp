# Claude Memory MCP

## Stack
- **Backend**: Python 3.12, FastAPI, SQLModel
- **Database**: SQLite (local) + PostgreSQL (remote sync)
- **Transport**: MCP over NDJSON (stdio)
- **Package manager**: uv
- **Dense recall**: `Qwen/Qwen3-Embedding-0.6B` as a quantised-free fp32 ONNX graph baked
  into the image, served by `onnxruntime-gpu` on the cluster's Tesla T4 with an in-process
  CPU fallback. 1024-d, so it reuses the existing `halfvec(1024)` column and HNSW index.

## Quick Start
```bash
uv sync
uv run python -m mcp.server  # Start MCP server
uv run pytest                 # Run tests -- from the MAIN checkout only, see below
```

**In a git worktree, `uv run pytest` tests the wrong source tree.** The worktree's `.venv`
carries no pytest (`asyncpg` and `prometheus-client` sit in the optional `api` extra that
`uv sync` skips), so uv falls through to system python3.12, which has a user-level editable
install pointing at the MAIN checkout's `src`. Your edits appear to do nothing and correct
fixes keep failing their tests. Run this instead, from the worktree root:

```bash
PYTHONPATH=$PWD/src python3 -m pytest tests/ -q   # PYTHONPATH is searched before .pth
PYTHONPATH=$PWD/src python3 -m mypy src/claude_memory/
python3 -c "import importlib.util as u; print(u.find_spec('claude_memory.api.recall').origin)"
```

The last line is the check: the path it prints must contain `.worktrees/`.

## Architecture
- `src/` — MCP server implementation
- `mcp/` — MCP protocol handlers
- `migrations/` — Alembic database migrations
- `hooks/` — Claude Code hook scripts
- `skills/` — Claude Code skills
- `openclaw-plugin/` — OpenClaw integration
- `benchmarks/` — offline retrieval harness. `data/`, `results/` and any
  `scripts/build_*_eval_set.py` are **gitignored on purpose**: they embed real memory text.
  Both preserved eval sets live in `~/.claude/claude-memory/benchmark-artifacts/`.

## Key Patterns
- **Non-blocking startup**: MCP server startup must not block on sync/HTTP calls (15s timeout)
- **Suppress stderr**: Any stderr during startup causes Claude Code to reject the server
- **NDJSON transport**: One JSON object per line, NOT Content-Length framing
- **Wrapper script**: Use `~/.local/bin/claude-memory-mcp-wrapper` to source secrets then exec
- **An embedding-backend change is gated on cosine-vs-reference, in the BUILD.** Dimension,
  unit norm, correct query/document conventions and plausible latency ALL pass on garbage
  vectors — int8 quantisation of this model scored 0.16-0.37 against the
  sentence-transformers reference and looked healthy on every cheap check.
  `docker/export_onnx.py export` embeds fixed probes (English and Bulgarian) through
  sentence-transformers and writes them to `$GATE_REFERENCE`; `docker/export_onnx.py gate`
  re-embeds them through the production embedder against the bytes in `$OUT_DIR` and fails
  the build under 0.99. Two commands, in two stages, so a `src/` commit re-runs the 572 s
  gate and not the 1,686 s export. Neither has a default mode: a caller that means one and
  silently gets the other would produce an image with an ungated graph in it.
- **onnxruntime does NOT raise when a requested execution provider is unavailable.** It logs
  to stderr, silently serves on CPU, and hands back a working session. Believe
  `session.get_providers()`, never construction success — otherwise a CPU-only pod reports
  itself GPU-served and the fallback metric stays at zero forever.
- **The nvidia pip wheels are not on the loader path.** They install under
  `site-packages/nvidia/<lib>/lib/`, so `ort.preload_dlls()` is required before building a
  session or the CUDA provider cannot `dlopen` and quietly downgrades to CPU.
- **Cache provider FAILURES as well as sessions.** Building a session reads the whole ~2.4 GB
  graph, so retrying a dead provider per call costs a full model load per embed.
- **`MEMORY_EMBEDDINGS_ENABLED=0` is the rollback for anything dense.** `api/recall.py`
  documents it as a true no-op to the lexical path, correct whatever the embedding column
  holds — which is what makes an in-place re-embed safe to attempt.
- **Recall latency lives in the LEXICAL leg, not the vector search.** Profiled on the live
  store 2026-09-01: dense pgvector HNSW over ~10,900 vectors is 3 ms, the AND-match is
  2 ms, and the OR-broaden fallback is 200-440 ms. The broaden fires whenever the AND-match
  returns fewer than `limit` rows, which for a long query is always, and the per-turn hook
  sends the whole user prompt (real prompts: p50 25 terms, p90 86, max 124). ORing that
  many terms matches 68-95% of the store, so Postgres correctly abandons the GIN index and
  sequentially scans every row computing `ts_rank`. Before reaching for `hnsw.ef_search` or
  anything dense, read the plan for the OR query.
- **Anything touching the OR-broaden must be checked for row-identity, not just speed.**
  Two changes were safe because they are provably result-preserving: the `OR_BROADEN_MIN_RANK`
  floor is applied in Python after `ORDER BY rank DESC ... LIMIT` (a threshold on the sort
  key, so filter-then-limit and limit-then-filter agree), and duplicate tokens are dropped
  (`a | a` is the same tsquery as `a`). Capping the term COUNT is not in that class — it
  changes which rows come back, and `benchmarks/data` tops out at 19 terms so the offline
  eval cannot measure the long-prompt regime a cap would affect.
- **Capping the OR-broaden's term count was measured and DECLINED on 2026-09-01. Don't
  re-propose it without new relevance evidence.** Measured over 40 real prompts through the
  full fused path, with query vectors embedded once so every arm shared identical dense
  input. Against the uncapped baseline: a 48-term cap moved the fused top-5 on 4 of 40
  prompts for 12 ms of p90, a 32-term cap on 6 of 40 for 97 ms, a 20-term cap on **14 of
  40** for 102 ms. Warm p90 uncapped is 203 ms while the live p90 is ~1,050 ms and the alert
  fires at 2,500 ms, so nearly all of the live tail is cold-start rather than this query and
  a cap cannot reach it. Beware measuring the lexical leg alone: in isolation a 20-term cap
  looks like 287 ms → 66 ms and only 10-of-25 row identity, both of which overstate the case
  because the dense leg absorbs much of the change and pays none of the cost.

## CI/CD
- **Build**: GitHub Actions → `ghcr.io/viktorbarzin/claude-memory-mcp` (ADR-0002, off-infra;
  NOT DockerHub). Three stages: `exporter` traces the ONNX graph with torch/optimum and
  depends only on `docker/export_onnx.py`; `gate` adds `src/` and scores the graph through
  the production embedder; the runtime stage carries onnxruntime and a tokenizer instead of
  a training framework. The runtime copies the model `--from=exporter` and the gate's marker
  `--from=gate` — that marker is the ONLY thing keeping the gate in the dependency graph,
  because BuildKit prunes stages nothing references. Delete the `COPY --from=gate` line and
  the fidelity gate silently stops running.
- **LAYER ORDER IN `docker/Dockerfile` IS LOAD-BEARING, and CI measures it.** Every
  source-independent layer sits ABOVE `COPY src/`, marked by an explicit comment line.
  BuildKit re-executes every layer above a cache miss, and a re-executed `COPY` re-tars its
  content with a fresh mtime even when the bytes are identical — so a `COPY` moved three
  lines up puts the 2,094.6 MB CUDA install or the 1,076.0 MB model layer back below `src/`
  and every commit re-ships it. Measured on the two production images either side of
  `62271e36`: **3,170.7 MB of 3,216.9 MB (98.6%) re-shipped for a source-only commit** before
  the reorder. `scripts/ci-layer-delta.py` reports the delta on every build and FAILS a
  commit that cannot have touched a source-independent layer yet re-ships >200 MB; the paths
  that legitimately DO change one are listed in that script's `PREFIX_INPUTS` — keep it in
  step with the Dockerfile. Note `pip install ".[api]"` cannot be used above `COPY src/`
  (hatchling builds the wheel from `src/`), which is why the dependency list is extracted
  from `pyproject.toml` with `tomllib` and the package itself is installed `--no-deps` below
  the line. Plan: `infra/docs/plans/2026-09-02-node1-large-image-handling.md` Phase 1.
- **Deploy**: Woodpecker CI (kubectl set image), repo ID 78. `rollout status --timeout=900s`,
  raised from the generated 300s: the pod is pinned to k8s-node1 by its `nvidia.com/gpu`
  request with strategy `Recreate`, and a cold pull of this image measured **6m24.561s** — so
  300s reported a timeout on rollouts that then succeeded.
- **Image tags**: 8-char git SHA
- **`onnxruntime-gpu` is pinned to `~=1.26.0`** — from 1.27 the GPU wheel targets CUDA 13,
  and k8s-node1 runs driver 570 / CUDA 12.8. `nvidia-cublas-cu12` is listed explicitly
  because ORT links cuBLAS without declaring it.
- **`mcp` is pinned `<2`** (2.x renamed `FastMCP`) and the ruff **rule set** is pinned in
  `pyproject.toml` rather than the ruff version, so a linter upgrade cannot turn CI red on
  unchanged code. Both were latent breakages found on 2026-09-01.
- **The gate is `ruff check` + `mypy src/claude_memory/` + `pytest tests/`, and NOT
  `ruff format`.** Master carries 13 files `ruff format` would rewrite, so running it turns
  an unrelated diff into a large one. Leave existing formatting alone.
