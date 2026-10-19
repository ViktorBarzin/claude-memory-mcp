# Claude Memory MCP

## Stack
- **Backend**: Python 3.12, FastAPI, SQLModel
- **Database**: SQLite (local) + PostgreSQL (remote sync)
- **Transport**: MCP over NDJSON (stdio)
- **Package manager**: uv
- **Dense recall**: `Qwen/Qwen3-Embedding-0.6B` as an fp16 ONNX graph (onnx-community's
  export, pinned by revision) baked into the image, served by `onnxruntime-gpu` on the
  cluster's Tesla T4 with an in-process CPU fallback. 1024-d, so it reuses the existing
  `halfvec(1024)` column and HNSW index. fp16 since 2026-10-11; its vectors match the
  earlier fp32 graph at cosine >= 0.99997, so rows embedded before then stay valid.
- **The CUDA arena is shrunk after every run.** Recall embeds the whole prompt and attention
  memory grows with the square of its length, so without the shrink one long prompt left
  the pod at its high-water mark until restart.

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
fixes keep failing their tests.

**Simplest fix, and the one that matches CI: sync the extras.** `uv sync --extra api --extra
dev --extra vault` in the worktree makes `uv run` resolve to the worktree's own source, so
every `uv run` command below behaves. Skipping it produced 25 failures on 2026-09-16 that
read as real defects and wasted an agent's work; the tell is `ModuleNotFoundError: No module
named 'fastapi'` from a direct import. Verify with
`uv run python -c "import claude_memory.api.auth as a; print(a.__file__)"` — the path must
contain `.worktrees/`. Avoid `--all-extras` on the devvm: it pulls torch and the OOM killer
took the run twice. CI has the headroom and uses `--all-extras`, so let it own that matrix.

The PYTHONPATH route works too, from the worktree root:

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
- **An API key carries a SCOPE, and the external one is fenced by an allowlist** (2026-09-16,
  `api/scopes.py`). `API_KEYS` parses two shapes: the flat `{"wizard": "key"}`, which means
  scope `admin`, and `{"muse": {"key": "…", "scope": "external"}}`. An external key may call
  only the ten `(METHOD, path)` pairs in `EXTERNAL_ALLOWED_OPERATIONS`, enforced by an
  app-wide `Depends(enforce_key_scope)` on the `FastAPI()` constructor. It is a dependency and
  NOT middleware because `request.scope["route"]` is unpopulated until routing has matched —
  `MCPAuthMiddleware` string-matches the raw path for that same reason. Being app-wide is what
  makes it fail closed: a route added later is absent from the set, so an external key gets
  403 on it with nobody remembering to close it. `/mcp/*` is closed separately in
  `MCPAuthMiddleware`, because the MCP tools reach `memory_share` and the REST writes and
  would otherwise make the allowlist bypassable. One caveat is measured and harmless: a new
  single-segment literal under `/api/memories/` is shadowed by the allowlisted
  `/api/memories/{memory_id}` template and 422s for everyone.
- **`EXTERNAL_ALLOWED_OPERATIONS` is read by THREE things and must stay one constant** — the
  auth dependency, the curated `/muse/openapi.json` document, and the tests that pin both. A
  second hand-kept list is the failure this design exists to prevent. The curated document is
  `app.openapi()` deep-copied and filtered; `app.openapi()` caches and returns the SAME dict by
  reference, so mutating it without a copy corrupts `/openapi.json` for every caller, and the
  schema filter must be a transitive `$ref` closure or `HTTPValidationError` ships pointing at
  a dropped `ValidationError`.
- **Writes from an external key are stamped `source:<user_id>` server-side and the client
  cannot forge one.** `scopes.stamp_origin` strips every origin claim from client-supplied tags
  before appending the real one, comparing after NFKC normalisation, case folding, homoglyph
  folding and whitespace removal — `source\twizard`, a non-breaking space and a Cyrillic `ѕ` all
  survived a plain string test. `stamp_stored_origin` is the separate path for a tags value the
  server read back, which must NOT be stripped or an importance-only update would delete
  somebody else's provenance. There is deliberately NO importance ceiling.
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
- **Cache provider FAILURES as well as sessions.** Building a session reads the whole ~1.2 GB
  graph, so retrying a dead provider per call costs a full model load per embed.
- **`MEMORY_EMBEDDINGS_ENABLED=0` is the rollback for anything dense.** `api/recall.py`
  documents it as a true no-op to the lexical path, correct whatever the embedding column
  holds — which is what makes an in-place re-embed safe to attempt.
- **The recall TAIL is cold Postgres pages, not query cost.** Measured 2026-10-19 with
  `memory_recall_leg_seconds{leg}` (one series per stage: acquire, and, or, embed, dense,
  links). Over 60 replayed real prompts the dense leg took 0.3-6.1 s and the AND-match
  0.4-3.9 s while the same SQL ran in 3-30 ms from a fresh connection, and both dropped to
  single-digit ms once their index pages were resident. pg-cluster's 2 GB shared_buffers is
  shared with tenants several GB larger, so the HNSW index (38 MB), the GIN index (15 MB)
  and the TOAST table (53 MB) get evicted between recalls, and k8s-node3's disk served each
  missing page in 27-170 ms. These changes address it, all result-preserving:
  `database.keep_recall_relations_warm` re-reads the recall relations (memories, its TOAST,
  both indexes, memory_links) with `pg_prewarm` every `MEMORY_PREWARM_INTERVAL_S` (default 60, `0` disables; ~12 ms per run when nothing
  was evicted, and `memory_prewarm_seconds` shows when it had to read); migration 007 turns
  off the GIN pending list, which every AND-match scanned in full (~250 pages); and the
  dense leg runs on its own pooled connection concurrently with the lexical leg. The pool
  also keeps idle connections open (`max_inactive_connection_lifetime=0`): with recalls
  minutes apart, asyncpg's 300 s default handed recalls fresh backends whose first plan
  re-read evicted catalog pages (0.8-2 s measured on a fresh session vs 3 ms warm). The same
  eviction problem drove infra#86 (shared_buffers 1 GB to 2 GB, autoprewarm) on 2026-09-02;
  autoprewarm only restores what was resident at shutdown.
- **Warm, the OR-broaden is the costliest leg.** Profiled on the live
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
  commit that cannot have touched a source-independent layer yet re-ships >200 MB.
  **A COLD GitHub Actions cache trips this guard and looks identical to a layer-order
  regression** (hit 2026-09-16): GHA evicts cache entries unused for 7 days, the previous build
  was 14 days earlier, and the rebuild re-shipped 3,190 MB of 3,236 MB for an eight-file source
  change. The guard's message only anticipates the ~1,076 MB model layer in that case, so two
  large NEW layers read as a Dockerfile problem. Diagnose with
  `git diff --name-only <prev> <head> | grep -v '^src/'` before touching the Dockerfile; empty
  output means the cache, not the layer order. The image is pushed BEFORE the guard runs, so a
  failure here skips only the deploy step, and a re-run with a warm cache passes — though it
  passes trivially, because the predecessor is then the image the failed run already pushed,
  so confirm no build input changed rather than treating the green re-run as the answer.
  The paths that legitimately DO change a source-independent layer are listed in that
  script's `PREFIX_INPUTS` — keep it in step with the Dockerfile. Note `pip install ".[api]"` cannot be used above `COPY src/`
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

# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd prime` for full workflow context.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work atomically
bd close <id>         # Complete work
bd dolt push          # Push beads data to remote
```

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Run `bd prime` for detailed command reference and session close protocol
<!-- END BEADS INTEGRATION -->
