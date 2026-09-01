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
uv run pytest                 # Run tests
```

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
  `docker/export_onnx.py` embeds fixed probes (English and Bulgarian) two ways and fails the
  build under 0.99, using the production embedder rather than a reimplementation of it.
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

## CI/CD
- **Build**: GitHub Actions → `ghcr.io/viktorbarzin/claude-memory-mcp` (ADR-0002, off-infra;
  NOT DockerHub). The image is a two-stage build: a throwaway `exporter` stage traces and
  gates the ONNX graph with torch/optimum, and the runtime stage carries onnxruntime and a
  tokenizer instead.
- **Deploy**: Woodpecker CI (kubectl set image), repo ID 78
- **Image tags**: 8-char git SHA
- **`onnxruntime-gpu` is pinned to `~=1.26.0`** — from 1.27 the GPU wheel targets CUDA 13,
  and k8s-node1 runs driver 570 / CUDA 12.8. `nvidia-cublas-cu12` is listed explicitly
  because ORT links cuBLAS without declaring it.
- **`mcp` is pinned `<2`** (2.x renamed `FastMCP`) and the ruff **rule set** is pinned in
  `pyproject.toml` rather than the ruff version, so a linter upgrade cannot turn CI red on
  unchanged code. Both were latent breakages found on 2026-09-01.
