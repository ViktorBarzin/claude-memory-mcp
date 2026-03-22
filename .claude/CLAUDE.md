# Claude Memory MCP

## Stack
- **Backend**: Python 3.12, FastAPI, SQLModel
- **Database**: SQLite (local) + PostgreSQL (remote sync)
- **Transport**: MCP over NDJSON (stdio)
- **Package manager**: uv

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

## Key Patterns
- **Non-blocking startup**: MCP server startup must not block on sync/HTTP calls (15s timeout)
- **Suppress stderr**: Any stderr during startup causes Claude Code to reject the server
- **NDJSON transport**: One JSON object per line, NOT Content-Length framing
- **Wrapper script**: Use `~/.local/bin/claude-memory-mcp-wrapper` to source secrets then exec

## CI/CD
- **Build**: GitHub Actions (Docker image push to DockerHub)
- **Deploy**: Woodpecker CI (kubectl set image), repo ID 78
- **Image tags**: 8-char git SHA
