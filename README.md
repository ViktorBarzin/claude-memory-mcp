# Claude Memory MCP

Give Claude persistent memory that survives across sessions, context compactions, and machines.

```bash
claude plugins install github:ViktorBarzin/claude-memory-mcp
```

That's it. Claude now remembers things.

## What It Does

```
You:    "remember I prefer Svelte for all new frontend apps"
Claude: Stored. ✓

  ── 3 weeks later, new session, different machine ──

You:    "build me a dashboard for this API"
Claude: I'll set up a SvelteKit project since that's your preference...
```

**No configuration needed** for single-machine use. Memories are stored in a local SQLite database with full-text search. For multi-machine sync, point it at an API server (details below).

## Features

### Slash Commands
- **`/remember <fact>`** — store a fact, preference, decision, or person detail
- **`/recall <query>`** — search memories by topic

### Automatic Behaviors (via hooks)
- **Auto-recall** — before responding, Claude checks stored memories for relevant context (preferences, past corrections, decisions). Completely invisible to the user.
- **Auto-learn** — after each response, a background process analyzes the conversation for corrections, preferences, and decisions worth remembering. Uses haiku-as-judge for conservative extraction.
- **Compaction survival** — when Claude's context window compacts, key memories are saved to a marker file and re-injected on the next prompt. No knowledge is lost.
- **Auto-approve** — memory tool calls are approved automatically, no permission prompts.

### MCP Tools
Claude has direct access to these tools during conversation:

| Tool | Description |
|------|-------------|
| `memory_store` | Store a fact with category, tags, importance, and semantic search keywords |
| `memory_recall` | Search memories using full-text search with expanded query terms |
| `memory_list` | List recent memories, optionally filtered by category |
| `memory_delete` | Delete a memory by ID |
| `secret_get` | Retrieve the decrypted content of a sensitive memory |

### Memory Categories
Memories are organized into: `facts`, `preferences`, `projects`, `people`, `decisions`

### Sensitive Memory Support
Mark memories as sensitive with `force_sensitive: true`. When Vault is configured, sensitive content is encrypted at rest and only decryptable via `secret_get`.

## Architecture

```
Claude Code Session
┌──────────────────────────────────────────────────────┐
│  Hooks                        MCP Server             │
│  ┌──────────────────┐        ┌────────────────────┐  │
│  │ auto-recall      │───────▶│ memory_store       │  │
│  │ auto-learn       │        │ memory_recall      │  │
│  │ compaction       │        │ memory_list        │  │
│  │ auto-approve     │        │ memory_delete      │  │
│  └──────────────────┘        │ secret_get         │  │
│                              └─────────┬──────────┘  │
│                                        │             │
│              ┌─────────────────────────┼──────────┐  │
│              │ Local SQLite            │          │  │
│              │ (cache + FTS5)    SyncEngine       │  │
│              │ ◄──── always ────(background, 60s) │  │
│              │       used       push queued writes │  │
│              │                  pull server changes│  │
│              │  pending_ops ──────────┘            │  │
│              │  (offline write queue)              │  │
│              └────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
                                │
              optional, for multi-machine sync
                                │
                     ┌──────────▼───────────┐
                     │  API Server (FastAPI) │
                     │  Docker / Kubernetes  │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │    PostgreSQL         │
                     │  (authoritative)      │
                     └──────────────────────┘
```

## Operating Modes

| Mode | When | What happens |
|------|------|--------------|
| **SQLite-only** | No env vars set | Everything local. Zero config. Works offline. |
| **Hybrid** | `MEMORY_API_KEY` set | Local SQLite for reads + background sync to API. Writes queue if API is down. |
| **HTTP-only** | `MEMORY_API_KEY` + `MEMORY_SYNC_DISABLE=1` | Direct API calls, no local cache. Legacy mode. |
| **Full** | Any mode + Vault configured | Above + Vault for encrypting sensitive memories at rest. |

## Setup

### Option 1: Plugin Install (recommended)

```bash
claude plugins install github:ViktorBarzin/claude-memory-mcp
```

Works immediately with SQLite-only mode. To enable multi-machine sync:

```bash
export MEMORY_API_URL="https://your-server.example.com"
export MEMORY_API_KEY="your-api-key"
```

### Option 2: Manual MCP Config

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "claude_memory": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "claude_memory.mcp_server"],
      "env": {
        "MEMORY_API_URL": "https://your-server.example.com",
        "MEMORY_API_KEY": "your-api-key"
      }
    }
  }
}
```

Omit the `env` block for SQLite-only mode. Requires `pip install claude-memory-mcp`.

### Verify

```
You: /remember "I prefer dark mode in all applications"
You: /recall "UI preferences"
```

## Running the API Server

Only needed for multi-machine sync. Skip this if you're using SQLite-only mode.

### Docker Compose

```bash
cd docker
docker compose up -d
```

Starts the API server + PostgreSQL. Optionally add Vault:

```bash
docker compose --profile vault up -d
```

### Manual

```bash
pip install claude-memory-mcp[api]
export DATABASE_URL="postgresql://user:pass@localhost:5432/claude_memory"
export API_KEY="your-secret-key"
alembic upgrade head
uvicorn claude_memory.api.app:app --host 0.0.0.0 --port 8000
```

### Kubernetes

Designed for high availability: 2 replicas with pod anti-affinity, PodDisruptionBudget, and startup probes.

```bash
helm install claude-memory deploy/helm/claude-memory \
  --set env.DATABASE_URL="postgresql://user:pass@host:5432/db" \
  --set env.API_KEY="your-key" \
  --set ingress.host="claude-memory.yourdomain.com"
```

Or raw manifests:

```bash
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl create secret generic claude-memory-secrets \
  -n claude-memory \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=api-key="your-key"
kubectl apply -f deploy/kubernetes/
```

## Multi-User Setup

For team deployments, use `API_KEYS` with a JSON mapping:

```bash
export API_KEYS='{"alice": "key-alice-xxx", "bob": "key-bob-yyy"}'
```

Each user gets isolated memory storage. Users cannot see each other's memories.

**Adding a user:**
1. Generate a key: `openssl rand -base64 32`
2. Add to `API_KEYS` JSON
3. Restart the API server
4. Share the key with the user

## Plugin Hooks

| Hook | Event | What it does |
|------|-------|-------------|
| `pre-compact-backup.sh` | PreCompact | Saves top 20 memories to a marker file before context compaction |
| `post-compact-recovery.sh` | UserPromptSubmit | Re-injects saved memories after compaction (one-time, then deletes marker) |
| `user-prompt-recall.py` | UserPromptSubmit | Tells Claude to check `memory_recall` before responding to each message |
| `auto-learn.py` | Stop (async) | Runs haiku-as-judge on the last exchange to extract durable facts worth storing |
| `auto-allow-memory-tools.py` | PermissionRequest | Auto-approves `memory_store`, `memory_recall`, `memory_list`, `memory_delete`, `secret_get` |

### Debug

```bash
# See what hooks are doing
export DEBUG_CLAUDE_MEMORY_HOOKS=1

# Disable auto-approve to see permission prompts
export DISABLE_CLAUDE_MEMORY_AUTO_APPROVE=1
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/memories` | POST | Store a memory |
| `/api/memories` | GET | List memories (`?category=facts&limit=20`) |
| `/api/memories/recall` | POST | Search memories by context and expanded query |
| `/api/memories/{id}` | DELETE | Soft-delete a memory |
| `/api/memories/sync` | GET | Incremental sync (`?since=ISO-timestamp`) |
| `/api/memories/{id}/secret` | POST | Get decrypted sensitive memory content |
| `/api/memories/migrate-secrets` | POST | Re-encrypt existing secrets with current Vault config |
| `/api/memories/import` | POST | Bulk import memories (JSON array) |

All endpoints except `/health` require `Authorization: Bearer <api-key>`.

## Environment Variables

### MCP Server (client-side)

| Variable | Description | Default |
|----------|-------------|---------|
| `MEMORY_API_URL` | API server URL | `http://localhost:8080` |
| `MEMORY_API_KEY` | API key (enables hybrid mode) | None |
| `MEMORY_HOME` | Local storage directory | `~/.claude/claude-memory` |
| `MEMORY_DB` | SQLite database path override | `$MEMORY_HOME/memory/memory.db` |
| `MEMORY_SYNC_INTERVAL` | Background sync interval in seconds | `60` |
| `MEMORY_SYNC_DISABLE` | Set to `1` for HTTP-only mode | None |

Aliases `CLAUDE_MEMORY_API_URL` and `CLAUDE_MEMORY_API_KEY` are also supported.

### API Server

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `API_KEY` | Single-user API key | None |
| `API_KEYS` | Multi-user JSON map `{"user": "key"}` | None |
| `VAULT_ADDR` | Vault server address | None |
| `VAULT_TOKEN` | Vault authentication token | None |
| `MEMORY_ENCRYPTION_KEY` | AES-256 key for non-Vault encryption | None |

## Database Migrations

Migrations run automatically on API server startup. To run manually:

```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/claude_memory"
alembic upgrade head
```

## Development

```bash
git clone https://github.com/ViktorBarzin/claude-memory-mcp.git
cd claude-memory-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"
pytest tests/ -v
ruff check src/ tests/
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
