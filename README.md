# Claude Memory MCP

A persistent memory layer for Claude Code that stores knowledge across sessions. Operates as an MCP (Model Context Protocol) server with optional PostgreSQL API backend, local SQLite cache with background sync, and Vault integration for secrets.

## Architecture

```
MCP Server (per Claude session)
┌─────────────────────────────────────────────┐
│  Tool call (store/recall/list/delete)       │
│         │                                   │
│         ▼                                   │
│  ┌─────────────┐    ┌──────────────────┐    │
│  │ Local SQLite │◄──│  SyncEngine      │    │
│  │ (cache+FTS5) │   │  (background     │    │
│  │ always used  │   │   thread, 60s)   │    │
│  └─────────────┘   └────────┬─────────┘    │
│                              │              │
│  ┌──────────────┐            │              │
│  │ pending_ops  │────────────┘              │
│  │ (offline     │   push queued writes      │
│  │  write queue)│   pull server changes     │
│  └──────────────┘                           │
└──────────────────────────────────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │  API Server (k8s)    │
                    │  2 replicas + PDB    │
                    │  pod anti-affinity   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    PostgreSQL        │
                    │  (authoritative)     │
                    └──────────────────────┘
```

In **hybrid mode** (the default when an API key is set), all reads hit local SQLite for instant results. Writes go to SQLite first, then attempt a synchronous API push — if the API is down, writes queue in a `pending_ops` table and sync on the next background cycle. The server is authoritative (server-wins conflict resolution).

## Operating Modes

| Mode | Condition | Storage | Use Case |
|------|-----------|---------|----------|
| **SQLite-only** | No `MEMORY_API_KEY` | SQLite + FTS5 | Single user, local Claude Code |
| **Hybrid** (default) | `MEMORY_API_KEY` set | Local SQLite cache + API sync | Multi-session with offline resilience |
| **HTTP-only** (legacy) | `MEMORY_API_KEY` + `MEMORY_SYNC_DISABLE=1` | PostgreSQL via HTTP API | Direct API, no local cache |
| **Full** | Any mode + Vault configured | Above + Vault for secrets | Multi-user, team deployment |

## Setting Up a New Agent

### 1. Install the package

```bash
pip install claude-memory-mcp
```

Or install from source for development:

```bash
git clone https://github.com/ViktorBarzin/claude-memory-mcp.git
cd claude-memory-mcp
pip install -e .
```

### 2. Choose your mode and configure

#### Local Mode (SQLite, zero config)

No server needed. Memories are stored in a local SQLite database.

Add to your Claude Code MCP settings (`~/.claude/settings.json` or project `.claude/settings.json`):

```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "claude_memory.mcp_server"]
    }
  }
}
```

The database defaults to `~/.claude/claude-memory/memory/memory.db`. Override with:

```json
{
  "env": {
    "MEMORY_HOME": "/path/to/memory/dir"
  }
}
```

#### Hybrid Mode (recommended for shared deployments)

Point the MCP server at a running API instance. Memories are cached locally in SQLite for fast reads and offline resilience, and synced to the API in the background.

```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "claude_memory.mcp_server"],
      "env": {
        "MEMORY_API_URL": "https://claude-memory.example.com",
        "MEMORY_API_KEY": "your-api-key"
      }
    }
  }
}
```

To disable local caching and use direct HTTP (legacy behavior), add `"MEMORY_SYNC_DISABLE": "1"` to `env`.

#### Full Mode (with Vault for secrets)

Same as Server Mode but with Vault for automatic credential detection and secure storage:

```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "python3",
      "args": ["-m", "claude_memory.mcp_server"],
      "env": {
        "MEMORY_API_URL": "https://claude-memory.example.com",
        "MEMORY_API_KEY": "your-api-key",
        "VAULT_ADDR": "https://vault.example.com",
        "VAULT_TOKEN": "your-vault-token"
      }
    }
  }
}
```

### 3. Verify it works

Start a Claude Code session and test:

```
> Store a test memory: "Claude Memory MCP is working"
> Recall memories about "Claude Memory"
```

You should see the MCP tools `memory_store`, `memory_recall`, `memory_list`, `memory_delete`, and `secret_get` available.

### Environment Variable Aliases

For backward compatibility, these aliases are supported:

| Primary | Alias |
|---------|-------|
| `MEMORY_API_URL` | `CLAUDE_MEMORY_API_URL` |
| `MEMORY_API_KEY` | `CLAUDE_MEMORY_API_KEY` |

## Running the API Server

### Docker Compose (recommended)

```bash
cd docker
docker compose up -d
```

This starts the API + PostgreSQL. Vault is available as an optional profile:

```bash
docker compose --profile vault up -d
```

### Manual

```bash
pip install claude-memory-mcp[api]

export DATABASE_URL="postgresql://user:pass@localhost:5432/claude_memory"
export API_KEY="your-secret-key"

# Run migrations
alembic upgrade head

# Start server
uvicorn claude_memory.api.app:app --host 0.0.0.0 --port 8000
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `API_KEY` | Single-user API key | None |
| `API_KEYS` | Multi-user JSON map `{"user": "key"}` | None |
| `VAULT_ADDR` | Vault server address | None |
| `VAULT_TOKEN` | Vault authentication token | None |
| `MEMORY_ENCRYPTION_KEY` | AES-256 key (hex or passphrase) for non-Vault encryption | None |
| `MEMORY_SYNC_INTERVAL` | Seconds between background sync cycles (hybrid mode) | `60` |
| `MEMORY_SYNC_DISABLE` | Set to `1` to disable local cache and sync (HTTP-only mode) | None |

## Multi-User Setup

For team deployments, use `API_KEYS` with a JSON mapping:

```bash
export API_KEYS='{"alice": "key-alice-xxx", "bob": "key-bob-yyy"}'
```

Each user gets isolated memory storage — users cannot see each other's memories. Each agent/user gets their own API key in their MCP config.

### Adding a new user

1. Generate a key: `openssl rand -base64 32`
2. Add to `API_KEYS`: `{"existing": "...", "newuser": "generated-key"}`
3. Restart the API server
4. Give the new user their MCP config with their key

## MCP Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a fact with category, tags, importance, and expanded keywords |
| `memory_recall` | Search memories using full-text search with expanded query terms |
| `memory_list` | List recent memories, optionally filtered by category |
| `memory_delete` | Delete a memory by ID |
| `secret_get` | Retrieve the actual content of a sensitive/redacted memory |

## API Reference

### Health Check
```
GET /health
```

### Store Memory
```
POST /api/memories
Authorization: Bearer <api-key>

{"content": "...", "category": "facts", "tags": "tag1,tag2", "expanded_keywords": "related terms", "importance": 0.8}
```

### Recall Memories
```
POST /api/memories/recall
Authorization: Bearer <api-key>

{"context": "search terms", "expanded_query": "additional search terms", "category": "facts", "limit": 10}
```

### List Memories
```
GET /api/memories?category=facts&limit=20
Authorization: Bearer <api-key>
```

### Delete Memory (soft delete)
```
DELETE /api/memories/{id}
Authorization: Bearer <api-key>
```

Sets `deleted_at` on the record rather than removing it, so sync clients can detect deletions.

### Sync Memories
```
GET /api/memories/sync?since=2026-03-14T10:00:00+00:00
Authorization: Bearer <api-key>
```

Returns all memories changed since the given timestamp (including soft-deleted ones). Without `since`, returns a full dump of non-deleted memories. Used by the SyncEngine for incremental sync.

### Get Secret Content
```
POST /api/memories/{id}/secret
Authorization: Bearer <api-key>
```

### Migrate Existing Secrets
```
POST /api/memories/migrate-secrets
Authorization: Bearer <api-key>
```

### Bulk Import
```
POST /api/memories/import
Authorization: Bearer <api-key>

[{"content": "...", "category": "facts"}, ...]
```

## Database Migrations

This project uses Alembic for database migrations. Migrations run automatically on API server startup.

To run manually:
```bash
export DATABASE_URL="postgresql://user:pass@localhost:5432/claude_memory"
alembic upgrade head
```

To create a new migration:
```bash
alembic revision -m "description of change"
```

## Kubernetes Deployment

The API server is designed for high availability:
- **2 replicas** with pod anti-affinity (spread across nodes)
- **PodDisruptionBudget** (`min_available: 1`) prevents both pods going down during voluntary disruptions
- **Startup probe** gives 60s for initial DB connection before the pod is killed

### Helm

```bash
helm install claude-memory deploy/helm/claude-memory \
  --set env.DATABASE_URL="postgresql://user:pass@host:5432/db" \
  --set env.API_KEY="your-key" \
  --set ingress.host="claude-memory.yourdomain.com"
```

### Raw Manifests

```bash
kubectl apply -f deploy/kubernetes/namespace.yaml
kubectl create secret generic claude-memory-secrets \
  -n claude-memory \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=api-key="your-key"
kubectl apply -f deploy/kubernetes/
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
