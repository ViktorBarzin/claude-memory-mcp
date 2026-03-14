# Claude Memory MCP

A persistent memory layer for Claude Code that stores knowledge across sessions. Operates as an MCP (Model Context Protocol) server with optional API and database backends.

## Operating Modes

| Mode | Storage | Auth | Use Case |
|------|---------|------|----------|
| **Local** | SQLite file | None | Single user, local Claude Code |
| **Server** | SQLite file | API key | Remote access, single user |
| **Full** | PostgreSQL | API keys + Vault | Multi-user, team deployment |

## Quick Start

### Local Mode (MCP stdio)

Install and configure Claude Code to use the MCP server directly:

```bash
pip install claude-memory-mcp
```

Add to your Claude Code MCP config (`~/.claude/plugins/`):

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "claude_memory.mcp_server"],
      "env": {
        "MEMORY_DB_PATH": "~/.claude/memory.db"
      }
    }
  }
}
```

### Server Mode (API)

Run the API server with SQLite:

```bash
pip install claude-memory-mcp[api]

export DATABASE_URL="sqlite:///./memory.db"
export API_KEY="your-secret-key"

uvicorn claude_memory.api.app:app --host 0.0.0.0 --port 8000
```

Configure Claude Code to connect via HTTP:

```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "claude_memory.mcp_server"],
      "env": {
        "MEMORY_API_URL": "http://localhost:8000",
        "MEMORY_API_KEY": "your-secret-key"
      }
    }
  }
}
```

### Full Mode (Docker Compose)

```bash
cd docker
docker compose up -d
```

This starts the API server with PostgreSQL. See [Docker Compose](#docker-compose) for details.

## Docker Compose

The dev environment includes the API server and PostgreSQL:

```bash
cd docker
docker compose up -d
```

To include HashiCorp Vault for secret management:

```bash
docker compose --profile vault up -d
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | Database connection string | `sqlite:///./memory.db` |
| `API_KEY` | Single-user API key | None |
| `API_KEYS` | Multi-user JSON map `{"user": "key"}` | None |
| `VAULT_ADDR` | Vault server address | None |
| `VAULT_TOKEN` | Vault authentication token | None |

## Multi-User Setup

For team deployments, use the `API_KEYS` environment variable with a JSON mapping of usernames to API keys:

```bash
export API_KEYS='{"alice": "key-alice-xxx", "bob": "key-bob-yyy"}'
```

Each user gets isolated memory storage. The username is extracted from the API key on each request.

## Vault Integration

For production deployments, store API keys in HashiCorp Vault instead of environment variables:

```bash
export VAULT_ADDR="https://vault.example.com"
export VAULT_TOKEN="s.xxxxxxxxxxxx"
```

The server reads API keys from the Vault KV store at `secret/claude-memory/api-keys`.

## API Reference

### Health Check

```
GET /health
```

### Store Memory

```
POST /api/v1/memories
Authorization: Bearer <api-key>
Content-Type: application/json

{
  "content": "The user prefers dark mode",
  "tags": ["preferences", "ui"],
  "source": "conversation"
}
```

### Recall Memories

```
GET /api/v1/memories?q=dark+mode&limit=10
Authorization: Bearer <api-key>
```

### List Memories

```
GET /api/v1/memories?tags=preferences&limit=20
Authorization: Bearer <api-key>
```

### Delete Memory

```
DELETE /api/v1/memories/{id}
Authorization: Bearer <api-key>
```

## Kubernetes Deployment

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
# Create secret with your credentials first:
kubectl create secret generic claude-memory-secrets \
  -n claude-memory \
  --from-literal=database-url="postgresql://user:pass@host:5432/db" \
  --from-literal=api-key="your-key"
kubectl apply -f deploy/kubernetes/
```

## Development

### Setup

```bash
git clone https://github.com/viktorbarzin/claude-memory-mcp.git
cd claude-memory-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e ".[api,dev]"
```

### Running Tests

```bash
pytest tests/ -v
```

### Linting

```bash
ruff check src/ tests/
mypy src/claude_memory/
```

### Building

```bash
pip install build
python -m build
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
