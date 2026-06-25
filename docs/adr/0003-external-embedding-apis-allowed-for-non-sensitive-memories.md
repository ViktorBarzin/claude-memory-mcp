# External embedding/extraction APIs allowed for non-sensitive memories

Embedding and concept extraction may use **hosted APIs** (e.g. OpenAI `text-embedding-3`,
Voyage, Cohere) for **non-sensitive** memories, to access a higher quality ceiling than
self-hosted models alone. **Sensitive / Vault-encrypted (secret) memories are never sent
externally** and are excluded from the corpus that gets embedded or extracted.

This is a deliberate relaxation of the homelab's usual local-only posture, made because the
quality gain is worth it for non-secret personal memory content. The research/benchmark may
still compare hosted vs self-hostable models (nomic-embed, bge-m3, gte-Qwen2, e5) so the
production choice is data-driven; this ADR only records that egress is *permitted* within the
sensitive-data boundary.

## Consequences

- The corpus-export step MUST filter out `is_sensitive` / secret memories before any external
  call.
- Production deployment needs an embedding API key (or falls back to the in-cluster
  llama-cpp model when absent).
