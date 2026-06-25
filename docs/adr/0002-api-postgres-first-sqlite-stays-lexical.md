# API/Postgres deployment gets semantics; SQLite-only stays lexical

The semantic + concept-graph layer targets the **API/Postgres** deployment only: embeddings
in pgvector on the (CNPG) Postgres, the Concept graph as node/edge tables in Postgres, and
embedding/extraction via reused cluster infra (llama-cpp on GPU, or a hosted API). The
**SQLite-only** mode keeps working but stays **lexical (FTS) only** — it gains no embeddings
or graph, degrading gracefully.

This is surprising because the README markets zero-config offline SQLite as the headline
feature. We accept that trade-off: the operator actually runs the remote API/Postgres store,
reuse-before-building favours cluster infra, and bundling a local embedding model into the
zero-config path would add heavy dependencies and double the build/test matrix for little
real-world benefit.

## Consequences

- All benchmark numbers are produced in API/Postgres mode.
- Offline zero-config users see no behaviour change.
- A future ADR may revisit offline semantics (e.g. via `sqlite-vec` + a small local model)
  if there is demand.
