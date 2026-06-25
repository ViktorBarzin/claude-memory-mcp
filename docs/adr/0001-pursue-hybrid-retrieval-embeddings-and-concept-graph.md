# Pursue hybrid retrieval: embeddings + concept graph over pure lexical

Today recall is **lexical only** (BM25 in SQLite, `tsvector`/`ts_rank` in Postgres over
content + LLM-generated `expanded_keywords`). It matches *tokens*, so it misses
paraphrase/synonym queries and cannot traverse between related Memories. We will pursue a
**hybrid** read path that adds dense-vector **Semantic recall** and a traversable **Concept
graph** (typed Relationships) alongside the existing Lexical recall.

This decision is **gated on a benchmark**: we adopt hybrid only if it shows a material
recall-quality uplift over the current lexical system on a stratified eval set (exact /
paraphrase / multi-hop). If the benchmark shows no improvement, a later ADR supersedes this
and we stay lexical.

## Considered options

- **Pure semantic (embeddings only)** — fixes paraphrase gaps but gives no real concept
  traversal; rejected as the *sole* mechanism.
- **Pure concept graph** — enables traversal but node-matching stays lexical, so paraphrase
  gaps remain; rejected as the *sole* mechanism.
- **Hybrid (chosen)** — embeddings for meaning + graph for traversal + existing FTS, fused
  into one ranked result. Highest ceiling; the GraphRAG / Zep-Graphiti / HippoRAG family.
