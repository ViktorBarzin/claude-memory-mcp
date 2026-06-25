# Claude Memory MCP

Persistent cross-session memory for Claude. Today it stores **Memories** as rows and
retrieves them by **lexical recall** (full-text keyword matching). This context is being
extended with **semantic recall** (embeddings) and a **concept graph** so retrieval works
by meaning and related memories become traversable.

## Language

**Memory**:
A single stored unit of knowledge — a fact, preference, decision, project note, or person
detail — with content plus metadata (category, tags, importance). The atomic thing a user
stores and recalls.

**Recall**:
Retrieving the Memories most relevant to a query. The read path.

**Lexical recall**:
The existing retrieval method — matches Memories whose words (content, tags, LLM-generated
keywords) overlap the query, ranked by BM25 / `ts_rank`. Matches *tokens*, not meaning.
_Avoid_: calling this "semantic search" — it is not semantic.

**Semantic recall**:
Retrieval by meaning via dense-vector **Embedding** similarity, so a query surfaces a Memory
even with zero shared words (e.g. "what UI library?" → "prefers Svelte").

**Embedding**:
A dense vector representation of a Memory's (or Concept's) meaning, used for Semantic recall.

**Concept**:
A distinct entity or idea that recurs across Memories (e.g. "Svelte", "Viktor", "TripIt",
"frontend framework"). A node in the Concept graph. Distinct from a Memory: one Memory can
mention several Concepts, and one Concept spans many Memories.

**Concept graph**:
The network of Concepts joined by typed **Relationships**, making the memory store
traversable — from one Memory or Concept to related ones.

**Relationship**:
A typed, directed edge in the Concept graph, between two Concepts or between a Memory and a
Concept (e.g. `prefers`, `is-a`, `used-in`, `mentions`).

**Hybrid retrieval**:
The target read path — combining Lexical recall, Semantic recall, and Concept-graph
traversal into one ranked result set.
