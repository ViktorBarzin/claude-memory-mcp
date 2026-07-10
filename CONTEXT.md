# Claude Memory MCP

Persistent cross-session memory for Claude. Today it stores **Memories** as rows and
retrieves them by **lexical recall** (full-text keyword matching). This context is being
extended with **semantic recall** (embeddings), explicit **Links** between Memories that
Recall follows, and a **concept graph** so retrieval works by meaning and related memories
become traversable.

## Language

**Memory**:
A single stored unit of knowledge — a fact, preference, decision, project note, or person
detail — with content plus metadata (category, tags, importance). The atomic thing a user
stores and recalls: there is no larger stored unit (no "document"), and no smaller one.
A Memory is **size-bounded** — small enough for Recall to deliver it whole — and must be
**Self-contained**. Knowledge too large for one Memory is split by the writer into several
self-contained Memories.

**Self-contained**:
The writing rule for every Memory: understandable on its own, with no reliance on
neighbouring Memories or unstated session context. Splitting long knowledge is a writing
act — each piece stands alone — never a mechanical chop.
_Avoid_: "part N of M" fragments; entries that begin mid-sentence.

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
_Distinct from_ a **Link**, which joins two Memories directly.

**Link**:
A typed, directed Memory→Memory edge, written explicitly when a Memory is stored or
updated. The closed set of types, each with its own Recall behaviour:
- `supersedes` (new→old): the successor is served *in place of* the old Memory whenever
  the old one would rank — stale vocabulary still finds current truth. Formalises
  tombstoning.
- `resolved-by` (symptom→current-truth): when the symptom-phrased Memory ranks, its
  target is attached to the result — the answer arrives without a second lookup.
- `part-of` (detail→hub): rendered as a one-line pointer in both directions; content
  fetched on demand.
- `see-also` (related): rendered as a one-line pointer; no other behaviour.
_Avoid_: open-vocabulary link types (the category-drift lesson); using `see-also` where a
more specific type applies.

**Hybrid retrieval**:
The target read path — combining Lexical recall, Semantic recall, and Concept-graph
traversal into one ranked result set.
