# Phase the hybrid: lexical + dense first, concept graph gated

The [benchmark](../research/benchmark-report.md) shows the hybrid read path beats lexical FTS on
every overall metric (recall@10 **+0.139**, driven by the statistically-robust paraphrase win recall@10
**+0.350**), with no recall regression on exact — so [ADR-0001](0001-pursue-hybrid-retrieval-embeddings-and-concept-graph.md)'s
quality gate **is met by the lexical + dense fusion**. The ablation also showed full-hybrid **identical
to three decimals** to FTS+dense alone — **but a post-run adversarial review proved this was a structural
artifact, not an empirical result**: the prototype's fusion barred the graph leg from the fused top-k by
construction (graph candidates excluded from the FTS∪dense base set; `w_graph=0.35` → max graph RRF
`0.0057` < base-leg min `0.0091`). So the concept graph was **never validly tested — it is *unevaluated*,
not disproven.**

We therefore **phase** the hybrid:

- **Phase 1 (adopt now):** lexical FTS ⊕ dense pgvector embeddings, fused with weighted RRF, the
  importance prior preserved as a post-fusion multiplier. This *is* the measured uplift.
- **Phase 2 (gated, NOT shipped):** the typed-relation concept graph. It stays designed
  ([integration design §A.5](../research/integration-design.md)) but disabled — its value is **unproven**
  and it carries real operational cost (LLM extraction + two extra Postgres tables + traversal).

The graph's prototype was a **zero-LLM keyword-co-occurrence** graph (concepts = tags ∪
expanded_keywords ∪ regex noun-phrases, edges from co-occurrence), **not** the LLM-extracted
typed-relation graph the production design specifies. So the null result kills *that cheap
construction* on *this eval set* — it does not prove an LLM-extracted graph is also null. The graph
is **gated, not killed**: re-open it only with evidence it helps — e.g. a multi-hop slice whose hops
are *not* semantically adjacent (where the dense leg can't shortcut), built from real typed-relation
extraction.

## Why the graph result is inconclusive

The ablation `A≡B` was **guaranteed by the fusion config** (graph candidates excluded from the FTS∪dense
base set; `w_graph=0.35` caps a graph-only id's RRF below any base-leg id), so it tested nothing about the
graph — the review even found a relevant memory the graph surfaced that both base legs missed, which
fusion then discarded. Separately, the multi-hop deltas (recall@10 +0.064) are **not statistically
significant** (3 of 4 CIs cross zero), so there is no distinguishable multi-hop win to attribute to
*either* leg. The graph is deferred on **cost + uncertainty**, not on evidence it fails.

## Consequences

- Production ships embeddings + RRF fusion; the graph schema is documented but not migrated.
- The concept-graph research (Zep/Graphiti split, HippoRAG PPR, EDC canonicalization) is preserved as
  the phase-2 blueprint, behind the gate.
- Phasing avoids paying the graph's cost while its value is unproven; the robust **lexical+dense**
  paraphrase win is what ADR-0001's gate actually surfaced. The graph stays a designed, gated follow-up
  pending a valid retest (graph candidates in the fused pool / swept weight, real typed-relation extraction).
