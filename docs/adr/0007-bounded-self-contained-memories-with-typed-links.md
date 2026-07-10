# Bounded self-contained Memories with typed Links

A month-long audit of real sessions (2026-07-09) showed the store's write side is healthy but its
*delivery* fails: 70% of entries are `part-N-of-M` mid-sentence fragments from a retired chopping
pipeline, a 50KB blob proves `content` is unbounded, 54% of hook injections exceeded the harness's
~10KB persist threshold (the model saw ~1 of 5 recalled memories), and the costliest verified
rediscovery happened while the answer sat one *unfollowed pointer* away (symptom-phrased #5972
outranked root-cause #6775 for 53 minutes; fixed the moment they were cross-linked). Separately,
the free-vocabulary `category` field drifted into singular/plural twins that hide 97% of gotchas
from exact-match filters.

## Decision

1. **Memory stays the only stored unit.** No Document/chunk entity. Knowledge too large for one
   Memory is split *by the writer* into several self-contained Memories — splitting is a writing
   act, never a mechanical chop.
2. **Hard content bound: 1,400 characters.** Derived from the delivery budget, not taste: the
   recall hook injects 5 results under a hard 8KB cap (below the ~10KB persist threshold);
   8KB/5 − ~150 chars metadata ≈ 1,400 chars arriving **whole**. The server rejects oversize
   store/update (422, with split-into-hub+parts guidance); the CLI pre-validates with the same
   message. Invariant: *a ranked Memory is always delivered complete.* Legacy oversize entries
   are clipped-with-pointer on read until the one-shot cleanup rewrites them.
3. **Links: typed, directed Memory→Memory edges, closed enum of four**, each with defined Recall
   behaviour (see CONTEXT.md "Link"):
   - `supersedes` — successor is served *in place of* the old entry (redirect); formalises
     tombstoning so stale vocabulary still finds current truth.
   - `resolved-by` — symptom→current-truth; target is auto-attached when the source ranks, so
     the answer arrives with zero extra calls.
   - `part-of` — detail→hub; one-line pointers both directions.
   - `see-also` — one-line pointer, no other behaviour.

## Considered options

- **First-class Document + auto-chunking** — writers never split, but it doubles every surface
  (store/get/update/embedding granularity) and mechanical chunking is exactly what produced the
  fragment pollution. Long-form documents already live in git (`infra/docs/`); memory stores
  pointers.
- **2,000-char cap + ~700-char read clip** — roomier writing, but every injected Memory arrives
  clipped, keeping the read-more step in the common path that sessions demonstrably skip.
- **Soft warning, no rejection** — zero friction; rejected because this store already proved
  that unenforced discipline drifts (blob, fragments, category twins).
- **Open link vocabulary** — flexible, but types without defined Recall semantics do nothing and
  the category-drift precedent says free vocabularies rot.
- **Pointer-only links** — simplest server; reproduces the verified unfollowed-pointer failure.
- **Auto-attach one hop for all types** — hub fan-out floods the 5-slot injection, re-creating
  the truncation problem this ADR exists to fix.

## Consequences

- New `memory_links` table (src, dst, type) with the closed enum enforced server-side; write
  clients gain link flags; a `get <id>` verb returns one full entry with its Links (both
  directions).
- All first-party writers (CLI, hooks, extraction prompts) must split long knowledge; the 422
  message teaches the pattern at the point of failure.
- The one-shot store cleanup can now encode reassembled fragment series as real `part-of`
  structures and historical supersessions as `supersedes` edges instead of freetext
  "[SUPERSEDED]" markers.
- Embeddings are computed per (bounded) Memory — one granularity; rewritten entries re-embed.
- The 245-session retrieval benchmark gains link-behaviour slices (redirect, auto-attach) as
  part of the landing gate; supersedes-redirect changes what "the right result" means for
  stale-vocabulary queries.
