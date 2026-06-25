"""CANONICALIZATION → typed concept graph (EDC define+canonicalize / KGGEN
clustering) — slice S3.

This is the SECOND stage of "the graph done right". S2 (``graph_extract``) turned
each memory into open ``(subject, relation, object)`` triples; this stage
CANONICALISES those free-text surface forms into a TYPED concept graph with
canonical nodes — the thing the prior keyword-co-occurrence prototype never built.

Why this exists (the run's reason to exist)
============================================
The earlier prototype linked memories by SHARED RAW KEYWORDS (``hybrid.py``'s
``_concepts_for`` / ``_build_graph``). On the 5452-memory corpus that produced a
**2,095,624-edge** all-token-clique graph (benchmark-report §): non-discriminative,
untyped, and so dense that the fusion never let a graph candidate compete. This
stage instead does what EDC ("Extract-Define-Canonicalize") and KGGEN prescribe:

1. **Define + canonicalize ENTITIES.** Every distinct subject/object surface form
   is embedded ONCE with the SAME encoder the dense leg uses (bge-large, injected
   so CI stays model-free), then clustered by **cosine-NN + a threshold** via
   greedy single-linkage. Alias surface forms (``Svelte`` ≈ ``SvelteKit``) collapse
   into ONE canonical :class:`Concept` node that keeps every surface form as an
   ``alias``; unrelated forms stay separate.

2. **Canonicalize RELATIONS into a BOUNDED typed vocabulary.** The open relation
   phrases (``prefers`` / ``likes`` / ``favors`` / …) map through
   :data:`_RELATION_SYNONYMS` to a SMALL fixed set :data:`CANONICAL_RELATIONS`;
   anything unrecognised falls back to the catch-all ``mentions``. So the edge
   *label* space can never blow up the way the keyword space did.

3. **Build the typed graph.** Each triple becomes a directed
   :class:`ConceptEdge` ``(canonical-src) -[relation]-> (canonical-dst)``.
   Edges with the same ``(src, dst, relation)`` MERGE — accumulating ``weight`` and
   the set of ``evidence_memory_ids`` they were derived from. Self-loops (subject
   and object canonicalising to the same concept) are dropped. A
   :class:`MemoryConcept` mention-link records, per memory, every concept it
   mentions and the typed relation it played, with the originating memory id as
   evidence.

The result (:class:`ConceptGraph`) mirrors the production schema the FINAL DESIGN
stages into Postgres — ``concepts`` / ``concept_edges`` / ``memory_concepts`` — but
lives in plain in-memory dataclasses here so the offline harness (and S4's PPR leg)
can consume it with no database. The build reports its **edge count** via
:meth:`ConceptGraph.stats` for the latency-sanity check: it must be FAR below the
prior 2,095,624 edges.

Model-free by construction
==========================
The clustering needs an embedder, but CI must never load bge-large. So
:func:`build_concept_graph` takes an INJECTABLE :class:`EmbedFn` (mirroring S2's
``BatchExtractFn``); tests pass a deterministic stub, and the offline run passes a
thin wrapper over the cached bge-large encoder. Cosine is computed in pure Python
so the module imports with NO required third-party dependency (ADR-0002).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:  # pragma: no cover - exercised by both import paths
    from retrievers.graph_extract import Triple
except ModuleNotFoundError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from retrievers.graph_extract import Triple

# A concept id is a small integer assigned in first-seen order.
ConceptId = int

# Default cosine threshold for the alias-collapse single-linkage clustering. bge
# embeddings of true aliases / near-synonyms typically sit well above 0.8; 0.82 is
# a conservative default that merges obvious variants without over-merging distinct
# topics. The offline run sweeps/tunes this against the real corpus; tests pass an
# explicit threshold so the default never silently changes their behaviour.
_DEFAULT_THRESHOLD = 0.82


@runtime_checkable
class EmbedFn(Protocol):
    """An injectable batch embedder.

    Given a list of surface-form strings, return a list of equal length where each
    element is that form's embedding vector (any fixed dimensionality, ideally
    L2-normalised — the builder normalises defensively regardless). The real
    implementation wraps the cached bge-large encoder; tests pass a deterministic
    stub so CI never loads a model.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]:
        ...


# ---------------------------------------------------------------------------
# bounded typed relation vocabulary (EDC "canonicalize relations" / KGGEN)
# ---------------------------------------------------------------------------
#
# A SMALL fixed set of typed relations. Open relation phrases from extraction map
# into this set via _RELATION_SYNONYMS; anything unknown becomes the catch-all
# ``mentions``. Keeping this set tiny is what stops the typed-edge label space from
# proliferating the way the raw-keyword space did.
CANONICAL_RELATIONS: frozenset[str] = frozenset(
    {
        "prefers",
        "is-a",
        "used-in",
        "part-of",
        "depends-on",
        "runs-on",
        "located-in",
        "has",
        "uses",
        "related-to",
        "mentions",  # catch-all; MUST stay in the set (tests + fallback rely on it)
    }
)

# Surface relation phrase (lowercased, hyphen/space-normalised) → canonical label.
# Only the synonyms that differ from their canonical target need listing; a phrase
# that already equals a canonical label is mapped through identity below.
_RELATION_SYNONYMS: dict[str, str] = {
    # preference cluster
    "prefers": "prefers",
    "prefer": "prefers",
    "likes": "prefers",
    "like": "prefers",
    "favors": "prefers",
    "favours": "prefers",
    "favorite": "prefers",
    "favourite": "prefers",
    "loves": "prefers",
    # is-a / type cluster
    "is-a": "is-a",
    "is": "is-a",
    "are": "is-a",
    "type-of": "is-a",
    "kind-of": "is-a",
    "instance-of": "is-a",
    "a": "is-a",
    "an": "is-a",
    # used-in cluster
    "used-in": "used-in",
    "used-for": "used-in",
    "used-by": "used-in",
    "for": "used-in",
    # part-of cluster
    "part-of": "part-of",
    "belongs-to": "part-of",
    "member-of": "part-of",
    "component-of": "part-of",
    # depends-on cluster
    "depends-on": "depends-on",
    "requires": "depends-on",
    "needs": "depends-on",
    "relies-on": "depends-on",
    # runs-on / hosting cluster
    "runs-on": "runs-on",
    "hosted-on": "runs-on",
    "deployed-on": "runs-on",
    "runs": "runs-on",
    # located-in cluster
    "located-in": "located-in",
    "location": "located-in",
    "lives-in": "located-in",
    "based-in": "located-in",
    "in": "located-in",
    "at": "located-in",
    # has / possession cluster
    "has": "has",
    "have": "has",
    "owns": "has",
    "contains": "has",
    "includes": "has",
    # generic "uses" cluster
    "uses": "uses",
    "use": "uses",
    "using": "uses",
    "utilises": "uses",
    "utilizes": "uses",
    # related-to (weak association)
    "related-to": "related-to",
    "associated-with": "related-to",
    "linked-to": "related-to",
    "connected-to": "related-to",
}


def _normalise_relation_phrase(relation: str) -> str:
    """Lowercase, trim, and collapse internal whitespace/underscores to single
    hyphens so ``"used in"`` / ``"used_in"`` / ``"USED-IN"`` all key the same."""
    t = relation.strip().lower()
    # unify separators: spaces and underscores → hyphen, collapse repeats
    out: list[str] = []
    prev_sep = False
    for ch in t:
        if ch in " _-":
            if not prev_sep:
                out.append("-")
            prev_sep = True
        else:
            out.append(ch)
            prev_sep = False
    return "".join(out).strip("-")


def canonicalize_relation(relation: str) -> str:
    """Map an open relation phrase to a label in :data:`CANONICAL_RELATIONS`.

    Idempotent on canonical labels; unknown phrases fall back to ``mentions`` so the
    typed-relation vocabulary is BOUNDED no matter what the extractor emitted.
    """
    norm = _normalise_relation_phrase(relation)
    if norm in _RELATION_SYNONYMS:
        return _RELATION_SYNONYMS[norm]
    if norm in CANONICAL_RELATIONS:
        return norm
    return "mentions"


# ---------------------------------------------------------------------------
# schema dataclasses (mirror the staged Postgres tables)
# ---------------------------------------------------------------------------


@dataclass
class Concept:
    """A canonical concept node = one cluster of alias surface forms.

    ``canonical_name`` is the cluster's representative surface form (first seen);
    ``aliases`` is every surface form that collapsed into it (canonical included).
    ``embedding`` is the representative's vector (the cluster centroid would also
    work; the representative is cheaper and sufficient for NN at query time).
    """

    id: ConceptId
    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    category: str = "concept"


@dataclass
class ConceptEdge:
    """A typed, directed, evidence-bearing edge between two canonical concepts.

    Edges with the same ``(src_id, dst_id, relation)`` are MERGED: ``weight`` counts
    the supporting triples and ``evidence_memory_ids`` is the de-duplicated set of
    memories that asserted the relation.
    """

    src_id: ConceptId
    dst_id: ConceptId
    relation: str
    weight: int = 0
    evidence_memory_ids: list[int] = field(default_factory=list)


@dataclass
class MemoryConcept:
    """A mention-link: memory ``memory_id`` mentioned concept ``concept_id`` while
    playing typed relation ``relation`` (the memory id is its own evidence)."""

    memory_id: int
    concept_id: ConceptId
    relation: str


@dataclass
class ConceptGraph:
    """The built typed concept graph — the offline mirror of the staged Postgres
    ``concepts`` / ``concept_edges`` / ``memory_concepts`` tables, plus the indices
    S4's PPR leg seeds from."""

    concepts: dict[ConceptId, Concept] = field(default_factory=dict)
    concept_edges: list[ConceptEdge] = field(default_factory=list)
    memory_concepts: list[MemoryConcept] = field(default_factory=list)
    # surface form (lowercased) → canonical concept id
    _surface_to_concept: dict[str, ConceptId] = field(default_factory=dict)
    # concept id → sorted list of memory ids that mention it
    _concept_to_memories: dict[ConceptId, list[int]] = field(default_factory=dict)

    def concept_of(self, surface_form: str) -> ConceptId | None:
        """Canonical concept id for a surface form (case-insensitive), or ``None``
        if the form never appeared in any triple."""
        return self._surface_to_concept.get(surface_form.strip().lower())

    def memories_for_concept(self, concept_id: ConceptId) -> list[int]:
        """Memory ids that mention a concept (the bipartite adjacency PPR seeds
        from). Empty list for an unknown concept."""
        return self._concept_to_memories.get(concept_id, [])

    def stats(self) -> dict[str, int]:
        """Build statistics for the latency-sanity log. ``edges`` is the headline:
        it MUST be far below the prior keyword-co-occurrence prototype's 2,095,624.
        """
        return {
            "concepts": len(self.concepts),
            "edges": len(self.concept_edges),
            "memory_concepts": len(self.memory_concepts),
            "surface_forms": len(self._surface_to_concept),
        }


# ---------------------------------------------------------------------------
# entity canonicalization (cosine-NN + threshold, greedy single-linkage)
# ---------------------------------------------------------------------------


def _l2_normalise(vec: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec))
    if n == 0.0:
        return [0.0 for _ in vec]
    return [x / n for x in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two vectors. Inputs are L2-normalised by the caller, so
    this is a plain dot product; guarded against length mismatch defensively."""
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class _EntityCanonicalizer:
    """Greedy single-linkage clustering of surface forms by cosine-NN + threshold.

    Surface forms are added one at a time (in stable first-seen order). Each new
    form is compared against the representative vector of every existing cluster; if
    its best cosine is ≥ ``threshold`` it joins that cluster (recording the surface
    form as an alias), otherwise it founds a new cluster. First-seen order makes the
    representative — and thus ``canonical_name`` — deterministic.

    This is the EDC "define + canonicalize" / KGGEN entity-clustering step.
    """

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold
        self._concepts: dict[ConceptId, Concept] = {}
        self._surface_to_concept: dict[str, ConceptId] = {}
        self._next_id = 0

    def assign(self, surface_form: str, embedding: list[float]) -> ConceptId:
        """Return the canonical concept id for ``surface_form``, creating or growing
        a cluster as needed. Idempotent per surface form (a repeated form returns
        its already-assigned id without re-clustering)."""
        key = surface_form.strip().lower()
        existing = self._surface_to_concept.get(key)
        if existing is not None:
            return existing

        vec = _l2_normalise(embedding)
        best_id: ConceptId | None = None
        best_sim = self.threshold  # must strictly meet/exceed the threshold to merge
        for cid, concept in self._concepts.items():
            sim = _cosine(vec, concept.embedding)
            if sim >= best_sim:
                best_sim = sim
                best_id = cid

        if best_id is None:
            cid = self._next_id
            self._next_id += 1
            self._concepts[cid] = Concept(
                id=cid,
                canonical_name=surface_form,
                aliases=[surface_form],
                embedding=vec,
            )
            self._surface_to_concept[key] = cid
            return cid

        # join the nearest cluster; record the new surface form as an alias.
        concept = self._concepts[best_id]
        if surface_form not in concept.aliases:
            concept.aliases.append(surface_form)
        self._surface_to_concept[key] = best_id
        return best_id

    def result(self) -> tuple[dict[ConceptId, Concept], dict[str, ConceptId]]:
        return self._concepts, self._surface_to_concept


# ---------------------------------------------------------------------------
# the build
# ---------------------------------------------------------------------------


def build_concept_graph(
    triples_by_memory: Mapping[int, list[Triple]],
    embed_fn: EmbedFn,
    *,
    threshold: float = _DEFAULT_THRESHOLD,
) -> ConceptGraph:
    """Canonicalise S2's open triples into a TYPED concept graph.

    Args:
        triples_by_memory: ``{memory_id: [(subject, relation, object), ...]}`` from
            :mod:`retrievers.graph_extract`.
        embed_fn: an injectable batch embedder (the cached bge-large encoder offline,
            a deterministic stub in tests). Each DISTINCT surface form is embedded
            exactly once.
        threshold: cosine-NN merge threshold for alias collapse.

    Returns:
        A :class:`ConceptGraph` with canonical ``concepts``, typed merged
        ``concept_edges`` carrying ``evidence_memory_ids``, ``memory_concepts``
        mention-links, and the concept↔memory index PPR (S4) seeds from.
    """
    graph = ConceptGraph()
    if not triples_by_memory:
        return graph

    # 1) collect every DISTINCT surface form (subject ∪ object) across all triples,
    #    preserving first-seen order for deterministic canonical representatives.
    seen: set[str] = set()
    ordered_forms: list[str] = []
    for triples in triples_by_memory.values():
        for subj, _rel, obj in triples:
            for form in (subj, obj):
                key = form.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    ordered_forms.append(form)

    if not ordered_forms:
        return graph

    # 2) embed each distinct form ONCE (de-duplicated above), then canonicalise by
    #    cosine-NN + threshold. The embedder is called as a single batch so the
    #    cached encoder amortises model load over the whole vocabulary.
    vectors = embed_fn(ordered_forms)
    if len(vectors) != len(ordered_forms):
        raise ValueError(
            f"embedder returned {len(vectors)} vectors for {len(ordered_forms)} surface forms"
        )
    canon = _EntityCanonicalizer(threshold)
    for form, vec in zip(ordered_forms, vectors):
        canon.assign(form, list(vec))
    concepts, surface_to_concept = canon.result()
    graph.concepts = concepts
    graph._surface_to_concept = surface_to_concept

    # 3) walk the triples again, canonicalising relations + projecting endpoints to
    #    their concept ids; merge edges by (src, dst, relation); accumulate evidence;
    #    record mention-links and the concept→memory index.
    edge_index: dict[tuple[ConceptId, ConceptId, str], ConceptEdge] = {}
    edge_evidence: dict[tuple[ConceptId, ConceptId, str], set[int]] = {}
    # de-dupe mention-links per (memory, concept, relation) so a repeated surface
    # form in one memory doesn't emit duplicate rows.
    seen_mentions: set[tuple[int, ConceptId, str]] = set()
    concept_to_mems: dict[ConceptId, set[int]] = {}

    for memory_id, triples in triples_by_memory.items():
        for subj, rel, obj in triples:
            subj_key = subj.strip().lower()
            obj_key = obj.strip().lower()
            if not subj_key or not obj_key:
                continue
            src = surface_to_concept[subj_key]
            dst = surface_to_concept[obj_key]
            relation = canonicalize_relation(rel)

            # mention-links (subject plays the relation as actor; object as target).
            for cid in (src, dst):
                mk = (memory_id, cid, relation)
                if mk not in seen_mentions:
                    seen_mentions.add(mk)
                    graph.memory_concepts.append(
                        MemoryConcept(memory_id=memory_id, concept_id=cid, relation=relation)
                    )
                concept_to_mems.setdefault(cid, set()).add(memory_id)

            # NO self-loops: subject and object that canonicalise to the same node
            # carry no traversal value and would inflate the edge count.
            if src == dst:
                continue

            ek = (src, dst, relation)
            edge = edge_index.get(ek)
            if edge is None:
                edge = ConceptEdge(src_id=src, dst_id=dst, relation=relation, weight=0)
                edge_index[ek] = edge
                edge_evidence[ek] = set()
                graph.concept_edges.append(edge)
            edge.weight += 1
            edge_evidence[ek].add(memory_id)

    # finalise evidence id lists (sorted, de-duplicated) and the concept→memory index.
    for ek, edge in edge_index.items():
        edge.evidence_memory_ids = sorted(edge_evidence[ek])
    graph._concept_to_memories = {
        cid: sorted(mems) for cid, mems in concept_to_mems.items()
    }

    return graph
