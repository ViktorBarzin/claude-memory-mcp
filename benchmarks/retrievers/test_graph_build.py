"""Unit tests for CANONICALIZATION → typed concept graph (slice S3).

These tests are STRICTLY MODEL-FREE. The real builder embeds each distinct
subject/object surface form with the SAME cached bge-large encoder used by the
dense leg, then clusters by cosine-NN + threshold. CI must NEVER load that model,
so the embedder is an INJECTABLE callable — an ``EmbedFn`` — and every test passes
a deterministic STUB that maps a fixed vocabulary of surface forms to hand-placed
unit vectors. Aliases (``Svelte`` / ``SvelteKit``) are stubbed CLOSE together so
cosine-NN collapses them into ONE canonical node; unrelated forms (``Postgres``)
are stubbed ORTHOGONAL so they stay separate. This lets the test assert the
canonicalization arithmetic — not the embedding model's quality.

What slice S3 guarantees (acceptance), exercised below:
  * TYPED graph, not keyword co-occurrence. The builder consumes S2's open
    ``(subject, relation, object)`` triples and produces canonical concept NODES
    and TYPED directed ``concept_edges`` carrying a bounded relation label.
  * ALIAS COLLAPSE. Distinct surface forms whose embeddings are within the cosine
    threshold (``Svelte`` ≈ ``SvelteKit``) become ONE concept node; the node keeps
    both surface forms as ``aliases``.
  * RELATION COLLAPSE → BOUNDED VOCAB. Relation variants (``prefers`` / ``likes``
    / ``favors``) canonicalize to ONE typed relation drawn from a SMALL fixed
    vocabulary; ``concept_edges`` between the same canonical (src, dst, relation)
    MERGE into one edge.
  * EVIDENCE. Every edge and every ``memory_concepts`` mention carries the
    ``evidence_memory_ids`` it was derived from, so the graph is auditable and the
    downstream PPR leg (S4) can attribute a hit to its source memories.
  * EDGE COUNT LOGGED. The build reports its concept/edge counts as a stat; the
    edge count must be FAR below the prior keyword-co-occurrence prototype's
    2,095,624 edges (the latency-sanity check, ADR-0005 / FINAL DESIGN).

Run:  cd benchmarks && ../.venv/bin/python -m pytest retrievers/test_graph_build.py -q
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from retrievers.graph_build import (
    CANONICAL_RELATIONS,
    ConceptGraph,
    EmbedFn,
    build_concept_graph,
    canonicalize_relation,
)
from retrievers.graph_extract import Triple


# ---------------------------------------------------------------------------
# a deterministic, model-free embedder
# ---------------------------------------------------------------------------
#
# Each surface form is assigned a 3-d unit vector. Forms that are ALIASES share a
# near-identical direction (tiny perturbation → cosine ≈ 1, inside the threshold);
# unrelated forms are placed on ORTHOGONAL axes (cosine 0, outside the threshold).
# An unseen form falls back to a per-string deterministic axis so the stub never
# raises — but every form the tests assert on is placed explicitly.

def _unit(vec: Sequence[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / n for x in vec]


# Three orthogonal "topics": svelte-ish, postgres-ish, viktor-ish.
_VOCAB: dict[str, list[float]] = {
    # svelte cluster — nearly the same direction (aliases)
    "svelte": _unit([1.0, 0.0, 0.0]),
    "sveltekit": _unit([0.99, 0.01, 0.0]),
    "svelte kit": _unit([0.98, 0.02, 0.0]),
    # postgres cluster — a different axis, with one alias
    "postgres": _unit([0.0, 1.0, 0.0]),
    "postgresql": _unit([0.0, 0.99, 0.01]),
    # a person — third axis (no alias)
    "viktor": _unit([0.0, 0.0, 1.0]),
}


class CallCountingEmbedder:
    """Wraps the stub embedder and records how many surface forms it embedded, so
    a test can assert the builder de-duplicates surface forms before embedding
    (each DISTINCT form embedded at most once, mirroring the real cached encoder
    where re-embedding the same string is wasted work)."""

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def __call__(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            self.embedded.append(t)
            key = t.lower().strip()
            if key in _VOCAB:
                out.append(list(_VOCAB[key]))
            else:
                # deterministic per-string orthogonal-ish fallback (never asserted on)
                seed = sum(ord(c) for c in key) or 1
                out.append(_unit([seed % 7 + 1, seed % 5 + 1, seed % 3 + 1]))
        return out


def _stub_embedder() -> EmbedFn:
    return CallCountingEmbedder()


# A generous threshold: the alias perturbations above sit at cosine > 0.999, the
# cross-topic forms at cosine 0.0, so 0.9 cleanly separates them.
_THRESHOLD = 0.9


# ---------------------------------------------------------------------------
# alias collapse — the headline acceptance
# ---------------------------------------------------------------------------

def test_alias_surface_forms_collapse_to_one_concept_node() -> None:
    """`Svelte` and `SvelteKit` (stubbed near-identical) collapse to ONE canonical
    concept node carrying BOTH surface forms as aliases."""
    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte")],
        2: [("Viktor", "likes", "SvelteKit")],
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)

    # the two svelte surface forms map to the SAME concept id
    cid_svelte = g.concept_of("Svelte")
    cid_sveltekit = g.concept_of("SvelteKit")
    assert cid_svelte is not None and cid_sveltekit is not None
    assert cid_svelte == cid_sveltekit, "Svelte/SvelteKit must collapse to one node"

    node = g.concepts[cid_svelte]
    aliases_lower = {a.lower() for a in node.aliases}
    assert {"svelte", "sveltekit"} <= aliases_lower, "node keeps both surface forms"


def test_distinct_topics_stay_separate_nodes() -> None:
    """Forms on orthogonal stub axes (Svelte vs Postgres vs Viktor) must NOT merge
    — the threshold is discriminative, not a global collapse."""
    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte"), ("Viktor", "uses", "Postgres")],
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)
    ids = {g.concept_of(x) for x in ("Viktor", "Svelte", "Postgres")}
    assert None not in ids
    assert len(ids) == 3, "three orthogonal forms → three distinct concepts"


def test_postgres_alias_also_collapses() -> None:
    """A second, independent alias pair (`Postgres`/`PostgreSQL`) collapses on its
    own axis — alias collapse is general, not special-cased to one cluster."""
    triples: dict[int, list[Triple]] = {
        1: [("app", "depends-on", "Postgres")],
        2: [("service", "runs-on", "PostgreSQL")],
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)
    assert g.concept_of("Postgres") == g.concept_of("PostgreSQL")


# ---------------------------------------------------------------------------
# relation collapse → bounded typed vocabulary
# ---------------------------------------------------------------------------

def test_relation_variants_canonicalize_to_one_typed_relation() -> None:
    """`prefers` / `likes` / `favors` all canonicalize to the SAME typed relation
    drawn from the bounded vocabulary."""
    rels = {canonicalize_relation(r) for r in ("prefers", "likes", "favors")}
    assert len(rels) == 1, "preference synonyms collapse to one typed relation"
    (canon,) = rels
    assert canon in CANONICAL_RELATIONS, "the canonical relation is in the bounded vocab"
    assert canon == "prefers"


def test_relation_vocabulary_is_bounded() -> None:
    """The canonical relation set is SMALL and fixed (a typed vocabulary, not the
    open-ended surface relations). Any unknown relation maps to the catch-all
    `mentions`, so the edge label space can never blow up."""
    assert isinstance(CANONICAL_RELATIONS, frozenset)
    assert len(CANONICAL_RELATIONS) <= 16, "relation vocabulary stays small"
    assert "mentions" in CANONICAL_RELATIONS, "there is a catch-all relation"
    # an unseen relation phrase falls back to the catch-all, never a new label
    assert canonicalize_relation("blorps-the-frobnicator") == "mentions"
    # every canonical label is itself stable under canonicalization (idempotent)
    for r in CANONICAL_RELATIONS:
        assert canonicalize_relation(r) == r


def test_edges_between_same_canonical_pair_and_relation_merge() -> None:
    """Two memories asserting the SAME (canonical-src, relation, canonical-dst) —
    even via different surface forms / relation synonyms — produce ONE merged edge
    whose weight and evidence accumulate (the de-duplication that keeps the edge
    count bounded)."""
    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte")],
        2: [("Viktor", "favors", "SvelteKit")],  # same canon src, rel, dst as #1
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)
    src = g.concept_of("Viktor")
    dst = g.concept_of("Svelte")
    assert src is not None and dst is not None

    matching = [
        e for e in g.concept_edges
        if e.src_id == src and e.dst_id == dst and e.relation == "prefers"
    ]
    assert len(matching) == 1, "the two assertions merge into one typed edge"
    edge = matching[0]
    assert edge.weight == 2, "merged edge weight counts both supporting triples"
    assert set(edge.evidence_memory_ids) == {1, 2}, "edge carries both evidence ids"


# ---------------------------------------------------------------------------
# evidence ids on mentions
# ---------------------------------------------------------------------------

def test_memory_concepts_carry_evidence_and_relation() -> None:
    """`memory_concepts` links a memory to each concept it mentions, tagged with the
    typed relation and the originating memory id (evidence)."""
    triples: dict[int, list[Triple]] = {
        7: [("Viktor", "prefers", "Svelte")],
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)

    mems_for_7 = [mc for mc in g.memory_concepts if mc.memory_id == 7]
    # memory 7 mentions both Viktor (subject) and Svelte (object)
    concept_ids = {mc.concept_id for mc in mems_for_7}
    assert g.concept_of("Viktor") in concept_ids
    assert g.concept_of("Svelte") in concept_ids
    # the object-side mention carries the typed relation it participated in
    obj_links = [mc for mc in mems_for_7 if mc.concept_id == g.concept_of("Svelte")]
    assert any(mc.relation == "prefers" for mc in obj_links)


def test_concept_to_memories_index_is_built() -> None:
    """The builder exposes a concept→memory index (the bipartite adjacency the PPR
    leg in S4 seeds from). A concept shared by two memories lists both."""
    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte")],
        2: [("team", "uses", "SvelteKit")],  # SvelteKit → same concept as Svelte
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)
    svelte = g.concept_of("Svelte")
    assert svelte is not None
    assert set(g.memories_for_concept(svelte)) == {1, 2}, (
        "a canonical concept reached via an alias links both source memories"
    )


# ---------------------------------------------------------------------------
# surface-form de-duplication before embedding
# ---------------------------------------------------------------------------

def test_each_distinct_surface_form_embedded_at_most_once() -> None:
    """A surface form repeated across many triples is embedded only ONCE — the
    builder de-duplicates before calling the (expensive) encoder."""
    embedder = CallCountingEmbedder()
    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte")],
        2: [("Viktor", "likes", "Svelte")],
        3: [("Viktor", "uses", "Svelte")],
    }
    build_concept_graph(triples, embedder, threshold=_THRESHOLD)
    # distinct forms: viktor, svelte → embedded twice total, not 6 times.
    assert sorted({s.lower() for s in embedder.embedded}) == ["svelte", "viktor"]
    assert len(embedder.embedded) == len(set(s.lower() for s in embedder.embedded)), (
        "no surface form is embedded more than once"
    )


# ---------------------------------------------------------------------------
# edge-count stat — the latency-sanity check
# ---------------------------------------------------------------------------

def test_build_reports_bounded_edge_count_stat() -> None:
    """The build exposes a stats dict (concept count, edge count, …) for the
    latency-sanity log. On this tiny fixture the edge count is a handful, and the
    contract is that it is reported AND far below the prior 2,095,624-edge
    keyword-co-occurrence prototype."""
    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte"), ("Viktor", "uses", "Postgres")],
        2: [("Viktor", "favors", "SvelteKit")],
        3: [("app", "depends-on", "PostgreSQL")],
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)
    stats = g.stats()
    assert set(stats) >= {"concepts", "edges", "memory_concepts", "surface_forms"}
    assert stats["edges"] == len(g.concept_edges)
    assert stats["concepts"] == len(g.concepts)
    # the whole point of the typed graph: edge count is tiny, NOT the 2.1M blow-up.
    assert stats["edges"] < 2_095_624
    assert stats["edges"] <= 10, "this 4-entity fixture yields only a few typed edges"


def test_empty_triples_yield_empty_graph() -> None:
    """No triples → an empty graph with zero concepts/edges and a clean stats dict
    (degrades to nothing, never raises)."""
    g = build_concept_graph({}, _stub_embedder(), threshold=_THRESHOLD)
    assert g.concepts == {}
    assert g.concept_edges == []
    assert g.memory_concepts == []
    assert g.stats()["edges"] == 0
    assert g.concept_of("anything") is None


def test_self_referential_triple_does_not_create_self_loop() -> None:
    """A triple whose subject and object canonicalize to the SAME concept (e.g. an
    alias pair on both sides) must NOT create a self-loop edge — self-loops add no
    traversal value and would inflate the edge count."""
    triples: dict[int, list[Triple]] = {
        1: [("Svelte", "is-a", "SvelteKit")],  # both → the same canonical concept
    }
    g = build_concept_graph(triples, _stub_embedder(), threshold=_THRESHOLD)
    assert all(e.src_id != e.dst_id for e in g.concept_edges), "no self-loops"


# ---------------------------------------------------------------------------
# return type contract
# ---------------------------------------------------------------------------

def test_build_returns_a_concept_graph() -> None:
    """`build_concept_graph` returns a `ConceptGraph` with the three relations the
    production schema mirrors: concepts, concept_edges, memory_concepts."""
    g = build_concept_graph({1: [("Viktor", "prefers", "Svelte")]}, _stub_embedder(), threshold=_THRESHOLD)
    assert isinstance(g, ConceptGraph)
    assert hasattr(g, "concepts") and hasattr(g, "concept_edges") and hasattr(g, "memory_concepts")


# ---------------------------------------------------------------------------
# numpy fast path — MUST be byte-equivalent to the pure-Python single-linkage
# ---------------------------------------------------------------------------
#
# The pure-Python canonicalizer is O(forms × clusters) cosines in interpreted
# Python — wall-clock-prohibitive over the real 24k-form corpus (measured: ~72s
# for 1000 forms, quadratic → hours for the full set). `build_concept_graph_fast`
# vectorises the SAME greedy single-linkage with numpy (one form-vs-all-reps matmul
# per form), so the full-corpus build is tractable WITHOUT sampling. It is only
# ever invoked offline (numpy present); the model-free slow path stays the
# ADR-0002 base. These tests pin the fast path to be IDENTICAL to the slow path —
# same concept assignment, same first-seen canonical_name, same aliases, same
# typed edges + evidence — INCLUDING the >=-threshold tie-break (a tie goes to the
# LATER-seen cluster).

def _embed_vocab(forms: list[str]) -> list[list[float]]:
    """Embed via the same _VOCAB stub the other tests use (deterministic)."""
    return CallCountingEmbedder()(forms)


def _assert_graphs_equivalent(a: ConceptGraph, b: ConceptGraph) -> None:
    # same surface-form → concept-id partition (ids may differ in label but the
    # PARTITION and the canonical representative must be identical because both
    # process forms in the same first-seen order).
    assert a.stats() == b.stats()
    # canonical_name + aliases per surface form must match exactly.
    for surface in a._surface_to_concept:
        ca = a.concepts[a.concept_of(surface)]  # type: ignore[index]
        cb = b.concepts[b.concept_of(surface)]  # type: ignore[index]
        assert ca.canonical_name == cb.canonical_name, surface
        assert sorted(s.lower() for s in ca.aliases) == sorted(s.lower() for s in cb.aliases), surface
    # typed edges (as canonical-name-keyed tuples) + evidence must match.
    def _edge_key(g: ConceptGraph):  # type: ignore[no-untyped-def]
        return sorted(
            (
                g.concepts[e.src_id].canonical_name,
                g.concepts[e.dst_id].canonical_name,
                e.relation,
                e.weight,
                tuple(e.evidence_memory_ids),
            )
            for e in g.concept_edges
        )
    assert _edge_key(a) == _edge_key(b)


def test_fast_path_matches_slow_path_on_vocab_fixture() -> None:
    """The numpy fast path produces an IDENTICAL graph to the pure-Python path on a
    multi-cluster fixture with alias collapse on two independent axes."""
    from retrievers.graph_build import build_concept_graph_fast

    triples: dict[int, list[Triple]] = {
        1: [("Viktor", "prefers", "Svelte"), ("Viktor", "uses", "Postgres")],
        2: [("Viktor", "favors", "SvelteKit"), ("app", "depends-on", "PostgreSQL")],
        3: [("team", "uses", "Svelte Kit"), ("service", "runs-on", "postgresql")],
    }
    slow = build_concept_graph(triples, _embed_vocab, threshold=_THRESHOLD)
    fast = build_concept_graph_fast(triples, _embed_vocab, threshold=_THRESHOLD)
    _assert_graphs_equivalent(slow, fast)


def test_fast_path_matches_slow_path_with_tie_break() -> None:
    """When a new form is EQUIDISTANT (cosine tie) from two existing clusters, BOTH
    paths must break the tie identically — the pure-Python `>=` loop takes the
    LATER-seen cluster, and the fast path must replicate that exact rule (NOT
    numpy's first-argmax)."""
    import math

    from retrievers.graph_build import build_concept_graph_fast

    # Two clusters founded first (axes A and B); then a probe form placed exactly
    # on the 45° bisector is equally similar (cos = 1/√2) to both. With threshold
    # below 1/√2 it must join — and BOTH paths must pick the LATER cluster.
    inv = 1.0 / math.sqrt(2.0)
    vocab = {
        "alpha": _unit([1.0, 0.0]),       # cluster 0
        "beta": _unit([0.0, 1.0]),        # cluster 1 (later-seen)
        "probe": _unit([inv, inv]),       # tie: cos(alpha)=cos(beta)=1/√2
    }

    def embed(forms: list[str]) -> list[list[float]]:
        return [list(vocab[f.lower()]) for f in forms]

    triples: dict[int, list[Triple]] = {
        1: [("alpha", "related-to", "beta")],
        2: [("probe", "related-to", "alpha")],
    }
    thr = 0.5  # below 1/√2 ≈ 0.707 so probe merges into a cluster
    slow = build_concept_graph(triples, embed, threshold=thr)
    fast = build_concept_graph_fast(triples, embed, threshold=thr)
    _assert_graphs_equivalent(slow, fast)
    # and concretely: probe joined beta (the LATER cluster), not alpha.
    assert slow.concept_of("probe") == slow.concept_of("beta")
    assert fast.concept_of("probe") == fast.concept_of("beta")


def test_fast_path_empty_triples() -> None:
    """The fast path degrades to an empty graph on empty input, like the slow path."""
    from retrievers.graph_build import build_concept_graph_fast

    g = build_concept_graph_fast({}, _embed_vocab, threshold=_THRESHOLD)
    assert g.concepts == {}
    assert g.concept_edges == []
    assert g.stats()["edges"] == 0
