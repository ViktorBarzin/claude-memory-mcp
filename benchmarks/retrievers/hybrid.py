"""HYBRID retriever (ADR-0001/0002/0003 prototype): lexical FTS + dense semantic
recall + a memory-node concept graph, fused with Reciprocal Rank Fusion (RRF).

This is the self-contained prototype the hybrid-recall ADOPTION decision is gated
on (ADR-0001): does dense embeddings + a concept graph beat the current lexical
FTS5/BM25 on recall@5/recall@10/nDCG@10/MRR? Quality decides; latency/storage are
reported but non-gating.

It implements the harness ``retrieve(query, k) -> list[int]`` contract and the
optional ``build_index(corpus)`` / ``index_size_bytes()`` / ``name`` hooks.

Three legs, mirroring the FINAL DESIGN
======================================

1. **Lexical (FTS5/BM25).** We reuse the *faithful* production reimplementation
   ``retrievers.fts.FtsRetriever`` verbatim — AND-first then OR-broaden over an
   FTS5(content, category, tags, expanded_keywords) index, ranked by the blended
   ``(-bm25*0.7 + importance*0.3)``. This is the exact "current system" the hybrid
   must beat, so the lexical leg of the hybrid IS that system (no drift).

2. **Dense (semantic).** Embeddings per FINAL DESIGN: a HOSTED API is used ONLY if
   its key is in the environment (``OPENAI_API_KEY`` / ``VOYAGE_API_KEY`` /
   ``CO_API_KEY``) AND the memory is non-sensitive (ADR-0003); otherwise the local
   default ``BAAI/bge-large-en-v1.5`` (1024-d, MIT, sentence-transformers). The
   benchmark corpus is already sensitive-free (``is_sensitive=1`` excluded at
   export, README privacy note), so here the choice is purely "hosted key present
   or not". Vectors are L2-normalised; similarity is cosine = dot product. The
   corpus matrix is cached to ``cache/`` (gitignored) keyed by model id + a corpus
   fingerprint, so re-runs skip re-embedding. BGE retrieval convention: the QUERY
   gets the instruction prefix "Represent this sentence for searching relevant
   passages: "; passages are embedded raw (per the official BAAI model card).

3. **Graph (concept expansion).** A memory-node concept graph built with the
   design's TRACTABLE extraction — NO 5452 sequential LLM calls. Concepts are the
   union of each memory's ``tags`` and its already-LLM-generated
   ``expanded_keywords`` (plus salient content noun-phrases via a lightweight
   regex/stop-word filter), normalised and de-pluralised. A concept that appears
   in 2..N memories (very common concepts above a document-frequency ceiling are
   dropped as non-discriminative) links those memories: ``memory -[shares
   concept c]- memory``. At query time we take the fused dense+lexical SEEDS, walk
   1 hop to neighbours that share *discriminative* concepts, and emit those
   neighbours as a third ranked list that joins the SHARED fused pool — NO base-set
   exclusion (ADR-0005 fusion fix), so a graph candidate can reinforce a base hit
   or compete as a graph-only id. This targets the **multihop** stratum (queries
   needing 2+ memories that share an entity/concept).

Fusion (``retrieval_fusion``)
=============================
Reciprocal Rank Fusion (Cormack et al., 2009): for a document *d* with rank
``r_leg(d)`` (1-based) in a leg's ranked list,

    RRF(d) = Σ_leg  w_leg / (k_rrf + r_leg(d))

with ``k_rrf = 60`` (the standard constant) and per-leg weights. RRF is
score-scale-free (no BM25-vs-cosine calibration), which is why the design floats
"RRF vs CC" and we pick RRF for the prototype. The dense and lexical legs carry
full weight; ``w_graph`` is a SWEPT attribute (ADR-0005). Because the graph leg now
competes in the SHARED fused pool with no exclusion, a graph-only rank-1 hit scores
only ``w_graph/(60+1)`` and must clear the TOP-10 fused boundary (≈ 0.0286 under
realistic dual-leg double-counting, NOT the weakest tail) — so the sweep extends
above 1.0 (to ~2.0+) to give the sparse graph leg a genuine shot. All three weights
are class attributes so the comparison can ablate the graph to zero (w_graph=0) or
sweep it up.

Graceful degradation (task requirement)
=======================================
If the embedding model cannot be loaded/used (missing package, download failure,
OOM), the dense leg is skipped, the failure is recorded in ``self.errors``, and the
retriever degrades to **FTS + graph** (or FTS-only if the graph also failed). The
harness still gets metrics for whatever worked.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

# ── package-relative imports that also work under direct execution ────────────
try:  # pragma: no cover - exercised by both import paths
    from harness.types import Memory, MemoryId
    from retrievers.fts import FtsRetriever
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.types import Memory, MemoryId
    from retrievers.fts import FtsRetriever

if TYPE_CHECKING:  # type-only: the typed graph is injected (S5 wiring), never imported
    from retrievers.graph_build import ConceptGraph  # at module import (ADR-0002 lazy).

_BENCH_ROOT = Path(__file__).resolve().parents[1]
_CACHE_DIR = _BENCH_ROOT / "cache"

# Local default embedding model (FINAL DESIGN: prototype default + sensitive-only
# fallback). 1024-d, MIT-licensed, strong on MTEB retrieval.
_LOCAL_MODEL = "BAAI/bge-large-en-v1.5"
# BGE retrieval query instruction (official BAAI model card recommendation; the
# v1.5 line relaxed it but it still helps short-query / long-passage asymmetry,
# which is exactly the paraphrase stratum). Applied to QUERIES only.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# RRF constant (Cormack/Clarke/Buettcher 2009). 60 is the canonical default.
_RRF_K = 60

# Concept-graph tuning.
#   _CONCEPT_MIN_DF : a concept must appear in >= this many memories to form edges
#                     (df==1 links nothing; we need a shared concept).
#   _CONCEPT_MAX_DF_FRAC : drop concepts appearing in more than this fraction of
#                     the corpus — they are non-discriminative hubs ("memory",
#                     "homelab") that would over-connect the graph (design risk:
#                     "over-merge").
#   _GRAPH_SEEDS    : how many fused seeds to expand from.
#   _GRAPH_NEIGHBOURS_PER_SEED : cap neighbours pulled per seed (keeps the graph
#                     leg from flooding the candidate pool).
_CONCEPT_MIN_DF = 2
_CONCEPT_MAX_DF_FRAC = 0.02
_GRAPH_SEEDS = 10
_GRAPH_NEIGHBOURS_PER_SEED = 25

# ── Personalized PageRank (PPR) over the TYPED concept graph (slice S4) ───────────
#   The HippoRAG-2 idea WITHOUT a per-query LLM: seed a restart vector from the
#   memory nodes the FTS∪dense fusion surfaced, then run PPR to convergence over the
#   bipartite memory↔concept graph and read off a graph-leg ranking of MEMORIES by
#   stationary probability. A non-seeded memory reaches the ranking purely via a
#   shared canonical concept with a seed — that is the entity-bridged hop, and the
#   only mechanism by which S1's "graph-only G reaches the fused top-k" is reachable.
#
#   _PPR_ALPHA : damping = probability of FOLLOWING an edge each step; (1-alpha) is
#                the restart/teleport probability back to the personalization (seed)
#                distribution. 0.5 is the HippoRAG default and keeps mass close to the
#                seeds (short-hop bias) rather than smearing it across the graph.
#   _PPR_TOL   : L1 convergence tolerance on successive iterates.
#   _PPR_MAX_ITER : hard iteration cap (the power iteration converges geometrically;
#                this just bounds a pathological graph). Fixed alpha + fixed seed ⇒
#                a DETERMINISTIC stationary vector (required for the comparison).
_PPR_ALPHA = 0.5
_PPR_TOL = 1e-9
_PPR_MAX_ITER = 200

# A small English stop-word set for the lightweight noun-phrase extraction. We
# deliberately keep this tiny + dependency-free (no spaCy/NLTK download on the hot
# path); the heavy lifting is done by the pre-computed ``expanded_keywords``.
_STOPWORDS = frozenset(
    """
    a an the of to in on at by for with from into over under and or but not is are
    was were be been being do does did has have had this that these those it its as
    if then than so such no yes can will would should could may might must i you he
    she they we me him her them us my your his their our about above after again all
    any because before below between both during each few more most other some only
    own same too very up down out off here there when where which who whom what how
    """.split()
)
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]{2,}")


def _normalise_concept(token: str) -> str:
    """Lowercase, strip surrounding punctuation, light de-plural so concept
    variants collapse to one node (e.g. 'decisions'->'decision',
    'addresses'->'address', 'policies'->'policy'). This is a heuristic collapser,
    not a linguistically perfect stemmer — its only job is to merge obvious
    plural/singular pairs so the graph links them; exactness is not load-bearing.
    Order matters: -ies, then -sses, then sibilant -es, then a bare trailing -s."""
    t = token.lower().strip(".,;:!?()[]{}\"'`")
    if len(t) > 4 and t.endswith("ies"):  # policies -> policy
        return t[:-3] + "y"
    if len(t) > 4 and t.endswith("sses"):  # addresses -> address, classes -> class
        return t[:-2]
    if len(t) > 4 and t.endswith(("ches", "shes", "xes", "zes", "ses")):  # boxes->box
        return t[:-2]
    if len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):  # tags->tag
        return t[:-1]
    return t


def _concepts_for(memory: Memory) -> set[str]:
    """Extract the concept set for one memory: tags ∪ expanded_keywords ∪ salient
    content tokens. ``expanded_keywords`` is already an LLM-generated keyword field
    in the corpus, so this is the design's 'tractable extraction' — we reuse the
    extraction that production already pays for instead of new LLM calls."""
    concepts: set[str] = set()
    # tags: comma-separated
    for tag in memory.tags.split(","):
        c = _normalise_concept(tag)
        if len(c) >= 3 and c not in _STOPWORDS:
            concepts.add(c)
    # expanded_keywords: space-separated, already curated
    for kw in memory.expanded_keywords.split():
        c = _normalise_concept(kw)
        if len(c) >= 3 and c not in _STOPWORDS:
            concepts.add(c)
    # salient content tokens (lightweight noun-phrase proxy: alpha tokens len>=3,
    # not stop-words). This is a cheap NER/noun-phrase stand-in per the design.
    for m in _WORD_RE.finditer(memory.content):
        c = _normalise_concept(m.group(0))
        if len(c) >= 3 and c not in _STOPWORDS:
            concepts.add(c)
    return concepts


def _corpus_fingerprint(corpus: Sequence[Memory]) -> str:
    """Stable hash over (id, content) so the embedding cache invalidates if the
    corpus changes but is reused across runs of the same corpus."""
    h = hashlib.sha256()
    for m in corpus:
        h.update(str(m.id).encode())
        h.update(b"\x00")
        h.update(m.content.encode("utf-8", "replace"))
        h.update(b"\x01")
    return h.hexdigest()[:16]


def _ppr_stationary(
    edges: Sequence[tuple[int, int]],
    n_nodes: int,
    seed_weights: Mapping[int, float],
    *,
    alpha: float = _PPR_ALPHA,
    tol: float = _PPR_TOL,
    max_iter: int = _PPR_MAX_ITER,
) -> list[float]:
    """Personalized PageRank over an UNDIRECTED graph, by sparse power iteration.

    This is the model-free numerical core of the S4 graph leg (HippoRAG-2 idea). It
    is deliberately a free function over a plain edge list + node count + seed map so
    it can be unit-tested on a hand-built bipartite memory↔concept graph with NO
    embedding model, NO corpus, and NO database.

    Args:
        edges: undirected edges as ``(u, v)`` index pairs into ``[0, n_nodes)``. Each
            pair is symmetrised internally; duplicates accumulate edge weight (so a
            doubly-asserted concept link pulls harder), matching the typed graph where
            an edge's ``weight`` counts supporting triples.
        n_nodes: total node count (memory nodes ∪ concept nodes share one index space).
        seed_weights: ``{node_index: restart_weight}`` — the personalization / restart
            distribution. NON-NEGATIVE weights; they are L1-normalised internally, so
            only the RELATIVE weighting matters. An EMPTY map means NO restart mass and
            returns an all-zero vector (the graph leg contributes nothing without
            seeds) rather than a uniform PageRank.

    Returns:
        A length-``n_nodes`` list of stationary probabilities (sums to 1 when there is
        restart mass; all-zero when ``seed_weights`` is empty). Every node is ranked —
        the seeds only set the restart vector — so a NON-SEEDED node reachable from a
        seed (through a shared concept) gets nonzero mass and a node in a disjoint
        component gets exactly zero (teleport targets seeds only). Deterministic for a
        fixed ``alpha`` and ``seed_weights``.

    The iteration (mass-conserving, with dangling-node handling):

        x_{t+1} = alpha · (Pᵀ x_t + d_t · p) + (1 - alpha) · p

    where ``P = D⁻¹ A`` is the row-stochastic random-walk matrix, ``p`` is the
    normalised restart vector, and ``d_t`` is the mass on dangling (degree-0) nodes at
    step t (redistributed to ``p`` so total mass stays 1). Note ``Pᵀ`` is built once
    per call; production would CACHE it across queries (the real hot-path cost — see
    ``HybridRetriever.measure_ppr_latency_ms``).
    """
    import numpy as np
    import scipy.sparse as sp

    if n_nodes <= 0 or not seed_weights:
        return [0.0] * max(n_nodes, 0)

    # Build the symmetric adjacency (duplicates sum → integer edge multiplicity/weight).
    if edges:
        rows: list[int] = []
        cols: list[int] = []
        for u, v in edges:
            rows.append(u)
            cols.append(v)
            rows.append(v)
            cols.append(u)
        data = np.ones(len(rows), dtype=np.float64)
        adj = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        adj.sum_duplicates()
    else:
        adj = sp.csr_matrix((n_nodes, n_nodes), dtype=np.float64)

    deg = np.asarray(adj.sum(axis=1)).ravel()
    dangling = deg == 0.0
    deg_safe = deg.copy()
    deg_safe[dangling] = 1.0
    d_inv = sp.diags(1.0 / deg_safe)
    # P = D⁻¹ A (row-stochastic); we iterate with its transpose (column-stochastic).
    p_transition = (d_inv @ adj).T.tocsr()

    pers = np.zeros(n_nodes, dtype=np.float64)
    for node, w in seed_weights.items():
        if 0 <= node < n_nodes and w > 0.0:
            pers[node] += float(w)
    total = pers.sum()
    if total <= 0.0:
        return [0.0] * n_nodes  # seeds out of range / all non-positive → empty leg
    pers /= total

    x = pers.copy()
    for _ in range(max_iter):
        dangling_mass = float(x[dangling].sum())
        x_new = alpha * (p_transition @ x + dangling_mass * pers) + (1.0 - alpha) * pers
        if float(np.abs(x_new - x).sum()) < tol:
            x = x_new
            break
        x = x_new
    return [float(v) for v in x]


class HybridRetriever:
    """Lexical FTS + dense (bge-large-en-v1.5 / hosted) + concept-graph expansion,
    fused with RRF. Degrades to FTS(+graph) if embeddings are unavailable."""

    #: Label surfaced in benchmark reports / the RUN schema.
    name = "hybrid"

    # Per-leg RRF weights. Dense + lexical carry full weight; graph is a
    # down-weighted recall extender (design: possible negative graph prior).
    w_dense = 1.0
    w_fts = 1.0
    w_graph = 0.35

    # Graph-leg traversal mode over the TYPED concept graph (slice S4):
    #   "ppr"  — Personalized PageRank over the bipartite memory↔concept graph
    #            (HippoRAG-2 idea; the DEFAULT). The fused seeds set the restart vector;
    #            the leg then ranks the NON-SEED memory nodes by stationary probability
    #            (seeds are excluded from the output — they reinforce via FTS+dense, so
    #            a bridged graph-only memory lands at graph-rank ~1 and can compete).
    #   "1hop" — the cheaper typed-1-hop traversal: from each seed, walk
    #            memory→concept→memory once and rank the bridged neighbours by shared-
    #            concept weight. Evaluated alongside PPR (the production-substrate
    #            choice is deferred to the measured-latency verdict).
    # Only consulted when a typed ConceptGraph has been set via set_concept_graph();
    # with no typed graph the leg falls back to the prior keyword-co-occurrence walk.
    graph_mode = "ppr"

    def __init__(self, model_name: str | None = None) -> None:
        self.errors: list[str] = []
        self.model_name = model_name or _LOCAL_MODEL
        self.embedding_backend: str = "none"  # "local:<model>" | "hosted:<provider>:<model>"
        self.embedding_dim: int | None = None

        # FTS leg (always available; pure stdlib sqlite).
        self._fts = FtsRetriever(sort_by="relevance")

        # Dense leg state.
        self._model = None  # SentenceTransformer or None
        self._np = None  # numpy module handle (set on successful dense build)
        self._emb = None  # (N, d) float32 L2-normalised matrix, row i ↔ self._ids[i]
        self._ids: list[MemoryId] = []  # row order of self._emb

        # Graph leg state — prior keyword-co-occurrence graph (fallback when no typed
        # graph is injected; the prototype this run supersedes).
        self._graph = None  # networkx.Graph or None
        self._concept_to_mems: dict[str, list[MemoryId]] = {}
        self._mem_concepts: dict[MemoryId, set[str]] = {}
        self._n_concepts_total = 0  # before df pruning, for reporting
        self._n_concepts_kept = 0
        self._n_edges = 0

        # TYPED concept graph (slice S3) + its bipartite memory↔concept index for PPR
        # (slice S4). Injected via set_concept_graph (S5 wiring); None ⇒ the leg uses
        # the keyword fallback above. The transition matrix is built lazily and CACHED
        # (the real per-query PPR hot-path is the iterate, not the rebuild).
        self._cgraph: ConceptGraph | None = None
        self._node_count = 0  # |memory nodes ∪ concept nodes| in the bipartite index
        self._mem_to_node: dict[MemoryId, int] = {}  # memory id → bipartite node index
        self._node_to_mem: dict[int, MemoryId] = {}  # inverse, for reading PPR results
        self._bipartite_edges: list[tuple[int, int]] = []  # undirected memory↔concept
        self._typed_edge_count = 0  # |concept_edges| in the typed graph, for reporting

        self._corpus_size = 0

    # ── lifecycle: build_index (timed by the runner) ─────────────────────────

    def build_index(self, corpus: Sequence[Memory]) -> None:
        corpus = list(corpus)
        self._corpus_size = len(corpus)
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # 1) lexical leg
        self._fts.build_index(corpus)

        # 2) dense leg (graceful)
        try:
            self._build_dense(corpus)
        except Exception as exc:  # pragma: no cover - defensive
            self.errors.append(f"dense leg disabled: {type(exc).__name__}: {exc}")
            self._model = None
            self._emb = None

        # 3) graph leg (graceful)
        try:
            self._build_graph(corpus)
        except Exception as exc:  # pragma: no cover - defensive
            self.errors.append(f"graph leg disabled: {type(exc).__name__}: {exc}")
            self._graph = None
            self._concept_to_mems = {}

    # ── dense leg ────────────────────────────────────────────────────────────

    def _select_embedding_backend(self) -> str:
        """Pick the embedding backend per FINAL DESIGN: hosted only if a key is in
        the env (non-sensitive corpus already guaranteed by export), else local.
        Returns a human label and sets self.model_name accordingly."""
        if os.environ.get("VOYAGE_API_KEY"):
            self.model_name = "voyage-3.5"
            return "hosted:voyage:voyage-3.5"
        if os.environ.get("OPENAI_API_KEY"):
            self.model_name = "text-embedding-3-large"
            return "hosted:openai:text-embedding-3-large"
        if os.environ.get("CO_API_KEY"):
            self.model_name = "embed-english-v3.0"
            return "hosted:cohere:embed-english-v3.0"
        self.model_name = _LOCAL_MODEL
        return f"local:{_LOCAL_MODEL}"

    def _build_dense(self, corpus: Sequence[Memory]) -> None:
        import numpy as np  # required for the dense leg

        self._np = np
        self.embedding_backend = self._select_embedding_backend()
        self._ids = [m.id for m in corpus]
        fp = _corpus_fingerprint(corpus)
        safe_model = self.model_name.replace("/", "_")
        emb_path = _CACHE_DIR / f"emb_{safe_model}_{fp}.npy"
        ids_path = _CACHE_DIR / f"emb_{safe_model}_{fp}.ids.npy"

        # cache hit?
        if emb_path.exists() and ids_path.exists():
            cached_ids = np.load(ids_path)
            if list(cached_ids.tolist()) == self._ids:
                self._emb = np.load(emb_path).astype(np.float32)
                self.embedding_dim = int(self._emb.shape[1])
                return  # cached embeddings reused

        # cache miss → embed
        if self.embedding_backend.startswith("hosted:"):
            vecs = self._embed_hosted([m.content for m in corpus])
        else:
            vecs = self._embed_local([m.content for m in corpus])
        vecs = vecs.astype(np.float32)
        # L2-normalise so dot product == cosine.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self._emb = vecs
        self.embedding_dim = int(vecs.shape[1])
        np.save(emb_path, vecs)
        np.save(ids_path, np.array(self._ids, dtype=np.int64))

    def _load_local_model(self):
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            # CPU is fine for ~5.5k short docs; force CPU to avoid CUDA init noise.
            self._model = SentenceTransformer(_LOCAL_MODEL, device="cpu")
            # Median memory is ~120 tokens; cap the window at 384 so the rare long
            # memory (1.6% > 512 tok) doesn't pad an entire batch to 512. bge's
            # native max is 512; 384 keeps ~p99 intact while bounding CPU cost.
            self._model.max_seq_length = min(self._model.max_seq_length, 384)
        return self._model

    def _embed_local(self, texts: list[str]):
        import numpy as np

        model = self._load_local_model()
        # Length-sort so each batch pads to a homogeneous length (big CPU win), then
        # restore original order. Passages embedded raw; the caller L2-normalises so
        # the local and hosted paths stay byte-for-byte consistent downstream.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sorted_texts = [texts[i] for i in order]
        out = model.encode(
            sorted_texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        out = np.asarray(out)
        # invert the permutation
        restored = np.empty_like(out)
        restored[np.asarray(order)] = out
        return restored

    def _embed_query_local(self, query: str):
        import numpy as np

        model = self._load_local_model()
        out = model.encode(
            [_BGE_QUERY_INSTRUCTION + query],
            convert_to_numpy=True,
            normalize_embeddings=True,  # query L2-normalised → cosine via dot
            show_progress_bar=False,
        )
        return np.asarray(out)[0]

    def _embed_hosted(self, texts: list[str]):
        """Batch-embed passages via the selected hosted API. Implemented for
        Voyage / OpenAI / Cohere; only reached when the matching key is set."""
        import numpy as np

        backend = self.embedding_backend
        if backend.startswith("hosted:voyage"):
            import voyageai

            client = voyageai.Client()
            vecs: list[list[float]] = []
            for i in range(0, len(texts), 128):
                batch = texts[i : i + 128]
                r = client.embed(batch, model="voyage-3.5", input_type="document")
                vecs.extend(r.embeddings)
            return np.asarray(vecs)
        if backend.startswith("hosted:openai"):
            from openai import OpenAI

            client = OpenAI()
            vecs = []
            for i in range(0, len(texts), 256):
                batch = texts[i : i + 256]
                r = client.embeddings.create(model="text-embedding-3-large", input=batch)
                vecs.extend([d.embedding for d in r.data])
            return np.asarray(vecs)
        if backend.startswith("hosted:cohere"):
            import cohere

            client = cohere.Client()
            vecs = []
            for i in range(0, len(texts), 96):
                batch = texts[i : i + 96]
                r = client.embed(texts=batch, model="embed-english-v3.0", input_type="search_document")
                vecs.extend(r.embeddings)
            return np.asarray(vecs)
        raise RuntimeError(f"unknown hosted backend {backend!r}")

    def _embed_query_hosted(self, query: str):
        import numpy as np

        backend = self.embedding_backend
        if backend.startswith("hosted:voyage"):
            import voyageai

            client = voyageai.Client()
            r = client.embed([query], model="voyage-3.5", input_type="query")
            v = np.asarray(r.embeddings[0], dtype=np.float32)
        elif backend.startswith("hosted:openai"):
            from openai import OpenAI

            client = OpenAI()
            r = client.embeddings.create(model="text-embedding-3-large", input=[query])
            v = np.asarray(r.data[0].embedding, dtype=np.float32)
        elif backend.startswith("hosted:cohere"):
            import cohere

            client = cohere.Client()
            r = client.embed(texts=[query], model="embed-english-v3.0", input_type="search_query")
            v = np.asarray(r.embeddings[0], dtype=np.float32)
        else:
            raise RuntimeError(f"unknown hosted backend {backend!r}")
        n = np.linalg.norm(v)
        return v / n if n else v

    def _dense_rank(self, query: str, k: int) -> list[MemoryId]:
        """Top-k corpus ids by cosine similarity to the query embedding."""
        if self._emb is None or self._np is None:
            return []
        np = self._np
        if self.embedding_backend.startswith("hosted:"):
            qv = self._embed_query_hosted(query)
        else:
            qv = self._embed_query_local(query)
        sims = self._emb @ qv  # (N,) cosine sims (both sides L2-normalised)
        kk = min(k, sims.shape[0])
        # argpartition for the top-kk, then sort those by score desc.
        idx = np.argpartition(-sims, kk - 1)[:kk]
        idx = idx[np.argsort(-sims[idx])]
        return [self._ids[i] for i in idx]

    # ── typed concept graph (S3) + PPR leg (S4) ──────────────────────────────

    def set_concept_graph(self, cgraph: ConceptGraph) -> None:
        """Inject the TYPED concept graph (slice S3) and pre-compute its bipartite
        memory↔concept index for the PPR / typed-1-hop graph leg (slice S4).

        Building the index here (once, at wiring time) keeps ``_graph_rank`` on the
        query hot path cheap: it only seeds a restart vector and iterates. The node
        index space holds BOTH memory nodes and concept nodes — memory ids are mapped
        to fresh node indices and concept ids are offset above them — and an undirected
        edge connects every memory to each concept it mentions (the bipartite
        adjacency PPR walks). This is the structure that lets memory→concept→memory be
        a single concept hop, so a non-seeded memory sharing a concept with a seed is
        reachable.
        """
        self._cgraph = cgraph
        self._typed_edge_count = len(cgraph.concept_edges)

        # 1) memory nodes: every memory that mentions at least one concept. Stable
        #    sorted order → deterministic node indices (and thus deterministic PPR).
        mem_ids = sorted({mc.memory_id for mc in cgraph.memory_concepts})
        self._mem_to_node = {mid: i for i, mid in enumerate(mem_ids)}
        self._node_to_mem = {i: mid for mid, i in self._mem_to_node.items()}
        n_mem = len(mem_ids)

        # 2) concept nodes: offset above the memory nodes in the shared index space.
        concept_ids = sorted(cgraph.concepts.keys())
        concept_to_node = {cid: n_mem + i for i, cid in enumerate(concept_ids)}
        self._node_count = n_mem + len(concept_ids)

        # 3) bipartite edges: memory ↔ each concept it mentions. De-duplicate (a memory
        #    may mention the same concept via several relations) — multiplicity here
        #    would over-weight a single mention; the typed concept↔concept edges carry
        #    the relational weight separately (added below).
        seen_mc: set[tuple[int, int]] = set()
        edges: list[tuple[int, int]] = []
        for mc in cgraph.memory_concepts:
            mnode = self._mem_to_node.get(mc.memory_id)
            cnode = concept_to_node.get(mc.concept_id)
            if mnode is None or cnode is None:
                continue
            key = (mnode, cnode)
            if key not in seen_mc:
                seen_mc.add(key)
                edges.append(key)

        # 4) typed concept↔concept edges (relational structure): a triple
        #    (src)-[rel]->(dst) connects two concept nodes. Repeated edges in the list
        #    accumulate weight inside the PPR adjacency, so a strongly-asserted relation
        #    pulls harder. Self-loops were already dropped by the S3 builder.
        for e in cgraph.concept_edges:
            s = concept_to_node.get(e.src_id)
            d = concept_to_node.get(e.dst_id)
            if s is None or d is None or s == d:
                continue
            edges.append((s, d))

        self._bipartite_edges = edges

    def _ppr_memory_ranking(self, seeds: list[MemoryId], k: int) -> list[MemoryId]:
        """PPR graph leg: seed the restart vector from the fused-seed MEMORY nodes, run
        PPR over the bipartite graph, then rank the NON-SEED memory nodes by stationary
        probability.

        Seed weight is rank-decayed (1/rank) so the best-agreed FTS∪dense seeds pull
        hardest — the same monotone seed prior the keyword leg used.

        SEED-LOCKOUT FIX (review finding): the SEED memory nodes are EXCLUDED from the
        returned ranking — only the bridged NON-SEED memories are emitted, mirroring the
        1hop leg's neighbours-only contract. PPR gives the restart-vector seeds the
        highest stationary mass, so leaving them in put the (already FTS∪dense-
        reinforced) seeds at graph-ranks 1..|seeds| and forced any purely-bridged
        graph-only memory below them; in ``retrieve()`` those seeds then sat in all
        three legs, so raising ``w_graph`` lifted the seeds in LOCKSTEP with the lone
        graph-only hit, which could therefore never reach the fused top-k at ANY weight.
        Dropping the seeds here puts a bridged graph-only memory at graph-rank ~1 (the
        profile the 0.0286 / w_graph≈2.0 boundary analysis assumes), so it genuinely
        competes in the shared pool. Nothing is lost: seed REINFORCEMENT still happens
        via the FTS+dense legs (the seeds are base hits already), exactly as before.

        Concept nodes are dropped from the output too (only memories are retrievable
        candidates), top ``k``."""
        if self._cgraph is None or not seeds or self._node_count == 0:
            return []
        seed_weights: dict[int, float] = {}
        for rank, s in enumerate(seeds[:_GRAPH_SEEDS], start=1):
            node = self._mem_to_node.get(s)
            if node is not None:
                seed_weights[node] = seed_weights.get(node, 0.0) + 1.0 / rank
        if not seed_weights:
            return []  # none of the seeds is a memory node in the typed graph

        seed_nodes = set(seed_weights)
        dist = _ppr_stationary(self._bipartite_edges, self._node_count, seed_weights)
        # rank NON-SEED memory nodes (drop seed + concept nodes) by stationary mass,
        # mass > 0 only — so a bridged graph-only memory is at graph-rank ~1.
        scored: list[tuple[float, MemoryId]] = []
        for node, mid in self._node_to_mem.items():
            if node in seed_nodes:
                continue  # seeds reinforce via FTS+dense, not via the graph leg
            p = dist[node]
            if p > 0.0:
                scored.append((p, mid))
        # sort by mass desc, then by id asc for a deterministic tie-break.
        scored.sort(key=lambda pm: (-pm[0], pm[1]))
        return [mid for _, mid in scored[:k]]

    def _typed_one_hop_ranking(self, seeds: list[MemoryId], k: int) -> list[MemoryId]:
        """Cheaper typed-1-hop graph-leg variant (behind ``graph_mode='1hop'``): from
        each seed memory, walk memory→concept→memory ONCE and rank the bridged
        NEIGHBOUR memories by shared-concept count, rank-decayed by the seed. No power
        iteration — this is the candidate production substrate that ports to a
        recursive Postgres CTE. Returns NEIGHBOURS only (a seed is not its own
        neighbour), so it mirrors the keyword leg's 1-hop contract."""
        if self._cgraph is None or not seeds:
            return []
        cg = self._cgraph
        # memory → set(concept ids) it mentions (built once from the mention-links).
        mem_concepts: dict[MemoryId, set[int]] = defaultdict(set)
        for mc in cg.memory_concepts:
            mem_concepts[mc.memory_id].add(mc.concept_id)

        scores: dict[MemoryId, float] = defaultdict(float)
        for rank, s in enumerate(seeds[:_GRAPH_SEEDS], start=1):
            seed_w = 1.0 / rank
            for cid in mem_concepts.get(s, set()):
                for nbr in cg.memories_for_concept(cid):
                    if nbr == s:
                        continue  # no self-neighbour
                    scores[nbr] += seed_w
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [mid for mid, _ in ranked[:k]]

    def measure_ppr_latency_ms(self, seeds: list[MemoryId], *, repeats: int = 5) -> float:
        """MEASURE the per-query PPR latency (the challenger-corrected honesty: the
        survey's ~2 ms is unproven; a 2.1M-edge graph would be ~140-280 ms). Times the
        FULL leg (build the restart vector + the power iteration over the current
        bipartite edges) and returns the MEDIAN milliseconds over ``repeats`` runs.
        Reported, never gating — but it IS the production hot-path cost if PPR is
        promoted, so it informs the substrate choice."""
        import time

        if self._cgraph is None or self._node_count == 0:
            return 0.0
        samples: list[float] = []
        for _ in range(max(1, repeats)):
            t0 = time.perf_counter()
            self._ppr_memory_ranking(seeds, k=max(self._node_count, 1))
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        return samples[len(samples) // 2]

    # ── graph leg (prior keyword-co-occurrence prototype — fallback) ──────────

    def _build_graph(self, corpus: Sequence[Memory]) -> None:
        import networkx as nx

        n = len(corpus)
        max_df = max(_CONCEPT_MIN_DF, int(_CONCEPT_MAX_DF_FRAC * n))

        # concept → set(memory ids)
        concept_to_mems: dict[str, set[MemoryId]] = defaultdict(set)
        mem_concepts: dict[MemoryId, set[str]] = {}
        for m in corpus:
            cs = _concepts_for(m)
            mem_concepts[m.id] = cs
            for c in cs:
                concept_to_mems[c].add(m.id)
        self._n_concepts_total = len(concept_to_mems)

        # Keep only discriminative concepts: appear in [_CONCEPT_MIN_DF, max_df]
        # memories. Below MIN_DF links nothing; above max_df is a non-specific hub.
        kept: dict[str, list[MemoryId]] = {}
        for c, mems in concept_to_mems.items():
            df = len(mems)
            if _CONCEPT_MIN_DF <= df <= max_df:
                kept[c] = sorted(mems)
        self._n_concepts_kept = len(kept)
        self._concept_to_mems = kept
        # restrict each memory's concept set to kept concepts (for neighbour scoring)
        self._mem_concepts = {
            mid: {c for c in cs if c in kept} for mid, cs in mem_concepts.items()
        }

        # Build a weighted memory-node graph: edge weight = # shared kept concepts.
        # We add edges via concept cliques but CAP per-concept fan-out to avoid an
        # O(df^2) blow-up on the densest kept concepts (design risk: over-merge).
        g = nx.Graph()
        g.add_nodes_from(m.id for m in corpus)
        edge_w: dict[tuple[MemoryId, MemoryId], int] = defaultdict(int)
        for c, mems in kept.items():
            # mems is small (<= max_df) by construction; full clique is fine.
            for i in range(len(mems)):
                a = mems[i]
                for j in range(i + 1, len(mems)):
                    b = mems[j]
                    key = (a, b) if a < b else (b, a)
                    edge_w[key] += 1
        for (a, b), w in edge_w.items():
            g.add_edge(a, b, weight=w)
        self._n_edges = g.number_of_edges()
        self._graph = g

    def _graph_rank(self, seeds: list[MemoryId], k: int) -> list[MemoryId]:
        """Rank graph-leg candidates from the fused seeds, returning up to ``k`` ids.

        DISPATCH (slice S4): if a TYPED concept graph has been injected
        (``set_concept_graph``), the leg runs over it — ``graph_mode='ppr'`` (default)
        does Personalized PageRank over the bipartite memory↔concept graph and ranks
        the NON-SEED memory nodes by stationary probability (the seeds set ONLY the
        restart vector, then are dropped from the output — so a non-seeded memory
        sharing a concept with a seed surfaces at graph-rank ~1 rather than below the
        high-mass seeds), while ``graph_mode='1hop'`` does the cheaper typed
        memory→concept→memory walk (neighbours only). With NO typed graph set, the leg
        falls back to the prior keyword-co-occurrence 1-hop walk (the prototype this run
        supersedes), so existing keyword-graph callers keep working unchanged.

        FUSION FIX (ADR-0005), preserved across all three modes: NO ``exclude`` — the
        graph leg is NOT carved out of the FTS∪dense base pool. Whatever it surfaces
        (a base-leg overlap that gets REINFORCED, or a graph-only id) enters the SHARED
        fused pool and competes on its own RRF mass. The prior ``exclude=base_set`` +
        fixed ``w_graph=0.35`` capped a graph-only hit below any realistic top-k
        boundary — the math artifact this run exists to undo."""
        if not seeds:
            return []
        # typed-graph path (the S4 substrate) takes precedence when wired in.
        if self._cgraph is not None:
            if self.graph_mode == "1hop":
                return self._typed_one_hop_ranking(seeds, k)
            return self._ppr_memory_ranking(seeds, k)

        # fallback: prior keyword-co-occurrence 1-hop walk (no typed graph injected).
        if self._graph is None:
            return []
        g = self._graph
        scores: dict[MemoryId, float] = defaultdict(float)
        for rank, s in enumerate(seeds[:_GRAPH_SEEDS], start=1):
            if s not in g:
                continue
            seed_w = 1.0 / rank  # earlier seeds contribute more
            nbrs = sorted(
                g[s].items(), key=lambda kv: kv[1].get("weight", 1), reverse=True
            )[:_GRAPH_NEIGHBOURS_PER_SEED]
            for nbr, data in nbrs:
                scores[nbr] += seed_w * float(data.get("weight", 1))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [mid for mid, _ in ranked[:k]]

    # ── fusion + retrieve ────────────────────────────────────────────────────

    @staticmethod
    def _rrf_accumulate(scores: dict[MemoryId, float], ranked: list[MemoryId], weight: float) -> None:
        # A zero-weight leg is a TRUE no-op: it must not even seed 0.0-score keys
        # into the pool, or an ABLATED leg (w_graph=0 in the +dense config,
        # w_dense=0 in the +graph config) would still leak its ids into the fused
        # tail when the surviving legs return < k candidates — polluting recall@k.
        if weight == 0.0:
            return
        for r, mid in enumerate(ranked, start=1):
            scores[mid] += weight / (_RRF_K + r)

    def retrieve(self, query: str, k: int) -> list[MemoryId]:
        """Fuse lexical + dense + graph-expansion via weighted RRF and return the
        top-k memory ids. Pulls each leg deeper than k so fusion has material to
        re-order, then truncates."""
        depth = max(k, 50)  # per-leg retrieval depth before fusion

        fts_ranked = self._fts.retrieve(query, depth)
        dense_ranked = self._dense_rank(query, depth)  # [] if dense disabled

        # Seeds for graph expansion: RRF of the two base legs (so the graph walks
        # from the best-agreed memories, not just one leg's view).
        seed_scores: dict[MemoryId, float] = defaultdict(float)
        self._rrf_accumulate(seed_scores, fts_ranked, self.w_fts)
        self._rrf_accumulate(seed_scores, dense_ranked, self.w_dense)
        seeds = [mid for mid, _ in sorted(seed_scores.items(), key=lambda kv: kv[1], reverse=True)]
        # FUSION FIX (ADR-0005): the graph leg's FULL ranking enters the SAME shared
        # fused pool as FTS+dense — NO base-set exclusion. Graph candidates that also
        # appear in a base leg are reinforced; graph-only candidates compete on their
        # own RRF mass. w_graph is a swept attribute so the sparse graph leg gets a
        # genuine shot at the top-k (the prior exclude=base_set + fixed w_graph=0.35
        # made "graph adds nothing" a math artifact rather than a tested result).
        graph_ranked = self._graph_rank(seeds, k=depth)

        # Final weighted RRF over all three legs.
        scores: dict[MemoryId, float] = defaultdict(float)
        self._rrf_accumulate(scores, fts_ranked, self.w_fts)
        self._rrf_accumulate(scores, dense_ranked, self.w_dense)
        self._rrf_accumulate(scores, graph_ranked, self.w_graph)

        fused = sorted(scores.items(), key=lambda kv: (kv[1], -kv[0]), reverse=True)
        return [mid for mid, _ in fused[:k]]

    # ── reporting hooks ───────────────────────────────────────────────────────

    def index_size_bytes(self) -> int:
        """Sum of the dense matrix bytes + FTS index bytes (graph is in-memory
        networkx; we approximate it via node+edge count * a small constant). Non-
        gating per ADR-0001; reported for the storage column."""
        total = 0
        if self._emb is not None:
            total += int(self._emb.nbytes)
        try:
            total += self._fts.index_size_bytes()
        except Exception:
            pass
        if self._graph is not None:
            # rough: ~64 B/node + ~96 B/edge accounting for python object overhead.
            total += self._graph.number_of_nodes() * 64 + self._graph.number_of_edges() * 96
        return total

    def graph_stats(self) -> dict[str, int]:
        """Graph-build statistics for the report. The keyword fields (``edges``,
        ``concepts_*``) describe the prior keyword-co-occurrence fallback; the
        ``typed_*`` fields (slice S4) describe the injected TYPED concept graph + its
        bipartite PPR index. ``typed_edges`` is the headline for the latency-sanity
        check — it MUST be far below the prior keyword prototype's 2,095,624 edges."""
        return {
            "nodes": self._graph.number_of_nodes() if self._graph is not None else 0,
            "edges": self._n_edges,
            "concepts_total": self._n_concepts_total,
            "concepts_kept": self._n_concepts_kept,
            # typed concept graph (S3) + bipartite PPR index (S4):
            "typed_concepts": len(self._cgraph.concepts) if self._cgraph is not None else 0,
            "typed_edges": self._typed_edge_count,
            "bipartite_nodes": self._node_count,
            "bipartite_edges": len(self._bipartite_edges),
        }

    def close(self) -> None:
        self._fts.close()


# Convenience for `run_eval.py --retriever retrievers.hybrid:HybridRetriever`.
