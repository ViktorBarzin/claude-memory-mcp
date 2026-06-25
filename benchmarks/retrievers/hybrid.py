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
   neighbours as a third ranked list. This targets the **multihop** stratum
   (queries needing 2+ memories that share an entity/concept) without re-ranking
   the precise hits the other legs already nail.

Fusion (``retrieval_fusion``)
=============================
Reciprocal Rank Fusion (Cormack et al., 2009): for a document *d* with rank
``r_leg(d)`` (1-based) in a leg's ranked list,

    RRF(d) = Σ_leg  w_leg / (k_rrf + r_leg(d))

with ``k_rrf = 60`` (the standard constant) and per-leg weights. RRF is
score-scale-free (no BM25-vs-cosine calibration), which is why the design floats
"RRF vs CC" and we pick RRF for the prototype. The dense and lexical legs carry
full weight; the graph leg is down-weighted (it is a RECALL extender for multihop,
and the design explicitly flags a possible negative graph prior — so it can add
documents but should not dethrone strong dense/lexical hits). All three weights
are class attributes so the kill-gate analysis can ablate the graph to zero.

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
import time
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

# ── package-relative imports that also work under direct execution ────────────
try:  # pragma: no cover - exercised by both import paths
    from harness.types import Memory, MemoryId
    from retrievers.fts import FtsRetriever
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from harness.types import Memory, MemoryId
    from retrievers.fts import FtsRetriever

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

        # Graph leg state.
        self._graph = None  # networkx.Graph or None
        self._concept_to_mems: dict[str, list[MemoryId]] = {}
        self._mem_concepts: dict[MemoryId, set[str]] = {}
        self._n_concepts_total = 0  # before df pruning, for reporting
        self._n_concepts_kept = 0
        self._n_edges = 0

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

    # ── graph leg ──────────────────────────────────────────────────────────

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

    def _graph_rank(self, seeds: list[MemoryId], exclude: set[MemoryId], k: int) -> list[MemoryId]:
        """From fused seeds, walk 1 hop and rank neighbour memories by accumulated
        edge weight (shared-concept strength), weighted by the seed's own rank so
        higher-confidence seeds pull harder. Returns up to k NEW ids (not in
        ``exclude``)."""
        if self._graph is None or not seeds:
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
                if nbr in exclude:
                    continue
                scores[nbr] += seed_w * float(data.get("weight", 1))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [mid for mid, _ in ranked[:k]]

    # ── fusion + retrieve ────────────────────────────────────────────────────

    @staticmethod
    def _rrf_accumulate(scores: dict[MemoryId, float], ranked: list[MemoryId], weight: float) -> None:
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
        base_set = set(seeds)
        graph_ranked = self._graph_rank(seeds, exclude=base_set, k=depth)

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
        return {
            "nodes": self._graph.number_of_nodes() if self._graph is not None else 0,
            "edges": self._n_edges,
            "concepts_total": self._n_concepts_total,
            "concepts_kept": self._n_concepts_kept,
        }

    def close(self) -> None:
        self._fts.close()


# Convenience for `run_eval.py --retriever retrievers.hybrid:HybridRetriever`.
