"""Shared recall + embed-on-write helpers for the Postgres entry points.

Both recall entry points in ``api/app.py`` — the REST ``recall_memories`` and the
FastMCP ``memory_recall`` tool — retrieve through the single :func:`_fused_recall`
helper here, and both store paths (REST ``store_memory`` + FastMCP ``memory_store``)
schedule embed-on-write through :func:`schedule_embedding`. Factoring the retrieval
and write-side embedding into one place each is the anti-drift guarantee of S8: the
two near-identical SQL bodies that previously lived inline can no longer diverge.

Degrade contract (challenger must_fix #3, ADR-0002/0004)
--------------------------------------------------------
When BOTH ``MEMORY_EMBEDDINGS_ENABLED`` and ``MEMORY_GRAPH_ENABLED`` are off (the
default, and the SQLite-only / pre-pgvector posture), :func:`_fused_recall` is a
**true no-op**: it runs the EXACT current ``ts_rank`` lexical SQL (the additive
``ts_rank*0.7 + importance*0.3`` blend, the ``plainto_tsquery`` AND-match, and the
relevance-bounded OR-broaden fallback) and returns the rows verbatim. It is NOT an
RRF collapse — RRF-rank × multiplicative-importance reorders relative to the additive
blend, so collapsing to a single-leg RRF would silently change default-sort ordering
and break the existing recall tests. Fusion engages only when ≥1 leg flag is on, and
that ordering change is acknowledged as flag-gated, never sold as a no-op.

Privacy (ADR-0003)
------------------
Sensitive rows are never embedded (:func:`schedule_embedding` refuses them) and never
appear in the dense leg (the dense CTE filters ``embedding IS NOT NULL`` and sensitive
rows keep a NULL embedding). The embedding column lives ONLY in Postgres — ``sync.py``
is untouched and the SQLite cache stays purely lexical.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from claude_memory.embeddings import Embedder, select_embedder

logger = logging.getLogger(__name__)

#: Strong references to in-flight embed-on-write tasks. ``asyncio.create_task`` only holds
#: a weak reference, so without this a fire-and-forget embed task could be garbage-collected
#: mid-flight; we keep the task here and drop it in the done-callback.
_background_tasks: "set[asyncio.Task[None]]" = set()

# ── Feature flags ────────────────────────────────────────────────────────────
#: Gates the dense (semantic) recall leg and embed-on-write. Default OFF.
EMBEDDINGS_FLAG_ENV = "MEMORY_EMBEDDINGS_ENABLED"
#: Gates the concept-graph recall leg (phase-2). Default OFF.
GRAPH_FLAG_ENV = "MEMORY_GRAPH_ENABLED"

# ── Fusion / lexical constants ───────────────────────────────────────────────
#: RRF constant (Cormack/Clarke/Buettcher 2009); 60 is the canonical default and
#: matches the offline harness (``benchmarks/retrievers/hybrid.py``).
_RRF_K = 60

#: Per-leg RRF weights. Lexical + dense each get full weight; the graph leg's weight is
#: swept offline (the production graph leg lands in a later slice).
_W_LEX = 1.0
_W_DENSE = 1.0

#: How many candidates the dense ANN leg pulls before fusion (mirrors the offline depth).
_DENSE_LIMIT = 50

#: OR-broadening fallback: when the precise AND-match is sparse we widen to an OR-match
#: to fill results, but only with rows whose relevance (ts_rank) clears this floor. Below
#: it a row merely contains one query word incidentally — noise. Shared by both entry
#: points so the lexical behaviour cannot drift between them.
OR_BROADEN_MIN_RANK = 0.01

# ── Link semantics (ADR-0007) ────────────────────────────────────────────────
#: Max hops when walking a supersedes chain — shared by the recall redirect walk
#: here and the link-create cycle check in ``api/app.py`` so the two bounds
#: cannot drift.
SUPERSEDES_DEPTH_CAP = 10

#: Cap on resolved-by auto-attachments per recall response. Attachments arrive
#: BEYOND the caller's limit, so an unbounded fan-out would flood the 5-slot
#: hook injection — the truncation problem ADR-0007 exists to fix.
MAX_RESOLVED_BY_ATTACHMENTS = 3


def _flag_on(env_name: str) -> bool:
    """Interpret an on/off feature-flag env var. Truthy: 1/true/yes/on (case-insensitive)."""
    return os.environ.get(env_name, "").strip().lower() in {"1", "true", "yes", "on"}


def embeddings_enabled() -> bool:
    """Whether the dense leg + embed-on-write are enabled (read live, default off)."""
    return _flag_on(EMBEDDINGS_FLAG_ENV)


def graph_enabled() -> bool:
    """Whether the concept-graph leg is enabled (read live, default off)."""
    return _flag_on(GRAPH_FLAG_ENV)


def _lexical_score_expr(sort_by: str) -> str:
    """The EXACT current hybrid relevance blend, keyed by ``sort_by``.

    Verbatim from the inline recall bodies: the additive ts_rank/importance blend that
    must be preserved byte-for-byte when both leg flags are off.
    """
    if sort_by == "importance":
        return "(ts_rank(search_vector, query) * 0.4 + importance * 0.6)"
    return "(ts_rank(search_vector, query) * 0.7 + importance * 0.3)"


async def _lexical_recall(
    conn: Any,
    *,
    user_id: str,
    query_text: str,
    sort_by: str,
    category: Optional[str],
    limit: int,
) -> list[Any]:
    """Run the CURRENT lexical recall SQL verbatim and return the ordered rows.

    This is the flags-off no-op path and leg 1 of the fused path. It reproduces, exactly,
    the AND-match + relevance-bounded OR-broaden fallback that previously lived inline in
    both ``recall_memories`` and ``memory_recall`` — same columns, same ordering, same
    OR-broaden floor — so flags-off behaviour is byte-identical to today.
    """
    hybrid_score = _lexical_score_expr(sort_by)
    order_clause = f"{hybrid_score} DESC"
    if sort_by == "recency":
        order_clause = "created_at DESC"

    category_filter = ""
    params: list[Any] = [user_id, query_text, limit]
    if category:
        category_filter = "AND category = $4"
        params.append(category)

    rows = await conn.fetch(
        f"""
        SELECT id, content, category, tags, importance, is_sensitive,
               ts_rank(search_vector, query) AS rank,
               created_at, updated_at, user_id AS owner,
               CASE WHEN user_id = $1 THEN NULL ELSE user_id END AS shared_by
        FROM memories, plainto_tsquery('english', $2) query
        WHERE deleted_at IS NULL
          AND (search_vector @@ query OR $2 = '')
          {category_filter}
        ORDER BY {order_clause}
        LIMIT $3
        """,
        *params,
    )

    all_rows = list(rows)

    # If AND-match returned too few results, broaden to OR-match.
    if len(all_rows) < limit and query_text:
        words = query_text.split()
        if len(words) > 1:
            or_tsquery = " | ".join(w for w in words if w)
            or_params: list[Any] = [user_id, or_tsquery, limit]
            or_cat_filter = ""
            if category:
                or_cat_filter = "AND category = $4"
                or_params.append(category)
            seen_ids = {r["id"] for r in all_rows}
            or_rows = await conn.fetch(
                f"""
                SELECT id, content, category, tags, importance, is_sensitive,
                       ts_rank(search_vector, query) AS rank,
                       created_at, updated_at, user_id AS owner,
                       CASE WHEN user_id = $1 THEN NULL ELSE user_id END AS shared_by
                FROM memories, to_tsquery('english', $2) query
                WHERE deleted_at IS NULL
                  AND search_vector @@ query
                  AND ts_rank(search_vector, query) > {OR_BROADEN_MIN_RANK}
                  {or_cat_filter}
                ORDER BY ts_rank(search_vector, query) DESC
                LIMIT $3
                """,
                *or_params,
            )
            all_rows = all_rows + [r for r in or_rows if r["id"] not in seen_ids]
            all_rows = all_rows[:limit]

    return all_rows


def _vector_literal(vec: list[float]) -> str:
    """Render a float vector as a pgvector/halfvec text literal (``[a,b,c]``)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


async def _dense_recall(
    conn: Any,
    *,
    user_id: str,
    qvec: list[float],
    category: Optional[str],
) -> list[Any]:
    """Run the dense ANN leg: nearest neighbours by cosine ``<=>`` over the HNSW index.

    Sensitive rows have a NULL embedding, so ``embedding IS NOT NULL`` is also the hard
    privacy gate that keeps them out of the dense leg entirely (ADR-0003). Selects the
    same projection as the lexical leg (``rank`` is left as the dense rank position,
    filled by the caller) so a dense-only candidate carries the full row fields the
    response serializers expect.
    """
    category_filter = ""
    # $1 user_id (owner/shared_by CASE), $2 qvec, $3 dense limit, [$4 category]
    params: list[Any] = [user_id, _vector_literal(qvec), _DENSE_LIMIT]
    if category:
        category_filter = "AND category = $4"
        params.append(category)

    return list(
        await conn.fetch(
            f"""
            SELECT id, content, category, tags, importance, is_sensitive,
                   0.0::float4 AS rank,
                   created_at, updated_at, user_id AS owner,
                   CASE WHEN user_id = $1 THEN NULL ELSE user_id END AS shared_by
            FROM memories
            WHERE embedding IS NOT NULL
              AND deleted_at IS NULL
              {category_filter}
            ORDER BY embedding <=> $2::halfvec
            LIMIT $3
            """,
            *params,
        )
    )


def _fuse(
    lexical_rows: list[Any],
    dense_rows: list[Any],
    *,
    limit: int,
) -> list[Any]:
    """Weighted-RRF fuse the lexical + dense legs, with importance as a POST-fusion multiplier.

    Each leg contributes ``weight / (_RRF_K + rank)`` to a memory's fused score; the
    SHARED candidate pool means a dense-only hit competes on its own RRF mass. importance
    is applied AFTER fusion as ``final = fused * (0.7 + 0.3*importance)`` — never fused as a
    leg (ADR-0005). Returns the original row objects (lexical preferred for full fields),
    reordered by ``final`` and capped to ``limit``. The sort is stable, so equal-score ids
    keep their first-seen (lexical-then-dense) order.
    """
    scores: dict[Any, float] = {}
    row_by_id: dict[Any, Any] = {}

    def accumulate(rows: list[Any], weight: float) -> None:
        for rank, row in enumerate(rows, start=1):
            mid = row["id"]
            scores[mid] = scores.get(mid, 0.0) + weight / (_RRF_K + rank)
            # Prefer the lexical row object (richer, already present); only fill from the
            # dense leg for ids the lexical leg never returned.
            if mid not in row_by_id:
                row_by_id[mid] = row

    accumulate(lexical_rows, _W_LEX)
    accumulate(dense_rows, _W_DENSE)

    def final_score(mid: Any) -> float:
        row = row_by_id[mid]
        importance = float(row["importance"]) if row["importance"] is not None else 0.0
        return scores[mid] * (0.7 + 0.3 * importance)

    ordered_ids = sorted(scores, key=final_score, reverse=True)
    return [row_by_id[mid] for mid in ordered_ids[:limit]]


async def _fused_recall(
    conn: Any,
    *,
    user_id: str,
    query_text: str,
    sort_by: str,
    category: Optional[str],
    limit: int,
    pool: Any = None,  # noqa: ARG001 - kept for call-site symmetry / future per-leg conns
    embedder: Optional[Embedder] = None,
) -> list[Any]:
    """The ONE recall helper both Postgres entry points call.

    Flags off (default) ⇒ verbatim lexical no-op (:func:`_lexical_recall`). With the
    embeddings flag on, the dense leg joins a shared weighted-RRF pool with importance as
    a post-fusion multiplier. The graph leg is gated by :func:`graph_enabled` and lands in
    a later slice; until then a graph-only flag still engages fusion (lexical leg only),
    which is harmless. Callers keep their own response-shaping loops; this returns the
    ordered rows.
    """
    lexical_rows = await _lexical_recall(
        conn,
        user_id=user_id,
        query_text=query_text,
        sort_by=sort_by,
        category=category,
        limit=limit,
    )

    if not (embeddings_enabled() or graph_enabled()):
        # TRUE no-op: byte-identical to the current lexical recall.
        return lexical_rows

    dense_rows: list[Any] = []
    if embeddings_enabled() and query_text:
        # Degrade-safe dense leg (asymmetry fix vs the write path). embed_query is a
        # BLOCKING call — a network round-trip for Voyage, a model load+inference for bge —
        # so it runs OFF the event loop via to_thread (a hung hosted API or slow local model
        # must not stall recall). ANY embedder/query failure (Voyage 5xx/timeout, a
        # sentence-transformers ImportError/model-load error, an HNSW/halfvec query error) is
        # swallowed: lexical_rows was ALREADY fetched above, and _fuse(lexical, []) reduces to
        # lexical order, so a dense-backend outage degrades to lexical-only — never a 500.
        # Recall runs on every prompt; the lexical leg must always be returnable.
        try:
            emb = embedder or select_embedder()
            qvec = await asyncio.to_thread(emb.embed_query, query_text)
            dense_rows = await _dense_recall(conn, user_id=user_id, qvec=qvec, category=category)
        except Exception as exc:  # noqa: BLE001 - degrade to lexical on any dense-leg failure
            logger.warning("dense recall leg failed, falling back to lexical: %s", exc)

    # graph leg: phase-2 (MEMORY_GRAPH_ENABLED) — production graph module lands later.
    return _fuse(lexical_rows, dense_rows, limit=limit)


# ── ADR-0007 link semantics: recall post-processing ─────────────────────────


async def _fetch_edges_touching(conn: Any, user_id: str, ids: list[int]) -> list[Any]:
    """One batched query for every edge of ``ids`` (either direction, all four types)."""
    return list(
        await conn.fetch(
            """
            SELECT id, src_id, dst_id, link_type, created_at
            FROM memory_links
            WHERE user_id = $1 AND (src_id = ANY($2::int[]) OR dst_id = ANY($2::int[]))
            """,
            user_id,
            ids,
        )
    )


async def apply_link_semantics(
    conn: Any,
    *,
    user_id: str,
    rows: list[Any],
    max_attachments: int = MAX_RESOLVED_BY_ATTACHMENTS,
) -> list[dict[str, Any]]:
    """Post-ranking step giving each ADR-0007 link type its Recall behaviour.

    Runs AFTER retrieval on the already-ranked ``rows`` (so every sort mode and
    every fusion path gets it) and returns one result dict per served memory:
    ``{"row": <record>, "redirected_from": id|None, "attached_via": {...}|None,
    "links": [{"type", "dir", "id"}, ...]}``. The caller keeps its own
    response-shaping (sensitive redaction applies uniformly to every served row).

    (a) **supersedes redirects**: a ranked memory with an INCOMING supersedes
        edge is replaced by its chain head (newest superseder at each fan-in;
        visited set + depth cap), at the same rank, marked ``redirected_from``;
        if the head also ranked, the duplicate folds into the best rank.
    (b) **resolved-by auto-attaches**: each surviving result's outgoing
        resolved-by targets are APPENDED as extra results (never consuming the
        caller's limit), marked ``attached_via``, capped per response.
    (c) **links summary**: every served result lists its edges, all four types,
        both directions.

    Edge lookups are batched — ONE touching-query for the ranked set, plus one
    per supersedes hop / new-node group — never per-row N+1. The no-links common
    case costs exactly one query.
    """
    if not rows:
        return []

    ids0 = [r["id"] for r in rows]
    row_by_id: dict[int, Any] = {r["id"]: r for r in rows}

    edges_seen: set[int] = set()
    incoming_sup: dict[int, list[Any]] = {}  # dst -> incoming supersedes edges
    outgoing_res: dict[int, list[Any]] = {}  # src -> outgoing resolved-by edges
    touching: dict[int, list[Any]] = {}  # node -> every edge touching it

    def _ingest(fetched: list[Any]) -> None:
        for e in fetched:
            if e["id"] in edges_seen:
                continue
            edges_seen.add(e["id"])
            if e["link_type"] == "supersedes":
                incoming_sup.setdefault(e["dst_id"], []).append(e)
            elif e["link_type"] == "resolved-by":
                outgoing_res.setdefault(e["src_id"], []).append(e)
            touching.setdefault(e["src_id"], []).append(e)
            touching.setdefault(e["dst_id"], []).append(e)

    _ingest(await _fetch_edges_touching(conn, user_id, ids0))

    # Pull the supersedes chains hop by hop (batched per hop, capped) so each
    # per-memory walk below has its full chain in hand.
    chain_nodes_fetched: set[int] = set(ids0)
    for _ in range(SUPERSEDES_DEPTH_CAP):
        frontier = sorted(
            {e["src_id"] for es in incoming_sup.values() for e in es} - chain_nodes_fetched
        )
        if not frontier:
            break
        _ingest(
            list(
                await conn.fetch(
                    """
                    SELECT id, src_id, dst_id, link_type, created_at
                    FROM memory_links
                    WHERE user_id = $1 AND link_type = 'supersedes' AND dst_id = ANY($2::int[])
                    """,
                    user_id,
                    frontier,
                )
            )
        )
        chain_nodes_fetched.update(frontier)

    def _chain_head(start: int) -> int:
        """Follow incoming supersedes edges to the newest chain head (bounded)."""
        current = start
        visited = {start}
        for _ in range(SUPERSEDES_DEPTH_CAP):
            sups = incoming_sup.get(current)
            if not sups:
                break
            newest = max(sups, key=lambda e: (e["created_at"], e["id"]))
            nxt = newest["src_id"]
            if nxt in visited:  # pre-existing cycle in data: stop where we are
                break
            visited.add(nxt)
            current = nxt
        return current

    # (a) substitute heads in place, then dedupe keeping the best (earliest) rank.
    slots: list[tuple[int, Optional[int]]] = []
    for r in rows:
        head = _chain_head(r["id"])
        slots.append((head, r["id"] if head != r["id"] else None))
    served: set[int] = set()
    final_slots: list[tuple[int, Optional[int]]] = []
    for fid, redirected_from in slots:
        if fid in served:
            continue
        served.add(fid)
        final_slots.append((fid, redirected_from))

    # Heads that never ranked need their edges (links summary + their own
    # resolved-by fan-out) — one batched query.
    head_ids_new = [fid for fid, _ in final_slots if fid not in row_by_id]
    if head_ids_new:
        _ingest(await _fetch_edges_touching(conn, user_id, head_ids_new))

    # (b) resolved-by attachments from every SURVIVING result, capped per response.
    final_ids = {fid for fid, _ in final_slots}
    attachments: list[tuple[int, int]] = []  # (target_id, source_id)
    attached: set[int] = set()
    for fid, _ in final_slots:
        if len(attachments) >= max_attachments:
            break
        for e in sorted(outgoing_res.get(fid, ()), key=lambda e: (e["created_at"], e["id"])):
            if len(attachments) >= max_attachments:
                break
            target = e["dst_id"]
            if target in final_ids or target in attached:
                continue
            attachments.append((target, fid))
            attached.add(target)

    # Attach targets outside every set fetched so far still need their edges.
    edges_covered = set(ids0) | set(head_ids_new)
    attach_ids_uncovered = [t for t, _ in attachments if t not in edges_covered]
    if attach_ids_uncovered:
        _ingest(await _fetch_edges_touching(conn, user_id, attach_ids_uncovered))

    # One row fetch for every served id that wasn't in the ranked set. Uses the
    # same projection as the retrieval legs (rank 0.0: these rows earned their
    # slot via a link, not a lexical score).
    missing_row_ids = [
        mid
        for mid in [fid for fid, _ in final_slots] + [t for t, _ in attachments]
        if mid not in row_by_id
    ]
    if missing_row_ids:
        fetched_rows = await conn.fetch(
            """
            SELECT id, content, category, tags, importance, is_sensitive,
                   0.0::float4 AS rank,
                   created_at, updated_at, user_id AS owner,
                   CASE WHEN user_id = $1 THEN NULL ELSE user_id END AS shared_by
            FROM memories
            WHERE deleted_at IS NULL AND id = ANY($2::int[])
            """,
            user_id,
            missing_row_ids,
        )
        for r in fetched_rows:
            row_by_id[r["id"]] = r

    def _links_summary(mid: int) -> list[dict[str, Any]]:
        summary = []
        for e in sorted(touching.get(mid, ()), key=lambda e: (e["created_at"], e["id"])):
            if e["src_id"] == mid:
                summary.append({"type": e["link_type"], "dir": "out", "id": e["dst_id"]})
            else:
                summary.append({"type": e["link_type"], "dir": "in", "id": e["src_id"]})
        return summary

    results: list[dict[str, Any]] = []
    for fid, redirected_from in final_slots:
        row = row_by_id.get(fid)
        if row is None:
            # The head row is gone (deleted after linking). Fall back to the
            # superseded original — it ranked; better stale than a dropped slot.
            if redirected_from is not None and redirected_from in row_by_id:
                results.append(
                    {
                        "row": row_by_id[redirected_from],
                        "redirected_from": None,
                        "attached_via": None,
                        "links": _links_summary(redirected_from),
                    }
                )
            continue
        results.append(
            {
                "row": row,
                "redirected_from": redirected_from,
                "attached_via": None,
                "links": _links_summary(fid),
            }
        )

    for target, source in attachments:
        row = row_by_id.get(target)
        if row is None:
            continue  # dangling resolved-by to a deleted memory
        results.append(
            {
                "row": row,
                "redirected_from": None,
                "attached_via": {"type": "resolved-by", "source": source},
                "links": _links_summary(target),
            }
        )

    return results


async def _embed_and_persist(pool: Any, memory_id: int, content: str, embedder: Embedder) -> None:
    """Embed ``content`` off the hot path and persist it to ``memories.embedding``.

    Runs the CPU-bound embedder in a threadpool, then UPDATEs the row on its OWN pooled
    connection. Best-effort: any failure is logged, never raised (the store already
    succeeded; a missing embedding just means that row falls back to lexical recall).
    """
    try:
        vec = await asyncio.to_thread(embedder.embed_document, content, is_sensitive=False)
        if vec is None:  # defensive — sensitive rows are filtered before scheduling
            return
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE memories SET embedding = $1 WHERE id = $2",
                _vector_literal(vec),
                memory_id,
            )
    except Exception as exc:  # noqa: BLE001 - never break the store on an embed failure
        logger.warning("embed-on-write failed for memory %s: %s", memory_id, exc)


def schedule_embedding(
    pool: Any,
    memory_id: int,
    content: str,
    *,
    is_sensitive: bool,
) -> "Optional[asyncio.Task[None]]":
    """Schedule embed-on-write for a freshly stored memory, OFF the hot path.

    Returns immediately (the store response is never blocked). Returns ``None`` — i.e.
    schedules nothing — when the embeddings flag is off OR the row is sensitive (sensitive
    content is never embedded; ADR-0003). Otherwise fires an ``asyncio`` task and returns
    it (so the event loop tracks it and tests can await it).
    """
    if is_sensitive or not embeddings_enabled():
        return None
    embedder = select_embedder()
    task = asyncio.create_task(_embed_and_persist(pool, memory_id, content, embedder))
    # Hold a strong ref until the task finishes so it can't be GC'd mid-flight.
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task
