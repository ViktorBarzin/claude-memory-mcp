"""Prometheus instrumentation for the memory service (ADR-0007 observability).

Everything here answers one question — "is memory delivery working better than
before?" — with server-side truth. The counters mirror the failure classes the
2026-07 session audit found (silent recall 5xx, importance-swamped ranking,
unfollowed pointers) and the new behaviours that replaced them (dense leg,
supersedes redirects, resolved-by attaches, bounded writes).

Metric names are stable API — Grafana panels and alert rules reference them.
"""
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

#: Recall traffic + latency, split by serving surface (REST endpoint vs FastMCP
#: tool) and requested sort. Latency buckets sized for the hook's 6s timeout.
RECALL_REQUESTS = Counter(
    "memory_recall_requests_total", "Recall requests served", ["surface", "sort"]
)
RECALL_ERRORS = Counter(
    "memory_recall_errors_total", "Recall requests that raised", ["surface"]
)
RECALL_LATENCY = Histogram(
    "memory_recall_seconds",
    "End-to-end recall handler latency",
    ["surface"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0),
)

#: Dense-leg contribution — the live "is semantic recall adding value" signal.
#: dense_candidates: recalls where the dense leg returned any candidate rows;
#: dense_only_top5: served results in the top 5 that ONLY the dense leg found
#: (a lexical-only deployment would have missed them).
DENSE_CANDIDATE_RECALLS = Counter(
    "memory_recall_dense_candidate_recalls_total",
    "Recalls where the dense leg contributed candidate rows",
)
DENSE_ONLY_TOP5 = Counter(
    "memory_recall_dense_only_top5_total",
    "Served top-5 results only the dense leg surfaced",
)

#: ADR-0007 link semantics firing in production. A redirect means stale
#: vocabulary landed on current truth; an attach means a symptom query got its
#: root cause without a second lookup.
LINK_REDIRECTS = Counter(
    "memory_link_redirects_total", "supersedes-redirects applied to served results"
)
LINK_ATTACHES = Counter(
    "memory_link_attaches_total", "resolved-by attachments added to responses"
)

#: Write-side health.
STORES = Counter("memory_store_total", "Memory stores", ["surface", "outcome"])
BOUND_REJECTS = Counter(
    "memory_bound_rejects_total",
    "Writes rejected by the 1,400-char Memory bound (per surface)",
    ["surface"],
)
LINKS_CREATED = Counter(
    "memory_links_created_total", "Typed Memory links created", ["link_type"]
)
EMBED_WRITE = Counter(
    "memory_embed_write_total", "Embed-on-write outcomes", ["status"]
)

#: Backfill / embed-lag visibility: set at scrape time by the /metrics handler.
EMBEDDINGS_PENDING = Gauge(
    "memory_embeddings_pending",
    "Non-sensitive live memories still lacking an embedding",
)
MEMORIES_TOTAL = Gauge("memory_entries_total", "Live (non-deleted) memories")


def exposition() -> tuple[bytes, str]:
    """Render the default registry in Prometheus text format."""
    return generate_latest(), CONTENT_TYPE_LATEST
