from typing import Any, Literal, Optional, get_args

from pydantic import BaseModel, Field, JsonValue, field_validator

# ── ADR-0007: typed Memory→Memory links ──────────────────────────────────────
#: The closed link-type enum. Each type has defined Recall behaviour
#: (CONTEXT.md "Link"): supersedes redirects, resolved-by auto-attaches,
#: part-of / see-also are pointer-only. No open vocabulary (the category-drift
#: lesson).
LinkType = Literal["part-of", "supersedes", "see-also", "resolved-by"]
LINK_TYPES: tuple[str, ...] = get_args(LinkType)

# ── ADR-0007: the Memory content bound ───────────────────────────────────────
#: Hard bound on Memory content, in UNICODE CHARACTERS (not bytes). Derived from
#: the delivery budget: the recall hook injects 5 results under a hard 8KB cap,
#: so 8KB/5 − ~150 chars metadata ≈ 1,400 chars arriving whole (ADR-0007).
MEMORY_CONTENT_MAX_CHARS = 1400

#: The exact 422 guidance — teaches the split-into-hub+parts pattern at the
#: point of failure. The CLI pre-validates with the same message.
CONTENT_BOUND_MESSAGE = (
    "content exceeds the 1,400-char Memory bound; split into a self-contained "
    "hub Memory plus part-of linked detail Memories (see ADR-0007)"
)


def validate_content_bound(content: str) -> str:
    """Reject content over the ADR-0007 bound, counting unicode chars (len), not bytes."""
    if len(content) > MEMORY_CONTENT_MAX_CHARS:
        raise ValueError(CONTENT_BOUND_MESSAGE)
    return content


# ── ADR-0007: category canonicalization on write ─────────────────────────────
#: The closed canonical category set. Free vocabulary drifted into
#: singular/plural twins that hid 97% of gotchas from exact-match filters, so
#: writes are canonicalized server-side (the same drift lesson as link types).
CANONICAL_CATEGORIES = frozenset({
    "facts", "decisions", "projects", "preferences", "gotchas", "references",
    "infrastructure", "runbook", "lessons", "operations", "post-mortems",
    "people", "incidents", "feedback", "process", "architecture", "sessions",
})

#: Known drift twins, folded silently on write.
CATEGORY_FOLD_MAP = {
    "gotcha": "gotchas",
    "project": "projects",
    "reference": "references",
    "infra": "infrastructure",
    "bug": "gotchas",
    "incident": "incidents",
    "procedures": "runbook",
}


#: The canonical set as a sorted list, for the JSON-Schema ``enum`` every category field
#: carries. A closed enum is what the curated /muse/openapi.json owes a connector builder:
#: link_type and sort_by already render as enums there, and a bare string next to them
#: reads as an open vocabulary when the server rejects anything outside this set with a
#: 422. The API is slightly MORE permissive than the enum — it also folds the drift twins
#: in CATEGORY_FOLD_MAP — which is the safe direction: every value listed here is accepted.
CATEGORY_ENUM: list[str] = sorted(CANONICAL_CATEGORIES)


def _category_schema(*, optional: bool) -> dict[str, JsonValue]:
    """The JSON-Schema extra that declares the closed category vocabulary on a field.

    ``optional`` adds null, because a field typed ``Optional[str]`` renders as an
    ``anyOf`` carrying null and a plain enum beside it would contradict that.
    """
    values: list[JsonValue] = [*CATEGORY_ENUM]
    if optional:
        values.append(None)
    return {"enum": values}


def canonicalize_category(category: str) -> str:
    """Fold a written category to its canonical form, or raise listing the allowed set.

    Case/whitespace are normalized first (``Facts`` → ``facts``), then the drift
    fold map applies silently; anything still outside the canonical set is a
    ``ValueError`` (→ 422 on the REST paths) naming every allowed value.
    """
    normalized = category.strip().lower()
    folded = CATEGORY_FOLD_MAP.get(normalized, normalized)
    if folded not in CANONICAL_CATEGORIES:
        allowed = ", ".join(sorted(CANONICAL_CATEGORIES))
        raise ValueError(f"category {category!r} is not canonical; allowed: {allowed}")
    return folded


class MemoryStore(BaseModel):
    content: str
    category: str = Field(default="facts", json_schema_extra=_category_schema(optional=False))
    tags: str = Field(default="", max_length=500)
    expanded_keywords: str = Field(default="", max_length=500)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    force_sensitive: bool = False

    @field_validator("content")
    @classmethod
    def _content_within_bound(cls, v: str) -> str:
        return validate_content_bound(v)

    @field_validator("category")
    @classmethod
    def _category_canonical(cls, v: str) -> str:
        return canonicalize_category(v)


class MemoryRecall(BaseModel):
    context: str
    expanded_query: str = ""
    # ``None`` is in the enum because the field is optional and omitting the filter is the
    # common case; a plain enum beside an ``anyOf`` carrying null would contradict it.
    category: Optional[str] = Field(default=None, json_schema_extra=_category_schema(optional=True))
    # Default flipped from "importance" to "relevance" (ADR-0005 amendment,
    # 2026-07-09): importance-sorted recall was the largest measured rediscovery
    # driver; sort_by="importance" stays available explicitly.
    sort_by: Literal["importance", "relevance", "recency"] = "relevance"
    # Default to a small top-N so recall returns the most relevant matches, not
    # the whole store. Ceiling stays high for callers that explicitly want more.
    limit: int = Field(default=30, ge=1, le=10000)

    @field_validator("category")
    @classmethod
    def _category_canonical(cls, v: Optional[str]) -> Optional[str]:
        """Fold and validate the filter exactly as the write paths fold and validate.

        Recall matches the column with ``AND category = $4`` (recall.py), so an unfolded
        ``Gotcha`` and an invented ``banana`` both returned 200 and zero rows — a filter
        that silently matches nothing reads as "no memories" rather than as a mistake,
        while the same two values are folded and rejected on store and update. An empty
        or blank string keeps meaning "no filter", which is what the SQL already did with
        it, so only a non-blank value is canonicalized.
        """
        if v is None or not v.strip():
            return None
        return canonicalize_category(v)


class MemoryResponse(BaseModel):
    id: int
    category: str
    importance: float


class SecretResponse(BaseModel):
    id: int
    content: str
    source: str  # "vault", "encrypted", "plaintext"


class SyncResponse(BaseModel):
    memories: list[dict[str, Any]]
    server_time: str


class ShareMemory(BaseModel):
    shared_with: str = Field(..., min_length=1, max_length=100)
    permission: Literal["read", "write"] = "read"


class ShareTag(BaseModel):
    tag: str = Field(..., min_length=1, max_length=100)
    shared_with: str = Field(..., min_length=1, max_length=100)
    permission: Literal["read", "write"] = "read"


class UnshareTag(BaseModel):
    tag: str = Field(..., min_length=1, max_length=100)
    shared_with: str = Field(..., min_length=1, max_length=100)


class LinkCreate(BaseModel):
    target_id: int
    link_type: LinkType


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    category: Optional[str] = Field(default=None, json_schema_extra=_category_schema(optional=True))
    tags: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    expanded_keywords: Optional[str] = None

    @field_validator("content")
    @classmethod
    def _content_within_bound(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return validate_content_bound(v)

    @field_validator("category")
    @classmethod
    def _category_canonical(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return canonicalize_category(v)
