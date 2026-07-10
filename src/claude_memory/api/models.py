from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

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


class MemoryStore(BaseModel):
    content: str
    category: str = "facts"
    tags: str = Field(default="", max_length=500)
    expanded_keywords: str = Field(default="", max_length=500)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    force_sensitive: bool = False

    @field_validator("content")
    @classmethod
    def _content_within_bound(cls, v: str) -> str:
        return validate_content_bound(v)


class MemoryRecall(BaseModel):
    context: str
    expanded_query: str = ""
    category: Optional[str] = None
    # Default flipped from "importance" to "relevance" (ADR-0005 amendment,
    # 2026-07-09): importance-sorted recall was the largest measured rediscovery
    # driver; sort_by="importance" stays available explicitly.
    sort_by: Literal["importance", "relevance", "recency"] = "relevance"
    # Default to a small top-N so recall returns the most relevant matches, not
    # the whole store. Ceiling stays high for callers that explicitly want more.
    limit: int = Field(default=30, ge=1, le=10000)


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


class MemoryUpdate(BaseModel):
    content: Optional[str] = None
    tags: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    expanded_keywords: Optional[str] = None

    @field_validator("content")
    @classmethod
    def _content_within_bound(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        return validate_content_bound(v)
