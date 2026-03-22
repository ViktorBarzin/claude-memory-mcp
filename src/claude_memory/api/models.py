from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


MAX_MEMORY_CHARS = 800


class MemoryStore(BaseModel):
    content: str = Field(..., max_length=MAX_MEMORY_CHARS)
    category: str = "facts"
    tags: str = Field(default="", max_length=500)
    expanded_keywords: str = Field(default="", max_length=500)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    force_sensitive: bool = False


class MemoryRecall(BaseModel):
    context: str
    expanded_query: str = ""
    category: Optional[str] = None
    sort_by: Literal["importance", "relevance", "recency"] = "importance"
    limit: int = Field(default=10, ge=1, le=500)


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
    content: Optional[str] = Field(None, max_length=MAX_MEMORY_CHARS)
    tags: Optional[str] = None
    importance: Optional[float] = Field(None, ge=0.0, le=1.0)
    expanded_keywords: Optional[str] = None
