from typing import Optional

from pydantic import BaseModel, Field


class MemoryStore(BaseModel):
    content: str
    category: str = "facts"
    tags: str = ""
    expanded_keywords: str = ""
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    force_sensitive: bool = False


class MemoryRecall(BaseModel):
    context: str
    expanded_query: str = ""
    category: Optional[str] = None
    sort_by: str = "importance"
    limit: int = 10


class MemoryResponse(BaseModel):
    id: int
    category: str
    importance: float


class SecretResponse(BaseModel):
    id: int
    content: str
    source: str  # "vault", "encrypted", "plaintext"
