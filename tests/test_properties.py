"""Property-based tests for Claude Memory models."""

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from claude_memory.api.models import (
    CANONICAL_CATEGORIES,
    MEMORY_CONTENT_MAX_CHARS,
    MemoryStore,
    MemoryRecall,
    ShareMemory,
)


# Strategy for valid MemoryStore (canonical categories only — ADR-0007 rejects drift)
valid_memory_store = st.builds(
    MemoryStore,
    content=st.text(min_size=1, max_size=800),
    category=st.sampled_from(sorted(CANONICAL_CATEGORIES)),
    tags=st.text(max_size=500),
    expanded_keywords=st.text(max_size=500),
    importance=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)


@given(mem=valid_memory_store)
@settings(max_examples=50)
def test_roundtrip_memory_store(mem):
    """Any valid MemoryStore can be serialized and deserialized identically."""
    data = mem.model_dump()
    restored = MemoryStore(**data)
    assert restored.content == mem.content
    assert restored.importance == mem.importance
    assert restored.tags == mem.tags


@given(content=st.text(min_size=501, max_size=MEMORY_CONTENT_MAX_CHARS))
@settings(max_examples=20)
def test_content_up_to_bound_accepted(content):
    """Content over 500 chars is accepted up to the ADR-0007 bound (1,400 chars)."""
    mem = MemoryStore(content=content)
    assert len(mem.content) <= MEMORY_CONTENT_MAX_CHARS


@given(content=st.text(min_size=MEMORY_CONTENT_MAX_CHARS + 1, max_size=3000))
@settings(max_examples=20)
def test_content_over_bound_rejected(content):
    """Content over the 1,400-char bound is rejected with the split guidance (ADR-0007)."""
    try:
        MemoryStore(content=content)
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        assert "1,400" in str(e)


@given(
    content=st.text(
        # Cyrillic block: every character is 2 UTF-8 bytes, so any text over 700
        # chars exceeds 1,400 BYTES while staying within the 1,400-CHAR bound.
        alphabet=st.characters(min_codepoint=0x0410, max_codepoint=0x044F),
        min_size=701,
        max_size=MEMORY_CONTENT_MAX_CHARS,
    )
)
@settings(max_examples=20)
def test_bound_counts_unicode_chars_not_bytes(content):
    """The bound is unicode characters, not bytes — multibyte text at the char
    bound is accepted even though its UTF-8 encoding exceeds 1,400 bytes."""
    assert len(content.encode()) > MEMORY_CONTENT_MAX_CHARS  # over the bound in BYTES
    mem = MemoryStore(content=content)  # ...yet accepted: chars are what count
    assert len(mem.content) <= MEMORY_CONTENT_MAX_CHARS


@given(importance=st.floats().filter(lambda x: x < 0.0 or x > 1.0).filter(lambda x: x == x))  # exclude NaN
@settings(max_examples=20)
def test_importance_out_of_bounds_rejected(importance):
    """Importance outside [0.0, 1.0] is rejected."""
    try:
        MemoryStore(content="test", importance=importance)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


@given(permission=st.text(min_size=1, max_size=20).filter(lambda x: x not in ("read", "write")))
@settings(max_examples=20)
def test_invalid_permission_rejected(permission):
    """Only 'read' or 'write' accepted for ShareMemory.permission."""
    try:
        ShareMemory(shared_with="user", permission=permission)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


@given(tags=st.text(max_size=200))
@settings(max_examples=50)
def test_tags_splitting_consistency(tags):
    """Tags splitting produces consistent results."""
    result1 = [t.strip() for t in tags.split(",") if t.strip()]
    result2 = [t.strip() for t in tags.split(",") if t.strip()]
    assert result1 == result2


@given(sort_by=st.sampled_from(["importance", "relevance", "recency"]))
def test_valid_sort_by_accepted(sort_by):
    """Valid sort_by values are accepted."""
    recall = MemoryRecall(context="test", sort_by=sort_by)
    assert recall.sort_by == sort_by


@given(sort_by=st.text(min_size=1, max_size=20).filter(lambda x: x not in ("importance", "relevance", "recency")))
@settings(max_examples=20)
def test_invalid_sort_by_rejected(sort_by):
    """Invalid sort_by values are rejected after model update."""
    try:
        MemoryRecall(context="test", sort_by=sort_by)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


@given(limit=st.integers(min_value=10001, max_value=50000))
@settings(max_examples=10)
def test_limit_too_high_rejected(limit):
    """Limit above 10000 is rejected after model update."""
    try:
        MemoryRecall(context="test", limit=limit)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


@given(limit=st.integers(min_value=-100, max_value=0))
@settings(max_examples=10)
def test_limit_zero_or_negative_rejected(limit):
    """Limit <= 0 is rejected."""
    try:
        MemoryRecall(context="test", limit=limit)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass
