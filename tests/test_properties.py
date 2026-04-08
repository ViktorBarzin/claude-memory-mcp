"""Property-based tests for Claude Memory models."""

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from claude_memory.api.models import MemoryStore, MemoryRecall, ShareMemory


# Strategy for valid MemoryStore
valid_memory_store = st.builds(
    MemoryStore,
    content=st.text(min_size=1, max_size=800),
    category=st.text(min_size=1, max_size=50),
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


@given(content=st.text(min_size=801, max_size=1000))
@settings(max_examples=20)
def test_content_over_max_rejected(content):
    """Content exceeding 800 chars is rejected."""
    try:
        MemoryStore(content=content)
        assert False, "Should have raised ValidationError"
    except ValidationError:
        pass


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
