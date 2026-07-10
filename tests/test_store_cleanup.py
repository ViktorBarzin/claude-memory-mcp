"""Tests for scripts/store_cleanup.py — the one-shot production store cleanup (ADR-0007).

Pure mechanical logic only: series grouping/ordering, reassembly, the category fold map,
importance rules, tombstone successor-id parsing, corrupted-entry helpers, LLM-output
parsing, and the runner's apply paths against injected fakes.

NO network and NO subprocess anywhere in here — API and LLM access are injected fakes.
"""

import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "store_cleanup.py"
_spec = importlib.util.spec_from_file_location("store_cleanup", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sc = importlib.util.module_from_spec(_spec)
sys.modules["store_cleanup"] = sc
_spec.loader.exec_module(sc)

NOW = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def mem(
    id: int,
    content: str = "some fact",
    category: str = "facts",
    tags: str = "",
    importance: float = 0.5,
    created_at: str = "2026-07-01T00:00:00+00:00",
    is_sensitive: bool = False,
    owner: str = "wizard",
) -> dict:
    return {
        "id": id,
        "content": content,
        "category": category,
        "tags": tags,
        "importance": importance,
        "created_at": created_at,
        "updated_at": created_at,
        "is_sensitive": is_sensitive,
        "owner": owner,
    }


# --- part-N-of-M tag parsing -------------------------------------------------


class TestParsePartTag:
    def test_plain_part_tag(self):
        assert sc.parse_part_tag("part-2-of-3") == (2, 3)

    def test_part_tag_within_tag_list(self):
        assert sc.parse_part_tag("session-summary,tripit,part-4-of-12") == (4, 12)

    def test_whitespace_around_tags(self):
        assert sc.parse_part_tag("session-summary, part-1-of-2") == (1, 2)

    def test_no_part_tag(self):
        assert sc.parse_part_tag("session-summary,tripit") is None

    def test_embedded_word_is_not_a_part_tag(self):
        # "counterpart-1-of-2" must NOT match — the tag must be exactly part-N-of-M.
        assert sc.parse_part_tag("counterpart-1-of-2") is None

    def test_empty_tags(self):
        assert sc.parse_part_tag("") is None

    def test_base_tags_strip_all_part_tags(self):
        assert sc.base_tags("session-summary,part-1-of-2,tripit") == ["session-summary", "tripit"]

    def test_base_tags_without_part_tag_unchanged(self):
        assert sc.base_tags("a, b") == ["a", "b"]


# --- series grouping and ordering --------------------------------------------


def series_mem(id: int, n: int, m: int, base: str = "session-summary,tripit", **kw) -> dict:
    return mem(id, tags=f"{base},part-{n}-of-{m}", **kw)


class TestSeriesGrouping:
    def test_fragments_group_by_base_tags_and_m(self):
        mems = [series_mem(1, 1, 2), series_mem(2, 2, 2), series_mem(3, 1, 2, base="other")]
        groups = sc.group_series(mems)
        sizes = sorted(len(v) for v in groups.values())
        assert sizes == [1, 2]

    def test_tag_order_does_not_split_a_series(self):
        a = mem(1, tags="alpha,beta,part-1-of-2")
        b = mem(2, tags="beta,alpha,part-2-of-2")
        groups = sc.group_series([a, b])
        assert len(groups) == 1

    def test_same_tags_different_m_are_different_series(self):
        mems = [series_mem(1, 1, 2), series_mem(2, 1, 3)]
        assert len(sc.group_series(mems)) == 2

    def test_different_category_splits_series(self):
        a = series_mem(1, 1, 2, category="facts")
        b = series_mem(2, 2, 2, category="decisions")
        assert len(sc.group_series([a, b])) == 2

    def test_non_fragments_are_ignored(self):
        groups = sc.group_series([mem(1, tags="plain")])
        assert groups == {}

    def test_order_series_sorts_by_n(self):
        entries = [(2, mem(20)), (1, mem(10)), (3, mem(30))]
        ordered = sc.order_series(entries, m=3)
        assert [f["id"] for f in ordered] == [10, 20, 30]

    def test_order_series_missing_part_is_incomplete(self):
        entries = [(1, mem(10)), (3, mem(30))]
        assert sc.order_series(entries, m=3) is None

    def test_order_series_duplicate_part_is_incomplete(self):
        entries = [(1, mem(10)), (1, mem(11)), (2, mem(20))]
        assert sc.order_series(entries, m=2) is None

    def test_order_series_single_part_series_is_rejected(self):
        assert sc.order_series([(1, mem(10))], m=1) is None


# --- reassembly ---------------------------------------------------------------


class TestReassemble:
    def test_paragraph_chunks_rejoin_with_blank_line(self):
        # Chunks that came from the paragraph-boundary path are < chop size.
        assert sc.reassemble(["para one.", "para two."], chop_chars=50) == "para one.\n\npara two."

    def test_hard_split_chunks_rejoin_seamlessly(self):
        # The legacy chopper hard-split oversized paragraphs at exactly chop_chars,
        # so a fragment of exactly that length continues mid-text in the next one.
        first, second = "a" * 50, "bbb"
        assert sc.reassemble([first, second], chop_chars=50) == first + second

    def test_mixed_boundaries(self):
        para = "p" * 30
        hard_head, hard_tail = "x" * 50, "y" * 10
        joined = sc.reassemble([para, hard_head, hard_tail], chop_chars=50)
        assert joined == para + "\n\n" + hard_head + hard_tail

    def test_single_fragment_is_identity(self):
        assert sc.reassemble(["only"], chop_chars=50) == "only"

    @settings(max_examples=50, deadline=None)
    @given(
        st.lists(
            st.text(alphabet="abcdefgh .", min_size=1, max_size=120).map(lambda s: s.strip(" \n") or "x"),
            min_size=1,
            max_size=6,
        )
    )
    def test_round_trips_the_legacy_chopper(self, paragraphs):
        """reassemble() inverts the retired _split_content chopper (paragraph-first,
        hard-split at chop_chars) whenever no paragraph-boundary chunk lands on
        exactly the chop size by coincidence (the one case the heuristic cannot
        distinguish — checked against the chopper's ground-truth boundary flags)."""
        chop = 50
        text = "\n\n".join(paragraphs)
        chunks, hard = legacy_split_with_flags(text, chop)
        if any(not hard[i] and len(chunks[i]) == chop for i in range(len(chunks) - 1)):
            return  # ambiguous input: heuristic cannot know; skip
        assert sc.reassemble(chunks, chop_chars=chop) == text


def legacy_split_with_flags(text: str, max_chars: int) -> tuple[list[str], list[bool]]:
    """Verbatim behaviour of the retired chopper (app.py _split_content), plus
    ground-truth flags: hard[i] is True iff the boundary AFTER chunk i is a
    mid-paragraph hard split (chunk i+1 continues chunk i with no separator)."""
    if len(text) <= max_chars:
        return [text], []
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    hard: list[bool] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
                hard.append(False)
            while len(para) > max_chars:
                chunks.append(para[:max_chars])
                hard.append(True)
                para = para[max_chars:]
            current = para
    if current:
        chunks.append(current)
    return chunks, hard[: max(0, len(chunks) - 1)]


# --- category fold map ---------------------------------------------------------


class TestFoldCategory:
    @pytest.mark.parametrize(
        ("raw", "canonical"),
        [
            ("gotcha", "gotchas"),
            ("project", "projects"),
            ("reference", "references"),
            ("infra", "infrastructure"),
            ("bug", "gotchas"),
            ("incident", "incidents"),
            ("procedures", "runbook"),
        ],
    )
    def test_fold_map(self, raw, canonical):
        assert sc.fold_category(raw) == canonical

    def test_canonical_categories_unchanged(self):
        for cat in ("gotchas", "projects", "references", "infrastructure", "incidents", "runbook", "facts"):
            assert sc.fold_category(cat) == cat


# --- importance rules -----------------------------------------------------------


class TestImportanceRules:
    def test_part_fragment_capped_at_0_4(self):
        m = mem(1, tags="session-summary,part-1-of-3", importance=0.9)
        assert sc.importance_new_value(m, NOW) == pytest.approx(0.4)

    def test_part_fragment_below_cap_untouched(self):
        m = mem(1, tags="x,part-1-of-2", importance=0.3)
        assert sc.importance_new_value(m, NOW) is None

    def test_old_session_summary_capped_at_0_6(self):
        m = mem(1, tags="session-summary", importance=0.9, created_at="2026-05-01T00:00:00+00:00")
        assert sc.importance_new_value(m, NOW) == pytest.approx(0.6)

    def test_recent_session_summary_untouched(self):
        m = mem(1, tags="session-summary", importance=0.9, created_at="2026-07-01T00:00:00+00:00")
        assert sc.importance_new_value(m, NOW) is None

    def test_old_dated_decision_capped(self):
        m = mem(1, category="decisions", tags="2026-01-15,foo", importance=0.95, created_at="2026-01-15T09:00:00+00:00")
        assert sc.importance_new_value(m, NOW) == pytest.approx(0.6)

    def test_old_dated_project_capped(self):
        m = mem(1, category="projects", tags="2026-02-01", importance=0.8, created_at="2026-02-01T00:00:00+00:00")
        assert sc.importance_new_value(m, NOW) == pytest.approx(0.6)

    def test_decision_without_date_tag_untouched(self):
        m = mem(1, category="decisions", tags="foo", importance=0.95, created_at="2026-01-15T09:00:00+00:00")
        assert sc.importance_new_value(m, NOW) is None

    def test_dated_entry_in_other_category_untouched(self):
        m = mem(1, category="facts", tags="2026-01-15", importance=0.95, created_at="2026-01-15T09:00:00+00:00")
        assert sc.importance_new_value(m, NOW) is None

    def test_date_tag_is_the_age_reference(self):
        # created_at recent but the date tag says the session was months ago -> capped.
        m = mem(1, tags="session-summary,2026-03-01", importance=0.9, created_at="2026-07-05T00:00:00+00:00")
        assert sc.importance_new_value(m, NOW) == pytest.approx(0.6)

    def test_invalid_date_tag_is_ignored(self):
        m = mem(1, category="decisions", tags="2026-99-99", importance=0.9, created_at="2026-01-01T00:00:00+00:00")
        assert sc.importance_new_value(m, NOW) is None

    def test_part_cap_wins_over_summary_cap(self):
        m = mem(
            1, tags="session-summary,part-2-of-4", importance=0.9, created_at="2026-01-01T00:00:00+00:00"
        )
        assert sc.importance_new_value(m, NOW) == pytest.approx(0.4)

    def test_plain_entry_untouched(self):
        assert sc.importance_new_value(mem(1, importance=0.95), NOW) is None

    def test_summary_at_cap_untouched(self):
        m = mem(1, tags="session-summary", importance=0.6, created_at="2026-01-01T00:00:00+00:00")
        assert sc.importance_new_value(m, NOW) is None


# --- tombstone successor-id parsing ----------------------------------------------


class TestParseSuccessorId:
    def test_see_hash_n(self):
        assert sc.parse_successor_id("[SUPERSEDED] old truth, see #123") == 123

    def test_superseded_by_hash_n(self):
        assert sc.parse_successor_id("[SUPERSEDED] superseded by #77 on 2026-05-01") == 77

    def test_bracketed_superseded_by(self):
        assert sc.parse_successor_id("[SUPERSEDED by #456] the old content") == 456

    def test_case_insensitive(self):
        assert sc.parse_successor_id("[superseded] SEE #9") == 9

    def test_no_successor(self):
        assert sc.parse_successor_id("[SUPERSEDED] gone, no pointer") is None

    def test_first_pointer_wins(self):
        assert sc.parse_successor_id("[SUPERSEDED] see #5; superseded by #6") == 5

    def test_is_tombstone(self):
        assert sc.is_tombstone("[SUPERSEDED] x")
        assert sc.is_tombstone("[SUPERSEDED by #4] x")
        assert not sc.is_tombstone("plain content [SUPERSEDED]")


# --- corrupted-entry helpers -------------------------------------------------------


class TestCorruptedHelpers:
    def test_strip_residue_cuts_from_marker(self):
        content = "real fact here  </content>\n<tool_call>junk</tool_call>"
        assert sc.strip_residue(content) == "real fact here"

    def test_strip_residue_without_marker(self):
        assert sc.strip_residue("clean content") is None

    def test_strip_residue_empty_remainder(self):
        assert sc.strip_residue("</content>junk") == ""

    def test_build_corrupted_content_prefixes_and_bounds(self):
        blob = "x" * 50_000
        out = sc.build_corrupted_content(blob)
        assert out.startswith("[CORRUPTED - superseded]")
        assert len(out) <= sc.MAX_CONTENT_CHARS

    def test_build_corrupted_content_keeps_short_content_whole(self):
        out = sc.build_corrupted_content("small blob")
        assert out.startswith("[CORRUPTED - superseded]")
        assert "small blob" in out


# --- LLM output parsing --------------------------------------------------------------


class TestComposeParsing:
    def test_parse_json_block_naked(self):
        assert sc.parse_json_block('{"hub": "h", "parts": []}') == {"hub": "h", "parts": []}

    def test_parse_json_block_fenced(self):
        text = 'Here you go:\n```json\n{"hub": "h", "parts": ["p"]}\n```\nthanks'
        assert sc.parse_json_block(text) == {"hub": "h", "parts": ["p"]}

    def test_parse_json_block_garbage_raises(self):
        with pytest.raises(sc.ComposeError):
            sc.parse_json_block("no json here")

    def test_validate_series_compose_happy(self):
        hub, parts = sc.validate_series_compose({"hub": "h", "parts": ["a", "b"]})
        assert hub == "h" and parts == ["a", "b"]

    def test_validate_series_compose_missing_parts_defaults_empty(self):
        hub, parts = sc.validate_series_compose({"hub": "h"})
        assert hub == "h" and parts == []

    def test_validate_series_compose_oversize_hub_rejected(self):
        with pytest.raises(sc.ComposeError):
            sc.validate_series_compose({"hub": "x" * (sc.MAX_CONTENT_CHARS + 1), "parts": []})

    def test_validate_series_compose_oversize_part_rejected(self):
        with pytest.raises(sc.ComposeError):
            sc.validate_series_compose({"hub": "h", "parts": ["y" * (sc.MAX_CONTENT_CHARS + 1)]})

    def test_validate_series_compose_empty_hub_rejected(self):
        with pytest.raises(sc.ComposeError):
            sc.validate_series_compose({"hub": "  ", "parts": []})

    def test_validate_single_compose(self):
        assert sc.validate_single_compose({"content": "c"}) == "c"
        with pytest.raises(sc.ComposeError):
            sc.validate_single_compose({"content": "z" * (sc.MAX_CONTENT_CHARS + 1)})

    def test_validate_single_compose_allows_empty_when_asked(self):
        assert sc.validate_single_compose({"content": ""}, allow_empty=True) == ""
        with pytest.raises(sc.ComposeError):
            sc.validate_single_compose({"content": ""})


# --- series planning: the under/over-1400 fork ------------------------------------------


class TestPlanSeries:
    def test_short_series_plans_direct_rewrite(self):
        mems = [
            series_mem(1, 1, 2, importance=0.9, content="first half."),
            series_mem(2, 2, 2, importance=0.7, content="second half."),
        ]
        units, notes = sc.plan_series(mems)
        assert len(units) == 1
        u = units[0]
        assert u.mode == "rewrite"
        assert u.joined == "first half.\n\nsecond half."
        assert [f["id"] for f in u.fragments] == [1, 2]
        assert u.new_tags == "session-summary,tripit"
        # importance carried from the fragments but capped at 0.6 (inflation cleanup)
        assert u.new_importance == pytest.approx(0.6)

    def test_long_series_plans_llm_compose(self):
        big = "z" * 900
        mems = [series_mem(1, 1, 2, content=big), series_mem(2, 2, 2, content=big)]
        units, _ = sc.plan_series(mems)
        assert units[0].mode == "llm"

    def test_boundary_exactly_1400_is_direct_rewrite(self):
        a = "a" * 700
        b = "b" * 698  # 700 + 2 (blank line) + 698 = 1400
        mems = [series_mem(1, 1, 2, content=a), series_mem(2, 2, 2, content=b)]
        units, _ = sc.plan_series(mems)
        assert units[0].mode == "rewrite"
        assert len(units[0].joined) == sc.MAX_CONTENT_CHARS

    def test_incomplete_series_skipped_with_note(self):
        mems = [series_mem(1, 1, 3), series_mem(2, 3, 3)]
        units, notes = sc.plan_series(mems)
        assert units == []
        assert any("incomplete" in n for n in notes)

    def test_sensitive_fragment_skips_series(self):
        mems = [series_mem(1, 1, 2, is_sensitive=True), series_mem(2, 2, 2)]
        units, notes = sc.plan_series(mems)
        assert units == []
        assert any("sensitive" in n for n in notes)

    def test_new_category_is_folded(self):
        mems = [series_mem(1, 1, 2, category="gotcha"), series_mem(2, 2, 2, category="gotcha")]
        units, _ = sc.plan_series(mems)
        assert units[0].new_category == "gotchas"

    def test_low_importance_series_keeps_its_importance(self):
        mems = [series_mem(1, 1, 2, importance=0.4), series_mem(2, 2, 2, importance=0.3)]
        units, _ = sc.plan_series(mems)
        assert units[0].new_importance == pytest.approx(0.4)


# --- fakes for runner tests -----------------------------------------------------------


class FakeClient:
    """In-memory stand-in for MemoryClient: dict-backed, records every write."""

    def __init__(self, memories=()):
        self.memories = {m["id"]: dict(m) for m in memories}
        self.next_id = (max(self.memories) if self.memories else 0) + 1000
        self.links = []  # (src, dst, type)
        self.stores = []
        self.updates = []

    def auth_user(self):
        return "wizard"

    def list_all(self):
        return [dict(m) for m in self.memories.values()]

    def get(self, memory_id):
        if memory_id not in self.memories:
            raise sc.ClientError(f"GET /api/memories/{memory_id} -> 404", status=404)
        row = dict(self.memories[memory_id])
        # Mirror the production GET /api/memories/{id} shape (links both directions).
        row["links_out"] = [{"id": d, "type": t} for s, d, t in self.links if s == memory_id]
        row["links_in"] = [{"id": s, "type": t} for s, d, t in self.links if d == memory_id]
        return row

    def store(self, content, category, tags, importance, expanded_keywords=""):
        new_id = self.next_id
        self.next_id += 1
        self.memories[new_id] = mem(
            new_id, content=content, category=category, tags=tags, importance=importance
        )
        self.stores.append(new_id)
        return new_id

    def update(self, memory_id, **fields):
        if memory_id not in self.memories:
            raise sc.ClientError(f"PUT /api/memories/{memory_id} -> 404")
        self.memories[memory_id].update(fields)
        self.updates.append((memory_id, fields))

    def create_link(self, src_id, target_id, link_type):
        self.links.append((src_id, target_id, link_type))


class FakeComposer:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def compose(self, prompt):
        self.prompts.append(prompt)
        if not self.responses:
            raise sc.ComposeError("no canned response left")
        return self.responses.pop(0)


def make_runner(client, composer=None, execute=True, checkpoint=None, tmp_path=None):
    cp = sc.Checkpoint(checkpoint)
    return sc.Runner(client=client, composer=composer or FakeComposer([]), execute=execute,
                     checkpoint=cp, sleep_s=0.0, workers=2)


# --- runner: series apply --------------------------------------------------------------


class TestRunnerSeries:
    def test_rewrite_series_stores_links_and_downgrades(self):
        frags = [
            series_mem(1, 1, 2, importance=0.9, content="first half."),
            series_mem(2, 2, 2, importance=0.9, content="second half."),
        ]
        client = FakeClient(frags)
        runner = make_runner(client)
        summary = runner.run_phase("series", client.list_all())

        assert summary["applied"] == 1
        assert len(client.stores) == 1
        hub_id = client.stores[0]
        hub = client.memories[hub_id]
        assert hub["content"] == "first half.\n\nsecond half."
        assert hub["tags"] == "session-summary,tripit"
        # hub supersedes every fragment
        assert (hub_id, 1, "supersedes") in client.links
        assert (hub_id, 2, "supersedes") in client.links
        # fragments dropped to 0.3, nothing else touched
        assert client.memories[1]["importance"] == pytest.approx(0.3)
        assert client.memories[2]["importance"] == pytest.approx(0.3)
        assert client.memories[1]["content"] == "first half."

    def test_llm_series_stores_hub_and_parts_with_part_of_links(self):
        big = "z" * 900
        frags = [series_mem(1, 1, 2, content=big), series_mem(2, 2, 2, content=big)]
        client = FakeClient(frags)
        composer = FakeComposer([{"hub": "the hub", "parts": ["detail one", "detail two"]}])
        runner = make_runner(client, composer=composer)
        summary = runner.run_phase("series", client.list_all())

        assert summary["applied"] == 1
        assert len(client.stores) == 3  # hub + 2 parts
        hub_id, p1, p2 = client.stores
        assert client.memories[hub_id]["content"] == "the hub"
        assert (p1, hub_id, "part-of") in client.links
        assert (p2, hub_id, "part-of") in client.links
        assert (hub_id, 1, "supersedes") in client.links
        assert (hub_id, 2, "supersedes") in client.links

    def test_compose_failure_is_logged_and_skipped_not_fatal(self):
        big = "z" * 900
        frags = [series_mem(1, 1, 2, content=big), series_mem(2, 2, 2, content=big)]
        client = FakeClient(frags)
        composer = FakeComposer([])  # raises ComposeError
        runner = make_runner(client, composer=composer)
        summary = runner.run_phase("series", client.list_all())

        assert summary["failed"] == 1
        assert client.stores == []
        assert client.memories[1]["importance"] == pytest.approx(0.5)  # untouched

    def test_checkpoint_skips_processed_series(self, tmp_path):
        frags = [series_mem(1, 1, 2, content="a."), series_mem(2, 2, 2, content="b.")]
        client = FakeClient(frags)
        cp_file = tmp_path / "cp.json"
        runner = make_runner(client, checkpoint=cp_file)
        first = runner.run_phase("series", client.list_all())
        assert first["applied"] == 1

        client2 = FakeClient(frags)
        runner2 = make_runner(client2, checkpoint=cp_file)
        second = runner2.run_phase("series", client2.list_all())
        assert second["applied"] == 0
        assert second["skipped"] >= 1
        assert client2.stores == []

    def test_dry_run_writes_nothing(self):
        frags = [series_mem(1, 1, 2, content="a."), series_mem(2, 2, 2, content="b.")]
        client = FakeClient(frags)
        runner = make_runner(client, execute=False)
        summary = runner.run_phase("series", client.list_all())
        assert summary["planned"] == 1
        assert client.stores == [] and client.updates == [] and client.links == []


# --- runner: the other phases -----------------------------------------------------------


class TestRunnerPhases:
    def test_importance_phase_only_touches_importance(self):
        m1 = mem(1, tags="x,part-1-of-2", importance=0.9)
        m2 = mem(2, importance=0.9)  # no rule matches
        client = FakeClient([m1, m2])
        runner = make_runner(client)
        summary = runner.run_phase("importance", client.list_all())
        assert summary["applied"] == 1
        assert client.updates == [(1, {"importance": 0.4})]

    def test_tombstone_phase_drops_importance_and_links_successor(self):
        old = mem(4, content="[SUPERSEDED] see #7", importance=0.8)
        successor = mem(7, content="current truth")
        client = FakeClient([old, successor])
        runner = make_runner(client)
        summary = runner.run_phase("tombstones", client.list_all())
        assert summary["applied"] == 1
        assert client.memories[4]["importance"] == pytest.approx(0.3)
        assert (7, 4, "supersedes") in client.links

    def test_tombstone_with_missing_successor_still_downgrades(self):
        old = mem(4, content="[SUPERSEDED] see #999", importance=0.8)
        client = FakeClient([old])
        runner = make_runner(client)
        summary = runner.run_phase("tombstones", client.list_all())
        assert summary["applied"] == 1
        assert client.memories[4]["importance"] == pytest.approx(0.3)
        assert client.links == []

    def test_categories_phase_folds_legacy_categories(self):
        m1 = mem(1, category="gotcha")
        m2 = mem(2, category="gotchas")
        m3 = mem(3, category="procedures")
        client = FakeClient([m1, m2, m3])
        runner = make_runner(client)
        summary = runner.run_phase("categories", client.list_all())
        assert summary["applied"] == 2
        assert client.memories[1]["category"] == "gotchas"
        assert client.memories[2]["category"] == "gotchas"
        assert client.memories[3]["category"] == "runbook"

    def test_dupes_phase_merges_known_cluster(self):
        members = [
            mem(676, content="NFS migration fact A", category="infra", importance=0.8),
            mem(677, content="NFS migration fact B", category="infrastructure", importance=0.9),
        ]
        client = FakeClient(members)
        composer = FakeComposer([{"content": "consolidated NFS migration memory"}])
        runner = make_runner(client, composer=composer)
        summary = runner.run_phase("dupes", client.list_all())
        assert summary["applied"] == 1
        assert summary["skipped"] == len(sc.DUPE_CLUSTERS) - 1  # other clusters absent from store
        new_id = client.stores[0]
        assert client.memories[new_id]["content"] == "consolidated NFS migration memory"
        assert client.memories[new_id]["category"] == "infrastructure"
        assert (new_id, 676, "supersedes") in client.links
        assert (new_id, 677, "supersedes") in client.links
        assert client.memories[676]["importance"] == pytest.approx(0.3)
        assert client.memories[677]["importance"] == pytest.approx(0.3)

    def test_corrupted_residue_strip_in_place(self):
        bad = mem(sc.CORRUPTED_RESIDUE_ID, content="the real fact</content><xml>junk</xml>")
        client = FakeClient([bad])
        runner = make_runner(client)
        summary = runner.run_phase("corrupted", client.list_all())
        assert summary["applied"] >= 1
        assert client.memories[sc.CORRUPTED_RESIDUE_ID]["content"] == "the real fact"

    def test_corrupted_blob_no_fact_gets_prefix_and_floor_importance(self):
        blob = mem(sc.CORRUPTED_XML_BLOB_ID, content="<xml>" + "x" * 3000, importance=0.8)
        client = FakeClient([blob])
        composer = FakeComposer([{"content": ""}])  # LLM finds no real fact
        runner = make_runner(client, composer=composer)
        summary = runner.run_phase("corrupted", client.list_all())
        assert summary["applied"] >= 1
        row = client.memories[sc.CORRUPTED_XML_BLOB_ID]
        assert row["content"].startswith("[CORRUPTED - superseded]")
        assert len(row["content"]) <= sc.MAX_CONTENT_CHARS
        assert row["importance"] == pytest.approx(0.1)
        assert client.stores == []

    def test_corrupted_blob_with_fact_stores_fresh_memory(self):
        blob = mem(sc.CORRUPTED_XML_BLOB_ID, content="<xml>real: registry is 10.0.20.10" + "x" * 3000)
        client = FakeClient([blob])
        composer = FakeComposer([{"content": "registry is 10.0.20.10"}])
        runner = make_runner(client, composer=composer)
        runner.run_phase("corrupted", client.list_all())
        assert len(client.stores) == 1
        new_id = client.stores[0]
        assert (new_id, sc.CORRUPTED_XML_BLOB_ID, "supersedes") in client.links
        assert client.memories[sc.CORRUPTED_XML_BLOB_ID]["importance"] == pytest.approx(0.1)

    def test_link_not_visible_after_create_fails_unit(self):
        class NoopLinkClient(FakeClient):
            def create_link(self, src_id, target_id, link_type):
                pass  # returns 2xx but the edge never lands -> links_out stays empty

        old = mem(4, content="[SUPERSEDED] see #7", importance=0.8)
        successor = mem(7, content="current truth")
        client = NoopLinkClient([old, successor])
        runner = make_runner(client)
        summary = runner.run_phase("tombstones", client.list_all())
        assert summary["failed"] == 1
        assert summary["applied"] == 0

    def test_duplicate_link_409_is_idempotent_success(self):
        class DuplicateLinkClient(FakeClient):
            def create_link(self, src_id, target_id, link_type):
                raise sc.ClientError("duplicate edge", status=409)

        old = mem(4, content="[SUPERSEDED] see #7", importance=0.8)
        successor = mem(7, content="current truth")
        client = DuplicateLinkClient([old, successor])
        client.links.append((7, 4, "supersedes"))  # a previous interrupted run already created it
        runner = make_runner(client)
        summary = runner.run_phase("tombstones", client.list_all())
        assert summary["applied"] == 1
        assert summary["failed"] == 0
        assert client.memories[4]["importance"] == pytest.approx(0.3)

    def test_verify_failure_marks_unit_failed(self):
        class LyingClient(FakeClient):
            def update(self, memory_id, **fields):
                self.updates.append((memory_id, fields))  # recorded but NOT applied

        m1 = mem(1, tags="x,part-1-of-2", importance=0.9)
        client = LyingClient([m1])
        runner = make_runner(client)
        summary = runner.run_phase("importance", client.list_all())
        assert summary["failed"] == 1
        assert summary["applied"] == 0


# --- checkpoint ---------------------------------------------------------------------------


class TestCheckpoint:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "cp.json"
        cp = sc.Checkpoint(path)
        assert not cp.seen("series:abc")
        cp.mark("series:abc")
        assert cp.seen("series:abc")
        cp2 = sc.Checkpoint(path)
        assert cp2.seen("series:abc")
        assert not cp2.seen("series:other")

    def test_none_path_is_memory_only(self):
        cp = sc.Checkpoint(None)
        cp.mark("k")
        assert cp.seen("k")

    def test_file_is_valid_json(self, tmp_path):
        path = tmp_path / "cp.json"
        cp = sc.Checkpoint(path)
        cp.mark("a")
        cp.mark("b")
        data = json.loads(path.read_text())
        assert set(data["processed"]) == {"a", "b"}
