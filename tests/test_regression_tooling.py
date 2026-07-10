"""Pure-logic tests for the POST-CLEANUP retrieval-regression tooling
(``benchmarks/scripts/snapshot_corpus.py`` + ``benchmarks/scripts/regression_run.py``).

The tooling re-validates retrieval quality on the post-cleanup store against the
preserved 5,452-memory / 119-query baseline (docs/runbooks/hybrid-recall-promotion.md,
docs/research/hybrid-build-report.md). These tests cover the pure logic ONLY — corpus
format writing, id-map loading/resolution/application (the ADR-0007 supersedes
redirect), gold-coverage pruning, and the PASS/FAIL threshold comparison — with NO
network, NO embedding model, and NO real data. The offline end-to-end test drives the
stdlib-only FTS retriever over a tiny synthetic dataset.

The scripts live in ``benchmarks/scripts/`` (not a package), so they are imported by
file path exactly as ``benchmarks/harness/test_matrix_runner.py`` imports run_eval.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = _REPO_ROOT / "benchmarks" / "scripts"


def _load_script(stem: str):
    """Import a benchmarks/scripts/ module by file path (scripts/ is not a package)."""
    path = _SCRIPTS_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"_bench_{stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


snapshot_corpus = _load_script("snapshot_corpus")
regression_run = _load_script("regression_run")


# ── snapshot_corpus: API rows → corpus.jsonl format ──────────────────────────


def _api_row(mem_id: int, content: str = "some fact", **over: object) -> dict:
    row = {
        "id": mem_id,
        "content": content,
        "category": "facts",
        "tags": "a,b",
        "expanded_keywords": "kw1 kw2",
        "importance": 0.7,
        "is_sensitive": False,
        "created_at": "2026-07-01T00:00:00+00:00",
        "updated_at": "2026-07-01T00:00:00+00:00",
        "deleted_at": None,
    }
    row.update(over)
    return row


class TestToCorpusRecords:
    def test_maps_api_fields_to_corpus_schema(self) -> None:
        records, _stats = snapshot_corpus.to_corpus_records([_api_row(7)])
        assert records == [
            {
                "id": 7,
                "content": "some fact",
                "category": "facts",
                "tags": "a,b",
                "expanded_keywords": "kw1 kw2",
                "importance": 0.7,
            }
        ]

    def test_excludes_sensitive_rows(self) -> None:
        rows = [_api_row(1), _api_row(2, is_sensitive=True), _api_row(3, is_sensitive=1)]
        records, stats = snapshot_corpus.to_corpus_records(rows)
        assert [r["id"] for r in records] == [1]
        assert stats["sensitive_excluded"] == 2

    def test_excludes_deleted_rows(self) -> None:
        rows = [_api_row(1), _api_row(2, deleted_at="2026-07-09T10:00:00+00:00")]
        records, stats = snapshot_corpus.to_corpus_records(rows)
        assert [r["id"] for r in records] == [1]
        assert stats["deleted_excluded"] == 1

    def test_normalises_null_optional_fields(self) -> None:
        rows = [_api_row(1, tags=None, expanded_keywords=None, category=None, importance=None)]
        records, _stats = snapshot_corpus.to_corpus_records(rows)
        assert records[0]["tags"] == ""
        assert records[0]["expanded_keywords"] == ""
        assert records[0]["category"] == "facts"
        assert records[0]["importance"] == 0.5

    def test_sorted_by_id(self) -> None:
        records, _stats = snapshot_corpus.to_corpus_records([_api_row(9), _api_row(2), _api_row(5)])
        assert [r["id"] for r in records] == [2, 5, 9]

    def test_stats_account_for_every_row(self) -> None:
        rows = [
            _api_row(1),
            _api_row(2, is_sensitive=True),
            _api_row(3, deleted_at="2026-07-09T10:00:00+00:00"),
            # sensitive AND deleted → counted once, under sensitive (the SQL
            # `WHERE is_sensitive=0` in export_corpus.py excludes it first).
            _api_row(4, is_sensitive=True, deleted_at="2026-07-09T10:00:00+00:00"),
        ]
        _records, stats = snapshot_corpus.to_corpus_records(rows)
        assert stats == {
            "total_rows": 4,
            "sensitive_excluded": 2,
            "deleted_excluded": 1,
            "written": 1,
        }


class TestWriteCorpusJsonl:
    def test_round_trips_through_harness_loader(self, tmp_path: Path) -> None:
        records, _ = snapshot_corpus.to_corpus_records(
            [_api_row(1, content="café ↔ naïve"), _api_row(2, content="plain")]
        )
        out = tmp_path / "snap" / "corpus.jsonl"
        n = snapshot_corpus.write_corpus_jsonl(records, out)
        assert n == 2

        # unicode is preserved raw (ensure_ascii=False), matching export_corpus.py
        assert "café" in out.read_text(encoding="utf-8")

        sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))
        from harness.dataset import load_corpus

        loaded = load_corpus(out)
        assert [m.id for m in loaded] == [1, 2]
        assert loaded[0].content == "café ↔ naïve"
        assert loaded[0].tags == "a,b"
        assert loaded[0].expanded_keywords == "kw1 kw2"
        assert loaded[0].importance == 0.7

    def test_one_json_object_per_line(self, tmp_path: Path) -> None:
        records, _ = snapshot_corpus.to_corpus_records([_api_row(1), _api_row(2)])
        out = tmp_path / "corpus.jsonl"
        snapshot_corpus.write_corpus_jsonl(records, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(isinstance(json.loads(ln), dict) for ln in lines)


class TestSnapshotDirSafety:
    """The preserved eval set MUST NOT be modified — the snapshot writer refuses
    to write anywhere inside it (including through a symlink like benchmarks/data)."""

    def test_refuses_writing_into_preserved_dir(self, tmp_path: Path) -> None:
        preserved = tmp_path / "preserved-data"
        preserved.mkdir()
        with pytest.raises(ValueError, match="preserved"):
            snapshot_corpus.assert_snapshot_dir_safe(preserved / "2026-07-10", preserved)

    def test_refuses_the_preserved_dir_itself(self, tmp_path: Path) -> None:
        preserved = tmp_path / "preserved-data"
        preserved.mkdir()
        with pytest.raises(ValueError, match="preserved"):
            snapshot_corpus.assert_snapshot_dir_safe(preserved, preserved)

    def test_refuses_via_symlink(self, tmp_path: Path) -> None:
        preserved = tmp_path / "preserved-data"
        preserved.mkdir()
        link = tmp_path / "data"
        link.symlink_to(preserved)
        with pytest.raises(ValueError, match="preserved"):
            snapshot_corpus.assert_snapshot_dir_safe(link / "2026-07-10", preserved)

    def test_allows_sibling_dir(self, tmp_path: Path) -> None:
        preserved = tmp_path / "preserved-data"
        preserved.mkdir()
        snapshot_corpus.assert_snapshot_dir_safe(tmp_path / "snapshots" / "2026-07-10", preserved)


class TestWriteSnapshot:
    def test_writes_corpus_and_meta(self, tmp_path: Path) -> None:
        records, stats = snapshot_corpus.to_corpus_records([_api_row(1), _api_row(2)])
        out_dir = tmp_path / "snapshots" / "2026-07-10"
        meta = snapshot_corpus.write_snapshot(records, stats, out_dir, source="test")
        assert (out_dir / "corpus.jsonl").exists()
        on_disk = json.loads((out_dir / "snapshot_meta.json").read_text())
        assert on_disk == meta
        assert meta["stats"]["written"] == 2
        assert meta["source"] == "test"
        # 16-hex corpus fingerprint (same scheme as the cached-embedding key)
        assert len(meta["corpus_fingerprint"]) == 16
        int(meta["corpus_fingerprint"], 16)

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        records, stats = snapshot_corpus.to_corpus_records([_api_row(1)])
        out_dir = tmp_path / "snap"
        snapshot_corpus.write_snapshot(records, stats, out_dir, source="test")
        with pytest.raises(ValueError, match="exists"):
            snapshot_corpus.write_snapshot(records, stats, out_dir, source="test")
        snapshot_corpus.write_snapshot(records, stats, out_dir, source="test", force=True)


# ── regression_run: id-map loading / resolution / application ────────────────


class TestLoadIdMap:
    def test_flat_object(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        p.write_text(json.dumps({"5972": 6775, "100": 200}))
        assert regression_run.load_id_map(p) == {5972: 6775, 100: 200}

    def test_list_of_pairs(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        p.write_text(json.dumps([[5972, 6775], [100, 200]]))
        assert regression_run.load_id_map(p) == {5972: 6775, 100: 200}

    def test_list_of_old_new_objects(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        p.write_text(json.dumps([{"old": 1, "new": 2}, {"old": 3, "new": 4}]))
        assert regression_run.load_id_map(p) == {1: 2, 3: 4}

    def test_nested_under_report_key(self, tmp_path: Path) -> None:
        p = tmp_path / "report.json"
        p.write_text(json.dumps({"deleted": [9], "id_map": {"1": 2}}))
        assert regression_run.load_id_map(p) == {1: 2}

    def test_plain_text_pairs(self, tmp_path: Path) -> None:
        p = tmp_path / "map.txt"
        p.write_text("12 34\n56,78\n\n# comment\n")
        assert regression_run.load_id_map(p) == {12: 34, 56: 78}

    def test_garbage_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "map.json"
        p.write_text(json.dumps({"not-an-id": "nope"}))
        with pytest.raises(ValueError):
            regression_run.load_id_map(p)


class TestResolveIdMap:
    def test_chain_collapses_to_final_successor(self) -> None:
        assert regression_run.resolve_id_map({1: 2, 2: 3}) == {1: 3, 2: 3}

    def test_identity_entries_dropped(self) -> None:
        assert regression_run.resolve_id_map({5: 5, 1: 2}) == {1: 2}

    def test_cycle_raises(self) -> None:
        with pytest.raises(ValueError, match="[Cc]ycle"):
            regression_run.resolve_id_map({1: 2, 2: 1})

    def test_long_chain(self) -> None:
        assert regression_run.resolve_id_map({1: 2, 2: 3, 3: 4}) == {1: 4, 2: 4, 3: 4}


class TestApplyIdMap:
    def test_qrels_gold_ids_remapped_and_merged(self) -> None:
        qrels = {"q1": {1, 2}, "q2": {3}}
        out = regression_run.apply_id_map_to_qrels(qrels, {1: 9, 2: 9})
        assert out == {"q1": {9}, "q2": {3}}
        assert qrels == {"q1": {1, 2}, "q2": {3}}  # input not mutated

    def test_redirect_ranked_serves_successor_in_place(self) -> None:
        # ADR-0007: supersedes REDIRECTS — the successor is served in place of the
        # superseded entry, at the superseded entry's rank.
        assert regression_run.redirect_ranked([5, 7, 3], {5: 9}) == [9, 7, 3]

    def test_redirect_ranked_dedups_keeping_first_rank(self) -> None:
        # old id 5 → 9 at rank 1; the raw 9 at rank 2 collapses into it.
        assert regression_run.redirect_ranked([5, 9, 7], {5: 9}) == [9, 7]

    def test_redirect_ranked_no_map_is_identity(self) -> None:
        assert regression_run.redirect_ranked([1, 2, 3], {}) == [1, 2, 3]


class TestGoldCoverage:
    def test_missing_gold_detected(self) -> None:
        missing = regression_run.gold_coverage({"q1": {1, 99}, "q2": {2}}, {1, 2})
        assert missing == {"q1": {99}}

    def test_prune_drops_missing_ids_and_empty_queries(self) -> None:
        qrels = {"q1": {1, 99}, "q2": {98}}
        pruned, dropped_queries, report = regression_run.prune_missing_gold(qrels, {1, 2})
        assert pruned == {"q1": {1}}
        assert dropped_queries == {"q2"}
        assert report == {"q1": {99}, "q2": {98}}


# ── regression_run: content bridge between id spaces ─────────────────────────
#
# Verified live (2026-07-10): the preserved eval set carries LOCAL-SQLite ids while
# API snapshots (and the cleanup report) carry REMOTE ids — 0/5,452 preserved
# (id, content) pairs matched the live store, but 137/139 gold ids matched by exact
# content under a different id. The bridge maps reference-corpus ids → snapshot ids
# by exact content so preserved qrels can score an API snapshot.


def _mem(mem_id: int, content: str):
    sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))
    from harness.types import Memory

    return Memory(id=mem_id, content=content)


class TestBuildContentBridge:
    def test_bridges_by_exact_content(self) -> None:
        bridge, report = regression_run.build_content_bridge(
            [_mem(5, "postgres tuning")], [_mem(50, "postgres tuning"), _mem(51, "other")]
        )
        assert bridge == {5: (50,)}
        assert report["bridged"] == 1
        assert report["unbridged"] == 0

    def test_ambiguous_content_maps_to_all_twins(self) -> None:
        # near-duplicate twins: any of them satisfies the query (qrels twin precedent)
        bridge, report = regression_run.build_content_bridge(
            [_mem(5, "twin fact")], [_mem(50, "twin fact"), _mem(51, "twin fact")]
        )
        assert bridge == {5: (50, 51)}
        assert report["ambiguous"] == 1

    def test_unmatched_reference_id_absent_from_bridge(self) -> None:
        bridge, report = regression_run.build_content_bridge(
            [_mem(5, "gone from the store")], [_mem(50, "something else")]
        )
        assert 5 not in bridge
        assert report["unbridged"] == 1

    def test_whitespace_trimmed_before_matching(self) -> None:
        bridge, _ = regression_run.build_content_bridge(
            [_mem(5, "fact\n")], [_mem(50, "fact")]
        )
        assert bridge == {5: (50,)}


class TestApplyBridge:
    def test_qrels_expand_to_all_twins(self) -> None:
        out, unbridged = regression_run.apply_bridge_to_qrels({"q": {5}}, {5: (50, 51)})
        assert out == {"q": {50, 51}}
        assert unbridged == {}

    def test_unbridged_gold_never_passes_through(self) -> None:
        # an unbridged local id must NOT survive raw — it could collide with an
        # unrelated snapshot id and silently score the wrong row.
        out, unbridged = regression_run.apply_bridge_to_qrels({"q": {5, 7}}, {5: (50,)})
        assert out == {"q": {50}}
        assert unbridged == {"q": {7}}

    def test_queries_relevant_ids_expanded_in_order(self) -> None:
        sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))
        from harness.types import Query

        q = Query(query_id="q", text="t", stratum="exact", relevant_ids=(5, 7))
        out = regression_run.apply_bridge_to_queries([q], {5: (50, 51), 7: (70,)})
        assert out[0].relevant_ids == (50, 51, 70)


# ── regression_run: threshold comparison (the PASS/FAIL gate) ────────────────


def _result(overall_r10: float, overall_n10: float = 0.65) -> dict:
    return {
        "overall": {"recall@5": 0.6, "recall@10": overall_r10, "ndcg@10": overall_n10, "mrr": 0.6},
        "per_stratum": {
            "exact": {"recall@10": 1.0, "ndcg@10": 0.99},
            "paraphrase": {"recall@10": 0.4, "ndcg@10": 0.3},
        },
    }


def _baseline(overall_r10: float) -> dict:
    return {
        "overall": {"recall@10": overall_r10, "ndcg@10": 0.65},
        "per_stratum": {
            "exact": {"recall@10": 1.0, "ndcg@10": 0.99},
            "paraphrase": {"recall@10": 0.4, "ndcg@10": 0.3},
        },
    }


class TestCompareToBaseline:
    def test_pass_when_within_threshold(self) -> None:
        cmp_ = regression_run.compare_to_baseline(
            {"fts": _result(0.69)}, {"fts": _baseline(0.70)}, threshold=0.02
        )
        assert cmp_["passed"] is True
        assert cmp_["retrievers"]["fts"]["passed"] is True

    def test_drop_of_exactly_threshold_passes(self) -> None:
        # the gate is "drops MORE THAN threshold" — the boundary itself passes.
        cmp_ = regression_run.compare_to_baseline(
            {"fts": _result(0.68)}, {"fts": _baseline(0.70)}, threshold=0.02
        )
        assert cmp_["passed"] is True

    def test_fail_when_overall_recall10_drops_beyond_threshold(self) -> None:
        cmp_ = regression_run.compare_to_baseline(
            {"fts": _result(0.60)}, {"fts": _baseline(0.70)}, threshold=0.02
        )
        assert cmp_["passed"] is False
        assert cmp_["retrievers"]["fts"]["passed"] is False
        assert cmp_["retrievers"]["fts"]["gate"]["delta"] == pytest.approx(-0.10)

    def test_improvement_passes(self) -> None:
        cmp_ = regression_run.compare_to_baseline(
            {"fts": _result(0.80)}, {"fts": _baseline(0.70)}, threshold=0.02
        )
        assert cmp_["passed"] is True

    def test_any_failing_retriever_fails_the_gate(self) -> None:
        cmp_ = regression_run.compare_to_baseline(
            {"fts": _result(0.70), "dense": _result(0.60)},
            {"fts": _baseline(0.70), "dense": _baseline(0.83)},
            threshold=0.02,
        )
        assert cmp_["retrievers"]["fts"]["passed"] is True
        assert cmp_["retrievers"]["dense"]["passed"] is False
        assert cmp_["passed"] is False

    def test_retriever_without_baseline_does_not_gate(self) -> None:
        cmp_ = regression_run.compare_to_baseline(
            {"mystery": _result(0.10)}, {"fts": _baseline(0.70)}, threshold=0.02
        )
        assert cmp_["passed"] is True
        assert cmp_["retrievers"]["mystery"]["gate"] is None

    def test_table_renders_slices_and_verdict(self) -> None:
        cmp_ = regression_run.compare_to_baseline(
            {"fts": _result(0.60)}, {"fts": _baseline(0.70)}, threshold=0.02
        )
        table = regression_run.format_comparison_table(cmp_)
        assert "FAIL" in table
        assert "overall" in table
        assert "paraphrase" in table
        assert "recall@10" in table
        assert "0.6000" in table and "0.7000" in table

    def test_default_baselines_match_the_build_report(self) -> None:
        # the committed aggregate numbers from docs/research/hybrid-build-report.md §2
        b = regression_run.DEFAULT_BASELINES
        assert b["fts"]["overall"]["recall@10"] == pytest.approx(0.6952, abs=1e-4)
        assert b["fts"]["overall"]["ndcg@10"] == pytest.approx(0.6507, abs=1e-4)
        assert b["dense"]["overall"]["recall@10"] == pytest.approx(0.8338, abs=1e-4)
        assert b["dense"]["per_stratum"]["paraphrase"]["recall@10"] == pytest.approx(0.7250, abs=1e-4)


# ── regression_run: offline end-to-end over a tiny synthetic dataset ─────────


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> dict[str, Path]:
    """Synthetic post-cleanup snapshot: gold id 99 was superseded by 3 (id_map 99→3);
    the tombstoned row 99 is still in the corpus, its successor 3 alongside it."""
    corpus = [
        {"id": 1, "content": "kubernetes ingress nginx routing", "category": "facts",
         "tags": "k8s", "expanded_keywords": "ingress nginx", "importance": 0.8},
        {"id": 2, "content": "grafana dashboards live under monitoring", "category": "facts",
         "tags": "grafana", "expanded_keywords": "dashboards", "importance": 0.6},
        {"id": 3, "content": "postgres autovacuum tuned via scale factor", "category": "facts",
         "tags": "postgres", "expanded_keywords": "vacuum autovacuum", "importance": 0.7},
        {"id": 99, "content": "postgres vacuum runs nightly cron", "category": "facts",
         "tags": "postgres", "expanded_keywords": "vacuum nightly", "importance": 0.5},
    ]
    queries = [
        {"query_id": "q_exact", "text": "kubernetes ingress nginx routing",
         "stratum": "exact", "relevant_ids": [1]},
        {"query_id": "q_super", "text": "postgres vacuum nightly",
         "stratum": "paraphrase", "relevant_ids": [99]},
    ]
    qrels = [
        {"query_id": "q_exact", "relevant_ids": [1]},
        {"query_id": "q_super", "relevant_ids": [99]},
    ]
    return {
        "corpus": _write_jsonl(tmp_path / "corpus.jsonl", corpus),
        "queries": _write_jsonl(tmp_path / "queries.jsonl", queries),
        "qrels": _write_jsonl(tmp_path / "qrels.jsonl", qrels),
    }


class TestRunRegressionOffline:
    def test_fts_with_id_map_redirect_scores_successor_as_hit(self, tiny_dataset: dict[str, Path]) -> None:
        out = regression_run.run_regression(
            corpus_path=tiny_dataset["corpus"],
            queries_path=tiny_dataset["queries"],
            qrels_path=tiny_dataset["qrels"],
            id_map={99: 3},
            retrievers=("fts",),
        )
        res = out["results"]["fts"]
        # q_exact hits 1; q_super's gold 99 remaps to 3, and whichever of {99, 3}
        # FTS surfaces, the redirect serves 3 → both queries are full hits.
        assert res["overall"]["recall@10"] == pytest.approx(1.0)
        assert res["per_stratum"]["paraphrase"]["recall@10"] == pytest.approx(1.0)

    def test_missing_gold_raises_without_drop(self, tiny_dataset: dict[str, Path]) -> None:
        # gold 99 remaps to 12345 which is not in the corpus → hard error.
        with pytest.raises(ValueError, match="12345"):
            regression_run.run_regression(
                corpus_path=tiny_dataset["corpus"],
                queries_path=tiny_dataset["queries"],
                qrels_path=tiny_dataset["qrels"],
                id_map={99: 12345},
                retrievers=("fts",),
            )

    def test_missing_gold_dropped_when_asked(self, tiny_dataset: dict[str, Path]) -> None:
        out = regression_run.run_regression(
            corpus_path=tiny_dataset["corpus"],
            queries_path=tiny_dataset["queries"],
            qrels_path=tiny_dataset["qrels"],
            id_map={99: 12345},
            retrievers=("fts",),
            drop_missing_gold=True,
        )
        assert out["gold"]["dropped_queries"] == ["q_super"]
        assert out["results"]["fts"]["n_queries"] == 1

    def test_bridge_composes_with_id_map(self, tmp_path: Path, tiny_dataset: dict[str, Path]) -> None:
        # Reference corpus in the LOCAL id space (matches tiny_dataset's queries/qrels);
        # snapshot in the REMOTE id space; the cleanup then superseded remote 1099 → 1003.
        reference = [
            {"id": 1, "content": "kubernetes ingress nginx routing", "category": "facts",
             "tags": "", "expanded_keywords": "", "importance": 0.8},
            {"id": 99, "content": "postgres vacuum runs nightly cron", "category": "facts",
             "tags": "", "expanded_keywords": "", "importance": 0.5},
        ]
        snapshot = [
            {"id": 1001, "content": "kubernetes ingress nginx routing", "category": "facts",
             "tags": "k8s", "expanded_keywords": "ingress nginx", "importance": 0.8},
            {"id": 1003, "content": "postgres autovacuum tuned via scale factor", "category": "facts",
             "tags": "postgres", "expanded_keywords": "vacuum autovacuum", "importance": 0.7},
            {"id": 1099, "content": "postgres vacuum runs nightly cron", "category": "facts",
             "tags": "postgres", "expanded_keywords": "vacuum nightly", "importance": 0.5},
        ]
        ref_path = _write_jsonl(tmp_path / "reference.jsonl", reference)
        snap_path = _write_jsonl(tmp_path / "snapshot.jsonl", snapshot)
        out = regression_run.run_regression(
            corpus_path=snap_path,
            queries_path=tiny_dataset["queries"],
            qrels_path=tiny_dataset["qrels"],
            bridge_corpus_path=ref_path,
            id_map={1099: 1003},  # cleanup map, REMOTE id space
            retrievers=("fts",),
        )
        # gold 1 → bridge → 1001 (hit); gold 99 → bridge → 1099 → cleanup map → 1003;
        # FTS surfaces the 1099 tombstone, the redirect serves 1003 → full hits.
        assert out["results"]["fts"]["overall"]["recall@10"] == pytest.approx(1.0)
        assert out["bridge"]["bridged"] == 2

    def test_unbridged_gold_raises_with_hint(self, tmp_path: Path, tiny_dataset: dict[str, Path]) -> None:
        reference = [{"id": 1, "content": "kubernetes ingress nginx routing", "category": "facts",
                      "tags": "", "expanded_keywords": "", "importance": 0.8}]
        snapshot = [{"id": 1001, "content": "kubernetes ingress nginx routing", "category": "facts",
                     "tags": "", "expanded_keywords": "", "importance": 0.8}]
        ref_path = _write_jsonl(tmp_path / "reference.jsonl", reference)
        snap_path = _write_jsonl(tmp_path / "snapshot.jsonl", snapshot)
        # gold 99 has no content match in the reference→snapshot bridge → hard error.
        with pytest.raises(ValueError, match="99"):
            regression_run.run_regression(
                corpus_path=snap_path,
                queries_path=tiny_dataset["queries"],
                qrels_path=tiny_dataset["qrels"],
                bridge_corpus_path=ref_path,
                retrievers=("fts",),
            )

    def test_gate_wiring_end_to_end(self, tiny_dataset: dict[str, Path]) -> None:
        out = regression_run.run_regression(
            corpus_path=tiny_dataset["corpus"],
            queries_path=tiny_dataset["queries"],
            qrels_path=tiny_dataset["qrels"],
            id_map={99: 3},
            retrievers=("fts",),
        )
        ok = regression_run.compare_to_baseline(
            out["results"], {"fts": _baseline(1.0)}, threshold=0.02
        )
        assert ok["passed"] is True
        bad = regression_run.compare_to_baseline(
            out["results"], {"fts": _baseline(1.5)}, threshold=0.02
        )
        assert bad["passed"] is False
