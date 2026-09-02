"""Tests for scripts/ci-layer-delta.py — the image layer-delta guard.

The guard exists because docker/Dockerfile's layer ORDER is worth 3,023.8 MB per commit
and nothing in a green pipeline shows when that order regresses. These tests pin the three
decisions it makes: which bytes count as re-shipped, when the threshold is enforced rather
than reported, and that a measurement it could not take never reads as a pass.

Pure logic only — no docker, no network, no subprocess.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci-layer-delta.py"
_spec = importlib.util.spec_from_file_location("ci_layer_delta", _SCRIPT)
assert _spec is not None and _spec.loader is not None
cld = importlib.util.module_from_spec(_spec)
sys.modules["ci_layer_delta"] = cld
_spec.loader.exec_module(cld)


def manifest(*layers: tuple[str, int]) -> str:
    return json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"digest": "sha256:cfg", "size": 1},
            "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip", "digest": d, "size": s} for d, s in layers],
        }
    )


# ── parse_manifest ────────────────────────────────────────────────────────────────
def test_parses_layers_in_order():
    assert cld.parse_manifest(manifest(("sha256:a", 10), ("sha256:b", 20)), "x") == [
        ("sha256:a", 10),
        ("sha256:b", 20),
    ]


def test_manifest_list_is_broken_not_empty():
    """An index would compare against the wrong thing, so it must not silently pass."""
    raw = json.dumps({"schemaVersion": 2, "manifests": [{"digest": "sha256:a"}]})
    with pytest.raises(cld.Broken, match="manifest LIST"):
        cld.parse_manifest(raw, "x")


@pytest.mark.parametrize(
    "raw",
    ["not json at all", "[]", "{}", json.dumps({"layers": []}), json.dumps({"layers": [{"digest": "sha256:a"}]})],
)
def test_unreadable_manifest_is_broken(raw):
    with pytest.raises(cld.Broken):
        cld.parse_manifest(raw, "x")


# ── resupplied ────────────────────────────────────────────────────────────────────
def test_only_layers_absent_from_the_predecessor_count():
    old = [("sha256:a", 100), ("sha256:b", 200)]
    new = [("sha256:a", 100), ("sha256:c", 7)]
    assert cld.resupplied(old, new) == (7, 107)


def test_a_layer_that_only_moved_position_is_not_re_shipped():
    """Set membership, not position: the node already holds that blob."""
    old = [("sha256:a", 100), ("sha256:b", 200)]
    new = [("sha256:b", 200), ("sha256:a", 100)]
    assert cld.resupplied(old, new) == (0, 300)


def test_a_full_rebuild_counts_every_byte():
    old = [("sha256:a", 100)]
    new = [("sha256:x", 100), ("sha256:y", 50)]
    assert cld.resupplied(old, new) == (150, 150)


@given(
    st.lists(st.tuples(st.text(min_size=1, max_size=4), st.integers(0, 10_000)), max_size=8),
    st.lists(st.tuples(st.text(min_size=1, max_size=4), st.integers(0, 10_000)), max_size=8),
)
def test_re_shipped_never_exceeds_total(old, new):
    fresh, total = cld.resupplied(old, new)
    assert 0 <= fresh <= total


# ── stable_prefix ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "a,b,expected",
    [
        (["1", "2", "3"], ["1", "2", "3"], 3),
        (["1", "2", "3"], ["1", "2", "9"], 2),
        (["1", "2", "3"], ["9", "2", "3"], 0),
        (["1", "2"], ["1", "2", "3"], 2),
        ([], ["1"], 0),
    ],
)
def test_stable_prefix(a, b, expected):
    assert cld.stable_prefix(a, b) == expected


# ── enforced ──────────────────────────────────────────────────────────────────────
def test_a_source_only_commit_is_enforced():
    ok, why = cld.enforced(["src/claude_memory/api/app.py", "tests/test_api.py"])
    assert ok is True
    assert "2 file(s)" in why


def test_a_commit_touching_pyproject_is_reported_not_enforced():
    ok, why = cld.enforced(["src/claude_memory/store.py", "pyproject.toml"])
    assert ok is False
    assert "pyproject.toml" in why


def test_a_commit_touching_the_dockerfile_is_reported_not_enforced():
    """The first build after a reorder re-ships everything by design."""
    ok, _ = cld.enforced(["docker/Dockerfile"])
    assert ok is False


def test_an_unknown_diff_base_is_reported_not_enforced():
    ok, why = cld.enforced([])
    assert ok is False
    assert "unknown" in why


def test_migrations_are_below_the_source_line_so_they_are_enforced():
    assert cld.enforced(["migrations/versions/007_x.py"])[0] is True


# ── the CLI's exit codes, which are the whole contract with CI ────────────────────
def test_missing_predecessor_exits_zero(tmp_path, capsys):
    new = tmp_path / "new.json"
    new.write_text(manifest(("sha256:a", 10)))
    assert cld.main(["manifest", str(tmp_path / "absent.json"), str(new)]) == 0
    assert "no predecessor" in capsys.readouterr().out


def test_present_but_unreadable_predecessor_exits_one(tmp_path, capsys):
    old, new = tmp_path / "old.json", tmp_path / "new.json"
    old.write_text("garbage")
    new.write_text(manifest(("sha256:a", 10)))
    assert cld.main(["manifest", str(old), str(new)]) == 1
    assert "BROKEN" in capsys.readouterr().out


def test_source_only_commit_over_the_limit_fails(tmp_path, capsys):
    old, new, changed = tmp_path / "old.json", tmp_path / "new.json", tmp_path / "changed.txt"
    old.write_text(manifest(("sha256:a", 2_000_000_000)))
    new.write_text(manifest(("sha256:z", 2_000_000_000)))
    changed.write_text("src/claude_memory/store.py\n")
    rc = cld.main(["manifest", str(old), str(new), "--changed-files", str(changed), "--max-new-mb", "200"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "2000.0 MB" in out


def test_source_only_commit_under_the_limit_passes(tmp_path, capsys):
    old, new, changed = tmp_path / "old.json", tmp_path / "new.json", tmp_path / "changed.txt"
    old.write_text(manifest(("sha256:a", 2_000_000_000)))
    new.write_text(manifest(("sha256:a", 2_000_000_000), ("sha256:z", 5_000_000)))
    changed.write_text("src/claude_memory/store.py\n")
    rc = cld.main(["manifest", str(old), str(new), "--changed-files", str(changed), "--max-new-mb", "200"])
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_a_dockerfile_commit_over_the_limit_still_exits_zero(tmp_path, capsys):
    """Reordering the Dockerfile re-ships the whole image once. That is not a failure."""
    old, new, changed = tmp_path / "old.json", tmp_path / "new.json", tmp_path / "changed.txt"
    old.write_text(manifest(("sha256:a", 3_000_000_000)))
    new.write_text(manifest(("sha256:z", 3_000_000_000)))
    changed.write_text("docker/Dockerfile\n")
    rc = cld.main(["manifest", str(old), str(new), "--changed-files", str(changed), "--max-new-mb", "200"])
    assert rc == 0
    assert "REPORT ONLY" in capsys.readouterr().out
