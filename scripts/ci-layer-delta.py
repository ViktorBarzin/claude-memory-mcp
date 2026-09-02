#!/usr/bin/env python3
"""Measure how many bytes this image re-ships against the image it replaces.

WHY THIS EXISTS
---------------
``docker/Dockerfile`` is ordered so that every source-independent layer sits above every
source-dependent one. That ordering is worth 3,023.8 MB of the image's 3,067 MB per
commit, and nothing in a green pipeline or a readable diff shows when it silently
regresses — a single ``COPY`` moved three lines up puts the 2,094.6 MB CUDA install and
the 1,076.0 MB model layer back below ``src/`` and no test notices.

So the property is measured rather than trusted. Every build reports its delta against
the image it replaces, and a commit that cannot have changed a source-independent layer
FAILS if it re-ships more than ``--max-new-mb``.

Layer sizes here are COMPRESSED registry bytes — the bytes a node actually pulls, which
is the quantity the plan this guard comes from is written in
(``infra/docs/plans/2026-09-02-node1-large-image-handling.md``).

USAGE
-----
    ci-layer-delta.py manifest OLD.json NEW.json [--changed-files F] [--max-new-mb 200]
        OLD.json / NEW.json come from `docker buildx imagetools inspect --raw <ref>`.
        A MISSING OLD.json means "no predecessor" and reports nothing, exit 0.
        A PRESENT but unreadable manifest is a broken guard and exits 1, because a guard
        that cannot measure must not read as a pass.

    ci-layer-delta.py local IMAGE_OLD IMAGE_NEW [--min-stable-prefix N]
        Elementwise uncompressed-layer-digest comparison of two LOCAL images, for verifying
        ordering by hand before it reaches CI. Uncompressed layer identity only, no byte
        accounting: `docker image inspect` exposes no per-layer size, and `docker history`
        cannot be mapped onto diff_ids reliably (a RUN that writes nothing — `chmod -R
        a+rX` measured 0.0 MB — is indistinguishable from a metadata instruction).

Stdlib only, by repo convention: this runs in CI with no pip step in front of it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: Paths that feed a layer ABOVE `COPY src/` in docker/Dockerfile. A commit touching any
#: of these legitimately re-ships the expensive layers, so the delta is reported and not
#: enforced. Everything else — src/, migrations/, tests/, docs/, deploy/, .github/ — can
#: only change layers below that line, which is what makes the threshold meaningful.
#: Keep this list in step with the Dockerfile: an out-of-date entry here turns the guard
#: into noise (a legitimate rebuild fails) or a blind spot (a regression passes).
PREFIX_INPUTS = frozenset(
    {
        "pyproject.toml",
        "README.md",
        "alembic.ini",
        "docker/Dockerfile",
        "docker/export_onnx.py",
    }
)


class Broken(Exception):
    """The guard cannot measure. Distinct from "measured, and it is bad"."""


def parse_manifest(raw: str, label: str) -> list[tuple[str, int]]:
    """Return [(digest, compressed_size)] from an OCI/Docker v2 image manifest."""
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Broken(f"{label}: not JSON ({exc})") from exc
    if not isinstance(doc, dict):
        raise Broken(f"{label}: expected a JSON object, got {type(doc).__name__}")
    if "manifests" in doc:
        raise Broken(
            f"{label}: this is a manifest LIST/index, not a single image manifest. The build "
            "pushes single-manifest images on purpose (provenance: false, one platform — "
            "ADR-0002), so an index here means that changed and the comparison would be "
            "against the wrong thing."
        )
    layers = doc.get("layers")
    if not isinstance(layers, list) or not layers:
        raise Broken(f"{label}: no 'layers' array")
    out: list[tuple[str, int]] = []
    for i, layer in enumerate(layers):
        if not isinstance(layer, dict) or "digest" not in layer or "size" not in layer:
            raise Broken(f"{label}: layer {i} has no digest/size")
        out.append((str(layer["digest"]), int(layer["size"])))
    return out


def resupplied(old: list[tuple[str, int]], new: list[tuple[str, int]]) -> tuple[int, int]:
    """(bytes in `new` whose layer is absent from `old`, total bytes in `new`).

    Set membership, not position: a layer that merely MOVED is still already on the node,
    and a layer whose digest is unchanged is not pulled again.
    """
    held = {digest for digest, _ in old}
    fresh = sum(size for digest, size in new if digest not in held)
    return fresh, sum(size for _, size in new)


def stable_prefix(a: list[str], b: list[str]) -> int:
    """How many leading entries the two lists share."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def enforced(changed: list[str]) -> tuple[bool, str]:
    """Should the threshold be enforced for this commit, and why."""
    if not changed:
        return False, "the changed-file set is unknown (no diff base), so report only"
    touched = sorted(set(changed) & PREFIX_INPUTS)
    if touched:
        return False, f"this commit touches {', '.join(touched)}, which feeds a layer above src/"
    return True, f"no source-independent layer input changed ({len(changed)} file(s) changed)"


def _mb(n: int) -> float:
    return n / 1_000_000


def cmd_manifest(args: argparse.Namespace) -> int:
    old_path, new_path = Path(args.old), Path(args.new)
    if not old_path.exists():
        print(f"[layer-delta] no predecessor manifest at {old_path} — nothing to compare (first build?)")
        return 0
    old = parse_manifest(old_path.read_text(), str(old_path))
    new = parse_manifest(new_path.read_text(), str(new_path))

    changed: list[str] = []
    if args.changed_files:
        cf = Path(args.changed_files)
        if cf.exists():
            changed = [line.strip() for line in cf.read_text().splitlines() if line.strip()]

    fresh, total = resupplied(old, new)
    held = {digest for digest, _ in old}
    print(f"[layer-delta] {len(old)} layer(s) before, {len(new)} after")
    for i, (digest, size) in enumerate(new):
        print(f"[layer-delta]   {i:2}  {digest[:26]}  {_mb(size):9.1f} MB  {'HELD' if digest in held else 'NEW'}")
    pct = (100.0 * fresh / total) if total else 0.0
    print(f"[layer-delta] re-shipped {_mb(fresh):.1f} MB of {_mb(total):.1f} MB ({pct:.1f}%)")

    do_enforce, why = enforced(changed)
    if not do_enforce:
        print(f"[layer-delta] REPORT ONLY: {why}")
        return 0
    if _mb(fresh) > args.max_new_mb:
        print(f"[layer-delta] FAIL: {why}, yet {_mb(fresh):.1f} MB re-shipped (limit {args.max_new_mb} MB).")
        print(
            "[layer-delta] Two things cause this. Most likely: docker/Dockerfile's layer order "
            "has regressed, so something that does not depend on src/ now sits below it and "
            "every commit re-ships it — move it back above the 'everything below this line "
            "depends on src/' marker. Otherwise, if the layer that re-shipped is the ~1,076 MB "
            "model layer alone, the GHA cache evicted the exporter stage and the re-run did not "
            "reproduce it byte for byte, which `touch -d @0` can only half fix (open question 6 "
            "in infra/docs/plans/2026-09-02-node1-large-image-handling.md). The layer table "
            "above says which case this is."
        )
        return 1
    print(f"[layer-delta] PASS: {why}, and {_mb(fresh):.1f} MB re-shipped (limit {args.max_new_mb} MB).")
    return 0


def _diff_ids(image: str) -> list[str]:
    """The image's uncompressed layer digests, bottom-up.

    The key moved: `docker image inspect` exposes these as `RootFS.Layers` on Docker 29
    and as `RootFS.Diff_IDs` on older CLIs, so read the whole object and take whichever is
    there rather than guessing — a missing key here would otherwise read as "no layers".
    """
    proc = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .RootFS}}", image],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise Broken(f"docker image inspect {image} failed: {proc.stderr.strip()}")
    rootfs = json.loads(proc.stdout)
    ids = rootfs.get("Layers") or rootfs.get("Diff_IDs")
    if not isinstance(ids, list) or not ids:
        raise Broken(f"{image}: RootFS carries no layer list (keys: {sorted(rootfs)})")
    return [str(x) for x in ids]


def cmd_local(args: argparse.Namespace) -> int:
    a, b = _diff_ids(args.old), _diff_ids(args.new)
    print(f"[layer-delta] {args.old}: {len(a)} layer(s)   {args.new}: {len(b)} layer(s)")
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "-"
        y = b[i] if i < len(b) else "-"
        print(f"[layer-delta]   {i:2}  {x[:26]}  {y[:26]}  {'HELD' if x == y else 'CHURNED'}")
    n = stable_prefix(a, b)
    print(f"[layer-delta] stable prefix: {n} of {len(a)} layer(s) identical from the base up")
    if n < args.min_stable_prefix:
        print(
            f"[layer-delta] FAIL: expected at least {args.min_stable_prefix} identical leading "
            f"layers, got {n}. A source-independent layer churned, so the Dockerfile's ordering "
            "has regressed."
        )
        return 1
    print(f"[layer-delta] PASS: at least {args.min_stable_prefix} leading layers held.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("manifest", help="compare two registry manifests (what CI runs)")
    m.add_argument("old")
    m.add_argument("new")
    m.add_argument("--changed-files", help="file holding one changed path per line")
    m.add_argument("--max-new-mb", type=float, default=200.0)
    m.set_defaults(func=cmd_manifest)

    lo = sub.add_parser("local", help="compare two local images' diff_ids")
    lo.add_argument("old")
    lo.add_argument("new")
    lo.add_argument("--min-stable-prefix", type=int, default=1)
    lo.set_defaults(func=cmd_local)
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except Broken as exc:
        print(f"[layer-delta] BROKEN: {exc}")
        print("[layer-delta] refusing to report a pass from a measurement that did not happen.")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
