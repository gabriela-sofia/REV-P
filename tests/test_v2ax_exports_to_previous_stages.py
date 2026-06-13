"""v2ax - exports and previous-stage preservation tests."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import scripts.v2ax_recife_geometry_intake_pack_engine as engine


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_blocked_state_exports_headers_but_no_false_readiness():
    names = [
        "v2ax_ready_to_feed_v2aw_patch_sources.csv", "v2ax_ready_to_feed_v2aw_event_sources.csv",
        "v2ax_ready_to_feed_v2av_patch_sources.csv", "v2ax_ready_to_feed_v2au_geometry_sources.csv",
    ]
    for name in names:
        path = ROOT / "datasets" / name
        assert path.is_file()
        assert path.read_text(encoding="utf-8").splitlines()[0]
        assert read_csv(path) == []


def test_manifest_never_claims_readiness_without_geometry():
    rows = read_csv(ROOT / "datasets" / "v2ax_recife_manual_intake_manifest.csv")
    assert len(rows) == 55
    assert all(row["ready_for_v2aw"] == "false" for row in rows)
    assert all(row["ready_for_v2av"] == "false" for row in rows)
    assert all(row["ready_for_v2au"] == "false" for row in rows)


def test_v2ax_replay_does_not_overwrite_v2aw_v2av_v2au_outputs():
    protected = [
        ROOT / "datasets" / "v2aw_patch_geometry_sources_template.csv",
        ROOT / "datasets" / "v2av_patch_boundary_recovery_queue.csv",
        ROOT / "datasets" / "v2au_overlay_review_queue.csv",
    ]
    before = {path: digest(path) for path in protected}
    code, _ = engine.run()
    assert code == 0
    assert {path: digest(path) for path in protected} == before
