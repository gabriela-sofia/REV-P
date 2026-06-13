"""v2az - deterministic dry-run and prior-output preservation tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import scripts.v2az_turning_point_replay_orchestrator as engine


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot():
    paths = list((ROOT / "datasets").glob("v2az_*.csv"))
    paths += list((ROOT / "datasets" / "schemas").glob("v2az_*.schema.json"))
    paths += list((ROOT / "docs").glob("v2az_*.md"))
    paths += list((ROOT / "outputs_public" / "execution_reports").glob("v2az_*.md"))
    paths += list((ROOT / "outputs_public" / "execution_reports").glob("v2az_*.json"))
    paths += list((ROOT / "outputs_public" / "logs_summary").glob("v2az_*.txt"))
    paths += list((ROOT / "datasets" / "manual_intake" / "recife_p1").glob("minimal_turning_point_*"))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def test_two_dry_runs_are_deterministic_and_preserve_prior_outputs():
    protected = [
        ROOT / "datasets" / "v2aw_patch_geometry_sources_template.csv",
        ROOT / "datasets" / "v2av_patch_boundary_recovery_queue.csv",
        ROOT / "datasets" / "v2au_overlay_review_queue.csv",
        ROOT / "datasets" / "v2ax_recife_manual_intake_manifest.csv",
        ROOT / "datasets" / "v2ay_turning_point_readiness_gate.csv",
    ]
    before = {path: digest(path) for path in protected}
    code, _ = engine.run("dry_run")
    assert code == 0
    first = snapshot()
    code, _ = engine.run("dry_run")
    assert code == 0
    assert snapshot() == first
    assert {path: digest(path) for path in protected} == before
