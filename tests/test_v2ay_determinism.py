"""v2ay - deterministic additive replay tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import scripts.v2ay_event_scope_reconciliation_turning_point_engine as engine


ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "datasets").glob("v2ay_*.csv"))
    paths += list((ROOT / "datasets" / "schemas").glob("v2ay_*.schema.json"))
    paths += list((ROOT / "docs").glob("v2ay_*.md"))
    paths += list((ROOT / "outputs_public" / "execution_reports").glob("v2ay_event_scope_*.md"))
    paths += list((ROOT / "outputs_public" / "execution_reports").glob("v2ay_event_scope_*.json"))
    paths += list((ROOT / "outputs_public" / "logs_summary").glob("v2ay_event_scope_*.txt"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def test_two_v2ay_runs_are_deterministic_and_do_not_touch_prior_outputs():
    protected = [
        ROOT / "datasets" / "v2aw_patch_geometry_sources_template.csv",
        ROOT / "datasets" / "v2av_patch_boundary_recovery_queue.csv",
        ROOT / "datasets" / "v2au_overlay_review_queue.csv",
        ROOT / "datasets" / "v2ax_recife_manual_intake_manifest.csv",
        ROOT / "datasets" / "protocolo_c" / "v2ay_orchestrator_manifest.csv",
    ]
    before_protected = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected}
    code, _ = engine.run()
    assert code == 0
    first = snapshot()
    code, _ = engine.run()
    assert code == 0
    assert snapshot() == first
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in protected} == before_protected
