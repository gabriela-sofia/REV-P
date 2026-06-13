"""v2ax - deterministic controlled replay tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import scripts.v2ax_recife_geometry_intake_pack_engine as engine


ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    roots = [
        ROOT / "datasets" / "manual_intake" / "recife_p1",
        ROOT / "datasets" / "examples" / "v2ax_recife_geometry_pack",
    ]
    paths = [
        ROOT / "datasets" / "v2ax_recife_manual_intake_manifest.csv",
        ROOT / "datasets" / "v2ax_recife_manual_intake_validation.csv",
        ROOT / "outputs_public" / "execution_reports" / "v2ax_recife_geometry_intake_pack_summary.json",
        ROOT / "outputs_public" / "execution_reports" / "v2ax_recife_geometry_intake_pack_report.md",
    ]
    for root in roots:
        paths.extend(path for path in root.rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def test_two_replays_are_deterministic():
    code, _ = engine.run()
    assert code == 0
    first = snapshot()
    code, _ = engine.run()
    assert code == 0
    assert snapshot() == first
