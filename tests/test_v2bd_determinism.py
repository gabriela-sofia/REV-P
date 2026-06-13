"""v2bd deterministic execution and staging tests."""

import hashlib
import subprocess
from pathlib import Path

import scripts.v2bd_sentinel_patch_footprint_recovery_drilldown as engine

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "datasets").glob("v2bd_*.csv"))
    paths += list((ROOT / "datasets" / "schemas").glob("v2bd_*.json"))
    paths += list((ROOT / "docs").glob("v2bd_*.md")) + list((ROOT / "outputs_public").rglob("v2bd_*"))
    paths += [ROOT / "datasets" / "external_sources" / "recife_minimal_tp" / "derived" / "patch_boundary_REC_00019_from_lineage.geojson"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def test_two_runs_are_deterministic_and_staging_unchanged():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    engine.run()
    first = snapshot()
    engine.run()
    assert snapshot() == first
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True) == staged
