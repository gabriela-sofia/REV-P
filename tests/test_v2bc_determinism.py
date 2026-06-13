"""v2bc deterministic execution and staging tests."""

import hashlib
import subprocess
from pathlib import Path

import scripts.v2bc_recife_gis_digitization_workbench as engine

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "datasets").glob("v2bc_*.csv"))
    paths += list((ROOT / "datasets" / "schemas").glob("v2bc_*.json"))
    paths += list((ROOT / "datasets" / "gis_workbench" / "recife_minimal_tp").rglob("*"))
    paths += list((ROOT / "docs").glob("v2bc_*.md")) + list((ROOT / "outputs_public").rglob("v2bc_*"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def test_two_runs_are_deterministic_and_staging_unchanged():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    engine.run()
    first = snapshot()
    engine.run()
    assert snapshot() == first
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True) == staged
