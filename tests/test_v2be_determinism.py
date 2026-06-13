"""v2be deterministic execution and staging tests."""

import hashlib
import subprocess
from pathlib import Path

import scripts.v2be_tp1_patch_boundary_integration_gate as engine

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "datasets").glob("v2be_*.csv"))
    paths += list((ROOT / "datasets/schemas").glob("v2be_*.json"))
    paths += list((ROOT / "docs").glob("v2be_*.md")) + list((ROOT / "outputs_public").rglob("v2be_*"))
    paths += [ROOT / "datasets/external_sources/recife_minimal_tp/patch_boundary_REC_00019/FILL_THIS_PATCH_BOUNDARY.autofill_tp1_candidate_v2be.csv"]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def test_two_runs_are_deterministic_and_staging_unchanged():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert engine.run()[0] == 0
    first = snapshot()
    assert engine.run()[0] == 0
    assert snapshot() == first
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True) == staged
