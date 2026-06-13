"""v2bf deterministic execution and staging tests."""

import hashlib
import subprocess
from pathlib import Path

import scripts.v2bf_recife_observed_event_polygon_tp2_gate as engine

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "datasets").glob("v2bf_*.csv")) + list((ROOT / "datasets/schemas").glob("v2bf_*.json"))
    paths += list((ROOT / "docs").glob("v2bf_*.md")) + list((ROOT / "outputs_public").rglob("v2bf_*"))
    paths += list((ROOT / "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30").glob("*v2bf.csv"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def test_two_full_runs_are_deterministic_and_staging_unchanged():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert engine.run(mode="full")[0] == 0
    first = snapshot()
    assert engine.run(mode="full")[0] == 0
    assert snapshot() == first
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True) == staged
