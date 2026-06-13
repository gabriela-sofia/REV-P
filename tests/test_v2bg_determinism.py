"""v2bg deterministic execution and staging tests."""

import hashlib
import subprocess
from pathlib import Path

import scripts.v2bg_charter758_deep_product_mining_tp2_recovery as engine

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT / "datasets").glob("v2bg_*.csv")) + list((ROOT / "datasets/schemas").glob("v2bg_*.json"))
    paths += list((ROOT / "docs").glob("v2bg_*.md")) + list((ROOT / "outputs_public").rglob("v2bg_*"))
    paths += list((ROOT / "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/digitization").glob("*"))
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths if path.is_file()}


def test_two_full_runs_are_deterministic_and_staging_unchanged():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert engine.run(mode="full")[0] == 0
    first = snapshot()
    assert engine.run(mode="full")[0] == 0
    assert snapshot() == first
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True) == staged
