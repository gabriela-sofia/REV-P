"""v2bb deterministic non-download modes and staging test."""

import hashlib
import subprocess
from pathlib import Path
import scripts.v2bb_public_geometry_retrieval_feed_builder as engine

ROOT = Path(__file__).resolve().parents[1]


def snapshot():
    paths = list((ROOT/"datasets").glob("v2bb_*.csv")) + list((ROOT/"datasets"/"schemas").glob("v2bb_*.json"))
    paths += list((ROOT/"docs").glob("v2bb_*.md")) + list((ROOT/"outputs_public").rglob("v2bb_*"))
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def test_two_scan_downloads_are_deterministic_and_staging_unchanged():
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    engine.run("scan_downloads")
    first = snapshot()
    engine.run("scan_downloads")
    assert snapshot() == first
    assert subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True) == staged
