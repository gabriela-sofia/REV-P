"""v2ba deterministic source-scan tests."""

import hashlib
from pathlib import Path

import scripts.v2ba_minimal_real_geometry_acquisition_workbench as engine


ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())


def snapshot():
    paths = list((ROOT / "datasets").glob("v2ba_*.csv"))
    paths += list((ROOT / "datasets" / "schemas").glob("v2ba_*.schema.json"))
    paths += list((ROOT / "docs").glob("v2ba_*.md"))
    paths += list((ROOT / "outputs_public").rglob("v2ba_*"))
    paths += list((ROOT / "datasets" / "external_sources" / "recife_minimal_tp").rglob("FILL_THIS_*.csv"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def test_two_source_scans_are_deterministic_and_do_not_touch_staging():
    before_staged = __import__("subprocess").check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    code, _ = engine.run("source_scan")
    assert code == 0
    first = snapshot()
    code, _ = engine.run("source_scan")
    assert code == 0 and snapshot() == first
    after_staged = __import__("subprocess").check_output(
        ["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert after_staged == before_staged
