"""v2bh determinism and Git boundary tests."""

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs_public/execution_reports/v2bh_charter758_recife_product_georeferencing_digitization_summary.json"
CANDIDATE = ROOT / "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/derived/event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_full_is_deterministic_and_v2bh_is_not_staged():
    before = (digest(SUMMARY), digest(CANDIDATE))
    subprocess.run([sys.executable, str(ROOT / "scripts/run_v2bh_charter758_recife_product_georeferencing_digitization.py"), "--mode", "full"], cwd=ROOT, check=True)
    assert before == (digest(SUMMARY), digest(CANDIDATE))
    staged = subprocess.check_output(["git", "diff", "--cached", "--name-only"], cwd=ROOT, text=True)
    assert "v2bh" not in staged.lower()
