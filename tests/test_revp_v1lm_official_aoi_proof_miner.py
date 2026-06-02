"""Tests for v1lm official AOI proof miner."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lm_official_aoi_proof_miner.py"
MATRIX = ROOT / "datasets/official_aoi_gate_matrix.csv"
REG = ROOT / "datasets/official_aoi_proof_candidate_registry.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert MATRIX.exists() and REG.exists()


def test_layer_bounds_alone_do_not_pass_official_aoi_gate() -> None:
    matrix = rows(MATRIX)[0]
    assert matrix["layer_bounds_only_gate"] == "FAIL"
    if matrix["official_aoi_gate"] != "PASS":
        assert matrix["decision"] != "OFFICIAL_AOI_CONFIRMED"


def test_public_outputs_have_no_private_paths() -> None:
    low = (MATRIX.read_text(encoding="utf-8") + REG.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
