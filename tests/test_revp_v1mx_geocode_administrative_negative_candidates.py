"""Tests for v1mx administrative negative geocoding."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mx_geocode_administrative_negative_candidates.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1mw_administrative_negative_candidate_miner.py"
REG = ROOT / "datasets/administrative_negative_geocoding_registry.csv"
MATRIX = ROOT / "datasets/administrative_negative_spatial_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_generic_review_area_does_not_pass_coordinate_gate() -> None:
    for row in rows(REG):
        if row["review_area_only_flag"] == "true":
            assert row["coordinate_or_precise_address_gate"] == "FAIL"


def test_public_outputs_are_sanitized() -> None:
    low = REG.read_text(encoding="utf-8").lower() + MATRIX.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
