"""Tests for v1ne gazette candidate geocoding gates."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ne_address_locality_extraction_geocoding_acts.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1nd_negative_semantics_strict_miner_from_acts.py"
GEO = ROOT / "datasets/gazette_negative_address_geocoding_registry.csv"
MATRIX = ROOT / "datasets/gazette_negative_spatial_specificity_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=180)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert GEO.exists() and MATRIX.exists()


def test_generic_neighborhood_is_not_formal_geocode() -> None:
    for row in rows(GEO):
        if row["review_area_only_flag"] == "true":
            assert row["coordinate_or_geocodable_address_gate"] == "FAIL"
            assert row["patch_extractability_gate"] == "FAIL"


def test_public_geocoding_registry_has_no_private_paths() -> None:
    low = GEO.read_text(encoding="utf-8").lower() + MATRIX.read_text(encoding="utf-8").lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
