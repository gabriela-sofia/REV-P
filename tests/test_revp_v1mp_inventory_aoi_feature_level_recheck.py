"""Tests for v1mp inventory/AOI feature-level recheck."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mp_inventory_aoi_feature_level_recheck.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1ml_full_structured_asset_row_exhaust.py"
REG = ROOT / "datasets/inventory_aoi_feature_level_recheck_registry.csv"
MATRIX = ROOT / "datasets/inventory_aoi_feature_level_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=240)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_extent_only_does_not_prove_aoi() -> None:
    for row in rows(REG):
        if row["extent_only_flag"] == "true":
            assert row["supports_official_aoi_gate"] == "FAIL"


def test_public_outputs_are_sanitized() -> None:
    low = (REG.read_text(encoding="utf-8") + MATRIX.read_text(encoding="utf-8")).lower()
    assert "c:\\" not in low and "c:/" not in low and "gabriela" not in low
