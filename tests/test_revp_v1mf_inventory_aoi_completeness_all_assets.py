"""Tests for v1mf inventory/AOI completeness from all assets."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mf_inventory_aoi_completeness_all_assets.py"
REG = ROOT / "datasets/inventory_aoi_completeness_all_assets_registry.csv"
MATRIX = ROOT / "datasets/inventory_aoi_completeness_all_assets_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_extent_or_context_does_not_prove_aoi() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["official_aoi_gate"] != "PASS":
        assert matrix["aoi_decision"] != "OFFICIAL_AOI_CONFIRMED"
    assert matrix["can_create_operational_label"] == "false"


def test_completeness_gate_strict() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["complete_inventory_gate"] != "PASS":
        assert matrix["decision"] != "COMPLETE_EVENT_INVENTORY_CONFIRMED"
