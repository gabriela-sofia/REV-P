"""Tests for v1lw inventory-based negative adjudication."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lw_inventory_based_negative_adjudication.py"
REG = ROOT / "datasets/inventory_based_negative_adjudication_registry.csv"
MATRIX = ROOT / "datasets/inventory_based_negative_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_no_inventory_negative_without_complete_inventory_and_aoi() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["complete_inventory_gate"] != "PASS" or matrix["official_aoi_gate"] != "PASS":
        assert matrix["formal_negative_candidate_count"] == "0"
        assert rows(REG)[0]["decision"].startswith("NEGATIVE_BLOCKED")


def test_stable_control_not_mentioned_as_formal_negative_source() -> None:
    low = REG.read_text(encoding="utf-8").lower()
    assert "hard_stable_control" not in low and "conservative_stable_control" not in low
