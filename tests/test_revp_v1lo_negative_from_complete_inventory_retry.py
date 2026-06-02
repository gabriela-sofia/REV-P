"""Tests for v1lo complete-inventory negative retry."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lo_negative_from_complete_inventory_retry.py"
REG = ROOT / "datasets/formal_negative_inventory_retry_registry.csv"
MATRIX = ROOT / "datasets/formal_negative_inventory_retry_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_negative_not_promoted_without_inventory_and_aoi_gates() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["complete_inventory_gate"] != "PASS" or matrix["official_aoi_gate"] != "PASS":
        assert matrix["formal_negative_inventory_candidate_count"] == "0"
        assert rows(REG)[0]["decision"].startswith("NEGATIVE_BLOCKED")


def test_no_operational_training_flags() -> None:
    assert rows(MATRIX)[0]["can_train_model"] == "false"
    assert rows(MATRIX)[0]["can_create_operational_label"] == "false"
