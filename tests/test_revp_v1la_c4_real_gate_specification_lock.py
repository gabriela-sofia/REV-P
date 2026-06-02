"""Tests for v1la C4 real gate specification."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1la_c4_real_gate_specification_lock.py"
MATRIX = ROOT / "datasets/c4_real_gate_specification_matrix.csv"
CONTRACT = ROOT / "datasets/c4_real_minimum_evidence_contract.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-contract"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert MATRIX.exists() and CONTRACT.exists()


def test_contract_separates_control_and_operational_c4() -> None:
    contract = {r["class_name"]: r for r in rows(CONTRACT)}
    assert contract["HARD_STABLE_CONTROL"]["can_be_formal_negative"] == "false"
    assert contract["C4_CONTROL_EXPERIMENT"]["can_open_c4_operational"] == "false"
    assert contract["C4_OPERATIONAL"]["can_open_c4_operational"] == "true"


def test_absence_of_record_cannot_satisfy_negative_gate() -> None:
    negative_gate = [r for r in rows(MATRIX) if r["gate_name"] == "formal_negative_gate"][0]
    assert "absence of record" in negative_gate["cannot_be_satisfied_by"]
