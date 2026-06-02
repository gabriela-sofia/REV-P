"""Tests for v1ln formal no-occurrence negative miner."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ln_formal_no_occurrence_negative_miner.py"
REG = ROOT / "datasets/formal_no_occurrence_negative_candidate_registry.csv"
MATRIX = ROOT / "datasets/formal_no_occurrence_negative_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_absence_of_record_never_becomes_formal_negative() -> None:
    data = rows(REG)
    for row in data:
        if row["explicit_no_occurrence_gate"] != "PASS":
            assert row["decision"] != "FORMAL_NEGATIVE_CANDIDATE"


def test_formal_negative_requires_patch_and_leakage_gates() -> None:
    for row in rows(REG):
        if row["decision"] == "FORMAL_NEGATIVE_CANDIDATE":
            assert row["patch_extractability_gate"] == "PASS"
            assert row["leakage_gate"] == "PASS"
    assert rows(MATRIX)[0]["can_train_model"] == "false"
