"""Tests for v1mr real C4 split/leakage recheck."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mr_real_c4_split_leakage_recheck.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1mo_formal_negative_feature_adjudication_strict.py"
REG = ROOT / "datasets/real_c4_split_leakage_recheck_registry.csv"
MATRIX = ROOT / "datasets/real_c4_split_leakage_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_split_blocks_without_formal_negative() -> None:
    row = rows(REG)[0]
    if row["formal_negative_count"] == "0":
        assert row["decision"] == "BLOCKED_NO_FORMAL_NEGATIVE"
        assert rows(MATRIX)[0]["split_leakage_gate"] == "FAIL"


def test_training_remains_false() -> None:
    assert rows(MATRIX)[0]["can_train_model"] == "false"
