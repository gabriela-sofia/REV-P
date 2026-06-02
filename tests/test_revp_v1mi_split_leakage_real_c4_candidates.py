"""Tests for v1mi split/leakage real C4 candidates."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mi_split_leakage_real_c4_candidates.py"
AUDIT = ROOT / "datasets/real_c4_split_leakage_audit.csv"
MATRIX = ROOT / "datasets/real_c4_split_readiness_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert AUDIT.exists() and MATRIX.exists()


def test_split_blocked_without_formal_negative() -> None:
    audit = rows(AUDIT)[0]
    matrix = rows(MATRIX)[0]
    if audit["formal_negative_count"] == "0":
        assert matrix["split_leakage_gate"] == "FAIL"
        assert matrix["remaining_blocker"] == "NO_FORMAL_NEGATIVES"


def test_no_training_flags() -> None:
    assert rows(MATRIX)[0]["can_train_model"] == "false"
