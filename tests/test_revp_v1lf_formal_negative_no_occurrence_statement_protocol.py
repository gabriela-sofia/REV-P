"""Tests for v1lf formal negatives from no-occurrence statements."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lf_formal_negative_no_occurrence_statement_protocol.py"
OUT = ROOT / "datasets/formal_negative_no_occurrence_statement_registry.csv"
GATES = ROOT / "datasets/formal_negative_no_occurrence_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-no-occurrence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and GATES.exists()


def test_no_occurrence_requires_explicit_official_location_date_and_leakage() -> None:
    assert sum(1 for r in rows(OUT) if r["decision"] == "FORMAL_NEGATIVE_CANDIDATE") == 0
    for row in rows(OUT):
        assert row["can_create_operational_label"] == "false"
        assert row["can_train_model"] == "false"
