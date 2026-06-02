"""Tests for v1me formal negative adjudication from all assets."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1me_formal_negative_adjudication_all_assets.py"
REG = ROOT / "datasets/formal_negative_all_assets_candidate_registry.csv"
MATRIX = ROOT / "datasets/formal_negative_all_assets_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_formal_negative_requires_all_hard_gates() -> None:
    for row in rows(REG):
        if row["decision"] == "FORMAL_NEGATIVE_CANDIDATE":
            for gate in ["explicit_negative_statement_gate", "phenomenon_specific_gate", "date_gate", "location_gate", "geometry_or_coordinate_gate", "patch_extractability_gate", "positive_buffer_exclusion_gate", "leakage_gate"]:
                assert row[gate] == "PASS"


def test_absence_and_stable_control_not_used() -> None:
    low = REG.read_text(encoding="utf-8").lower()
    assert "absence" not in low and "stable_control" not in low
    assert rows(MATRIX)[0]["can_train_model"] == "false"
