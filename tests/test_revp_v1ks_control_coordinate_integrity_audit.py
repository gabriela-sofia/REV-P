"""Tests for v1ks control coordinate integrity audit."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ks_control_coordinate_integrity_audit.py"
OUT = ROOT / "datasets/control_coordinate_integrity_audit.csv"
DECISION = ROOT / "datasets/control_coordinate_repair_decision_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-audit"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert DECISION.exists()


def test_coordinates_are_valid_but_latlon_was_missing() -> None:
    decision = rows(DECISION)[0]
    assert int(decision["valid_projected_coordinate_count"]) == 50
    assert int(decision["derived_latlon_count"]) == 50
    assert decision["primary_coordinate_blocker"] == "LATLON_ABSENT_BUT_DERIVABLE_FROM_EPSG31983"


def test_invalid_or_missing_coordinates_do_not_create_labels() -> None:
    audit = rows(OUT)
    assert all(r["can_create_operational_label"] == "false" for r in audit)
    assert all(r["can_train_model"] == "false" for r in audit)
    assert all(r["can_be_formal_negative"] == "false" for r in audit)
