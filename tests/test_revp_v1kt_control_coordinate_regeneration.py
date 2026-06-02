"""Tests for v1kt control coordinate regeneration."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/regenerated_control_coordinate_registry.csv"
AUDIT = ROOT / "datasets/regenerated_control_sampling_audit.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    for name, args in [
        ("revp_v1ks_control_coordinate_integrity_audit.py", ["--force", "--emit-audit"]),
        ("revp_v1kt_control_coordinate_regeneration.py", ["--force", "--limit", "50", "--emit-regenerated"]),
    ]:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert AUDIT.exists()


def test_regenerated_coordinates_are_review_controls_not_negatives() -> None:
    regs = rows(OUT)
    assert len(regs) == 50
    assert all(r["coordinate_status"] == "VALID_DERIVED_EPSG4326" for r in regs)
    assert all(r["control_status"] == "CONSERVATIVE_STABLE_CONTROL_REVIEW" for r in regs)
    assert all(r["can_be_formal_negative"] == "false" for r in regs)
