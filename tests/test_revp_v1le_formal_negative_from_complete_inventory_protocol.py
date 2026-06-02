"""Tests for v1le formal negatives from complete inventory."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1le_formal_negative_from_complete_inventory_protocol.py"
OUT = ROOT / "datasets/formal_negative_from_inventory_candidate_registry.csv"
GATES = ROOT / "datasets/formal_negative_from_inventory_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-negative"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and GATES.exists()


def test_inventory_negatives_block_without_completeness_and_aoi() -> None:
    assert sum(1 for r in rows(OUT) if r["decision"] == "FORMAL_NEGATIVE_CANDIDATE") == 0
    assert all(r["can_create_operational_label"] == "false" for r in rows(OUT))
    assert any(r["complete_inventory_gate"] == "FAIL" for r in rows(GATES))
