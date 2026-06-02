"""Tests for v1lc official AOI and inventory scope builder."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lc_official_aoi_inventory_scope_builder.py"
OUT = ROOT / "datasets/official_inventory_aoi_scope_registry.csv"
GATES = ROOT / "datasets/official_inventory_scope_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-scope"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and GATES.exists()


def test_layer_bounds_do_not_create_official_aoi_gate() -> None:
    gate = rows(GATES)[0]
    assert gate["official_aoi_gate"] == "FAIL"
    assert gate["negative_derivation_scope_gate"] == "FAIL"


def test_review_scope_cannot_support_formal_negative_derivation() -> None:
    scope = rows(OUT)[0]
    assert scope["can_support_formal_negative_derivation"] == "false"
