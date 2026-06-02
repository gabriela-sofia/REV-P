"""Tests for v1mg positive expansion from new assets."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mg_formal_positive_expansion_new_assets.py"
REG = ROOT / "datasets/formal_positive_expanded_candidate_registry.csv"
MATRIX = ROOT / "datasets/formal_positive_expanded_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert REG.exists() and MATRIX.exists()


def test_positive_expansion_does_not_create_operational_label() -> None:
    assert all(r["can_create_operational_label"] == "false" for r in rows(REG))
    assert rows(MATRIX)[0]["can_train_model"] == "false"


def test_positive_candidate_needs_patch_qa() -> None:
    for row in rows(REG):
        if row["decision"] == "FORMAL_POSITIVE_CANDIDATE_NEEDS_PATCH_QA":
            assert row["patch_extractability_gate"] in {"UNKNOWN", "PASS"}
