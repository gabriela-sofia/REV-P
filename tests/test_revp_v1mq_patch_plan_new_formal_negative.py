"""Tests for v1mq patch plan for new formal negatives."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1mq_patch_plan_new_formal_negative.py"
DEP = ROOT / "scripts/protocolo_c/revp_v1mo_formal_negative_feature_adjudication_strict.py"
PLAN = ROOT / "datasets/new_formal_negative_patch_plan_registry.csv"
MATRIX = ROOT / "datasets/new_formal_negative_patch_extractability_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    subprocess.run([sys.executable, str(DEP), "--force", "--emit-evidence"], cwd=ROOT, check=True, timeout=120)
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-evidence"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert PLAN.exists() and MATRIX.exists()


def test_no_formal_negative_blocks_patch_plan() -> None:
    matrix = rows(MATRIX)[0]
    if matrix["formal_negative_count"] == "0":
        assert matrix["decision"] == "NO_FORMAL_NEGATIVE_TO_PATCH"
        assert matrix["patch_extractability_gate"] == "FAIL"


def test_no_operational_label_created() -> None:
    assert rows(MATRIX)[0]["can_create_operational_label"] == "false"
    assert rows(MATRIX)[0]["can_train_model"] == "false"
