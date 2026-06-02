"""Tests for v1kx hard stable control reevaluation."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1kx_hard_stable_control_reevaluation_after_real_patches.py"
OUT = ROOT / "datasets/hard_stable_control_after_real_patch_registry.csv"
GATES = ROOT / "datasets/hard_stable_control_after_real_patch_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-reevaluation"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert GATES.exists()


def test_hard_and_strong_controls_are_not_operational_labels() -> None:
    regs = rows(OUT)
    assert len([r for r in regs if r["final_control_class"] in {"HARD_STABLE_CONTROL", "STRONG_STABLE_CONTROL_REVIEW"}]) >= 9
    assert all(r["can_be_formal_negative"] == "false" for r in regs)
    assert all(r["can_create_operational_label"] == "false" for r in regs)
    assert all(r["can_train_model"] == "false" for r in regs)
