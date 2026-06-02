"""Tests for v1lh C4 real transition decision."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1lh_c4_real_transition_decision_after_gt_recovery.py"
OUT = ROOT / "datasets/c4_real_transition_decision_matrix.csv"
TRAIN = ROOT / "datasets/c4_real_training_readiness_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-decision"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists() and TRAIN.exists()


def test_c4_operational_requires_positive_negative_and_split() -> None:
    decision = rows(OUT)[0]
    assert decision["formal_positive_count"] == "9"
    assert decision["formal_negative_count"] == "0"
    assert decision["decision"] == "C4_BLOCKED_NO_FORMAL_NEGATIVES"
    assert decision["can_create_operational_label"] == "false"
    assert decision["can_train_model"] == "false"
    assert rows(TRAIN)[0]["can_save_weights"] == "false"
