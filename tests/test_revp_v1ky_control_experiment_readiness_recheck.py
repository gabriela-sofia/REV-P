"""Tests for v1ky control experiment readiness recheck."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = ROOT / "scripts/protocolo_c/revp_v1ky_control_experiment_readiness_recheck.py"
OUT = ROOT / "datasets/control_experiment_readiness_recheck_matrix.csv"
BOUNDARY = ROOT / "datasets/control_experiment_claim_boundary_recheck.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force", "--emit-recheck"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert BOUNDARY.exists()


def test_control_experiment_ready_only_with_features_controls_and_leakage() -> None:
    matrix = rows(OUT)[0]
    if matrix["decision"] == "CONTROL_EXPERIMENT_READY_FOR_LOCAL_SANDBOX":
        assert int(matrix["feature_complete_control_count"]) >= 9
        assert int(matrix["hard_stable_control_count"]) + int(matrix["strong_stable_control_review_count"]) >= 9
        assert matrix["leakage_gate"] == "PASS"
    assert matrix["c4_operational_status"] == "BLOCKED"
    assert matrix["can_create_operational_label"] == "false"
    assert matrix["can_train_model"] == "false"
