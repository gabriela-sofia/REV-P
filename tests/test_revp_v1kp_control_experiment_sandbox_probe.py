"""Tests for v1kp guarded sandbox probe."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/control_experiment_sandbox_probe_registry.csv"
BOUNDARY = ROOT / "datasets/control_experiment_probe_boundary_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    chain = [
        ("revp_v1kk_control_candidate_pool_expansion.py", ["--force", "--limit", "50", "--emit-pool"]),
        ("revp_v1kl_control_multimodal_patch_acquisition.py", ["--force", "--emit-patch-qa"]),
        ("revp_v1km_positive_control_numeric_feature_table.py", ["--force", "--emit-feature-table"]),
        ("revp_v1kn_hard_stable_control_numeric_gate.py", ["--force", "--emit-gates"]),
        ("revp_v1ko_control_experiment_split_leakage_protocol.py", ["--force", "--emit-split"]),
        ("revp_v1kp_control_experiment_sandbox_probe.py", ["--force", "--emit-probe"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert BOUNDARY.exists()


def test_probe_does_not_run_when_readiness_is_blocked() -> None:
    probe = rows(OUT)[0]
    if probe["readiness_decision"] != "CONTROL_EXPERIMENT_READY_FOR_LOCAL_SANDBOX":
        assert probe["probe_executed"] == "false"
        assert probe["model_family"] == "NOT_RUN"
    assert probe["weights_saved"] == "false"


def test_claim_boundaries_are_false_for_operational_use() -> None:
    probe = rows(OUT)[0]
    assert probe["can_claim_detection"] == "false"
    assert probe["can_claim_prediction"] == "false"
    assert probe["can_train_operational_model"] == "false"
    assert probe["can_create_operational_label"] == "false"
