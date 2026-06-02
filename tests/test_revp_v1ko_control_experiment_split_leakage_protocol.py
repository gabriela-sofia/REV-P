"""Tests for v1ko control experiment split/leakage protocol."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/control_experiment_split_precheck_registry.csv"
READY = ROOT / "datasets/control_experiment_sandbox_readiness_matrix.csv"


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
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert READY.exists()


def test_control_experiment_requires_features_and_strong_controls() -> None:
    ready = rows(READY)[0]
    if int(ready["feature_ready_control_count"]) < 9:
        assert ready["readiness_decision"] == "CONTROL_EXPERIMENT_BLOCKED_FEATURES"
    assert ready["can_create_operational_label"] == "false"
    assert ready["can_train_model"] == "false"


def test_temporal_self_control_is_never_used_as_negative() -> None:
    split_rows = rows(OUT)
    assert all(r["temporal_self_control_gate"] == "PASS" for r in split_rows)
    assert all(r["can_be_formal_negative"] == "false" for r in split_rows)
