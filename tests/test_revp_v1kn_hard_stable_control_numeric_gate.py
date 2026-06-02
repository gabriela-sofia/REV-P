"""Tests for v1kn hard stable control numeric gate."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/hard_stable_control_final_registry.csv"
GATES = ROOT / "datasets/hard_stable_control_numeric_gate_matrix.csv"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_runs_with_force() -> None:
    chain = [
        ("revp_v1kk_control_candidate_pool_expansion.py", ["--force", "--limit", "50", "--emit-pool"]),
        ("revp_v1kl_control_multimodal_patch_acquisition.py", ["--force", "--emit-patch-qa"]),
        ("revp_v1km_positive_control_numeric_feature_table.py", ["--force", "--emit-feature-table"]),
        ("revp_v1kn_hard_stable_control_numeric_gate.py", ["--force", "--emit-gates"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert GATES.exists()


def test_hard_stable_control_is_never_formal_negative() -> None:
    all_rows = rows(OUT)
    assert all(r["can_be_formal_negative"] == "false" for r in all_rows)
    assert all(r["can_create_operational_label"] == "false" for r in all_rows)
    assert all(r["can_train_model"] == "false" for r in all_rows)


def test_insufficient_numeric_features_block_hard_control() -> None:
    gate_rows = rows(GATES)
    assert gate_rows
    for row in gate_rows:
        if row["patch_extractability_gate"] != "PASS":
            assert row["final_control_class"] != "HARD_STABLE_CONTROL"
