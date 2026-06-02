"""Tests for v1kq operational vs control resolution decision."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPTS = ROOT / "scripts/protocolo_c"
OUT = ROOT / "datasets/negative_resolution_final_decision_matrix.csv"
BOUNDARY = ROOT / "datasets/protocol_c_control_experiment_claim_boundary.csv"


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
        ("revp_v1kq_operational_vs_control_resolution_decision.py", ["--force", "--emit-decision"]),
    ]
    for name, args in chain:
        result = subprocess.run([sys.executable, str(SCRIPTS / name), *args], cwd=ROOT, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout
    assert OUT.exists()
    assert BOUNDARY.exists()


def test_c4_operational_stays_blocked_without_formal_negative() -> None:
    operational = [r for r in rows(OUT) if r["decision_layer"] == "C4_OPERATIONAL_GROUND_TRUTH"][0]
    if operational["formal_negative_count"] == "0":
        assert operational["decision"] == "BLOCKED"
    assert operational["can_create_operational_label"] == "false"
    assert operational["can_train_model"] == "false"


def test_stable_controls_are_not_promoted_to_formal_negatives() -> None:
    text = BOUNDARY.read_text(encoding="utf-8")
    assert "Stable controls cannot be renamed as formal negatives." in text
