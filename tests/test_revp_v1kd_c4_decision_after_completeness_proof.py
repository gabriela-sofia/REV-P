"""Tests for REV-P v1kd C4 decision after completeness proof."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCRIPTS = REVP_ROOT / "scripts/protocolo_c"
COMMANDS = [
    [sys.executable, str(SCRIPTS / "revp_v1ka_official_inventory_completeness_proof_scan.py"), "--scan-completeness-proof", "--emit-phrase-audit"],
    [sys.executable, str(SCRIPTS / "revp_v1kb_inventory_coverage_geometry_audit.py"), "--audit-coverage-geometry", "--emit-sampling-area-decision"],
    [sys.executable, str(SCRIPTS / "revp_v1kc_complete_inventory_gate_reevaluation.py"), "--reevaluate-complete-inventory-gate", "--emit-promotion-audit"],
    [sys.executable, str(SCRIPTS / "revp_v1kd_c4_decision_after_completeness_proof.py"), "--decide-c4-after-completeness-proof", "--emit-final-status"],
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    for command in COMMANDS:
        result = subprocess.run(command, cwd=str(REVP_ROOT), capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stderr + result.stdout


def test_script_runs_and_outputs_exist() -> None:
    run_once()
    for path in [
        DATASETS / "c4_decision_after_inventory_completeness_proof.csv",
        DATASETS / "protocol_c_negative_resolution_final_status.csv",
        DATASETS / "schemas/c4_decision_after_inventory_completeness_proof_schema.csv",
        DATASETS / "schemas/protocol_c_negative_resolution_final_status_schema.csv",
    ]:
        assert path.exists(), path


def test_c4_remains_zero_when_complete_gate_fails() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_decision_after_inventory_completeness_proof.csv")[0]
    if row["complete_inventory_status"] != "PASS":
        assert row["c4_decision"] == "C4_STILL_BLOCKED_COMPLETENESS_NOT_PROVEN"
        assert row["c4_ready_count"] == "0"
        assert row["can_create_training_label"] == "false"


def test_can_train_model_false_always() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_decision_after_inventory_completeness_proof.csv")[0]
    status = read_csv(DATASETS / "protocol_c_negative_resolution_final_status.csv")[0]
    assert row["can_train_model"] == "false"
    assert status["can_train_model"] == "false"


def test_summary_fields_present() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_decision_after_inventory_completeness_proof.csv")[0]
    for field in [
        "complete_inventory_status",
        "negative_candidate_count",
        "formal_negative_candidate_count",
        "training_label_candidate_count",
        "c4_ready_count",
        "remaining_blocker",
        "next_real_action",
    ]:
        assert row[field] != ""


def test_public_outputs_have_no_private_paths_or_raw_artifacts() -> None:
    run_once()
    for path in [
        DATASETS / "c4_decision_after_inventory_completeness_proof.csv",
        DATASETS / "protocol_c_negative_resolution_final_status.csv",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "C:/" not in text
        assert "gabriela" not in text.lower()
        assert ".tif" not in text.lower()
        assert ".npy" not in text.lower()
        assert ".npz" not in text.lower()
