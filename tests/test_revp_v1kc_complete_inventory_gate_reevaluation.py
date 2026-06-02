"""Tests for REV-P v1kc complete inventory gate reevaluation."""

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
]
sys.path.insert(0, str(SCRIPTS))

from revp_v1kc_complete_inventory_gate_reevaluation import evaluate_complete_inventory_gate


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
        DATASETS / "complete_inventory_gate_reevaluation_matrix.csv",
        DATASETS / "inventory_derived_negative_promotion_audit.csv",
        DATASETS / "schemas/complete_inventory_gate_reevaluation_matrix_schema.csv",
        DATASETS / "schemas/inventory_derived_negative_promotion_audit_schema.csv",
    ]:
        assert path.exists(), path


def test_complete_inventory_gate_requires_all_core_gates() -> None:
    result = evaluate_complete_inventory_gate("PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS")
    assert result["complete_inventory_gate"] == "PASS"
    assert result["can_create_training_label"] == "true"
    assert result["can_train_model"] == "false"
    missing_completeness = evaluate_complete_inventory_gate("PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "FAIL", "PASS")
    assert missing_completeness["complete_inventory_gate"] == "FAIL"
    assert missing_completeness["can_create_training_label"] == "false"


def test_actual_derived_negatives_remain_review_only_if_gate_fails() -> None:
    run_once()
    matrix = read_csv(DATASETS / "complete_inventory_gate_reevaluation_matrix.csv")[0]
    promotions = read_csv(DATASETS / "inventory_derived_negative_promotion_audit.csv")
    if matrix["complete_inventory_gate"] != "PASS":
        assert matrix["decision"] == "COMPLETENESS_NOT_PROVEN_C4_BLOCKED"
        assert all(row["can_be_formal_negative_candidate"] == "false" for row in promotions)
        assert all(row["can_create_training_label"] == "false" for row in promotions)
    assert all(row["can_train_model"] == "false" for row in promotions)


def test_public_outputs_have_no_private_paths() -> None:
    run_once()
    for path in [
        DATASETS / "complete_inventory_gate_reevaluation_matrix.csv",
        DATASETS / "inventory_derived_negative_promotion_audit.csv",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "C:/" not in text
        assert "gabriela" not in text.lower()
