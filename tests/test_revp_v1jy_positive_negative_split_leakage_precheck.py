"""Tests for REV-P v1jy positive/negative split leakage precheck."""

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
    [sys.executable, str(SCRIPTS / "revp_v1jv_official_inventory_completeness_audit_for_negatives.py"), "--audit-official-inventory", "--emit-completeness-gates"],
    [sys.executable, str(SCRIPTS / "revp_v1jw_inventory_derived_negative_candidate_generator.py"), "--read-inventory-audit", "--generate-inventory-negative-candidates", "--emit-sampling-design"],
    [sys.executable, str(SCRIPTS / "revp_v1jx_negative_candidate_multimodal_patch_qa.py"), "--read-inventory-negative-candidates", "--plan-multimodal-patch-qa", "--emit-patch-qa"],
    [sys.executable, str(SCRIPTS / "revp_v1jy_positive_negative_split_leakage_precheck.py"), "--read-positive-negative-candidates", "--precheck-split-leakage", "--emit-c4-label-pair-readiness"],
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    for command in COMMANDS:
        result = subprocess.run(command, cwd=str(REVP_ROOT), capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stderr + result.stdout


def test_script_runs_and_outputs_exist() -> None:
    run_once()
    for path in [
        DATASETS / "positive_negative_split_leakage_precheck_registry.csv",
        DATASETS / "c4_label_pair_readiness_matrix.csv",
        DATASETS / "schemas/positive_negative_split_leakage_precheck_schema.csv",
        DATASETS / "schemas/c4_label_pair_readiness_matrix_schema.csv",
    ]:
        assert path.exists(), path


def test_split_is_precheck_not_random_training_split() -> None:
    run_once()
    rows = read_csv(DATASETS / "positive_negative_split_leakage_precheck_registry.csv")
    assert rows
    assert all(row["proposed_split_role"] in {"PRECHECK_ONLY_NO_TRAIN_SPLIT", "NO_SPLIT"} for row in rows)
    assert all("not random patch" in row["event_grouping_rule"] for row in rows)


def test_temporal_self_control_never_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "positive_negative_split_leakage_precheck_registry.csv")
    ready = read_csv(DATASETS / "c4_label_pair_readiness_matrix.csv")[0]
    assert all(row["temporal_self_control_status"] == "BLOCKED_AS_NEGATIVE" for row in rows)
    assert int(ready["temporal_self_control_blocked_count"]) >= 1


def test_c4_label_pair_blocked_without_complete_inventory() -> None:
    run_once()
    ready = read_csv(DATASETS / "c4_label_pair_readiness_matrix.csv")[0]
    if ready["complete_inventory_gate"] != "PASS":
        assert ready["c4_label_pair_decision"] == "C4_LABEL_PAIR_BLOCKED_INVENTORY_COMPLETENESS"
        assert ready["can_create_training_label"] == "false"
    assert ready["can_train_model"] == "false"
