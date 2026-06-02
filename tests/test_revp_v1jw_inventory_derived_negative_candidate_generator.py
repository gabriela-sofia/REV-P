"""Tests for REV-P v1jw inventory-derived negative candidates."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCRIPT_V1JV = REVP_ROOT / "scripts/protocolo_c/revp_v1jv_official_inventory_completeness_audit_for_negatives.py"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jw_inventory_derived_negative_candidate_generator.py"
COMMANDS = [
    [sys.executable, str(SCRIPT_V1JV), "--audit-official-inventory", "--emit-completeness-gates"],
    [
        sys.executable,
        str(SCRIPT),
        "--read-inventory-audit",
        "--generate-inventory-negative-candidates",
        "--emit-sampling-design",
    ],
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
    assert SCRIPT.exists()
    run_once()
    for path in [
        DATASETS / "inventory_derived_negative_candidate_registry.csv",
        DATASETS / "inventory_negative_candidate_gate_matrix.csv",
        DATASETS / "inventory_negative_sampling_design_registry.csv",
    ]:
        assert path.exists(), path


def test_negative_candidates_only_from_inventory_gates() -> None:
    run_once()
    inventory = read_csv(DATASETS / "official_inventory_completeness_audit_registry.csv")[0]
    rows = read_csv(DATASETS / "inventory_derived_negative_candidate_registry.csv")
    real_candidates = [row for row in rows if row["can_be_negative_candidate"] == "true"]
    if inventory["decision"] == "PARTIAL_INVENTORY_USABLE_FOR_REVIEW_ONLY":
        assert real_candidates
        assert all(row["candidate_classification"] == "REVIEW_NEGATIVE_CANDIDATE_FROM_PARTIAL_INVENTORY" for row in real_candidates)
    else:
        assert not real_candidates


def test_no_background_or_pseudo_absence_promoted_to_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "inventory_derived_negative_candidate_registry.csv")
    assert all("PSEUDO" not in row["candidate_classification"] for row in rows)
    assert all(row["candidate_classification"] != "BACKGROUND_UNLABELED" or row["can_be_negative_candidate"] == "false" for row in rows)
    assert all(row["can_train_model"] == "false" for row in rows)


def test_complete_inventory_required_for_training_label() -> None:
    run_once()
    rows = read_csv(DATASETS / "inventory_derived_negative_candidate_registry.csv")
    for row in rows:
        if row["complete_inventory_gate"] != "PASS":
            assert row["can_create_training_label"] == "false"


def test_positive_buffer_exclusion_works() -> None:
    run_once()
    gates = read_csv(DATASETS / "inventory_negative_candidate_gate_matrix.csv")
    candidates = [row for row in gates if row["can_be_negative_candidate"] == "true"]
    assert candidates
    assert all(row["positive_anchor_buffer_exclusion_gate"] == "PASS" for row in candidates)
    assert all(row["temporal_self_control_gate"] == "PASS" for row in candidates)
