"""Tests for REV-P v1jx negative candidate multimodal patch QA."""

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
        DATASETS / "inventory_negative_multimodal_patch_qa_registry.csv",
        DATASETS / "inventory_negative_multimodal_readiness_matrix.csv",
        DATASETS / "schemas/inventory_negative_multimodal_patch_qa_schema.csv",
        DATASETS / "schemas/inventory_negative_multimodal_readiness_matrix_schema.csv",
    ]:
        assert path.exists(), path


def test_patch_qa_is_public_metadata_only() -> None:
    run_once()
    rows = read_csv(DATASETS / "inventory_negative_multimodal_patch_qa_registry.csv")
    assert rows
    assert all(row["raw_artifact_policy"] == "RAW_ONLY_LOCAL_RUNS" for row in rows)
    assert all(row["public_artifact_policy"] == "PUBLIC_METADATA_ONLY" for row in rows)
    assert all(row["cloud_local_status"] in {"NOT_COMPUTED_METADATA_ONLY", "NOT_COMPUTED"} for row in rows)


def test_patch_qa_does_not_unlock_training() -> None:
    run_once()
    rows = read_csv(DATASETS / "inventory_negative_multimodal_patch_qa_registry.csv")
    ready = read_csv(DATASETS / "inventory_negative_multimodal_readiness_matrix.csv")[0]
    assert all(row["can_train_model"] == "false" for row in rows)
    assert ready["can_train_model"] == "false"
    assert ready["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_intersections_and_positive_buffers_are_blocked() -> None:
    run_once()
    rows = [row for row in read_csv(DATASETS / "inventory_negative_multimodal_patch_qa_registry.csv") if row["qa_status"] == "QA_PASS_METADATA_ONLY_REVIEW"]
    assert rows
    assert all(row["intersects_inventory_polygon"] == "false" for row in rows)
    assert all(row["inside_positive_buffer"] == "false" for row in rows)
