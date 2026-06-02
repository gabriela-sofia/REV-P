"""Tests for REV-P v1jt negative control patch precheck."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jt_negative_control_patch_precheck.py"
COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--read-negative-gates",
    "--precheck-negative-control-patches",
    "--emit-patch-precheck",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (DATASETS / "negative_control_patch_precheck_registry.csv").exists()


def test_patch_precheck_outputs_exist() -> None:
    run_once()
    for path in [
        DATASETS / "negative_control_patch_precheck_registry.csv",
        DATASETS / "negative_control_multimodal_readiness_matrix.csv",
        DATASETS / "schemas/negative_control_patch_precheck_schema.csv",
        DATASETS / "schemas/negative_control_multimodal_readiness_matrix_schema.csv",
    ]:
        assert path.exists(), path


def test_no_candidate_blocks_patch_generation() -> None:
    run_once()
    rows = read_csv(DATASETS / "negative_control_patch_precheck_registry.csv")
    assert rows
    assert all(row["would_generate_patch"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)
    assert all(row["can_train_model"] == "false" for row in rows)


def test_readiness_matrix_does_not_unlock_c4() -> None:
    run_once()
    update = read_csv(DATASETS / "negative_control_multimodal_readiness_matrix.csv")[0]
    assert update["would_unlock_c4"] == "false"
    assert update["can_create_training_label"] == "false"
    assert update["can_train_model"] == "false"
    assert update["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_blocker_is_refined_when_no_candidate_exists() -> None:
    run_once()
    row = read_csv(DATASETS / "negative_control_patch_precheck_registry.csv")[0]
    if row["candidate_classification"] == "NO_CANDIDATE":
        assert "NO_EXPLICIT_ABSENCE" in row["source_gap_exact"]
        assert "official field sheet" in row["missing_document_type"]
