"""Tests for REV-P v1ju C4 readiness after negative harvest."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1ju_c4_readiness_after_negative_harvest.py"
COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--read-negative-harvest-results",
    "--update-c4-readiness",
    "--emit-ground-truth-evolution-summary",
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
    assert (DATASETS / "c4_readiness_after_negative_harvest_matrix.csv").exists()


def test_outputs_exist() -> None:
    run_once()
    for path in [
        DATASETS / "c4_readiness_after_negative_harvest_matrix.csv",
        DATASETS / "protocol_c_ground_truth_evolution_summary_v1ju.csv",
        DATASETS / "schemas/c4_readiness_after_negative_harvest_matrix_schema.csv",
        DATASETS / "schemas/protocol_c_ground_truth_evolution_summary_v1ju_schema.csv",
        DOCS / "protocolo_c_atualizacao_prontidao_c4_pos_colheita_negativa_v1ju.md",
        DOCS / "protocolo_c_relatorio_atualizacao_prontidao_c4_pos_colheita_negativa_v1ju.md",
    ]:
        assert path.exists(), path


def test_c3_stable_and_c4_blocked() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_readiness_after_negative_harvest_matrix.csv")[0]
    assert row["confirmed_c3_events"] == "9"
    assert row["c4_ready_events"] == "0"
    assert row["c4_changed_after_negative_harvest"] == "false"
    assert "C4_BLOCKED" in row["summary_decision"]


def test_training_label_and_dino_remain_blocked() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_readiness_after_negative_harvest_matrix.csv")[0]
    assert row["can_create_training_label"] == "false"
    assert row["can_train_model"] == "false"
    assert row["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_evolution_summary_records_stages() -> None:
    run_once()
    stages = {row["stage"] for row in read_csv(DATASETS / "protocol_c_ground_truth_evolution_summary_v1ju.csv")}
    assert {"v1jq", "v1jr", "v1js", "v1jt", "v1ju"}.issubset(stages)
