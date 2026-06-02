"""Tests for REV-P v1js formal negative statement extraction and gate audit."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1js_formal_negative_statement_extraction_and_gate_audit.py"
COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--read-harvest",
    "--extract-statements",
    "--audit-formal-negative-gates",
    "--emit-gate-update",
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
    assert (DATASETS / "formal_negative_statement_extraction_registry.csv").exists()


def test_outputs_exist() -> None:
    run_once()
    for path in [
        DATASETS / "formal_negative_statement_extraction_registry.csv",
        DATASETS / "formal_negative_candidate_gate_matrix.csv",
        DATASETS / "c4_negative_gate_update_matrix.csv",
        DATASETS / "schemas/formal_negative_statement_extraction_schema.csv",
        DATASETS / "schemas/formal_negative_candidate_gate_matrix_schema.csv",
        DATASETS / "schemas/c4_negative_gate_update_matrix_schema.csv",
        DOCS / "protocolo_c_extracao_declaracao_negativa_formal_v1js.md",
        DOCS / "protocolo_c_relatorio_extracao_declaracao_negativa_formal_v1js.md",
    ]:
        assert path.exists(), path


def test_ten_gates_are_present() -> None:
    run_once()
    row = read_csv(DATASETS / "formal_negative_candidate_gate_matrix.csv")[0]
    for gate in [
        "official_source_gate",
        "explicit_absence_or_stability_gate",
        "phenomenon_compatibility_gate",
        "temporal_compatibility_gate",
        "spatial_specificity_gate",
        "independent_control_area_gate",
        "positive_anchor_buffer_exclusion_gate",
        "patch_extractability_gate",
        "leakage_precheck_gate",
        "provenance_reproducibility_gate",
    ]:
        assert gate in row


def test_invalid_assumptions_and_incomplete_gates_do_not_create_label() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_candidate_gate_matrix.csv")
    assert all(row["can_create_training_label"] == "false" for row in rows)
    assert all(row["can_train_model"] == "false" for row in rows)
    assert all(row["can_unfreeze_dino_for_scientific_claim"] == "false" for row in rows)


def test_c4_update_remains_blocked_without_ready_negative() -> None:
    run_once()
    update = read_csv(DATASETS / "c4_negative_gate_update_matrix.csv")[0]
    assert update["formal_negative_ready_count"] == "0"
    assert update["c4_ready_after_statement_audit"] == "false"
    assert update["can_create_training_label"] == "false"
    assert "C4_STILL_BLOCKED" in update["summary_decision"]


def test_public_outputs_no_private_paths() -> None:
    run_once()
    for path in [
        DATASETS / "formal_negative_statement_extraction_registry.csv",
        DATASETS / "formal_negative_candidate_gate_matrix.csv",
        DATASETS / "c4_negative_gate_update_matrix.csv",
    ]:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert r"C:\Users\gabriela" not in text
        assert "Documents\\REV-P" not in text
