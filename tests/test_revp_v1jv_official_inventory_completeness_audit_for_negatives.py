"""Tests for REV-P v1jv official inventory completeness audit."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jv_official_inventory_completeness_audit_for_negatives.py"
COMMAND = [sys.executable, str(SCRIPT), "--audit-official-inventory", "--emit-completeness-gates"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_runs_and_outputs_exist() -> None:
    assert SCRIPT.exists()
    run_once()
    for path in [
        DATASETS / "official_inventory_completeness_audit_registry.csv",
        DATASETS / "official_inventory_completeness_gate_matrix.csv",
        DATASETS / "schemas/official_inventory_completeness_audit_schema.csv",
        DATASETS / "schemas/official_inventory_completeness_gate_matrix_schema.csv",
        DOCS / "protocolo_c_completude_inventario_negativos_v1jv.md",
        DOCS / "protocolo_c_relatorio_completude_inventario_negativos_v1jv.md",
    ]:
        assert path.exists(), path


def test_inventory_layer_is_partial_not_complete() -> None:
    run_once()
    row = read_csv(DATASETS / "official_inventory_completeness_audit_registry.csv")[0]
    assert row["feature_count"] == "444"
    assert row["crs"] == "EPSG:31983"
    assert row["decision"] in {"PARTIAL_INVENTORY_USABLE_FOR_REVIEW_ONLY", "INVENTORY_COMPLETENESS_NOT_PROVEN"}
    assert row["complete_inventory_gate"] == "FAIL"
    assert row["can_create_training_label"] == "false"


def test_completeness_and_temporal_gates_fail_closed() -> None:
    run_once()
    row = read_csv(DATASETS / "official_inventory_completeness_gate_matrix.csv")[0]
    assert row["inventory_completeness_statement_gate"] == "FAIL"
    assert row["event_or_survey_temporal_link_gate"] == "FAIL"
    assert row["can_train_model"] == "false"


def test_public_outputs_have_no_private_paths() -> None:
    run_once()
    for path in [
        DATASETS / "official_inventory_completeness_audit_registry.csv",
        DATASETS / "official_inventory_completeness_gate_matrix.csv",
        DOCS / "protocolo_c_completude_inventario_negativos_v1jv.md",
        DOCS / "protocolo_c_relatorio_completude_inventario_negativos_v1jv.md",
    ]:
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "C:/" not in text
        assert "gabriela" not in text.lower()
        assert "PROJETO" not in text
