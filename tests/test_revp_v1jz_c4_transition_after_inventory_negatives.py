"""Tests for REV-P v1jz C4 transition after inventory negatives."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPTS = REVP_ROOT / "scripts/protocolo_c"
COMMANDS = [
    [sys.executable, str(SCRIPTS / "revp_v1jv_official_inventory_completeness_audit_for_negatives.py"), "--audit-official-inventory", "--emit-completeness-gates"],
    [sys.executable, str(SCRIPTS / "revp_v1jw_inventory_derived_negative_candidate_generator.py"), "--read-inventory-audit", "--generate-inventory-negative-candidates", "--emit-sampling-design"],
    [sys.executable, str(SCRIPTS / "revp_v1jx_negative_candidate_multimodal_patch_qa.py"), "--read-inventory-negative-candidates", "--plan-multimodal-patch-qa", "--emit-patch-qa"],
    [sys.executable, str(SCRIPTS / "revp_v1jy_positive_negative_split_leakage_precheck.py"), "--read-positive-negative-candidates", "--precheck-split-leakage", "--emit-c4-label-pair-readiness"],
    [sys.executable, str(SCRIPTS / "revp_v1jz_c4_transition_after_inventory_negatives.py"), "--read-inventory-negative-chain", "--emit-c4-transition"],
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
        DATASETS / "c4_transition_after_inventory_negative_protocol.csv",
        DATASETS / "schemas/c4_transition_after_inventory_negative_protocol_schema.csv",
        DOCS / "protocolo_c_transicao_c4_negativos_inventario_v1jz.md",
        DOCS / "protocolo_c_relatorio_transicao_c4_negativos_inventario_v1jz.md",
    ]:
        assert path.exists(), path


def test_c4_does_not_change_if_completeness_not_proven() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_transition_after_inventory_negative_protocol.csv")[0]
    if row["complete_inventory_gate"] != "PASS":
        assert row["c4_transition_decision"] == "C4_STILL_BLOCKED"
        assert row["c4_changed"] == "false"
        assert row["blocking_reason"] == "INVENTORY_COMPLETENESS_NOT_PROVEN"


def test_training_remains_blocked_in_this_stage() -> None:
    run_once()
    row = read_csv(DATASETS / "c4_transition_after_inventory_negative_protocol.csv")[0]
    assert row["can_train_model"] == "false"
    assert row["can_unfreeze_dino_for_scientific_claim"] == "false"
    if row["complete_inventory_gate"] != "PASS":
        assert row["can_create_training_label"] == "false"


def test_public_outputs_have_no_private_paths_or_raw_extensions() -> None:
    run_once()
    public_files = [
        DATASETS / "c4_transition_after_inventory_negative_protocol.csv",
        DOCS / "protocolo_c_transicao_c4_negativos_inventario_v1jz.md",
        DOCS / "protocolo_c_relatorio_transicao_c4_negativos_inventario_v1jz.md",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8")
        assert "C:\\" not in text and "C:/" not in text
        assert "gabriela" not in text.lower()
        assert ".tif" not in text.lower()
        assert ".npy" not in text.lower()
        assert ".npz" not in text.lower()
        assert ".shp" not in text.lower()
