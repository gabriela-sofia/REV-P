"""Integration tests for REV-P v1je-v1jh ground reference expansion."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

COMMANDS = [
    [sys.executable, "scripts/protocolo_c/revp_v1je_pdf_metadata_coordinate_recovery.py", "--force"],
    [sys.executable, "scripts/protocolo_c/revp_v1jf_documented_locality_patch_candidate_builder.py", "--force"],
    [sys.executable, "scripts/protocolo_c/revp_v1jg_batch_sentinel_sar_candidate_acquisition.py", "--force"],
    [sys.executable, "scripts/protocolo_c/revp_v1jh_training_gate_weak_supervision_boundary.py", "--force"],
]

PRIVATE_FRAGMENTS = [
    r"C:\Users\gabriela",
    "gabriela",
    r"Documents\REV-P",
    "Documents/REV-P",
    r"Documents\PROJETO",
    "Documents/PROJETO",
]

FORBIDDEN_DOC_TERMS = [
    "flood detection",
    "landslide detection",
    "flood prediction",
    "landslide prediction",
    "detecÃ§Ã£o",
    "prediÃ§Ã£o",
    "predicao",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_all_once() -> None:
    for command in COMMANDS:
        result = subprocess.run(command, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr


def test_scripts_exist_and_run() -> None:
    for command in COMMANDS:
        assert (REVP_ROOT / command[1]).exists()
    run_all_once()


def test_libs_absent_or_present_do_not_break_coordinate_recovery() -> None:
    run_all_once()
    qa = {row["check"]: row["status"] for row in read_csv(REVP_ROOT / "local_runs/protocolo_c/v1je/v1je_qa.csv")}
    assert qa["libs_absent_do_not_break"] == "PASS"


def test_explicit_coordinate_becomes_anchor_candidate() -> None:
    run_all_once()
    rows = read_csv(DATASETS / "official_coordinate_recovery_hardened_registry.csv")
    candidates = [row for row in rows if row["can_be_official_anchor_candidate"] == "true"]
    assert candidates
    assert any(row["coordinate_format"] in {"DECIMAL_LAT_LON", "DEGREES_MINUTES_SECONDS"} for row in candidates)
    assert all(row["can_create_training_label"] == "false" for row in candidates)


def test_locality_without_coordinate_does_not_become_label() -> None:
    run_all_once()
    rows = read_csv(DATASETS / "documented_locality_patch_review_candidate_registry.csv")
    assert rows
    assert all(row["review_area_status"] == "REVIEW_AREA_ONLY" for row in rows)
    assert all(row["can_be_positive_label"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)


def test_review_area_and_controls_do_not_become_labels() -> None:
    run_all_once()
    controls = read_csv(DATASETS / "review_control_candidate_registry.csv")
    patch_rows = read_csv(DATASETS / "multimodal_patch_candidate_batch_registry.csv")
    assert all(row["can_be_negative_label"] == "false" for row in controls)
    assert all(row["can_be_positive_label"] == "false" for row in patch_rows)
    assert all(row["can_be_negative_label"] == "false" for row in patch_rows)


def test_sentinel_sar_batch_does_not_create_label() -> None:
    run_all_once()
    rows = read_csv(DATASETS / "multimodal_patch_candidate_batch_registry.csv")
    assert rows
    assert any(row["s2_status"] == "CONFIRMED_ANCHOR_PATCH_READY" for row in rows)
    assert all(row["label_status"] != "POSITIVE_LABEL_READY" for row in rows)
    assert all(row["can_train_model"] == "false" for row in rows)


def test_training_gate_requires_split_and_leakage_protocol() -> None:
    run_all_once()
    decision = read_csv(DATASETS / "training_gate_decision_matrix.csv")[0]
    assert decision["supervised_training_gate_status"] == "SUPERVISED_TRAINING_BLOCKED"
    assert decision["leakage_risk_status"] == "LEAKAGE_PROTOCOL_REQUIRED"
    assert decision["negative_labels_ready_count"] == "0"
    assert decision["can_train_model"] == "false"
    assert decision["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_no_private_path_in_public_outputs() -> None:
    run_all_once()
    public_files = [
        DATASETS / "official_coordinate_recovery_hardened_registry.csv",
        SCHEMAS / "official_coordinate_recovery_hardened_schema.csv",
        DATASETS / "documented_locality_patch_review_candidate_registry.csv",
        SCHEMAS / "documented_locality_patch_review_candidate_schema.csv",
        DATASETS / "multimodal_patch_candidate_batch_registry.csv",
        SCHEMAS / "multimodal_patch_candidate_batch_schema.csv",
        DATASETS / "ground_reference_candidate_master_registry.csv",
        SCHEMAS / "ground_reference_candidate_master_schema.csv",
        DATASETS / "training_gate_decision_matrix.csv",
        SCHEMAS / "training_gate_decision_matrix_schema.csv",
    ]
    for path in public_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_all_once()
    docs = [
        DOCS / "protocolo_c_expansao_ground_reference_v1je_v1jh.md",
        DOCS / "protocolo_c_relatorio_expansao_ground_reference_v1je_v1jh.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "supervised_training_blocked" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"


def test_local_runs_not_versioned() -> None:
    run_all_once()
    result = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    bad = [line for line in result.stdout.splitlines() if "local_runs/" in line.lower() or r"local_runs\\" in line.lower()]
    assert not bad
