"""
Tests for REV-P v1jd targeted PDF coordinate recovery for multi-anchors.

The tests cover the real local run plus parser-level recognition of decimal,
UTM, and invalid coordinates without creating labels, training permission, or
private-path leakage.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1jd_pdf_coordinate_recovery_multi_anchor.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jd"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1ir-units",
    "--scan-official-pdfs",
    "--extract-native-text",
    "--extract-tables",
    "--try-ocr-if-available",
    "--recover-coordinates",
    "--validate-coordinates",
    "--emit-anchor-update",
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
def load_script_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revp_v1jd", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def run_script_once() -> subprocess.CompletedProcess[str]:
    return subprocess.run(RUN_CMD, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)


def test_script_exists_compiles_and_runs() -> None:
    assert SCRIPT.exists()
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(SCRIPT)],
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr
    run_result = run_script_once()
    assert run_result.returncode == 0, run_result.stderr


def test_local_outputs_are_generated() -> None:
    run_script_once()
    required = [
        "v1jd_pdf_inventory.csv",
        "v1jd_text_extraction_quality.csv",
        "v1jd_coordinate_expression_inventory.csv",
        "v1jd_coordinate_validation_audit.csv",
        "v1jd_multi_anchor_update_decision.csv",
        "v1jd_summary.json",
        "v1jd_qa.csv",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1jd output: {name}"


def test_decimal_coordinate_is_recognized_and_validated() -> None:
    module = load_script_module()
    expressions = module.recover_coordinate_expressions("Coordenadas GPS Latitude -22.484251 Longitude -43.211257")
    assert expressions
    validation, confidence = module.validate_expression(expressions[0])
    assert expressions[0]["coordinate_format"] == "DECIMAL_DEGREES"
    assert validation == "VALID_PETROPOLIS_APPROX_RANGE"
    assert confidence == "EXPLICIT_COORDINATE_HIGH"


def test_utm_coordinate_is_recognized_or_registered_for_review() -> None:
    module = load_script_module()
    expressions = module.recover_coordinate_expressions("Coordenadas UTM SIRGAS 685000 7510000")
    utm = [expr for expr in expressions if expr["coordinate_format"] == "UTM"]
    assert utm
    validation, confidence = module.validate_expression(utm[0])
    assert validation in {
        "VALID_PETROPOLIS_APPROX_RANGE",
        "VALID_RJ_NEEDS_LOCALITY_REVIEW",
        "UTM_RECOGNIZED_CONVERSION_UNAVAILABLE",
        "INVALID_COORDINATE",
    }
    assert confidence in {
        "EXPLICIT_COORDINATE_HIGH",
        "EXPLICIT_COORDINATE_NEEDS_REVIEW",
        "INVALID_COORDINATE",
    }


def test_invalid_coordinate_is_rejected() -> None:
    module = load_script_module()
    validation, confidence = module.validate_decimal_coordinate(0.0, 0.0)
    assert validation == "INVALID_COORDINATE"
    assert confidence == "INVALID_COORDINATE"


def test_bairro_without_coordinate_does_not_become_anchor() -> None:
    run_script_once()
    rows = read_csv(DATASETS / "official_pdf_coordinate_recovery_registry.csv")
    no_coordinate = [row for row in rows if row["coordinate_confidence"] == "NO_COORDINATE_FOUND"]

    assert no_coordinate
    assert all(row["can_be_official_anchor_candidate"] == "false" for row in no_coordinate)
    assert all(row["can_create_training_label"] == "false" for row in no_coordinate)


def test_ocr_absent_does_not_break_script() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jd_summary.json").read_text(encoding="utf-8"))
    extraction_rows = read_csv(LOCAL_RUNS / "v1jd_text_extraction_quality.csv")
    qa = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1jd_qa.csv")}

    assert summary["ocr_status"] in {"OCR_NOT_AVAILABLE", "OCR_AVAILABLE"}
    assert qa["ocr_absence_controlled"] == "PASS"
    assert all(row["ocr_status"] in {"OCR_NOT_AVAILABLE", "OCR_AVAILABLE"} for row in extraction_rows)


def test_public_registries_are_created_when_coordinate_or_negative_audit_is_useful() -> None:
    run_script_once()
    required = [
        DATASETS / "official_pdf_coordinate_recovery_registry.csv",
        SCHEMAS / "official_pdf_coordinate_recovery_schema.csv",
        DATASETS / "official_multi_anchor_update_registry.csv",
        SCHEMAS / "official_multi_anchor_update_schema.csv",
    ]
    for path in required:
        assert path.exists(), f"Missing public v1jd output: {path.name}"


def test_valid_coordinate_remains_anchor_candidate_not_label() -> None:
    run_script_once()
    recovery_rows = read_csv(DATASETS / "official_pdf_coordinate_recovery_registry.csv")
    update_rows = read_csv(DATASETS / "official_multi_anchor_update_registry.csv")
    anchor = next(row for row in recovery_rows if row["documented_event_unit_id"] == "PET2022_CPRM_ANEXOII_19022022")
    update = next(row for row in update_rows if row["documented_event_unit_id"] == "PET2022_CPRM_ANEXOII_19022022")

    assert anchor["coordinate_confidence"] == "EXPLICIT_COORDINATE_HIGH"
    assert anchor["can_be_official_anchor_candidate"] == "true"
    assert update["anchor_status"] == "OFFICIAL_ANCHOR_CONFIRMED"
    assert update["can_be_positive_reference_candidate"] == "true"
    assert update["can_be_positive_training_label"] == "false"
    assert update["can_create_training_label"] == "false"


def test_training_unfreeze_and_protocol_b_remain_blocked() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jd_summary.json").read_text(encoding="utf-8"))
    recovery_rows = read_csv(DATASETS / "official_pdf_coordinate_recovery_registry.csv")
    update_rows = read_csv(DATASETS / "official_multi_anchor_update_registry.csv")
    qa = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1jd_qa.csv")}

    assert summary["training_boundary_status"] == "TRAINING_BLOCKED_INSUFFICIENT_LABELS"
    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_unfreeze_dino_for_scientific_claim"] is False
    assert all(row["can_create_training_label"] == "false" for row in recovery_rows)
    assert all(row["can_train_model"] == "false" for row in recovery_rows)
    assert all(row["can_unfreeze_dino_for_scientific_claim"] == "false" for row in update_rows)
    assert all(row["can_reopen_protocol_b"] == "false" for row in update_rows)
    assert qa["can_create_training_label_false"] == "PASS"
    assert qa["can_train_model_false"] == "PASS"
    assert qa["can_unfreeze_dino_for_scientific_claim_false"] == "PASS"


def test_no_private_path_in_public_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_pdf_coordinate_recovery_registry.csv",
        SCHEMAS / "official_pdf_coordinate_recovery_schema.csv",
        DATASETS / "official_multi_anchor_update_registry.csv",
        SCHEMAS / "official_multi_anchor_update_schema.csv",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_script_once()
    doc_files = [
        DOCS / "protocolo_c_recuperacao_coordenadas_pdf_v1jd.md",
        DOCS / "protocolo_c_relatorio_recuperacao_coordenadas_pdf_v1jd.md",
    ]
    for path in doc_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "ocr_not_available" in text
        assert "treino" in text
        assert "label" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
