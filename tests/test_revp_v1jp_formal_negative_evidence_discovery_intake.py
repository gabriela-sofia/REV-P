"""Tests for REV-P v1jp formal negative evidence discovery and intake."""

from __future__ import annotations

import csv
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jp_formal_negative_evidence_discovery_intake.py"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-c4-queues",
    "--scan-local-documents",
    "--scan-registries",
    "--extract-negative-evidence-candidates",
    "--classify-negative-evidence",
    "--emit-intake",
]

PRIVATE_FRAGMENTS = [r"C:\Users\gabriela", "gabriela", r"Documents\REV-P", "Documents/REV-P", "local_runs/"]
FORBIDDEN_DOC_TERMS = ["flood detection", "landslide detection", "flood prediction", "landslide prediction", "detecao", "deteccao", "predicao"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False, timeout=180)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (DATASETS / "formal_negative_evidence_intake_registry.csv").exists()


def test_absence_of_record_does_not_become_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_evidence_intake_registry.csv")
    absence_rows = [row for row in rows if "sem registro" in row["negative_evidence_phrase_sanitized"].lower()]
    assert rows
    assert all(row["can_be_formal_negative"] == "false" for row in absence_rows)
    assert all(row["negative_evidence_status"] != "FORMAL_NEGATIVE_READY" for row in absence_rows)


def test_low_risk_does_not_become_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_evidence_intake_registry.csv")
    low_risk_rows = [row for row in rows if "baixo risco" in row["negative_evidence_phrase_sanitized"].lower()]
    assert low_risk_rows
    assert all(row["negative_evidence_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION" for row in low_risk_rows)
    assert all(row["can_create_negative_label"] == "false" for row in low_risk_rows)


def test_pseudo_absence_does_not_become_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_evidence_intake_registry.csv")
    pseudo_rows = [row for row in rows if "pseudo" in row["negative_evidence_phrase_sanitized"].lower()]
    assert pseudo_rows
    assert all(row["can_be_formal_negative"] == "false" for row in pseudo_rows)
    assert all(row["can_create_training_label"] == "false" for row in pseudo_rows)


def test_formal_negative_requires_explicit_statement_and_complete_gates() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_evidence_intake_registry.csv")
    ready = [row for row in rows if row["negative_evidence_status"] == "FORMAL_NEGATIVE_READY"]
    for row in ready:
        assert row["explicit_absence_gate_status"] == "PASS"
        assert row["temporal_gate_status"] == "PASS"
        assert row["spatial_gate_status"].startswith("PASS")
        assert row["phenomenon_gate_status"] == "PASS"
        assert row["patch_linkage_possible"] == "true"
        assert row["leakage_risk_status"] == "LEAKAGE_REVIEW_REQUIRED"


def test_training_and_model_remain_blocked_when_no_formal_negative_complete() -> None:
    run_once()
    update = read_csv(DATASETS / "c4_negative_evidence_intake_update_matrix.csv")[0]
    rows = read_csv(DATASETS / "formal_negative_evidence_intake_registry.csv")
    if update["formal_negative_ready_count"] == "0":
        assert update["c4_ready_after_intake"] == "false"
        assert "C4_STILL_BLOCKED" in update["summary_decision"]
        assert all(row["can_create_training_label"] == "false" for row in rows)
        assert all(row["can_train_model"] == "false" for row in rows)


def test_decision_audit_does_not_unlock_c4_without_ready_negative() -> None:
    run_once()
    decisions = read_csv(DATASETS / "formal_negative_candidate_decision_audit.csv")
    update = read_csv(DATASETS / "c4_negative_evidence_intake_update_matrix.csv")[0]
    if update["formal_negative_ready_count"] == "0":
        assert all(row["can_unlock_c4"] == "false" for row in decisions)
        assert all("TRAINING_LABEL" in row["forbidden_use"] for row in decisions)


def test_outputs_and_schemas_exist() -> None:
    run_once()
    paths = [
        DATASETS / "formal_negative_evidence_intake_registry.csv",
        DATASETS / "formal_negative_candidate_decision_audit.csv",
        DATASETS / "c4_negative_evidence_intake_update_matrix.csv",
        SCHEMAS / "formal_negative_evidence_intake_schema.csv",
        SCHEMAS / "formal_negative_candidate_decision_audit_schema.csv",
        SCHEMAS / "c4_negative_evidence_intake_update_schema.csv",
        DOCS / "protocolo_c_busca_evidencia_negativa_formal_v1jp.md",
        DOCS / "protocolo_c_relatorio_busca_evidencia_negativa_formal_v1jp.md",
    ]
    for path in paths:
        assert path.exists(), path


def test_public_outputs_do_not_leak_private_paths() -> None:
    run_once()
    paths = [
        DATASETS / "formal_negative_evidence_intake_registry.csv",
        DATASETS / "formal_negative_candidate_decision_audit.csv",
        DATASETS / "c4_negative_evidence_intake_update_matrix.csv",
        DOCS / "protocolo_c_busca_evidencia_negativa_formal_v1jp.md",
        DOCS / "protocolo_c_relatorio_busca_evidencia_negativa_formal_v1jp.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment in text]
        assert not leaks, f"Private fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_detection_prediction_claims() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_busca_evidencia_negativa_formal_v1jp.md",
        DOCS / "protocolo_c_relatorio_busca_evidencia_negativa_formal_v1jp.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"


def test_no_formal_negative_matrix_records_correct_blocker() -> None:
    run_once()
    update = read_csv(DATASETS / "c4_negative_evidence_intake_update_matrix.csv")[0]
    if update["formal_negative_ready_count"] == "0":
        assert update["summary_decision"] == "NEGATIVE_INTAKE_NO_FORMAL_NEGATIVES_FOUND;C4_STILL_BLOCKED"
        assert update["supervised_training_status_after_intake"] == "SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVE_LABELS"
