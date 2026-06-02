"""Tests for REV-P v1jq official open negative evidence source discovery."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jq_official_open_negative_evidence_source_discovery.py"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-negative-evidence-queues",
    "--discover-official-open-sources",
    "--audit-source-relevance",
    "--classify-negative-evidence_potential",
    "--emit-source_discovery",
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
    assert (DATASETS / "official_open_negative_evidence_source_registry.csv").exists()


def test_source_registry_and_update_matrix_exist() -> None:
    run_once()
    paths = [
        DATASETS / "official_open_negative_evidence_source_registry.csv",
        DATASETS / "official_open_negative_source_intake_audit.csv",
        DATASETS / "c4_external_negative_source_update_matrix.csv",
        SCHEMAS / "official_open_negative_evidence_source_registry_schema.csv",
        SCHEMAS / "official_open_negative_source_intake_audit_schema.csv",
        SCHEMAS / "c4_external_negative_source_update_matrix_schema.csv",
        DOCS / "protocolo_c_descoberta_fontes_negativas_oficiais_v1jq.md",
        DOCS / "protocolo_c_relatorio_descoberta_fontes_negativas_oficiais_v1jq.md",
    ]
    for path in paths:
        assert path.exists(), path


def test_sources_are_classified_without_absence_of_record_promotion() -> None:
    run_once()
    rows = read_csv(DATASETS / "official_open_negative_evidence_source_registry.csv")
    s2id = [row for row in rows if row["source_id"] == "SRC_V1JQ_S2ID_ATLAS_DIGITAL"][0]
    assert s2id["source_relevance_status"] == "CONTEXT_ONLY_SOURCE"
    assert s2id["can_support_formal_negative"] == "false"
    assert "absence" in s2id["notes"].lower() or "absence" in s2id["cannot_be_used_reason"].lower()


def test_context_and_low_risk_mapping_do_not_become_formal_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "official_open_negative_evidence_source_registry.csv")
    risk_rows = [row for row in rows if row["source_id"] == "SRC_V1JQ_DRM_RJ_CARTA_RISCO_DADOS_ABERTOS"]
    assert risk_rows
    assert risk_rows[0]["source_relevance_status"] == "CONTEXT_ONLY_SOURCE"
    assert risk_rows[0]["can_support_formal_negative"] == "false"
    assert "RISK" in risk_rows[0]["source_type"]


def test_high_potential_source_is_extraction_target_not_ready_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "official_open_negative_source_intake_audit.csv")
    high = [row for row in rows if row["source_id"] == "SRC_V1JQ_DRM_RJ_CARTOGRAFIA_RISCO_FICHAS"]
    assert high
    assert high[0]["candidate_status"] == "EXTRACTION_TARGET_REVIEW"
    assert high[0]["can_be_formal_negative_candidate"] == "false"
    assert high[0]["blocking_reason"] == "NO_EXPLICIT_NEGATIVE_STATEMENT_EXTRACTED"


def test_no_formal_candidate_and_c4_remains_blocked() -> None:
    run_once()
    update = read_csv(DATASETS / "c4_external_negative_source_update_matrix.csv")[0]
    assert update["formal_negative_ready_count"] == "0"
    assert update["formal_negative_review_count"] == "0"
    assert update["c4_ready_after_external_discovery"] == "false"
    assert "C4_STILL_BLOCKED" in update["summary_decision"]


def test_training_and_dino_remain_blocked() -> None:
    run_once()
    update = read_csv(DATASETS / "c4_external_negative_source_update_matrix.csv")[0]
    assert update["can_create_training_label"] == "false"
    assert update["can_train_model"] == "false"
    assert update["can_unfreeze_dino_for_scientific_claim"] == "false"
    assert update["supervised_training_status_after_external_discovery"] == "SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVE_LABELS"


def test_public_outputs_do_not_leak_private_paths() -> None:
    run_once()
    paths = [
        DATASETS / "official_open_negative_evidence_source_registry.csv",
        DATASETS / "official_open_negative_source_intake_audit.csv",
        DATASETS / "c4_external_negative_source_update_matrix.csv",
        DOCS / "protocolo_c_descoberta_fontes_negativas_oficiais_v1jq.md",
        DOCS / "protocolo_c_relatorio_descoberta_fontes_negativas_oficiais_v1jq.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment in text]
        assert not leaks, f"Private fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_detection_prediction_claims() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_descoberta_fontes_negativas_oficiais_v1jq.md",
        DOCS / "protocolo_c_relatorio_descoberta_fontes_negativas_oficiais_v1jq.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
