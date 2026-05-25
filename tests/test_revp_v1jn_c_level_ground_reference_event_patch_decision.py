"""Integration tests for REV-P v1jn C-level ground reference layer."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jn_c_level_ground_reference_event_patch_decision.py"
LOCAL = REVP_ROOT / "local_runs/protocolo_c/v1jn"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-ground-reference-inputs",
    "--build-event-registry",
    "--build-event-patch-linkage",
    "--build-candidate-decision-audit",
    "--emit-c-level-summary",
]

PRIVATE_FRAGMENTS = [r"C:\Users\gabriela", "gabriela", r"Documents\REV-P", "Documents/REV-P", "local_runs/"]
FORBIDDEN_DOC_TERMS = ["flood detection", "landslide detection", "flood prediction", "landslide prediction", "detecao", "deteccao", "predicao"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (LOCAL / "v1jn_c_level_summary.json").exists()


def test_event_registry_created_with_nine_official_anchors() -> None:
    run_once()
    rows = read_csv(DATASETS / "ground_reference_event_registry.csv")
    assert len(rows) == 9
    assert all(row["event_id"].startswith("EVENT_PET2022_CPRM_") for row in rows)
    assert all(row["source_institution"] == "SGB/CPRM" for row in rows)


def test_events_with_patch_qa_enter_c3() -> None:
    run_once()
    events = read_csv(DATASETS / "ground_reference_event_registry.csv")
    linkages = read_csv(DATASETS / "event_patch_linkage_registry.csv")
    assert all(row["c_level"] == "C3_EVENT_PATCH_LINKED" for row in events)
    assert all(row["c_level_after_linkage"] == "C3_EVENT_PATCH_LINKED" for row in linkages)
    assert all(row["s2_pair_status"] == "QA_PASS" for row in linkages)
    assert all(row["dem_status"] == "QA_PASS" for row in linkages)
    assert all(row["dino_status"] == "DINO_QA_PASS" for row in linkages)


def test_no_event_becomes_c4_without_negative_split_leakage_gates() -> None:
    run_once()
    events = read_csv(DATASETS / "ground_reference_event_registry.csv")
    decisions = read_csv(DATASETS / "ground_truth_candidate_decision_audit.csv")
    summary = read_csv(DATASETS / "protocol_c_c_level_summary_registry.csv")[0]
    assert not [row for row in events if row["c_level"] == "C4_OPERATIONAL_LABEL_CANDIDATE"]
    assert summary["c4_operational_label_candidate_count"] == "0"
    assert all("FORMAL_NEGATIVES_ZERO" in row["blocking_reason"] for row in decisions)


def test_training_label_and_model_training_remain_blocked() -> None:
    run_once()
    decisions = read_csv(DATASETS / "ground_truth_candidate_decision_audit.csv")
    events = read_csv(DATASETS / "ground_reference_event_registry.csv")
    assert all(row["can_create_training_label"] == "false" for row in events)
    assert all(row["can_create_training_label"] == "false" for row in decisions)
    assert all(row["can_train_model"] == "false" for row in decisions)
    assert all(row["can_unfreeze_dino_for_scientific_claim"] == "false" for row in decisions)


def test_pseudo_absence_does_not_become_negative() -> None:
    run_once()
    pseudo = read_csv(DATASETS / "pseudo_absence_candidate_registry.csv")
    decisions = read_csv(DATASETS / "ground_truth_candidate_decision_audit.csv")
    assert pseudo
    assert all(row["can_be_formal_negative"] == "false" for row in pseudo)
    assert all(row["can_create_training_label"] == "false" for row in pseudo)
    assert all(row["can_be_negative_label"] == "false" for row in decisions)
    assert all(row["negative_evidence_status"] == "FORMAL_NEGATIVES_ZERO" for row in decisions)


def test_dino_does_not_create_label() -> None:
    run_once()
    dino = read_csv(DATASETS / "multi_anchor_dino_review_embedding_registry.csv")
    decisions = read_csv(DATASETS / "ground_truth_candidate_decision_audit.csv")
    assert all(row["embedding_dim"] == "768" for row in dino)
    assert all(row["dino_status"] == "DINO_QA_PASS" for row in dino)
    assert all(row["can_be_positive_label"] == "false" for row in decisions)
    assert all("DINO_UNFREEZE" in row["forbidden_use"] for row in decisions)


def test_public_outputs_and_schemas_exist() -> None:
    run_once()
    paths = [
        DATASETS / "ground_reference_event_registry.csv",
        DATASETS / "event_patch_linkage_registry.csv",
        DATASETS / "ground_truth_candidate_decision_audit.csv",
        DATASETS / "protocol_c_c_level_summary_registry.csv",
        SCHEMAS / "ground_reference_event_registry_schema.csv",
        SCHEMAS / "event_patch_linkage_registry_schema.csv",
        SCHEMAS / "ground_truth_candidate_decision_audit_schema.csv",
        SCHEMAS / "protocol_c_c_level_summary_schema.csv",
        DOCS / "protocolo_c_camada_c1_c4_ground_reference_v1jn.md",
        DOCS / "protocolo_c_relatorio_camada_c1_c4_ground_reference_v1jn.md",
    ]
    for path in paths:
        assert path.exists(), path


def test_no_private_path_in_public_outputs() -> None:
    run_once()
    paths = [
        DATASETS / "ground_reference_event_registry.csv",
        DATASETS / "event_patch_linkage_registry.csv",
        DATASETS / "ground_truth_candidate_decision_audit.csv",
        DATASETS / "protocol_c_c_level_summary_registry.csv",
        DOCS / "protocolo_c_camada_c1_c4_ground_reference_v1jn.md",
        DOCS / "protocolo_c_relatorio_camada_c1_c4_ground_reference_v1jn.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_claim_terms() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_camada_c1_c4_ground_reference_v1jn.md",
        DOCS / "protocolo_c_relatorio_camada_c1_c4_ground_reference_v1jn.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "c4 e treino seguem bloqueados" in text or "c4 permanece bloqueado" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
