"""Tests for REV-P v1jo C4 gate gap analysis and action queues."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jo_c4_gate_gap_analysis_action_queue.py"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-c-level-registries",
    "--audit-c4-gaps",
    "--rank-blockers",
    "--build-action-queues",
    "--emit-c4-readiness",
]

PRIVATE_FRAGMENTS = [r"C:\Users\gabriela", "gabriela", r"Documents\REV-P", "Documents/REV-P", "local_runs/"]
FORBIDDEN_DOC_TERMS = [
    "flood detection",
    "landslide detection",
    "flood prediction",
    "landslide prediction",
    "detecao",
    "deteccao",
    "predicao",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False, timeout=120)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (DATASETS / "c4_transition_readiness_matrix.csv").exists()


def test_c3_and_c4_counts_preserved() -> None:
    run_once()
    transition = read_csv(DATASETS / "c4_transition_readiness_matrix.csv")[0]
    gaps = read_csv(DATASETS / "c4_gate_gap_analysis_registry.csv")
    assert transition["confirmed_c3_events"] == "9"
    assert transition["c4_ready_events"] == "0"
    assert len(gaps) == 9
    assert all(row["current_c_level"] == "C3_EVENT_PATCH_LINKED" for row in gaps)


def test_formal_negatives_zero_is_primary_blocker() -> None:
    run_once()
    gaps = read_csv(DATASETS / "c4_gate_gap_analysis_registry.csv")
    ranking = read_csv(DATASETS / "c4_blocker_priority_ranking.csv")
    assert all(row["primary_blocker"] == "FORMAL_NEGATIVES_ZERO" for row in gaps)
    assert ranking[0]["blocker"] == "FORMAL_NEGATIVES_ZERO"
    assert ranking[0]["affected_event_count"] == "9"
    assert ranking[0]["blocks_c4"] == "true"


def test_s1_partial_is_secondary_not_primary() -> None:
    run_once()
    gaps = read_csv(DATASETS / "c4_gate_gap_analysis_registry.csv")
    s1_rows = [row for row in gaps if row["requires_s1_completion"] == "true"]
    assert len(s1_rows) == 8
    assert all(row["primary_blocker"] == "FORMAL_NEGATIVES_ZERO" for row in s1_rows)
    assert all("S1_PARTIAL_COVERAGE" in row["secondary_blockers"] for row in s1_rows)


def test_pseudo_absence_does_not_unlock_c4() -> None:
    run_once()
    transition = read_csv(DATASETS / "c4_transition_readiness_matrix.csv")[0]
    gaps = read_csv(DATASETS / "c4_gate_gap_analysis_registry.csv")
    assert int(transition["pseudo_absence_count"]) > 0
    assert transition["formal_negative_count"] == "0"
    assert all(row["gate_negative_evidence"] == "FAIL_FORMAL_NEGATIVES_ZERO" for row in gaps)
    assert all(row["can_create_training_label"] == "false" for row in gaps)


def test_negative_evidence_queue_rejects_absence_of_record_as_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "c4_negative_evidence_search_queue.csv")
    assert rows
    text = "\n".join(";".join(row.values()) for row in rows).lower()
    assert "ausencia de registro como negativo" not in text
    assert "ausência de registro como negativo" not in text
    assert "explicit" in text or "explicita" in text or "explícita" in text
    assert "pseudo-absence" in text or "pseudo-ausencia" in text or "pseudo-ausência" in text


def test_s1_completion_queue_strengthens_c3_but_never_unlocks_c4() -> None:
    run_once()
    rows = read_csv(DATASETS / "c4_s1_completion_queue.csv")
    assert len(rows) == 8
    assert all(row["would_unlock_c4"] == "false" for row in rows)
    assert all(row["would_strengthen_c3"] == "true" for row in rows)


def test_split_leakage_queue_waits_for_formal_labels() -> None:
    run_once()
    rows = read_csv(DATASETS / "c4_split_leakage_precondition_queue.csv")
    transition = read_csv(DATASETS / "c4_transition_readiness_matrix.csv")[0]
    assert rows
    assert transition["formal_negative_count"] == "0"
    assert any("FORMAL" in row["status"] or "FORMAL" in row["notes"] for row in rows)
    assert all(row["current_positive_c3_count"] == "9" for row in rows)
    assert all(row["current_formal_negative_count"] == "0" for row in rows)


def test_training_and_dino_unfreeze_remain_blocked() -> None:
    run_once()
    transition = read_csv(DATASETS / "c4_transition_readiness_matrix.csv")[0]
    assert transition["can_create_training_label"] == "false"
    assert transition["can_train_model"] == "false"
    assert transition["can_unfreeze_dino_for_scientific_claim"] == "false"
    assert transition["summary_decision"] == "C3_LAYER_STABLE_C4_BLOCKED_BY_NEGATIVE_EVIDENCE"


def test_public_outputs_and_schemas_exist() -> None:
    run_once()
    paths = [
        DATASETS / "c4_gate_gap_analysis_registry.csv",
        DATASETS / "c4_blocker_priority_ranking.csv",
        DATASETS / "c4_negative_evidence_search_queue.csv",
        DATASETS / "c4_s1_completion_queue.csv",
        DATASETS / "c4_split_leakage_precondition_queue.csv",
        DATASETS / "c4_transition_readiness_matrix.csv",
        SCHEMAS / "c4_gate_gap_analysis_registry_schema.csv",
        SCHEMAS / "c4_blocker_priority_ranking_schema.csv",
        SCHEMAS / "c4_negative_evidence_search_queue_schema.csv",
        SCHEMAS / "c4_s1_completion_queue_schema.csv",
        SCHEMAS / "c4_split_leakage_precondition_queue_schema.csv",
        SCHEMAS / "c4_transition_readiness_matrix_schema.csv",
        DOCS / "protocolo_c_priorizacao_c4_v1jo.md",
        DOCS / "protocolo_c_relatorio_priorizacao_c4_v1jo.md",
    ]
    for path in paths:
        assert path.exists(), path


def test_no_private_path_in_public_v1jo_outputs() -> None:
    run_once()
    paths = [
        DATASETS / "c4_gate_gap_analysis_registry.csv",
        DATASETS / "c4_blocker_priority_ranking.csv",
        DATASETS / "c4_negative_evidence_search_queue.csv",
        DATASETS / "c4_s1_completion_queue.csv",
        DATASETS / "c4_split_leakage_precondition_queue.csv",
        DATASETS / "c4_transition_readiness_matrix.csv",
        DOCS / "protocolo_c_priorizacao_c4_v1jo.md",
        DOCS / "protocolo_c_relatorio_priorizacao_c4_v1jo.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment in text]
        assert not leaks, f"Private fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_detection_prediction_claims() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_priorizacao_c4_v1jo.md",
        DOCS / "protocolo_c_relatorio_priorizacao_c4_v1jo.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "c4 nao esta liberado" in text or "nenhum evento foi promovido a c4" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
