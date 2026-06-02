"""
Tests for REV-P v1jc multi-anchor, control, and label readiness protocol.

The tests enforce that explicit coordinates can become anchor candidates, while
neighborhood-only records and review controls cannot become labels, negatives,
training data, or DINO unfreeze permission.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REVP_ROOT / "scripts" / "protocolo_c" / "revp_v1jc_multi_anchor_control_label_readiness.py"
LOCAL_RUNS = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jc"
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"

RUN_CMD = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-official-event-units",
    "--recover-explicit-coordinates",
    "--build-anchor-registry",
    "--build-control-candidates",
    "--build-label-readiness",
    "--emit-training-boundary",
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
        "v1jc_explicit_coordinate_recovery.csv",
        "v1jc_multi_anchor_candidate_audit.csv",
        "v1jc_control_candidate_audit.csv",
        "v1jc_label_readiness_matrix.csv",
        "v1jc_training_boundary_decision.csv",
        "v1jc_summary.json",
        "v1jc_qa.csv",
    ]
    for name in required:
        assert (LOCAL_RUNS / name).exists(), f"Missing local v1jc output: {name}"


def test_public_registries_and_schemas_are_created() -> None:
    run_script_once()
    required = [
        DATASETS / "official_multi_anchor_candidate_registry.csv",
        SCHEMAS / "official_multi_anchor_candidate_schema.csv",
        DATASETS / "review_control_candidate_registry.csv",
        SCHEMAS / "review_control_candidate_schema.csv",
        DATASETS / "label_and_training_readiness_matrix.csv",
        SCHEMAS / "label_and_training_readiness_schema.csv",
    ]
    for path in required:
        assert path.exists(), f"Missing public metadata output: {path.name}"


def test_explicit_coordinate_can_be_confirmed_anchor_candidate() -> None:
    run_script_once()
    rows = read_csv(DATASETS / "official_multi_anchor_candidate_registry.csv")
    anchor = next(row for row in rows if row["documented_event_unit_id"] == "PET2022_CPRM_ANEXOII_19022022")

    assert anchor["coordinate_available"] == "true"
    assert anchor["latitude"] == "-22.484251"
    assert anchor["longitude"] == "-43.211257"
    assert anchor["anchor_status"] == "OFFICIAL_ANCHOR_CONFIRMED"
    assert anchor["can_be_positive_reference_candidate"] == "true"
    assert anchor["can_be_positive_training_label"] == "false"


def test_neighborhood_without_coordinate_does_not_become_spatial_anchor() -> None:
    run_script_once()
    rows = read_csv(DATASETS / "official_multi_anchor_candidate_registry.csv")
    serra_velha = next(row for row in rows if row["documented_event_unit_id"] == "PET2022_CPRM_ANEXOIII_20022022")

    assert serra_velha["spatial_precision"] == "NEIGHBORHOOD"
    assert serra_velha["coordinate_available"] == "false"
    assert serra_velha["anchor_status"] == "DOCUMENTARY_EVENT_NO_COORDINATE"
    assert serra_velha["can_be_positive_reference_candidate"] == "false"
    assert serra_velha["blocking_reason"] == "NO_EXPLICIT_COORDINATE"


def test_insufficient_document_is_not_promoted_to_anchor() -> None:
    run_script_once()
    rows = read_csv(DATASETS / "official_multi_anchor_candidate_registry.csv")
    main = next(row for row in rows if row["documented_event_unit_id"] == "PET2022_CPRM_ANEXOMAIN_NODATE")

    assert main["anchor_status"] == "INSUFFICIENT_SPATIAL_PRECISION"
    assert main["can_be_positive_reference_candidate"] == "false"
    assert main["can_train_model"] == "false"


def test_control_candidates_do_not_become_negative_labels() -> None:
    run_script_once()
    rows = read_csv(DATASETS / "review_control_candidate_registry.csv")
    assert rows
    assert {row["control_type"] for row in rows} >= {
        "TEMPORAL_SELF_CONTROL",
        "SPATIAL_CONTEXT_CONTROL_CANDIDATE",
        "EXISTING_PATCH_BACKGROUND_CANDIDATE",
        "INVALID_NEGATIVE_LABEL",
    }
    assert all(row["can_be_negative_label"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)
    invalid = next(row for row in rows if row["control_type"] == "INVALID_NEGATIVE_LABEL")
    assert invalid["absence_claim_made"] == "true"
    assert invalid["can_be_review_control"] == "false"


def test_label_training_and_unfreeze_are_blocked() -> None:
    run_script_once()
    summary = json.loads((LOCAL_RUNS / "v1jc_summary.json").read_text(encoding="utf-8"))
    readiness = read_csv(DATASETS / "label_and_training_readiness_matrix.csv")[0]
    anchor_rows = read_csv(DATASETS / "official_multi_anchor_candidate_registry.csv")
    qa_rows = {row["check"]: row["status"] for row in read_csv(LOCAL_RUNS / "v1jc_qa.csv")}

    assert summary["training_boundary_status"] == "TRAINING_BLOCKED_INSUFFICIENT_LABELS"
    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False
    assert summary["can_unfreeze_dino_for_scientific_claim"] is False
    assert readiness["negative_labels_ready_count"] == "0"
    assert readiness["can_create_training_label"] == "false"
    assert readiness["can_train_model"] == "false"
    assert readiness["can_unfreeze_dino_for_scientific_claim"] == "false"
    assert all(row["can_create_training_label"] == "false" for row in anchor_rows)
    assert all(row["can_train_model"] == "false" for row in anchor_rows)
    assert qa_rows["can_create_training_label_false"] == "PASS"
    assert qa_rows["can_train_model_false"] == "PASS"
    assert qa_rows["can_unfreeze_dino_for_scientific_claim_false"] == "PASS"


def test_review_only_can_continue_without_training_ready() -> None:
    run_script_once()
    readiness = read_csv(DATASETS / "label_and_training_readiness_matrix.csv")[0]

    assert readiness["review_only_status"] == "REVIEW_ONLY_READY"
    assert readiness["split_readiness_status"] == "SPLIT_BLOCKED_SINGLE_CONFIRMED_ANCHOR"
    assert readiness["leakage_risk_status"] == "LEAKAGE_PROTOCOL_REQUIRED"
    assert readiness["training_boundary_status"] == "TRAINING_BLOCKED_INSUFFICIENT_LABELS"


def test_no_private_path_in_public_outputs() -> None:
    run_script_once()
    public_files = [
        DATASETS / "official_multi_anchor_candidate_registry.csv",
        SCHEMAS / "official_multi_anchor_candidate_schema.csv",
        DATASETS / "review_control_candidate_registry.csv",
        SCHEMAS / "review_control_candidate_schema.csv",
        DATASETS / "label_and_training_readiness_matrix.csv",
        SCHEMAS / "label_and_training_readiness_schema.csv",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_local_runs_and_local_only_are_not_versioned() -> None:
    run_script_once()
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    bad_lines = [
        line
        for line in result.stdout.splitlines()
        if "local_runs/" in line.lower() or r"local_runs\\" in line.lower() or "local_only" in line.lower()
    ]
    assert not bad_lines, f"Local-only artifacts are visible to git: {bad_lines}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_script_once()
    doc_files = [
        DOCS / "protocolo_c_multiplos_anchors_controles_labels_v1jc.md",
        DOCS / "protocolo_c_relatorio_multiplos_anchors_controles_labels_v1jc.md",
    ]
    for path in doc_files:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "controle candidato" in text
        assert "negativo formal" in text
        assert "dino" in text
        assert "frozen" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"
