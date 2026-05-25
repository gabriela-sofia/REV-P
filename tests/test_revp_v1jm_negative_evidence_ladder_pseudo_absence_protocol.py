"""Integration tests for REV-P v1jm negative evidence ladder protocol."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jm_negative_evidence_ladder_pseudo_absence_protocol.py"
LOCAL = REVP_ROOT / "local_runs/protocolo_c/v1jm"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--audit-formal-negative-evidence",
    "--build-pseudo-absence-candidates",
    "--build-background-unlabeled-candidates",
    "--evaluate-pu-boundary",
    "--evaluate-external-benchmark-option",
    "--emit-negative-ladder",
]

PRIVATE_FRAGMENTS = [r"C:\Users\gabriela", "gabriela", r"Documents\REV-P", "Documents/REV-P"]
FORBIDDEN_DOC_TERMS = ["flood detection", "landslide detection", "flood prediction", "landslide prediction", "detecao", "deteccao", "predicao"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (LOCAL / "v1jm_summary.json").exists()


def test_formal_negative_requires_explicit_absence_or_stability() -> None:
    run_once()
    rows = read_csv(DATASETS / "negative_evidence_ladder_registry.csv")
    ready = [row for row in rows if row["negative_evidence_status"] == "FORMAL_NEGATIVE_READY"]
    assert not ready
    assert all(row["can_be_formal_negative"] == "false" for row in rows if row["explicit_absence_or_stability_evidence"] == "false")


def test_absence_of_record_never_becomes_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "negative_evidence_ladder_registry.csv")
    invalid = [row for row in rows if row["negative_evidence_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"]
    assert invalid
    assert all(row["can_be_formal_negative"] == "false" for row in invalid)
    assert all(row["can_create_training_label"] == "false" for row in invalid)


def test_distance_from_anchor_does_not_create_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "negative_evidence_ladder_registry.csv")
    distance_rows = [row for row in rows if row["distance_to_nearest_positive_anchor_m"]]
    assert distance_rows
    assert all(row["can_be_formal_negative"] == "false" for row in distance_rows)


def test_pseudo_absence_never_becomes_label() -> None:
    run_once()
    rows = read_csv(DATASETS / "pseudo_absence_candidate_registry.csv")
    assert rows
    assert all(row["candidate_type"] == "PSEUDO_ABSENCE_REVIEW_ONLY" for row in rows)
    assert all(row["can_be_formal_negative"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)


def test_background_never_becomes_label() -> None:
    run_once()
    rows = read_csv(DATASETS / "background_unlabeled_candidate_registry.csv")
    assert rows
    assert all(row["candidate_type"] == "BACKGROUND_UNLABELED" for row in rows)
    assert all(row["can_be_formal_negative"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)


def test_pu_sandbox_does_not_unlock_supervised_claim() -> None:
    run_once()
    row = read_csv(DATASETS / "positive_unlabeled_boundary_matrix.csv")[0]
    assert row["pu_boundary_status"] == "PU_SANDBOX_LOCAL_ONLY_READY"
    assert row["can_train_pu_sandbox"] == "true"
    assert row["can_train_supervised_model"] == "false"
    assert row["can_create_training_label"] == "false"
    assert row["model_artifact_status"] == "NO_MODEL_OR_WEIGHTS_SAVED"


def test_supervised_training_false_if_formal_negatives_zero() -> None:
    run_once()
    row = read_csv(DATASETS / "positive_unlabeled_boundary_matrix.csv")[0]
    assert row["formal_negative_ready_count"] == "0"
    assert row["supervised_training_status"] == "SUPERVISED_TRAINING_BLOCKED_NO_FORMAL_NEGATIVES"
    assert row["can_train_supervised_model"] == "false"


def test_can_create_training_label_false_if_gates_incomplete() -> None:
    run_once()
    ladder = read_csv(DATASETS / "negative_evidence_ladder_registry.csv")
    pu = read_csv(DATASETS / "positive_unlabeled_boundary_matrix.csv")[0]
    assert all(row["can_create_training_label"] == "false" for row in ladder)
    assert pu["can_create_training_label"] == "false"


def test_external_benchmark_not_local_ground_truth() -> None:
    run_once()
    rows = read_csv(DATASETS / "external_benchmark_transfer_option_registry.csv")
    assert rows
    assert all(row["role_in_revp"] == "EXTERNAL_SUPERVISED_PRETRAINING_OPTION" for row in rows)
    assert all(row["local_ground_truth_status"] == "EXTERNAL_NEGATIVE_NOT_LOCAL_GROUND_TRUTH" for row in rows)
    assert all(row["can_supply_local_negative_ground_truth"] == "false" for row in rows)


def test_public_outputs_and_schemas_exist() -> None:
    run_once()
    paths = [
        DATASETS / "negative_evidence_ladder_registry.csv",
        DATASETS / "pseudo_absence_candidate_registry.csv",
        DATASETS / "background_unlabeled_candidate_registry.csv",
        DATASETS / "positive_unlabeled_boundary_matrix.csv",
        DATASETS / "external_benchmark_transfer_option_registry.csv",
        SCHEMAS / "negative_evidence_ladder_schema.csv",
        SCHEMAS / "pseudo_absence_candidate_schema.csv",
        SCHEMAS / "background_unlabeled_candidate_schema.csv",
        SCHEMAS / "positive_unlabeled_boundary_schema.csv",
        SCHEMAS / "external_benchmark_transfer_option_schema.csv",
    ]
    for path in paths:
        assert path.exists(), path


def test_no_private_path_in_public_outputs() -> None:
    run_once()
    paths = [
        DATASETS / "negative_evidence_ladder_registry.csv",
        DATASETS / "pseudo_absence_candidate_registry.csv",
        DATASETS / "background_unlabeled_candidate_registry.csv",
        DATASETS / "positive_unlabeled_boundary_matrix.csv",
        DATASETS / "external_benchmark_transfer_option_registry.csv",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_negativos_pseudoausencia_v1jm.md",
        DOCS / "protocolo_c_relatorio_negativos_pseudoausencia_v1jm.md",
    ]
    for path in docs:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "supervised_training_blocked_no_formal_negatives" in text or "supervised_training_blocked_no_negatives" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"


def test_local_runs_and_heavy_outputs_not_versioned() -> None:
    run_once()
    result = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    bad = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "local_runs/" in lowered or r"local_runs\\" in lowered:
            bad.append(line)
        if lowered.endswith((".tif", ".tiff", ".npy", ".npz", ".local_geotiff", ".pkl", ".joblib")):
            bad.append(line)
    assert not bad
