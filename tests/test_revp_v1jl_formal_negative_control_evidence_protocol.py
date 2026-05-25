"""Integration tests for REV-P v1jl formal negative/control evidence protocol."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jl_formal_negative_control_evidence_protocol.py"
LOCAL = REVP_ROOT / "local_runs/protocolo_c/v1jl"

PRIVATE_FRAGMENTS = [r"C:\Users\gabriela", "gabriela", r"Documents\REV-P", "Documents/REV-P"]
FORBIDDEN_DOC_TERMS = ["flood detection", "landslide detection", "flood prediction", "landslide prediction", "detecao", "deteccao", "predicao"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--force"], cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (LOCAL / "v1jl_summary.json").exists()


def test_absence_of_record_never_becomes_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_control_evidence_registry.csv")
    invalid = [row for row in rows if row["control_strength_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"]
    assert invalid
    assert all(row["can_be_negative_label"] == "false" for row in invalid)
    assert all("ABSENCE" in row["blocking_reason"] for row in invalid)


def test_pre_event_same_anchor_not_independent_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_control_evidence_registry.csv")
    temporal = [row for row in rows if row["candidate_type"] == "TEMPORAL_SELF_CONTROL"]
    assert temporal
    assert all(row["control_strength_status"] == "STRONG_CONTROL_CANDIDATE" for row in temporal)
    assert all(row["can_be_negative_label"] == "false" for row in temporal)


def test_cross_region_not_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_control_evidence_registry.csv")
    cross = [row for row in rows if row["candidate_type"] == "CROSS_REGION_CONTEXT_CANDIDATE"]
    assert cross
    assert all(row["can_be_negative_label"] == "false" for row in cross)
    assert all(row["negative_label_status"] == "NO_NEGATIVE_LABEL_CROSS_REGION_CONTEXT" for row in cross)


def test_formal_negative_requires_explicit_absence_or_stability() -> None:
    run_once()
    rows = read_csv(DATASETS / "formal_negative_control_evidence_registry.csv")
    ready = [row for row in rows if row["control_strength_status"] == "FORMAL_NEGATIVE_READY"]
    assert not ready
    assert all(row["official_absence_evidence"] == "false" for row in rows)
    assert all(row["official_stability_evidence"] == "false" for row in rows)


def test_if_negatives_zero_training_remains_blocked() -> None:
    run_once()
    readiness = read_csv(DATASETS / "negative_label_readiness_matrix.csv")[0]
    gate = read_csv(DATASETS / "supervised_training_minimum_gate_matrix.csv")[0]
    assert readiness["formal_negative_ready_count"] == "0"
    assert gate["supervised_training_boundary_status"] == "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES"
    assert gate["can_train_model"] == "false"
    assert gate["can_create_training_label"] == "false"
    assert gate["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_public_registries_and_schemas_exist() -> None:
    run_once()
    paths = [
        DATASETS / "formal_negative_control_evidence_registry.csv",
        DATASETS / "negative_label_readiness_matrix.csv",
        DATASETS / "supervised_training_minimum_gate_matrix.csv",
        SCHEMAS / "formal_negative_control_evidence_schema.csv",
        SCHEMAS / "negative_label_readiness_schema.csv",
        SCHEMAS / "supervised_training_minimum_gate_schema.csv",
    ]
    for path in paths:
        assert path.exists(), path


def test_no_private_path_in_public_outputs() -> None:
    run_once()
    paths = [
        DATASETS / "formal_negative_control_evidence_registry.csv",
        DATASETS / "negative_label_readiness_matrix.csv",
        DATASETS / "supervised_training_minimum_gate_matrix.csv",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_negativos_controles_formais_v1jl.md",
        DOCS / "protocolo_c_relatorio_negativos_controles_formais_v1jl.md",
    ]
    for path in docs:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "supervised_training_blocked_no_negatives" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"


def test_local_runs_ignored() -> None:
    run_once()
    result = subprocess.run(["git", "status", "--short", "--untracked-files=all"], cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    bad = [line for line in result.stdout.splitlines() if "local_runs/" in line.lower() or r"local_runs\\" in line.lower()]
    assert not bad
