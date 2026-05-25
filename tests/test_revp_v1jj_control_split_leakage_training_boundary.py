"""Integration tests for REV-P v1jj control/split/leakage boundary."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jj_control_split_leakage_training_boundary.py"
LOCAL = REVP_ROOT / "local_runs/protocolo_c/v1jj"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1ji-batch",
    "--build-control-candidates",
    "--design-split-leakage-protocol",
    "--evaluate-sandbox-boundary",
    "--emit-training-boundary",
]

PRIVATE_FRAGMENTS = [
    r"C:\Users\gabriela",
    "gabriela",
    r"Documents\REV-P",
    "Documents/REV-P",
]

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
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@lru_cache(maxsize=1)
def run_once() -> None:
    result = subprocess.run(COMMAND, cwd=str(REVP_ROOT), capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (LOCAL / "v1jj_summary.json").exists()


def test_dino_count_is_normalized_as_pair_diagnostics() -> None:
    run_once()
    rows = read_csv(LOCAL / "v1jj_dino_batch_count_audit.csv")
    assert rows[0]["pre_embedding_count"] == "9"
    assert rows[0]["post_embedding_count"] == "9"
    assert rows[0]["pair_diagnostic_count"] == "9"
    assert rows[0]["embedding_dim"] == "768"
    assert rows[0]["qa_status"] == "QA_PASS"


def test_control_candidate_never_becomes_negative() -> None:
    run_once()
    rows = read_csv(DATASETS / "control_candidate_expansion_registry.csv")
    assert rows
    assert all(row["can_be_negative_label"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)
    assert any(row["control_type"] == "INVALID_NEGATIVE_LABEL" for row in rows)


def test_absence_of_record_is_blocked_as_label() -> None:
    run_once()
    rows = read_csv(DATASETS / "control_candidate_expansion_registry.csv")
    invalid = [row for row in rows if row["control_type"] == "INVALID_NEGATIVE_LABEL"]
    assert invalid
    assert invalid[0]["absence_claim_made"] == "true"
    assert invalid[0]["can_be_negative_label"] == "false"
    assert "ABSENCE_OF_RECORD" in invalid[0]["blocking_reason"]


def test_split_requires_anchor_event_locality_rules() -> None:
    run_once()
    rows = read_csv(DATASETS / "split_leakage_protocol_registry.csv")
    split_units = {row["split_unit"] for row in rows}
    assert {"DOCUMENTED_EVENT_UNIT", "LOCALITY", "EVENT_DATE"} <= split_units
    assert all(row["can_train_model"] == "false" for row in rows)
    assert all(row["leakage_risk_status"] == "LEAKAGE_PROTOCOL_REQUIRED" for row in rows)


def test_same_anchor_pre_post_cannot_be_split_as_independent_samples() -> None:
    run_once()
    rows = read_csv(DATASETS / "split_leakage_protocol_registry.csv")
    assert all("same anchor" in row["same_anchor_pair_rule"].lower() for row in rows)
    controls = read_csv(DATASETS / "control_candidate_expansion_registry.csv")
    temporal = [row for row in controls if row["control_type"] == "TEMPORAL_SELF_CONTROL"]
    assert temporal
    assert all(row["leakage_risk_status"] == "LEAKAGE_RISK_HIGH" for row in temporal)


def test_training_and_unfreeze_remain_blocked() -> None:
    run_once()
    row = read_csv(DATASETS / "sandbox_training_boundary_registry.csv")[0]
    assert row["supervised_training_ready"] == "false"
    assert row["can_train_model"] == "false"
    assert row["can_create_training_label"] == "false"
    assert row["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_sandbox_is_local_only_and_invalid_for_claim() -> None:
    run_once()
    row = read_csv(DATASETS / "sandbox_training_boundary_registry.csv")[0]
    assert row["weak_label_sandbox_allowed_local_only"] == "true"
    assert row["one_class_prototype_sandbox_allowed"] == "true"
    assert "INVALID_FOR_SCIENTIFIC_CLAIM" in row["notes"]


def test_public_registries_and_schemas_exist() -> None:
    run_once()
    paths = [
        DATASETS / "control_candidate_expansion_registry.csv",
        DATASETS / "split_leakage_protocol_registry.csv",
        DATASETS / "sandbox_training_boundary_registry.csv",
        SCHEMAS / "control_candidate_expansion_schema.csv",
        SCHEMAS / "split_leakage_protocol_schema.csv",
        SCHEMAS / "sandbox_training_boundary_schema.csv",
    ]
    for path in paths:
        assert path.exists(), path


def test_no_private_path_in_public_outputs() -> None:
    run_once()
    paths = [
        DATASETS / "control_candidate_expansion_registry.csv",
        DATASETS / "split_leakage_protocol_registry.csv",
        DATASETS / "sandbox_training_boundary_registry.csv",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_controles_split_leakage_v1jj.md",
        DOCS / "protocolo_c_relatorio_controles_split_leakage_v1jj.md",
    ]
    for path in docs:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "can_train_model=false" in text or "treino supervisionado" in text
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
        if lowered.endswith((".tif", ".tiff", ".npy", ".npz", ".local_geotiff")):
            bad.append(line)
    assert not bad


def test_summary_keeps_formal_negatives_zero() -> None:
    run_once()
    summary = json.loads((LOCAL / "v1jj_summary.json").read_text(encoding="utf-8"))
    assert summary["negative_labels_ready_count"] == 0
    assert summary["split_readiness_status"] == "SPLIT_NOT_READY_INSUFFICIENT_LABELS"
    assert summary["leakage_risk_status"] == "LEAKAGE_PROTOCOL_REQUIRED"
    assert summary["can_train_model"] is False
