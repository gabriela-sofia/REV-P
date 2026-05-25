"""Integration tests for REV-P v1ji multi-anchor multimodal readiness."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from functools import lru_cache
from pathlib import Path


REVP_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
DATASETS = REVP_ROOT / "datasets"
SCHEMAS = DATASETS / "schemas"
DOCS = REVP_ROOT / "docs" / "metodologia_cientifica"
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1ji_multi_anchor_batch_multimodal_readiness.py"
LOCAL = REVP_ROOT / "local_runs/protocolo_c/v1ji"
RUN_INTEGRATION = os.environ.get("RUN_REVP_INTEGRATION") == "1"
SCRIPT_COMMAND = [sys.executable, str(SCRIPT), "--force"] if RUN_INTEGRATION else [sys.executable, str(SCRIPT), "--metadata-only-test"]
SCRIPT_TIMEOUT_SECONDS = 1800 if RUN_INTEGRATION else 60

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
    result = subprocess.run(
        SCRIPT_COMMAND,
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
        timeout=SCRIPT_TIMEOUT_SECONDS,
    )
    assert result.returncode == 0, result.stderr + result.stdout


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    run_once()
    assert (LOCAL / "v1ji_summary.json").exists()


def test_anchor_deduplication_explains_nine_official_units() -> None:
    run_once()
    rows = read_csv(LOCAL / "v1ji_anchor_deduplication_audit.csv")
    assert len(rows) >= 9
    statuses = {row["dedup_status"] for row in rows}
    assert statuses <= {"OFFICIAL_ANCHOR_CONFIRMED", "DUPLICATE_COORDINATE_MERGED", "COORDINATE_REVIEW_REQUIRED", "REVIEW_AREA_ONLY"}
    assert any(row["coordinate_expression_count"] != "1" for row in rows)


def test_public_registries_and_schemas_exist() -> None:
    run_once()
    public_files = [
        DATASETS / "official_multi_anchor_registry.csv",
        DATASETS / "multi_anchor_multimodal_patch_registry.csv",
        DATASETS / "multi_anchor_dino_review_embedding_registry.csv",
        DATASETS / "multi_anchor_training_gate_matrix.csv",
        SCHEMAS / "official_multi_anchor_schema.csv",
        SCHEMAS / "multi_anchor_multimodal_patch_schema.csv",
        SCHEMAS / "multi_anchor_dino_review_embedding_schema.csv",
        SCHEMAS / "multi_anchor_training_gate_schema.csv",
    ]
    for path in public_files:
        assert path.exists(), path


def test_s2_batch_does_not_create_labels() -> None:
    run_once()
    rows = read_csv(DATASETS / "multi_anchor_multimodal_patch_registry.csv")
    assert rows
    assert all(row["positive_label_ready"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)
    assert all(row["can_train_model"] == "false" for row in rows)


def test_s1_and_dem_absence_or_failure_is_controlled() -> None:
    run_once()
    s1_rows = read_csv(LOCAL / "v1ji_s1_patch_quality_audit.csv")
    dem_rows = read_csv(LOCAL / "v1ji_dem_terrain_quality_audit.csv")
    assert s1_rows or dem_rows
    assert all(row.get("qa_status") in {"QA_PASS", "QA_FAIL", "PATCH_NOT_AVAILABLE"} for row in s1_rows)
    assert all(row.get("qa_status") in {"QA_PASS", "QA_FAIL", "PATCH_NOT_AVAILABLE"} for row in dem_rows)


def test_dino_is_frozen_review_only_and_optional() -> None:
    run_once()
    rows = read_csv(DATASETS / "multi_anchor_dino_review_embedding_registry.csv")
    assert rows
    for row in rows:
        if row["dino_status"] == "DINO_QA_PASS":
            assert row["embedding_dim"] == "768"
        assert row["can_create_training_label"] == "false"
        assert row["can_train_model"] == "false"


def test_negatives_zero_and_training_blocked() -> None:
    run_once()
    gate = read_csv(DATASETS / "multi_anchor_training_gate_matrix.csv")[0]
    assert gate["negative_labels_ready_count"] == "0"
    assert gate["training_gate_status"] == "SUPERVISED_TRAINING_BLOCKED"
    assert gate["leakage_risk_status"] == "LEAKAGE_PROTOCOL_REQUIRED"
    assert gate["can_create_training_label"] == "false"
    assert gate["can_train_model"] == "false"
    assert gate["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_summary_matches_public_training_boundary() -> None:
    run_once()
    summary = json.loads((LOCAL / "v1ji_summary.json").read_text(encoding="utf-8"))
    assert summary["official_unique_anchor_count"] == 9
    if not RUN_INTEGRATION:
        assert summary["execution_mode"] == "METADATA_ONLY_TEST"
        assert summary["gee_status"] == "METADATA_ONLY_TEST_NO_GEE"
    assert summary["negative_labels_ready_count"] == 0
    assert summary["training_gate_status"] == "SUPERVISED_TRAINING_BLOCKED"
    assert summary["can_create_training_label"] is False
    assert summary["can_train_model"] is False


def test_no_private_path_in_public_outputs() -> None:
    run_once()
    public_files = [
        DATASETS / "official_multi_anchor_registry.csv",
        DATASETS / "multi_anchor_multimodal_patch_registry.csv",
        DATASETS / "multi_anchor_dino_review_embedding_registry.csv",
        DATASETS / "multi_anchor_training_gate_matrix.csv",
    ]
    for path in public_files:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_batch_multimodal_anchors_oficiais_v1ji.md",
        DOCS / "protocolo_c_relatorio_batch_multimodal_anchors_oficiais_v1ji.md",
    ]
    for path in docs:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "can_train_model=false" in text or "supervisionado bloqueado" in text
        for term in FORBIDDEN_DOC_TERMS:
            assert term not in text, f"Forbidden wording {term!r} found in {path.name}"


def test_raster_npy_npz_and_local_runs_not_versioned() -> None:
    run_once()
    result = subprocess.run(
        ["git", "status", "--short", "--untracked-files=all"],
        cwd=str(REVP_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    forbidden_suffixes = (".tif", ".tiff", ".npy", ".npz", ".local_geotiff")
    bad = []
    for line in result.stdout.splitlines():
        lowered = line.lower()
        if "local_runs/" in lowered or r"local_runs\\" in lowered:
            bad.append(line)
        if lowered.endswith(forbidden_suffixes):
            bad.append(line)
    assert not bad
