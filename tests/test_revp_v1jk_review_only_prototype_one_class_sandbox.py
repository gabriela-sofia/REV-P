"""Integration tests for REV-P v1jk review-only sandbox."""

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
SCRIPT = REVP_ROOT / "scripts/protocolo_c/revp_v1jk_review_only_prototype_one_class_sandbox.py"
LOCAL = REVP_ROOT / "local_runs/protocolo_c/v1jk"

COMMAND = [
    sys.executable,
    str(SCRIPT),
    "--force",
    "--read-v1ji-batch",
    "--read-v1jj-boundary",
    "--build-feature-table",
    "--run-review-only-prototypes",
    "--run-one-class-sandbox",
    "--emit-sandbox-report",
]

PRIVATE_FRAGMENTS = [r"C:\Users\gabriela", "gabriela", r"Documents\REV-P", "Documents/REV-P"]
FORBIDDEN_DOC_TERMS = ["flood detection", "landslide detection", "flood prediction", "landslide prediction", "detecao", "deteccao", "predicao"]


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
    assert (LOCAL / "v1jk_summary.json").exists()


def test_feature_table_created_for_official_anchors() -> None:
    run_once()
    rows = read_csv(LOCAL / "v1jk_feature_table.csv")
    assert len(rows) == 9
    assert all(row["s2_qa_status"] == "S2_PRE_POST_QA_PASS" for row in rows)
    assert all(row["dino_euclidean_distance"] for row in rows)


def test_prototypes_and_groups_do_not_become_classes() -> None:
    run_once()
    pca = read_csv(LOCAL / "v1jk_pca_projection.csv")
    ranking = read_csv(LOCAL / "v1jk_anchor_change_ranking.csv")
    assert pca
    assert ranking
    assert all("CLASS" not in row["exploratory_group"].upper() for row in pca)
    assert all("class" not in row["notes"].lower() or "not" in row["notes"].lower() for row in ranking)


def test_one_class_sandbox_invalid_for_claim_if_it_runs() -> None:
    run_once()
    rows = read_csv(LOCAL / "v1jk_one_class_sandbox_log.csv")
    assert rows
    assert all(row["model_saved"] == "false" for row in rows)
    assert all(row["can_create_training_label"] == "false" for row in rows)
    assert all(row["can_train_model"] == "false" for row in rows)
    statuses = {row["sandbox_status"] for row in rows}
    assert any("INVALID_FOR_SCIENTIFIC_CLAIM" in status or "SKLEARN_UNAVAILABLE" in status for status in statuses)


def test_public_registry_keeps_training_and_labels_blocked() -> None:
    run_once()
    row = read_csv(DATASETS / "review_only_multimodal_sandbox_registry.csv")[0]
    assert row["scientific_claim_status"] == "INVALID_FOR_SUPERVISED_CLAIM"
    assert row["supervised_training_status"] == "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES"
    assert row["can_create_training_label"] == "false"
    assert row["can_train_model"] == "false"
    assert row["can_unfreeze_dino_for_scientific_claim"] == "false"


def test_public_registry_and_schema_exist() -> None:
    run_once()
    assert (DATASETS / "review_only_multimodal_sandbox_registry.csv").exists()
    assert (SCHEMAS / "review_only_multimodal_sandbox_schema.csv").exists()


def test_no_private_path_in_public_outputs() -> None:
    run_once()
    paths = [DATASETS / "review_only_multimodal_sandbox_registry.csv", SCHEMAS / "review_only_multimodal_sandbox_schema.csv"]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        leaks = [fragment for fragment in PRIVATE_FRAGMENTS if fragment.lower() in text]
        assert not leaks, f"Private path fragment leaked in {path.name}: {leaks}"


def test_docs_do_not_use_forbidden_terms() -> None:
    run_once()
    docs = [
        DOCS / "protocolo_c_sandbox_review_only_v1jk.md",
        DOCS / "protocolo_c_relatorio_sandbox_review_only_v1jk.md",
    ]
    for path in docs:
        assert path.exists()
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert "classe" in text or "label" in text
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
