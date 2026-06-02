from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(r"C:\Users\gabriela\Documents\REV-P")
SCRIPT = REPO_ROOT / "scripts" / "protocolo_c" / "revp_v1iw_patch_corpus_taxonomy_reconciliation.py"
LOCAL_OUT = REPO_ROOT / "local_runs" / "protocolo_c" / "v1iw"
REGISTRY = REPO_ROOT / "datasets" / "patch_corpus_taxonomy_registry.csv"
SCHEMA = REPO_ROOT / "datasets" / "schemas" / "patch_corpus_taxonomy_schema.csv"
DOCS = [
    REPO_ROOT / "docs" / "metodologia_cientifica" / "protocolo_c_taxonomia_corpus_patches_v1iw.md",
    REPO_ROOT / "docs" / "metodologia_cientifica" / "protocolo_c_relatorio_taxonomia_corpus_patches_v1iw.md",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--force",
            "--read-dataset-registries",
            "--read-patch-registries",
            "--read-dino-manifests",
            "--emit-taxonomy",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def registry_by_id() -> dict[str, dict[str, str]]:
    return {row["taxonomy_id"]: row for row in read_csv(REGISTRY)}


def test_script_exists_and_runs() -> None:
    assert SCRIPT.exists()
    result = run_script()
    assert result.returncode == 0, result.stderr
    for name in [
        "v1iw_patch_count_sources.csv",
        "v1iw_patch_taxonomy_matrix.csv",
        "v1iw_count_reconciliation_report.csv",
        "v1iw_summary.json",
        "v1iw_qa.csv",
    ]:
        assert (LOCAL_OUT / name).exists(), f"Missing local output: {name}"


def test_registry_and_schema_are_created() -> None:
    assert REGISTRY.exists()
    assert SCHEMA.exists()
    schema_fields = {row["field"] for row in read_csv(SCHEMA)}
    for field in ["taxonomy_id", "corpus_layer", "count_total", "can_create_training_label"]:
        assert field in schema_fields


def test_territorial_59_counts() -> None:
    row = registry_by_id()["TERRITORIAL_CONSOLIDATED_PATCH_CORPUS"]
    assert row["count_total"] == "59"
    assert row["recife_count"] == "18"
    assert row["petropolis_count"] == "27"
    assert row["curitiba_count"] == "14"
    assert row["unit_type"] == "territorial_patch"


def test_sentinel_128_counts() -> None:
    row = registry_by_id()["SENTINEL_CANDIDATE_ASSET_MANIFEST"]
    assert row["count_total"] == "128"
    assert row["recife_count"] == "37"
    assert row["petropolis_count"] == "48"
    assert row["curitiba_count"] == "43"
    assert row["unit_type"] == "sentinel_candidate_asset"


def test_59_and_128_are_not_ground_truth() -> None:
    rows = registry_by_id()
    for key in ["TERRITORIAL_CONSOLIDATED_PATCH_CORPUS", "SENTINEL_CANDIDATE_ASSET_MANIFEST"]:
        assert rows[key]["can_be_used_as_ground_truth"] == "false"


def test_can_create_training_label_false_always() -> None:
    for row in read_csv(REGISTRY):
        assert row["can_create_training_label"] == "false"
    summary = json.loads((LOCAL_OUT / "v1iw_summary.json").read_text(encoding="utf-8"))
    assert summary["can_create_training_label"] is False


def test_docs_explain_territorial_vs_sentinel_manifest() -> None:
    content = "\n".join(path.read_text(encoding="utf-8") for path in DOCS)
    assert "59" in content
    assert "128" in content
    assert "corpus territorial" in content.lower()
    assert "manifesto sentinel" in content.lower()
    assert "não são contagens concorrentes" in content.lower() or "nao sao contagens concorrentes" in content.lower()


def test_docs_do_not_use_forbidden_detection_prediction_terms() -> None:
    content = "\n".join(path.read_text(encoding="utf-8").lower() for path in DOCS)
    forbidden = ["detection", "prediction", "detecção", "predição", "dataset rotulado", "classe"]
    for term in forbidden:
        assert term not in content


def test_no_private_path_in_public_outputs() -> None:
    for path in [REGISTRY, SCHEMA, *DOCS]:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        assert r"c:\users\gabriela".lower() not in text
        assert "documents\\rev-p" not in text
        assert "documents/rev-p" not in text
