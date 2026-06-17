from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cl_observed_geometry_validator import build_validation, main  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    write_csv(root / "outputs_public/tables/revp_digitization_task_queue_v2ck.csv",
              ["task_id", "candidate_id", "region", "source_name", "source_reference", "input_evidence_type"],
              [{"task_id": "T", "candidate_id": "A", "region": "Recife", "source_name": "src",
                "source_reference": "ref", "input_evidence_type": "GEOMETRIA_CANDIDATA"}])
    return root


def test_absence_of_vector_geometry_blocks(tmp_path: Path) -> None:
    rows = build_validation(repo(tmp_path), deps_available=False)
    assert rows[0]["validation_status"] == "NO_OBSERVED_VECTOR_GEOMETRY"


def test_missing_crs_blocks(tmp_path: Path) -> None:
    root = repo(tmp_path)
    g = root / "datasets/observed_geometry/A.geojson"
    g.parent.mkdir(parents=True)
    g.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    rows = build_validation(root, deps_available=True)
    assert rows[0]["validation_status"] == "BLOCKED_MISSING_CRS"


def test_missing_provenance_blocks(tmp_path: Path) -> None:
    root = repo(tmp_path)
    g = root / "datasets/observed_geometry/A.geojson"
    g.parent.mkdir(parents=True)
    g.write_text('{"type":"FeatureCollection","features":[],"crs":{"properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    (root / "datasets/observed_geometry/A.geojson.crs").write_text("EPSG:4326", encoding="utf-8")
    rows = build_validation(root, deps_available=True)
    assert rows[0]["validation_status"] == "BLOCKED_MISSING_PROVENANCE"


def test_missing_hash_is_false_when_no_file(tmp_path: Path) -> None:
    rows = build_validation(repo(tmp_path), deps_available=False)
    assert rows[0]["hash_available"] == "false"


def test_dependency_unavailable_blocks(tmp_path: Path) -> None:
    root = repo(tmp_path)
    g = root / "datasets/observed_geometry/A.geojson"
    g.parent.mkdir(parents=True)
    g.write_text('{"type":"FeatureCollection","features":[],"crs":{"properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    (root / "datasets/observed_geometry/A.geojson.crs").write_text("EPSG:4326", encoding="utf-8")
    (root / "datasets/observed_geometry/A.geojson.md").write_text("fonte", encoding="utf-8")
    rows = build_validation(root, deps_available=False)
    assert rows[0]["validation_status"] == "BLOCKED_VALIDATOR_DEPENDENCY_UNAVAILABLE"


def test_validated_geometry_is_candidate_only(tmp_path: Path) -> None:
    root = repo(tmp_path)
    g = root / "datasets/observed_geometry/A.geojson"
    g.parent.mkdir(parents=True)
    g.write_text('{"type":"FeatureCollection","features":[],"crs":{"properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    (root / "datasets/observed_geometry/A.geojson.crs").write_text("EPSG:4326", encoding="utf-8")
    (root / "datasets/observed_geometry/A.geojson.md").write_text("fonte", encoding="utf-8")
    rows = build_validation(root, deps_available=True)
    assert rows[0]["validation_status"] == "VALIDATED_OBSERVED_GEOMETRY_CANDIDATE"
    assert "ground_truth_operacional" in rows[0]["forbidden_claim"]


def test_outputs_and_guardrails_written(tmp_path: Path) -> None:
    root = repo(tmp_path)
    assert main(["--repo-root", str(root), "--force"]) == 0
    assert (root / "outputs_public/tables/revp_observed_geometry_validation_v2cl.csv").exists()
    assert (root / "outputs_public/logs_summary/revp_observed_geometry_validation_guardrails_v2cl.csv").exists()


def test_report_is_conservative(tmp_path: Path) -> None:
    root = repo(tmp_path)
    main(["--repo-root", str(root), "--force"])
    text = (root / "outputs_public/execution_reports/revp_observed_geometry_validation_report_v2cl.md").read_text()
    assert "nunca" in text
    assert "ground truth operacional" in text

