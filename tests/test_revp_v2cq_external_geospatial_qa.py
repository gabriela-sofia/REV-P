from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cn_to_v2cr_common as common  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def manifest(root: Path, local_rel: str) -> None:
    write_csv(root / "datasets/external_evidence/external_evidence_manifest_v2cp.csv", common.MANIFEST_FIELDS, [{
        "evidence_id": "EVID_A",
        "source_id": "SRC_A",
        "source_family": "COPERNICUS_EMS",
        "region": "Recife",
        "event_name": "event",
        "local_path": local_rel,
        "public_path_allowed": "true",
        "file_exists": "true",
        "file_size_bytes": "1",
        "sha256": "hash",
        "mime_or_extension": Path(local_rel).suffix,
        "retrieval_mode": "fixture",
        "retrieved_at_utc": "",
        "source_url": "",
        "license_status": "KNOWN",
        "license_reference": "",
        "redistribution_allowed": "true",
        "geospatial_validation_required": "true",
        "human_review_required": "true",
        "evidence_status": "READY_FOR_GEOSPATIAL_QA",
        "blocking_reason": "",
    }])


@pytest.mark.parametrize("suffix", [".png", ".jpg", ".pdf", ".txt", ".md"])
def test_non_geospatial_formats_do_not_validate(tmp_path: Path, suffix: str) -> None:
    root = tmp_path / "repo"
    local = root / f"datasets/external_evidence/raw/SRC_A/file{suffix}"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("not geometry", encoding="utf-8")
    manifest(root, common.rel(root, local))
    row = common.build_geospatial_qa(root, deps_available=True)[0]
    assert row["qa_status"] == "BLOCKED_NOT_GEOSPATIAL"
    assert row["tp2_candidate_allowed"] == "false"


def test_geojson_without_crs_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.geojson"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    manifest(root, common.rel(root, local))
    assert common.build_geospatial_qa(root, deps_available=True)[0]["qa_status"] == "BLOCKED_MISSING_CRS"


def test_wkt_csv_without_crs_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.wkt.csv"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text("wkt\nPOINT (0 0)\n", encoding="utf-8")
    manifest(root, common.rel(root, local))
    assert common.build_geospatial_qa(root, deps_available=True)[0]["qa_status"] == "BLOCKED_MISSING_CRS"


def test_dependency_unavailable_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.geojson"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"type":"FeatureCollection","bbox":[0,0,1,1],"features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]},"properties":{}}],"crs":{"type":"name","properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    manifest(root, common.rel(root, local))
    assert common.build_geospatial_qa(root, deps_available=False)[0]["qa_status"] == "BLOCKED_DEPENDENCY_UNAVAILABLE"


def test_validated_evidence_remains_candidate_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.geojson"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"type":"FeatureCollection","bbox":[0,0,1,1],"features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]},"properties":{}}],"crs":{"type":"name","properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    manifest(root, common.rel(root, local))
    row = common.build_geospatial_qa(root, deps_available=True)[0]
    assert row["qa_status"] == "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE"
    assert "ground_truth_operacional" in row["forbidden_claim"]


def test_geotiff_is_context_only(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.tif"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"TIFF")
    local.with_suffix(".crs.txt").write_text("EPSG:4326", encoding="utf-8")
    manifest(root, common.rel(root, local))
    assert common.build_geospatial_qa(root, deps_available=True)[0]["qa_status"] == "GEOSPATIAL_CONTEXT_ONLY"


def test_invalid_geojson_blocks_when_crs_exists(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.geojson"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"type":"FeatureCollection","features":[],"crs":{"type":"name","properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    manifest(root, common.rel(root, local))
    assert common.build_geospatial_qa(root, deps_available=True)[0]["qa_status"] == "BLOCKED_INVALID_GEOMETRY"


def test_json_geojson_with_crs_can_validate(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    local = root / "datasets/external_evidence/raw/SRC_A/file.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"type":"Feature","geometry":{"type":"Point","coordinates":[0,0]},"properties":{},"crs":{"type":"name","properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    manifest(root, common.rel(root, local))
    assert common.build_geospatial_qa(root, deps_available=True)[0]["qa_status"] == "VALIDATED_EXTERNAL_GEOMETRY_CANDIDATE"


def test_missing_local_file_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest(root, "datasets/external_evidence/raw/SRC_A/missing.geojson")
    assert common.build_geospatial_qa(root, deps_available=True)[0]["qa_status"] == "BLOCKED_NOT_GEOSPATIAL"


def test_qa_report_and_guardrails_are_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    manifest(root, "datasets/external_evidence/raw/SRC_A/missing.geojson")
    common.run_geospatial_qa(root, force=True)
    assert (root / "outputs_public/execution_reports/revp_external_geospatial_qa_report_v2cq.md").exists()
    assert (root / "outputs_public/logs_summary/revp_external_geospatial_qa_guardrails_v2cq.csv").exists()
