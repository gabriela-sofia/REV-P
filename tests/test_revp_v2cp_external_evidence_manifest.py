from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cn_to_v2cr_common as common  # noqa: E402


FORBIDDEN = ["GROUND_TRUTH_READY", "LABEL_READY", "TRAINING_READY", "MODEL_VALIDATED", "DETECTION_CONFIRMED", "PREDICTION_VALIDATED"]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def registry_row(**overrides: str) -> dict[str, str]:
    row = {
        "source_id": "SRC_A",
        "source_family": "COPERNICUS_EMS",
        "region": "Recife",
        "event_name": "event",
        "url": "file:///fixture.geojson",
        "expected_file_type": "GEOJSON",
        "license_status": "KNOWN",
        "license_reference": "fixture",
        "download_allowed": "true",
        "public_repo_allowed": "true",
        "manual_review_required": "true",
        "notes": "fixture",
    }
    row.update(overrides)
    return row


def prepare(root: Path, local_rel: str = "datasets/external_evidence/raw/SRC_A/SRC_A.geojson", **overrides: str) -> Path:
    local = root / local_rel
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text('{"type":"FeatureCollection","features":[],"crs":{"type":"name","properties":{"name":"EPSG:4326"}}}', encoding="utf-8")
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", common.SOURCE_FIELDS, [registry_row(**overrides)])
    write_csv(root / "datasets/external_evidence/metadata/acquisition_manifest_v2co.csv", common.ACQUISITION_FIELDS, [{**registry_row(**overrides), "acquisition_mode": "allow-downloads", "acquisition_status": "DOWNLOADED_UNVALIDATED", "retrieved_at_utc": "2026-01-01T00:00:00Z", "local_path": local_rel, "sha256": "", "file_size_bytes": "", "mime_or_extension": ".geojson", "blocking_reason": ""}])
    return local


def test_manifest_calculates_sha256_when_file_exists(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root)
    rows = common.build_manifest(root)
    assert rows[0]["sha256"]
    assert rows[0]["file_exists"] == "true"


def test_missing_file_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", common.SOURCE_FIELDS, [registry_row()])
    write_csv(root / "datasets/external_evidence/metadata/acquisition_manifest_v2co.csv", common.ACQUISITION_FIELDS, [{**registry_row(), "local_path": "datasets/external_evidence/raw/missing.geojson"}])
    rows = common.build_manifest(root)
    assert rows[0]["evidence_status"] == "BLOCKED_NO_LOCAL_FILE"


def test_missing_hash_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    prepare(root)
    monkeypatch.setattr(common, "sha256_file", lambda path: "")
    rows = common.build_manifest(root)
    assert rows[0]["evidence_status"] == "BLOCKED_NO_HASH"


def test_unknown_license_blocks_manifest(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, license_status="UNKNOWN")
    rows = common.build_manifest(root)
    assert rows[0]["evidence_status"] == "BLOCKED_LICENSE_UNKNOWN"


def test_public_manifest_hides_raw_path_when_redistribution_false(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, public_repo_allowed="false")
    rows = common.public_manifest(common.build_manifest(root))
    assert rows[0]["local_path"] == ""


def test_ready_for_geospatial_qa_when_vector_file_is_complete(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root)
    rows = common.build_manifest(root)
    assert rows[0]["evidence_status"] == "READY_FOR_GEOSPATIAL_QA"


def test_manifest_outputs_do_not_contain_forbidden_status_tokens(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root)
    common.run_manifest(root, force=True)
    text = (root / "datasets/external_evidence/external_evidence_manifest_v2cp.csv").read_text(encoding="utf-8")
    assert all(token not in text for token in FORBIDDEN)


def test_license_hash_rollup_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root)
    common.run_manifest(root, force=True)
    rows = read_csv(root / "outputs_public/logs_summary/revp_external_evidence_license_hash_rollup_v2cp.csv")
    assert rows[0]["sha256_available"] == "true"


def test_human_review_required_is_preserved(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root, manual_review_required="true")
    rows = common.build_manifest(root)
    assert rows[0]["human_review_required"] == "true"


def test_manifest_report_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    prepare(root)
    common.run_manifest(root, force=True)
    assert "manifesto" in (root / "outputs_public/execution_reports/revp_external_evidence_manifest_report_v2cp.md").read_text(encoding="utf-8")
