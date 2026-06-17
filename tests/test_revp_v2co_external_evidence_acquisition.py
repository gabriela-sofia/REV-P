from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

from revp_v2cn_to_v2cr_common import SOURCE_FIELDS, build_acquisition, run_acquisition  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def source_row(url: str = "", **overrides: str) -> dict[str, str]:
    row = {
        "source_id": "SRC_TEST",
        "source_family": "COPERNICUS_EMS",
        "region": "Recife",
        "event_name": "event",
        "url": url,
        "expected_file_type": "GEOJSON",
        "license_status": "KNOWN",
        "license_reference": "local fixture",
        "download_allowed": "true",
        "public_repo_allowed": "true",
        "manual_review_required": "true",
        "notes": "fixture",
    }
    row.update(overrides)
    return row


def test_default_offline_downloads_nothing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    rows, code = build_acquisition(root, allow_downloads=False)
    assert code == 0
    assert all(row["acquisition_status"] == "REGISTERED_OFFLINE_ONLY" for row in rows)
    assert not (root / "datasets/external_evidence/raw").exists()


def test_allow_downloads_downloads_only_registered_url(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    fixture = tmp_path / "fixture.geojson"
    fixture.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row(fixture.as_uri())])
    assert run_acquisition(root, allow_downloads=True, force=True) == 0
    rows = read_csv(root / "datasets/external_evidence/metadata/acquisition_manifest_v2co.csv")
    assert rows[0]["acquisition_status"] == "DOWNLOADED_UNVALIDATED"
    assert rows[0]["sha256"]


def test_registry_is_required_for_download_mode(tmp_path: Path) -> None:
    rows, code = build_acquisition(tmp_path / "repo", allow_downloads=True)
    assert rows == []
    assert code == 1


def test_unknown_license_blocks_download(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row("file:///tmp/a.geojson", license_status="UNKNOWN")])
    rows, code = build_acquisition(root, allow_downloads=True)
    assert code == 1
    assert rows[0]["acquisition_status"] == "BLOCKED_LICENSE_UNKNOWN"


def test_download_allowed_false_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row("file:///tmp/a.geojson", download_allowed="false")])
    rows, code = build_acquisition(root, allow_downloads=True)
    assert code == 1
    assert rows[0]["acquisition_status"] == "BLOCKED_DOWNLOAD_NOT_ALLOWED"


def test_raw_file_never_goes_to_outputs_public(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    fixture = tmp_path / "fixture.geojson"
    fixture.write_text("{}", encoding="utf-8")
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row(fixture.as_uri())])
    run_acquisition(root, allow_downloads=True, force=True)
    assert list((root / "datasets/external_evidence/raw").rglob("*.geojson"))
    assert not list((root / "outputs_public").rglob("*.geojson"))


def test_public_repo_false_blocks_download(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row("file:///tmp/a.geojson", public_repo_allowed="false")])
    rows, code = build_acquisition(root, allow_downloads=True)
    assert code == 1
    assert rows[0]["acquisition_status"] == "BLOCKED_PUBLIC_REPO_NOT_ALLOWED"


def test_unallowed_source_family_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row("file:///tmp/a.geojson", source_family="NEWS_BLOG")])
    rows, code = build_acquisition(root, allow_downloads=True)
    assert code == 1
    assert rows[0]["acquisition_status"] == "BLOCKED_SOURCE_FAMILY_NOT_ALLOWED"


def test_no_overwrite_without_force_blocks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    fixture = tmp_path / "fixture.geojson"
    fixture.write_text("{}", encoding="utf-8")
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row(fixture.as_uri())])
    assert run_acquisition(root, allow_downloads=True, force=True) == 0
    assert run_acquisition(root, allow_downloads=True, force=False) == 1


def test_acquisition_report_is_generated(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write_csv(root / "datasets/external_evidence/sources_registry_v2co.csv", SOURCE_FIELDS, [source_row(download_allowed="false")])
    run_acquisition(root, allow_downloads=False, force=True)
    report = root / "outputs_public/execution_reports/revp_external_evidence_acquisition_report_v2co.md"
    assert "nao faz busca livre" in report.read_text(encoding="utf-8")
