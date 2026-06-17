from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "multimodal"))

import revp_v2cs_to_v2cw_common as common  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def prepared(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    common.run_seeding(root, force=True)
    return root


def test_unknown_licenses_block_raw_downloads(tmp_path: Path) -> None:
    rows = common.build_triage(prepared(tmp_path))
    assert all(row["raw_download_allowed"] == "false" for row in rows)


def test_public_raw_outputs_blocked(tmp_path: Path) -> None:
    rows = common.build_triage(prepared(tmp_path))
    assert all(row["raw_public_output_allowed"] == "false" for row in rows)


def test_documentary_source_metadata_public_allowed(tmp_path: Path) -> None:
    rows = common.build_triage(prepared(tmp_path))
    charter = next(row for row in rows if "CHARTER_751" in row["source_id"])
    assert charter["metadata_public_allowed"] == "true"


def test_manual_access_required_for_charter_pages(tmp_path: Path) -> None:
    rows = common.build_triage(prepared(tmp_path))
    charter_rows = [row for row in rows if "CHARTER" in row["source_id"]]
    assert all(row["license_triage_status"] == "MANUAL_ACCESS_REQUIRED" for row in charter_rows)


def test_product_discovery_required_for_source_family(tmp_path: Path) -> None:
    rows = common.build_triage(prepared(tmp_path))
    gfm = next(row for row in rows if row["source_family"] == "COPERNICUS_GFM")
    assert gfm["product_discovery_required"] == "true"


def test_license_guardrail_rollup_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_triage(root, force=True)
    rows = read_csv(root / "outputs_public/logs_summary/revp_source_license_guardrails_v2ct.csv")
    assert any(row["guardrail"] == "unknown_license_blocks_public_download" for row in rows)


def test_triage_report_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_triage(root, force=True)
    assert "Downloads brutos bloqueados" in (root / "outputs_public/execution_reports/revp_source_license_triage_report_v2ct.md").read_text(encoding="utf-8")


def test_triage_has_required_columns(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_triage(root, force=True)
    row = read_csv(common.triage_path(root))[0]
    assert set(common.TRIAGE_FIELDS).issubset(row.keys())


def test_drm_license_reference_records_unspecified_license(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    drm = next(row for row in common.seed_rows() if row["source_family"] == "DRM_RJ")
    assert "licenca nao especificada" in drm["license_reference"]


def test_no_triage_row_allows_label_claim(tmp_path: Path) -> None:
    rows = common.build_triage(prepared(tmp_path))
    assert all("label_binario" in row["forbidden_claim"] for row in rows)
