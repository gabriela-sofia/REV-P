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
    common.run_triage(root, force=True)
    return root


def test_sync_writes_v2cu_registry_not_v2co(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_registry_sync(root, force=True)
    assert common.synced_registry_path(root).exists()
    assert not (root / "datasets/external_evidence/sources_registry_v2co.csv").exists()


def test_unknown_license_sets_download_allowed_false(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    assert all(row["download_allowed"] == "false" for row in rows)


def test_redistribution_unconfirmed_sets_public_repo_false(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    assert all(row["public_repo_allowed"] == "false" for row in rows)


def test_source_family_only_rows_have_no_direct_download_url(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    family_rows = [row for row in rows if row["source_family"] in {"COPERNICUS_EMS", "COPERNICUS_GFM"}]
    assert all(row["url"] == "" for row in family_rows)


def test_charter_rows_keep_reference_urls(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    charter = next(row for row in rows if "CHARTER_751" in row["source_id"])
    assert "activation-751" in charter["url"]


def test_public_sync_table_generated(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_registry_sync(root, force=True)
    assert (root / "outputs_public/tables/revp_external_source_registry_public_v2cu.csv").exists()


def test_sync_report_mentions_no_v2co_overwrite(tmp_path: Path) -> None:
    root = prepared(tmp_path)
    common.run_registry_sync(root, force=True)
    assert "nao e sobrescrito" in (root / "outputs_public/execution_reports/revp_external_registry_sync_report_v2cu.md").read_text(encoding="utf-8")


def test_methodological_diff_recorded(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    assert all(row["methodological_diff_from_v2co"] for row in rows)


def test_manual_review_required_is_preserved(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    assert all(row["manual_review_required"] == "true" for row in rows)


def test_sync_rows_have_required_v2co_compatible_fields(tmp_path: Path) -> None:
    rows = common.build_synced_registry(prepared(tmp_path))
    assert set(["source_id", "source_family", "region", "event_name", "url", "download_allowed"]).issubset(rows[0].keys())
