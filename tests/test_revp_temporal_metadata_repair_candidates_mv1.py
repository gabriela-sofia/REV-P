from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_temporal_metadata_repair_candidates_mv1 import CSV_COLUMNS, execute


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_prior_outputs(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv",
        [
            "canonical_patch_id",
            "region",
            "current_clean_dates_count",
            "current_acquisition_dates",
            "needed_dates_to_reach_3",
            "asset_count_total",
            "clean_asset_count",
            "sentinel_2_asset_count",
            "sentinel_1_asset_count",
            "dinov2_asset_count",
            "unknown_asset_count",
            "requires_s2_cloud_cover_review",
            "missing_acquisition_date",
            "missing_region",
            "missing_patch_asset_link",
            "missing_numeric_cloud_cover_s2_only",
            "backfill_priority",
            "backfill_action",
            "blocking_reason",
            "eligible_after_backfill_if_completed",
        ],
        [
            {
                "canonical_patch_id": "PATCH_DATE",
                "region": "Recife",
                "current_clean_dates_count": "0",
                "current_acquisition_dates": "unknown",
                "needed_dates_to_reach_3": "3",
                "asset_count_total": "1",
                "clean_asset_count": "0",
                "sentinel_2_asset_count": "1",
                "sentinel_1_asset_count": "0",
                "dinov2_asset_count": "0",
                "unknown_asset_count": "0",
                "requires_s2_cloud_cover_review": "true",
                "missing_acquisition_date": "true",
                "missing_region": "false",
                "missing_patch_asset_link": "false",
                "missing_numeric_cloud_cover_s2_only": "true",
                "backfill_priority": "P2_HAS_ASSETS_NO_CLEAN_DATES",
                "backfill_action": "RECOVER_ACQUISITION_DATE_FROM_MANIFEST_OR_FILENAME_ONLY_IF_UNAMBIGUOUS",
                "blocking_reason": "MISSING_ACQUISITION_DATE;MISSING_NUMERIC_CLOUD_COVER_S2_ONLY",
                "eligible_after_backfill_if_completed": "true",
            },
            {
                "canonical_patch_id": "PATCH_UNKNOWN",
                "region": "unknown",
                "current_clean_dates_count": "0",
                "current_acquisition_dates": "unknown",
                "needed_dates_to_reach_3": "3",
                "asset_count_total": "1",
                "clean_asset_count": "0",
                "sentinel_2_asset_count": "0",
                "sentinel_1_asset_count": "0",
                "dinov2_asset_count": "0",
                "unknown_asset_count": "1",
                "requires_s2_cloud_cover_review": "false",
                "missing_acquisition_date": "true",
                "missing_region": "true",
                "missing_patch_asset_link": "false",
                "missing_numeric_cloud_cover_s2_only": "false",
                "backfill_priority": "P3_METADATA_ONLY_OR_UNKNOWN",
                "backfill_action": "CLASSIFY_ASSET_TYPE_BEFORE_TEMPORAL_USE",
                "blocking_reason": "MISSING_ACQUISITION_DATE;MISSING_REGION;UNKNOWN_ASSET_TYPE",
                "eligible_after_backfill_if_completed": "false",
            },
            {
                "canonical_patch_id": "PATCH_CONFLICT",
                "region": "unknown",
                "current_clean_dates_count": "0",
                "current_acquisition_dates": "unknown",
                "needed_dates_to_reach_3": "3",
                "asset_count_total": "2",
                "clean_asset_count": "0",
                "sentinel_2_asset_count": "0",
                "sentinel_1_asset_count": "0",
                "dinov2_asset_count": "0",
                "unknown_asset_count": "2",
                "requires_s2_cloud_cover_review": "false",
                "missing_acquisition_date": "true",
                "missing_region": "true",
                "missing_patch_asset_link": "false",
                "missing_numeric_cloud_cover_s2_only": "false",
                "backfill_priority": "P3_METADATA_ONLY_OR_UNKNOWN",
                "backfill_action": "CLASSIFY_ASSET_TYPE_BEFORE_TEMPORAL_USE",
                "blocking_reason": "MISSING_ACQUISITION_DATE;MISSING_REGION;UNKNOWN_ASSET_TYPE",
                "eligible_after_backfill_if_completed": "false",
            },
            {
                "canonical_patch_id": "PATCH_S1",
                "region": "Curitiba",
                "current_clean_dates_count": "0",
                "current_acquisition_dates": "unknown",
                "needed_dates_to_reach_3": "3",
                "asset_count_total": "1",
                "clean_asset_count": "0",
                "sentinel_2_asset_count": "0",
                "sentinel_1_asset_count": "1",
                "dinov2_asset_count": "0",
                "unknown_asset_count": "0",
                "requires_s2_cloud_cover_review": "false",
                "missing_acquisition_date": "true",
                "missing_region": "false",
                "missing_patch_asset_link": "false",
                "missing_numeric_cloud_cover_s2_only": "false",
                "backfill_priority": "P2_HAS_ASSETS_NO_CLEAN_DATES",
                "backfill_action": "RECOVER_ACQUISITION_DATE_FROM_MANIFEST_OR_FILENAME_ONLY_IF_UNAMBIGUOUS",
                "blocking_reason": "MISSING_ACQUISITION_DATE",
                "eligible_after_backfill_if_completed": "true",
            },
        ],
    )
    (tmp_path / "outputs_public/metrics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json").write_text(
        json.dumps({"step_a_next_status": "STEP_A_METADATA_REPAIR_REQUIRED"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "outputs_public/execution_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs_public/execution_reports/revp_temporal_asset_backfill_queue_mv1.md").write_text(
        "# backfill\n",
        encoding="utf-8",
    )
    write_csv(
        tmp_path / "outputs_public/tables/revp_temporal_asset_readiness_mv1.csv",
        ["canonical_patch_id", "region", "asset_count_total", "clean_asset_count", "acquisition_dates_count"],
        [{"canonical_patch_id": "PATCH_DATE", "region": "Recife", "asset_count_total": "1", "clean_asset_count": "0", "acquisition_dates_count": "0"}],
    )
    (tmp_path / "outputs_public/metrics/revp_temporal_asset_readiness_mv1.json").write_text(
        json.dumps({"step_a_global_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL"}) + "\n",
        encoding="utf-8",
    )


def test_repair_candidates_outputs_rules_conflicts_and_determinism(tmp_path: Path) -> None:
    write_prior_outputs(tmp_path)
    manifest = tmp_path / "manifests/temporal_repair/source_manifest.csv"
    write_csv(
        manifest,
        ["canonical_patch_id", "asset_id", "region", "sensor", "asset_path_reference", "acquisition_datetime", "cloud_cover"],
        [
            {
                "canonical_patch_id": "PATCH_DATE",
                "asset_id": "S2_A",
                "region": "Recife",
                "sensor": "sentinel-2 optical",
                "asset_path_reference": "S2A_MSIL2A_20220304T123456_patch.tif",
                "acquisition_datetime": "missing",
                "cloud_cover": "12.5",
            },
            {
                "canonical_patch_id": "PATCH_UNKNOWN",
                "asset_id": "U_A",
                "region": "unknown",
                "sensor": "unknown",
                "asset_path_reference": "ambiguous_20220101_20220202.tif",
                "acquisition_datetime": "unknown",
                "cloud_cover": "missing",
            },
            {
                "canonical_patch_id": "PATCH_CONFLICT",
                "asset_id": "C_A",
                "region": "Recife",
                "sensor": "unknown",
                "asset_path_reference": "asset.tif",
                "acquisition_datetime": "unknown",
                "cloud_cover": "missing",
            },
            {
                "canonical_patch_id": "PATCH_CONFLICT",
                "asset_id": "C_B",
                "region": "Curitiba",
                "sensor": "unknown",
                "asset_path_reference": "asset2.tif",
                "acquisition_datetime": "unknown",
                "cloud_cover": "missing",
            },
            {
                "canonical_patch_id": "PATCH_S1",
                "asset_id": "S1_A",
                "region": "Curitiba",
                "sensor": "sentinel-1 sar",
                "asset_path_reference": "S1_20220405_scene.tif",
                "acquisition_datetime": "unknown",
                "cloud_cover": "99",
            },
        ],
    )
    before = manifest.read_text(encoding="utf-8")

    first_metrics = execute(tmp_path)
    second_metrics = execute(tmp_path)

    table = tmp_path / "outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_metadata_repair_candidates_mv1.json"
    report = tmp_path / "outputs_public/execution_reports/revp_temporal_metadata_repair_candidates_mv1.md"

    assert table.exists()
    assert metrics_path.exists()
    assert report.exists()
    assert first_metrics == second_metrics
    assert manifest.read_text(encoding="utf-8") == before

    rows = read_csv(table)
    assert list(rows[0].keys()) == CSV_COLUMNS
    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "step_a_repair_status" in persisted

    assert any(
        row["canonical_patch_id"] == "PATCH_DATE"
        and row["repair_field"] == "acquisition_date"
        and row["repair_rule"] == "UNAMBIGUOUS_DATE_FROM_FILENAME"
        and row["normalized_value"] == "2022-03-04"
        and row["is_applicable"] == "true"
        for row in rows
    )
    assert not any(row["canonical_patch_id"] == "PATCH_UNKNOWN" and row["normalized_value"] in {"unknown", "missing"} and row["is_applicable"] == "true" for row in rows)
    assert any(row["canonical_patch_id"] == "PATCH_CONFLICT" and row["rejection_reason"] == "AMBIGUOUS_CONFLICT" for row in rows)
    assert any(row["canonical_patch_id"] == "PATCH_DATE" and row["repair_field"] == "cloud_cover" and row["is_applicable"] == "true" for row in rows)
    assert any(row["canonical_patch_id"] == "PATCH_S1" and row["repair_field"] == "cloud_cover" and row["is_applicable"] == "false" for row in rows)
    assert not any(row["canonical_patch_id"] == "PATCH_UNKNOWN" and row["source_field"] == "asset_path_reference" and row["repair_rule"] == "UNAMBIGUOUS_DATE_FROM_FILENAME" for row in rows)

    output_text = table.read_text(encoding="utf-8") + metrics_path.read_text(encoding="utf-8")
    for forbidden in ["final_label", "formal_negative", "class_0", "operational_ground_truth"]:
        assert forbidden not in output_text
    assert "Guardrails preservados" in report.read_text(encoding="utf-8")
