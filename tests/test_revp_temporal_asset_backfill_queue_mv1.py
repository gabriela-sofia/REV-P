from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_temporal_asset_backfill_queue_mv1 import CSV_COLUMNS, execute


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_readiness_fixture(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "outputs_public/tables/revp_temporal_asset_readiness_mv1.csv",
        [
            "canonical_patch_id",
            "region",
            "asset_count_total",
            "clean_asset_count",
            "acquisition_dates_count",
            "acquisition_dates",
            "cloud_cover_min",
            "cloud_cover_max",
            "cloud_cover_mean",
            "temporal_bucket_coverage",
            "has_1_date",
            "has_2_dates",
            "has_3plus_dates",
            "eligible_for_temporal_drift",
            "step_a_status",
            "blocking_reason",
        ],
        [
            {
                "canonical_patch_id": "PATCH_P0",
                "region": "Recife",
                "asset_count_total": "2",
                "clean_asset_count": "2",
                "acquisition_dates_count": "2",
                "acquisition_dates": "2022-01-01;2022-01-15",
                "cloud_cover_min": "5",
                "cloud_cover_max": "10",
                "cloud_cover_mean": "7.5",
                "temporal_bucket_coverage": "2_dates",
                "has_1_date": "false",
                "has_2_dates": "true",
                "has_3plus_dates": "false",
                "eligible_for_temporal_drift": "false",
                "step_a_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL",
                "blocking_reason": "FEWER_THAN_3_CLEAN_DATES",
            },
            {
                "canonical_patch_id": "PATCH_P1",
                "region": "Curitiba",
                "asset_count_total": "1",
                "clean_asset_count": "1",
                "acquisition_dates_count": "1",
                "acquisition_dates": "2022-02-01",
                "cloud_cover_min": "missing",
                "cloud_cover_max": "missing",
                "cloud_cover_mean": "missing",
                "temporal_bucket_coverage": "1_date",
                "has_1_date": "true",
                "has_2_dates": "false",
                "has_3plus_dates": "false",
                "eligible_for_temporal_drift": "false",
                "step_a_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL",
                "blocking_reason": "FEWER_THAN_3_CLEAN_DATES",
            },
            {
                "canonical_patch_id": "PATCH_S2",
                "region": "Petropolis",
                "asset_count_total": "1",
                "clean_asset_count": "0",
                "acquisition_dates_count": "0",
                "acquisition_dates": "unknown",
                "cloud_cover_min": "missing",
                "cloud_cover_max": "missing",
                "cloud_cover_mean": "missing",
                "temporal_bucket_coverage": "0_dates",
                "has_1_date": "false",
                "has_2_dates": "false",
                "has_3plus_dates": "false",
                "eligible_for_temporal_drift": "false",
                "step_a_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL",
                "blocking_reason": "MISSING_ACQUISITION_DATE",
            },
            {
                "canonical_patch_id": "PATCH_S1",
                "region": "Petropolis",
                "asset_count_total": "1",
                "clean_asset_count": "0",
                "acquisition_dates_count": "0",
                "acquisition_dates": "unknown",
                "cloud_cover_min": "missing",
                "cloud_cover_max": "missing",
                "cloud_cover_mean": "missing",
                "temporal_bucket_coverage": "0_dates",
                "has_1_date": "false",
                "has_2_dates": "false",
                "has_3plus_dates": "false",
                "eligible_for_temporal_drift": "false",
                "step_a_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL",
                "blocking_reason": "MISSING_ACQUISITION_DATE",
            },
            {
                "canonical_patch_id": "PATCH_UNKNOWN",
                "region": "unknown",
                "asset_count_total": "1",
                "clean_asset_count": "0",
                "acquisition_dates_count": "0",
                "acquisition_dates": "unknown",
                "cloud_cover_min": "missing",
                "cloud_cover_max": "missing",
                "cloud_cover_mean": "missing",
                "temporal_bucket_coverage": "0_dates",
                "has_1_date": "false",
                "has_2_dates": "false",
                "has_3plus_dates": "false",
                "eligible_for_temporal_drift": "false",
                "step_a_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL",
                "blocking_reason": "MISSING_ACQUISITION_DATE;MISSING_REGION",
            },
        ],
    )
    (tmp_path / "outputs_public/metrics").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs_public/metrics/revp_temporal_asset_readiness_mv1.json").write_text(
        json.dumps({"step_a_global_status": "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "outputs_public/execution_reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs_public/execution_reports/revp_temporal_asset_readiness_mv1.md").write_text(
        "# readiness\n",
        encoding="utf-8",
    )


def write_asset_fixture(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "manifests/assets/sentinel_asset_manifest.csv",
        ["canonical_patch_id", "asset_id", "sensor", "platform", "acquisition_datetime", "cloud_cover"],
        [
            {
                "canonical_patch_id": "PATCH_P0",
                "asset_id": "S2_P0_A",
                "sensor": "sentinel-2 optical",
                "platform": "S2",
                "acquisition_datetime": "2022-01-01T00:00:00Z",
                "cloud_cover": "5",
            },
            {
                "canonical_patch_id": "PATCH_P1",
                "asset_id": "DINO_P1_A",
                "sensor": "dinov2_embedding",
                "platform": "DINOv2",
                "acquisition_datetime": "2022-02-01",
                "cloud_cover": "missing",
            },
            {
                "canonical_patch_id": "PATCH_S2",
                "asset_id": "S2_PATCH_A",
                "sensor": "sentinel-2 optical",
                "platform": "S2",
                "acquisition_datetime": "unknown",
                "cloud_cover": "missing",
            },
            {
                "canonical_patch_id": "PATCH_S1",
                "asset_id": "S1_PATCH_A",
                "sensor": "sentinel-1 sar",
                "platform": "S1",
                "acquisition_datetime": "unknown",
                "cloud_cover": "missing",
            },
            {
                "canonical_patch_id": "PATCH_UNKNOWN",
                "asset_id": "ASSET_UNKNOWN_A",
                "sensor": "unknown",
                "platform": "unknown",
                "acquisition_datetime": "unknown",
                "cloud_cover": "missing",
            },
        ],
    )


def test_backfill_queue_outputs_priorities_s2_cloud_rule_and_determinism(tmp_path: Path) -> None:
    write_readiness_fixture(tmp_path)
    write_asset_fixture(tmp_path)

    first_metrics = execute(tmp_path)
    second_metrics = execute(tmp_path)

    table = tmp_path / "outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json"
    report = tmp_path / "outputs_public/execution_reports/revp_temporal_asset_backfill_queue_mv1.md"

    assert table.exists()
    assert metrics_path.exists()
    assert report.exists()
    assert first_metrics == second_metrics

    rows = read_csv(table)
    assert list(rows[0].keys()) == CSV_COLUMNS
    by_patch = {row["canonical_patch_id"]: row for row in rows}
    assert by_patch["PATCH_P0"]["backfill_priority"] == "P0_NEAR_READY"
    assert by_patch["PATCH_P0"]["needed_dates_to_reach_3"] == "1"
    assert by_patch["PATCH_P1"]["backfill_priority"] == "P1_PARTIAL_TEMPORAL"
    assert by_patch["PATCH_P1"]["needed_dates_to_reach_3"] == "2"
    assert by_patch["PATCH_S2"]["missing_numeric_cloud_cover_s2_only"] == "true"
    assert by_patch["PATCH_S1"]["missing_numeric_cloud_cover_s2_only"] == "false"
    assert by_patch["PATCH_UNKNOWN"]["current_clean_dates_count"] == "0"
    assert by_patch["PATCH_UNKNOWN"]["current_acquisition_dates"] == "unknown"

    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert persisted["step_a_next_status"] == first_metrics["step_a_next_status"]
    assert "step_a_next_status" in persisted
    assert persisted["p0_p1_patch_count"] == 2
    assert persisted["guardrails_preserved"]

    output_text = table.read_text(encoding="utf-8") + metrics_path.read_text(encoding="utf-8")
    for forbidden in ["final_label", "formal_negative", "class_0", "operational_ground_truth"]:
        assert forbidden not in output_text
    assert "Guardrails preservados" in report.read_text(encoding="utf-8")
