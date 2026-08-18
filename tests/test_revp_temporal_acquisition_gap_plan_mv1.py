from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ground_truth"))

from revp_temporal_acquisition_gap_plan_mv1 import GAP_PLAN_COLUMNS, REQUIREMENTS_COLUMNS, execute


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_normalized_inputs(tmp_path: Path) -> None:
    asset_columns = [
        "normalized_asset_id",
        "canonical_patch_id",
        "asset_id",
        "region",
        "asset_type",
        "acquisition_date",
        "cloud_cover",
        "cloud_cover_required",
        "cloud_cover_available",
        "patch_asset_link_status",
        "temporal_usability_status",
        "is_clean_temporal_asset",
        "source_files",
        "applied_repair_fields",
        "applied_repair_rules",
        "applied_repair_confidences",
        "rejected_repair_count",
        "lineage_status",
        "guardrail_status",
    ]
    write_csv(
        tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv",
        asset_columns,
        [
            {
                "normalized_asset_id": "PATCH_1::A1",
                "canonical_patch_id": "PATCH_1",
                "asset_id": "A1",
                "region": "Recife",
                "asset_type": "sentinel_2_optical",
                "acquisition_date": "2022-01-01",
                "cloud_cover": "5",
                "cloud_cover_required": "true",
                "cloud_cover_available": "true",
                "patch_asset_link_status": "LINKED",
                "temporal_usability_status": "CLEAN_TEMPORAL_ASSET",
                "is_clean_temporal_asset": "true",
                "source_files": "fixture",
                "applied_repair_fields": "all",
                "applied_repair_rules": "fixture",
                "applied_repair_confidences": "HIGH_EXPLICIT_FIELD",
                "rejected_repair_count": "0",
                "lineage_status": "APPLIED_DETERMINISTIC_SHADOW",
                "guardrail_status": "GUARDRAILS_PRESERVED",
            },
            {
                "normalized_asset_id": "PATCH_2::A1",
                "canonical_patch_id": "PATCH_2",
                "asset_id": "A1",
                "region": "Curitiba",
                "asset_type": "sentinel_2_optical",
                "acquisition_date": "2022-01-01",
                "cloud_cover": "5",
                "cloud_cover_required": "true",
                "cloud_cover_available": "true",
                "patch_asset_link_status": "LINKED",
                "temporal_usability_status": "CLEAN_TEMPORAL_ASSET",
                "is_clean_temporal_asset": "true",
                "source_files": "fixture",
                "applied_repair_fields": "all",
                "applied_repair_rules": "fixture",
                "applied_repair_confidences": "HIGH_EXPLICIT_FIELD",
                "rejected_repair_count": "0",
                "lineage_status": "APPLIED_DETERMINISTIC_SHADOW",
                "guardrail_status": "GUARDRAILS_PRESERVED",
            },
            {
                "normalized_asset_id": "PATCH_2::A2",
                "canonical_patch_id": "PATCH_2",
                "asset_id": "A2",
                "region": "Curitiba",
                "asset_type": "sentinel_2_optical",
                "acquisition_date": "2022-01-15",
                "cloud_cover": "7",
                "cloud_cover_required": "true",
                "cloud_cover_available": "true",
                "patch_asset_link_status": "LINKED",
                "temporal_usability_status": "CLEAN_TEMPORAL_ASSET",
                "is_clean_temporal_asset": "true",
                "source_files": "fixture",
                "applied_repair_fields": "all",
                "applied_repair_rules": "fixture",
                "applied_repair_confidences": "HIGH_EXPLICIT_FIELD",
                "rejected_repair_count": "0",
                "lineage_status": "APPLIED_DETERMINISTIC_SHADOW",
                "guardrail_status": "GUARDRAILS_PRESERVED",
            },
            {
                "normalized_asset_id": "PATCH_3::A1",
                "canonical_patch_id": "PATCH_3",
                "asset_id": "A1",
                "region": "Petropolis",
                "asset_type": "sentinel_1_sar",
                "acquisition_date": "2022-01-01",
                "cloud_cover": "missing",
                "cloud_cover_required": "false",
                "cloud_cover_available": "false",
                "patch_asset_link_status": "LINKED",
                "temporal_usability_status": "CLEAN_TEMPORAL_ASSET",
                "is_clean_temporal_asset": "true",
                "source_files": "fixture",
                "applied_repair_fields": "all",
                "applied_repair_rules": "fixture",
                "applied_repair_confidences": "HIGH_EXPLICIT_FIELD",
                "rejected_repair_count": "0",
                "lineage_status": "APPLIED_DETERMINISTIC_SHADOW",
                "guardrail_status": "GUARDRAILS_PRESERVED",
            },
        ],
    )
    coverage_columns = [
        "canonical_patch_id",
        "region",
        "normalized_asset_count",
        "clean_temporal_asset_count",
        "unique_acquisition_dates_count",
        "unique_acquisition_dates",
        "s2_clean_asset_count",
        "s1_clean_asset_count",
        "dinov2_clean_asset_count",
        "has_1_date",
        "has_2_dates",
        "has_3plus_dates",
        "eligible_for_temporal_embedding_drift",
        "temporal_coverage_status",
        "remaining_blockers",
        "next_required_action",
    ]
    write_csv(
        tmp_path / "outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv",
        coverage_columns,
        [
            {
                "canonical_patch_id": "PATCH_1",
                "region": "Recife",
                "normalized_asset_count": "1",
                "clean_temporal_asset_count": "1",
                "unique_acquisition_dates_count": "1",
                "unique_acquisition_dates": "2022-01-01",
                "s2_clean_asset_count": "1",
                "s1_clean_asset_count": "0",
                "dinov2_clean_asset_count": "0",
                "has_1_date": "true",
                "has_2_dates": "false",
                "has_3plus_dates": "false",
                "eligible_for_temporal_embedding_drift": "false",
                "temporal_coverage_status": "PARTIAL_NEEDS_2_DATES",
                "remaining_blockers": "NONE",
                "next_required_action": "ADD_TWO_CLEAN_TEMPORAL_ACQUISITIONS",
            },
            {
                "canonical_patch_id": "PATCH_2",
                "region": "Curitiba",
                "normalized_asset_count": "2",
                "clean_temporal_asset_count": "2",
                "unique_acquisition_dates_count": "2",
                "unique_acquisition_dates": "2022-01-01;2022-01-15",
                "s2_clean_asset_count": "2",
                "s1_clean_asset_count": "0",
                "dinov2_clean_asset_count": "0",
                "has_1_date": "false",
                "has_2_dates": "true",
                "has_3plus_dates": "false",
                "eligible_for_temporal_embedding_drift": "false",
                "temporal_coverage_status": "NEAR_READY_NEEDS_1_DATE",
                "remaining_blockers": "NONE",
                "next_required_action": "ADD_ONE_CLEAN_TEMPORAL_ACQUISITION",
            },
            {
                "canonical_patch_id": "PATCH_3",
                "region": "Petropolis",
                "normalized_asset_count": "3",
                "clean_temporal_asset_count": "3",
                "unique_acquisition_dates_count": "3",
                "unique_acquisition_dates": "2022-01-01;2022-01-15;2022-02-01",
                "s2_clean_asset_count": "0",
                "s1_clean_asset_count": "3",
                "dinov2_clean_asset_count": "0",
                "has_1_date": "false",
                "has_2_dates": "false",
                "has_3plus_dates": "true",
                "eligible_for_temporal_embedding_drift": "true",
                "temporal_coverage_status": "READY_FOR_TEMPORAL_DRIFT",
                "remaining_blockers": "NONE",
                "next_required_action": "NO_DRIFT_ACTION_IN_THIS_STEP",
            },
        ],
    )
    for rel in [
        "outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    report = tmp_path / "outputs_public/execution_reports/revp_normalized_temporal_asset_manifest_mv1.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# normalized\n", encoding="utf-8")


def test_acquisition_gap_plan_outputs_slots_no_dates_and_determinism(tmp_path: Path) -> None:
    write_normalized_inputs(tmp_path)
    original_asset_manifest = (tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv").read_text(encoding="utf-8")

    first = execute(tmp_path)
    second = execute(tmp_path)

    plan_path = tmp_path / "outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv"
    requirements_path = tmp_path / "outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_acquisition_gap_plan_mv1.json"
    report_path = tmp_path / "outputs_public/execution_reports/revp_temporal_acquisition_gap_plan_mv1.md"

    assert first == second
    assert plan_path.exists()
    assert requirements_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert (tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv").read_text(encoding="utf-8") == original_asset_manifest

    plan_rows = read_csv(plan_path)
    req_rows = read_csv(requirements_path)
    assert list(plan_rows[0].keys()) == GAP_PLAN_COLUMNS
    assert list(req_rows[0].keys()) == REQUIREMENTS_COLUMNS
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "step_a_acquisition_plan_status" in metrics

    by_patch = {row["canonical_patch_id"]: row for row in plan_rows}
    assert by_patch["PATCH_1"]["missing_dates_to_3"] == "2"
    assert by_patch["PATCH_1"]["acquisition_priority"] == "P1_NEEDS_2_DATES"
    assert by_patch["PATCH_2"]["missing_dates_to_3"] == "1"
    assert by_patch["PATCH_2"]["acquisition_priority"] == "P0_NEEDS_1_DATE"
    assert by_patch["PATCH_3"]["missing_dates_to_3"] == "0"
    assert by_patch["PATCH_3"]["acquisition_gap_status"] == "READY_NO_GAP"

    assert sum(1 for row in req_rows if row["canonical_patch_id"] == "PATCH_1") == 2
    assert sum(1 for row in req_rows if row["canonical_patch_id"] == "PATCH_2") == 1
    assert sum(1 for row in req_rows if row["canonical_patch_id"] == "PATCH_3") == 0
    assert all(row["required_new_date_status"] == "UNSPECIFIED_REAL_ACQUISITION_NEEDED" for row in req_rows)
    assert all(row["requires_no_label_creation"] == "true" for row in req_rows)
    assert all(row["requires_no_event_assumption"] == "true" for row in req_rows)

    output_text = plan_path.read_text(encoding="utf-8") + requirements_path.read_text(encoding="utf-8") + metrics_path.read_text(encoding="utf-8")
    for forbidden in ["final_label", "formal_negative", "class_0", "operational_ground_truth"]:
        assert forbidden not in output_text
    assert "Guardrails preservados" in report_path.read_text(encoding="utf-8")
