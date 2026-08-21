from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_temporal_backfill_request_manifest_mv1 import BATCH_COLUMNS, REQUEST_COLUMNS, execute


PLAN_COLUMNS = [
    "canonical_patch_id",
    "region",
    "current_clean_dates_count",
    "current_clean_dates",
    "missing_dates_to_3",
    "current_clean_temporal_asset_count",
    "s2_clean_asset_count",
    "s1_clean_asset_count",
    "dinov2_clean_asset_count",
    "eligible_for_temporal_embedding_drift_now",
    "acquisition_gap_status",
    "acquisition_priority",
    "is_actionable_for_temporal_backfill",
    "requires_metadata_repair_before_acquisition",
    "remaining_blockers",
    "minimum_new_acquisitions_required",
    "recommended_sensor_family",
    "recommended_processing_contract",
    "next_required_action",
    "guardrail_status",
]

REQ_INPUT_COLUMNS = [
    "canonical_patch_id",
    "region",
    "required_slot_index",
    "existing_dates",
    "required_new_date_status",
    "required_sensor_family",
    "required_minimum_metadata",
    "requires_same_patch_geometry",
    "requires_same_preprocessing_contract",
    "requires_dinov2_frozen_inference_later",
    "requires_cloud_cover_if_s2",
    "requires_no_label_creation",
    "requires_no_event_assumption",
    "requirement_status",
    "blocking_reason",
]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def actionable_plan_row(index: int) -> dict[str, str]:
    patch_id = f"PATCH_{index:02d}"
    return {
        "canonical_patch_id": patch_id,
        "region": "Recife" if index % 2 else "Curitiba",
        "current_clean_dates_count": "1",
        "current_clean_dates": "2022-01-01",
        "missing_dates_to_3": "2",
        "current_clean_temporal_asset_count": "1",
        "s2_clean_asset_count": "1",
        "s1_clean_asset_count": "0",
        "dinov2_clean_asset_count": "0",
        "eligible_for_temporal_embedding_drift_now": "false",
        "acquisition_gap_status": "NEEDS_2_MORE_DATES",
        "acquisition_priority": "P1_NEEDS_2_DATES",
        "is_actionable_for_temporal_backfill": "true",
        "requires_metadata_repair_before_acquisition": "false",
        "remaining_blockers": "NONE",
        "minimum_new_acquisitions_required": "2",
        "recommended_sensor_family": "same_family_as_existing_clean_asset",
        "recommended_processing_contract": "SAME_PATCH_GEOMETRY_SAME_PREPROCESSING_DINOV2_FROZEN_LATER",
        "next_required_action": "PLAN_TWO_REAL_TEMPORAL_ACQUISITION_SLOTS",
        "guardrail_status": "GUARDRAILS_PRESERVED_ACQUISITION_PLAN_ONLY",
    }


def actionable_requirement_rows(plan_row: dict[str, str]) -> list[dict[str, str]]:
    rows = []
    for slot in ["1", "2"]:
        rows.append(
            {
                "canonical_patch_id": plan_row["canonical_patch_id"],
                "region": plan_row["region"],
                "required_slot_index": slot,
                "existing_dates": "2022-01-01",
                "required_new_date_status": "UNSPECIFIED_REAL_ACQUISITION_NEEDED",
                "required_sensor_family": "same_family_as_existing_clean_asset",
                "required_minimum_metadata": "canonical_patch_id|region|asset_type|acquisition_date|patch_asset_link|s2_cloud_cover_when_required",
                "requires_same_patch_geometry": "true",
                "requires_same_preprocessing_contract": "true",
                "requires_dinov2_frozen_inference_later": "true",
                "requires_cloud_cover_if_s2": "true",
                "requires_no_label_creation": "true",
                "requires_no_event_assumption": "true",
                "requirement_status": "ACTIONABLE_REQUIREMENT",
                "blocking_reason": "NONE",
            }
        )
    return rows


def write_inputs(tmp_path: Path) -> tuple[str, str]:
    plan_rows = [actionable_plan_row(index) for index in range(1, 31)]
    blocked_patch = {
        **actionable_plan_row(31),
        "canonical_patch_id": "PATCH_BLOCKED",
        "current_clean_dates_count": "0",
        "current_clean_dates": "unknown",
        "missing_dates_to_3": "3",
        "acquisition_gap_status": "BLOCKED_BY_METADATA_REPAIR",
        "acquisition_priority": "P3_REPAIR_METADATA_FIRST",
        "is_actionable_for_temporal_backfill": "false",
        "requires_metadata_repair_before_acquisition": "true",
        "remaining_blockers": "BLOCKED_MISSING_ASSET_TYPE",
        "minimum_new_acquisitions_required": "0",
        "recommended_sensor_family": "requires_manual_sensor_selection",
        "next_required_action": "REPAIR_METADATA_BEFORE_TEMPORAL_ACQUISITION",
    }
    plan_rows.append(blocked_patch)
    requirement_rows: list[dict[str, str]] = []
    for row in plan_rows[:30]:
        requirement_rows.extend(actionable_requirement_rows(row))
    for slot in ["1", "2", "3"]:
        requirement_rows.append(
            {
                "canonical_patch_id": "PATCH_BLOCKED",
                "region": "Recife",
                "required_slot_index": slot,
                "existing_dates": "unknown",
                "required_new_date_status": "UNSPECIFIED_REAL_ACQUISITION_NEEDED",
                "required_sensor_family": "requires_manual_sensor_selection",
                "required_minimum_metadata": "canonical_patch_id|region|asset_type|acquisition_date|patch_asset_link|s2_cloud_cover_when_required",
                "requires_same_patch_geometry": "true",
                "requires_same_preprocessing_contract": "true",
                "requires_dinov2_frozen_inference_later": "true",
                "requires_cloud_cover_if_s2": "false",
                "requires_no_label_creation": "true",
                "requires_no_event_assumption": "true",
                "requirement_status": "BLOCKED_REQUIREMENT",
                "blocking_reason": "BLOCKED_MISSING_ASSET_TYPE",
            }
        )

    plan_path = tmp_path / "outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv"
    req_path = tmp_path / "outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv"
    write_csv(plan_path, PLAN_COLUMNS, plan_rows)
    write_csv(req_path, REQ_INPUT_COLUMNS, requirement_rows)
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_acquisition_gap_plan_mv1.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"total_missing_acquisition_slots": len(requirement_rows)}) + "\n", encoding="utf-8")
    return plan_path.read_text(encoding="utf-8"), req_path.read_text(encoding="utf-8")


def test_backfill_request_manifest_dry_run_batches_and_guardrails(tmp_path: Path) -> None:
    original_plan, original_requirements = write_inputs(tmp_path)

    first = execute(tmp_path)
    second = execute(tmp_path)

    request_path = tmp_path / "outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv"
    batch_path = tmp_path / "outputs_public/tables/revp_temporal_backfill_batch_targets_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_backfill_request_manifest_mv1.json"
    report_path = tmp_path / "outputs_public/execution_reports/revp_temporal_backfill_request_manifest_mv1.md"

    assert first == second
    assert request_path.exists()
    assert batch_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert (tmp_path / "outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv").read_text(encoding="utf-8") == original_plan
    assert (tmp_path / "outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv").read_text(encoding="utf-8") == original_requirements

    request_rows = read_csv(request_path)
    batch_rows = read_csv(batch_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert list(request_rows[0].keys()) == REQUEST_COLUMNS
    assert list(batch_rows[0].keys()) == BATCH_COLUMNS
    assert metrics["step_a_backfill_request_status"] == "STEP_A_BACKFILL_REQUESTS_READY_FOR_30_PATCH_DRY_RUN"
    assert metrics["actionable_patch_count"] == 30
    assert metrics["batch_targets"]["BATCH_MIN_1_PATCH_TEMPORAL_DRIFT_PILOT"]["required_new_acquisitions"] == 2
    assert metrics["batch_targets"]["BATCH_MIN_20_PATCHES_TEMPORAL_DRIFT_PILOT"]["required_new_acquisitions"] == 40
    assert metrics["batch_targets"]["BATCH_MIN_30_PATCHES_TEMPORAL_DRIFT_READY"]["required_new_acquisitions"] == 60

    patch_1_requests = [row for row in request_rows if row["canonical_patch_id"] == "PATCH_01"]
    assert len(patch_1_requests) == 2
    assert {row["request_status"] for row in patch_1_requests} == {"READY_FOR_FUTURE_SEARCH"}
    assert all(row["is_actionable_request"] == "true" for row in patch_1_requests)
    assert all(row["future_search_backend"] == "STAC_OR_GEE_FUTURE_SEARCH_NOT_EXECUTED" for row in patch_1_requests)

    blocked_rows = [row for row in request_rows if row["canonical_patch_id"] == "PATCH_BLOCKED"]
    assert len(blocked_rows) == 3
    assert all(row["is_actionable_request"] == "false" for row in blocked_rows)
    assert {row["request_status"] for row in blocked_rows} == {"BLOCKED_BY_UNSUPPORTED_SENSOR_SELECTION"}
    assert "PATCH_BLOCKED" not in {row["canonical_patch_id"] for row in batch_rows if row["batch_blocking_reason"] == "NONE"}

    one_patch_batch = [
        row for row in batch_rows if row["batch_id"] == "BATCH_MIN_1_PATCH_TEMPORAL_DRIFT_PILOT"
    ]
    assert len(one_patch_batch) == 1
    assert one_patch_batch[0]["required_new_acquisitions_for_patch"] == "2"
    assert one_patch_batch[0]["projected_clean_dates_after_batch"] == "3"
    assert one_patch_batch[0]["would_be_eligible_after_batch"] == "true"

    output_text = (
        request_path.read_text(encoding="utf-8")
        + batch_path.read_text(encoding="utf-8")
        + metrics_path.read_text(encoding="utf-8")
        + report_path.read_text(encoding="utf-8")
    )
    for forbidden in [
        "final_label",
        "formal_negative",
        "class_0",
        "operational_ground_truth",
        "label final",
        "negativo formal",
        "classe 0",
        "ground truth operacional",
    ]:
        assert forbidden not in output_text
    assert "required_new_date_status" not in request_path.read_text(encoding="utf-8")
    assert "Guardrails preservados" in report_path.read_text(encoding="utf-8")
