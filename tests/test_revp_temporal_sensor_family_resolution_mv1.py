from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_temporal_sensor_family_resolution_mv1 import (
    RESOLVED_REQUIREMENT_EXTRA_COLUMNS,
    SENSOR_RESOLUTION_COLUMNS,
    execute,
)


REQ_COLUMNS = [
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

GAP_COLUMNS = [
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

ASSET_COLUMNS = [
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


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def requirement(patch_id: str, region: str = "Recife", slot: str = "1", blocker: str = "NONE") -> dict[str, str]:
    return {
        "canonical_patch_id": patch_id,
        "region": region,
        "required_slot_index": slot,
        "existing_dates": "2022-01-01",
        "required_new_date_status": "UNSPECIFIED_REAL_ACQUISITION_NEEDED",
        "required_sensor_family": "requires_manual_sensor_selection",
        "required_minimum_metadata": "canonical_patch_id|region|asset_type|acquisition_date|patch_asset_link|s2_cloud_cover_when_required",
        "requires_same_patch_geometry": "true",
        "requires_same_preprocessing_contract": "true",
        "requires_dinov2_frozen_inference_later": "true",
        "requires_cloud_cover_if_s2": "false",
        "requires_no_label_creation": "true",
        "requires_no_event_assumption": "true",
        "requirement_status": "ACTIONABLE_REQUIREMENT" if blocker == "NONE" else "BLOCKED_REQUIREMENT",
        "blocking_reason": blocker,
    }


def gap_row(
    patch_id: str,
    region: str = "Recife",
    blockers: str = "NONE",
    repair: str = "false",
    missing: str = "1",
) -> dict[str, str]:
    return {
        "canonical_patch_id": patch_id,
        "region": region,
        "current_clean_dates_count": str(3 - int(missing)),
        "current_clean_dates": "2022-01-01",
        "missing_dates_to_3": missing,
        "current_clean_temporal_asset_count": "1",
        "s2_clean_asset_count": "0",
        "s1_clean_asset_count": "0",
        "dinov2_clean_asset_count": "0",
        "eligible_for_temporal_embedding_drift_now": "false",
        "acquisition_gap_status": "NEEDS_1_MORE_DATE" if blockers == "NONE" else "BLOCKED_BY_METADATA_REPAIR",
        "acquisition_priority": "P0_NEEDS_1_DATE",
        "is_actionable_for_temporal_backfill": "true" if blockers == "NONE" else "false",
        "requires_metadata_repair_before_acquisition": repair,
        "remaining_blockers": blockers,
        "minimum_new_acquisitions_required": missing,
        "recommended_sensor_family": "requires_manual_sensor_selection",
        "recommended_processing_contract": "SAME_PATCH_GEOMETRY_SAME_PREPROCESSING_DINOV2_FROZEN_LATER",
        "next_required_action": "PLAN_ONE_REAL_TEMPORAL_ACQUISITION_SLOT",
        "guardrail_status": "GUARDRAILS_PRESERVED_ACQUISITION_PLAN_ONLY",
    }


def asset(patch_id: str, asset_type: str, clean: str = "true") -> dict[str, str]:
    return {
        "normalized_asset_id": f"{patch_id}::{asset_type}",
        "canonical_patch_id": patch_id,
        "asset_id": f"{patch_id}_{asset_type}_asset",
        "region": "Recife",
        "asset_type": asset_type,
        "acquisition_date": "2022-01-01" if clean == "true" else "unknown",
        "cloud_cover": "5" if asset_type == "sentinel_2_optical" else "missing",
        "cloud_cover_required": "true" if asset_type == "sentinel_2_optical" else "false",
        "cloud_cover_available": "true" if asset_type == "sentinel_2_optical" else "false",
        "patch_asset_link_status": "LINKED",
        "temporal_usability_status": "CLEAN_TEMPORAL_ASSET" if clean == "true" else "BLOCKED_MISSING_ASSET_TYPE",
        "is_clean_temporal_asset": clean,
        "source_files": "fixture",
        "applied_repair_fields": "fixture",
        "applied_repair_rules": "fixture",
        "applied_repair_confidences": "HIGH_EXPLICIT_FIELD",
        "rejected_repair_count": "0",
        "lineage_status": "APPLIED_DETERMINISTIC_SHADOW",
        "guardrail_status": "GUARDRAILS_PRESERVED",
    }


def write_inputs(tmp_path: Path) -> tuple[str, str, str, str]:
    req_rows = [
        requirement("PATCH_S2"),
        requirement("PATCH_DINO"),
        requirement("PATCH_UNKNOWN", blocker="BLOCKED_MISSING_ASSET_TYPE"),
        requirement("PATCH_AMBIG"),
        requirement("PATCH_S1"),
    ]
    gap_rows = [
        gap_row("PATCH_S2"),
        gap_row("PATCH_DINO"),
        gap_row("PATCH_UNKNOWN", blockers="BLOCKED_MISSING_ASSET_TYPE", repair="true"),
        gap_row("PATCH_AMBIG"),
        gap_row("PATCH_S1"),
    ]
    asset_rows = [
        asset("PATCH_S2", "sentinel_2_optical"),
        asset("PATCH_DINO", "dinov2_embedding"),
        asset("PATCH_UNKNOWN", "unknown", clean="false"),
        asset("PATCH_AMBIG", "sentinel_2_optical"),
        asset("PATCH_AMBIG", "sentinel_1_sar"),
        asset("PATCH_S1", "sentinel_1_sar"),
    ]
    coverage_rows = [
        {
            "canonical_patch_id": row["canonical_patch_id"],
            "region": row["region"],
            "normalized_asset_count": "1",
            "clean_temporal_asset_count": "1",
            "unique_acquisition_dates_count": row["current_clean_dates_count"],
            "unique_acquisition_dates": row["current_clean_dates"],
            "s2_clean_asset_count": "0",
            "s1_clean_asset_count": "0",
            "dinov2_clean_asset_count": "0",
            "has_1_date": "false",
            "has_2_dates": "true",
            "has_3plus_dates": "false",
            "eligible_for_temporal_embedding_drift": "false",
            "temporal_coverage_status": "fixture",
            "remaining_blockers": row["remaining_blockers"],
            "next_required_action": row["next_required_action"],
        }
        for row in gap_rows
    ]
    request_manifest_rows = [
        {
            "request_id": f"REQ_{index}",
            "canonical_patch_id": row["canonical_patch_id"],
            "region": row["region"],
            "required_slot_index": row["required_slot_index"],
            "current_clean_dates_count": "2",
            "existing_dates": row["existing_dates"],
            "missing_dates_to_3": "1",
            "required_sensor_family": row["required_sensor_family"],
            "request_mode": "BLOCKED_METADATA_REPAIR_REQUIRED",
            "request_status": "BLOCKED_BY_UNSUPPORTED_SENSOR_SELECTION",
            "is_actionable_request": "false",
            "blocked_before_request": "true",
            "blocking_reason": "requires_manual_sensor_selection",
            "minimum_required_metadata": row["required_minimum_metadata"],
            "requires_same_patch_geometry": "true",
            "requires_same_preprocessing_contract": "true",
            "requires_dinov2_frozen_inference_later": "true",
            "requires_cloud_cover_if_s2": "false",
            "requires_no_label_creation": "true",
            "requires_no_event_assumption": "true",
            "future_search_backend": "UNDEFINED_BACKEND_BLOCKED",
            "future_search_contract_status": "BLOCKED_BEFORE_FUTURE_SEARCH_CONTRACT",
            "guardrail_status": "GUARDRAILS_PRESERVED_DRY_RUN_REQUEST_ONLY",
        }
        for index, row in enumerate(req_rows, start=1)
    ]

    req_path = tmp_path / "outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv"
    gap_path = tmp_path / "outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv"
    assets_path = tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv"
    coverage_path = tmp_path / "outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv"
    request_manifest_path = tmp_path / "outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv"
    write_csv(req_path, REQ_COLUMNS, req_rows)
    write_csv(gap_path, GAP_COLUMNS, gap_rows)
    write_csv(assets_path, ASSET_COLUMNS, asset_rows)
    write_csv(coverage_path, list(coverage_rows[0].keys()), coverage_rows)
    write_csv(request_manifest_path, list(request_manifest_rows[0].keys()), request_manifest_rows)
    for rel in [
        "outputs_public/metrics/revp_temporal_backfill_request_manifest_mv1.json",
        "outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    return (
        req_path.read_text(encoding="utf-8"),
        gap_path.read_text(encoding="utf-8"),
        assets_path.read_text(encoding="utf-8"),
        request_manifest_path.read_text(encoding="utf-8"),
    )


def test_sensor_family_resolution_outputs_rules_and_guardrails(tmp_path: Path) -> None:
    original_req, original_gap, original_assets, original_request_manifest = write_inputs(tmp_path)

    first = execute(tmp_path)
    second = execute(tmp_path)

    resolution_path = tmp_path / "outputs_public/tables/revp_temporal_sensor_family_resolution_mv1.csv"
    resolved_req_path = tmp_path / "outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_sensor_family_resolution_mv1.json"
    report_path = tmp_path / "outputs_public/execution_reports/revp_temporal_sensor_family_resolution_mv1.md"

    assert first == second
    assert resolution_path.exists()
    assert resolved_req_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert (tmp_path / "outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv").read_text(encoding="utf-8") == original_req
    assert (tmp_path / "outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv").read_text(encoding="utf-8") == original_gap
    assert (tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv").read_text(encoding="utf-8") == original_assets
    assert (tmp_path / "outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv").read_text(encoding="utf-8") == original_request_manifest

    resolution_rows = read_csv(resolution_path)
    resolved_req_rows = read_csv(resolved_req_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert list(resolution_rows[0].keys()) == SENSOR_RESOLUTION_COLUMNS
    for column in REQ_COLUMNS:
        assert column in resolved_req_rows[0]
    for column in RESOLVED_REQUIREMENT_EXTRA_COLUMNS:
        assert column in resolved_req_rows[0]
    assert "step_a_sensor_resolution_status" in metrics

    by_patch = {row["canonical_patch_id"]: row for row in resolution_rows}
    assert by_patch["PATCH_S2"]["resolved_required_sensor_family"] == "sentinel_2_optical_preferred"
    assert by_patch["PATCH_S2"]["sensor_resolution_status"] == "RESOLVED_SENTINEL_2_OPTICAL"
    assert by_patch["PATCH_S2"]["is_request_potentially_actionable_after_resolution"] == "true"
    assert by_patch["PATCH_DINO"]["resolved_required_sensor_family"] == "blocked_derived_embedding_without_source_sensor"
    assert by_patch["PATCH_DINO"]["sensor_resolution_status"] == "BLOCKED_DERIVED_EMBEDDING_WITHOUT_SOURCE_SENSOR"
    assert by_patch["PATCH_UNKNOWN"]["sensor_resolution_status"] == "BLOCKED_UNKNOWN_ASSET_TYPE"
    assert by_patch["PATCH_AMBIG"]["sensor_resolution_status"] == "BLOCKED_AMBIGUOUS_SENSOR_FAMILY"
    assert by_patch["PATCH_S1"]["resolved_required_sensor_family"] == "sentinel_1_sar_optional_support"
    assert by_patch["PATCH_S1"]["is_request_potentially_actionable_after_resolution"] == "false"

    resolved_by_patch = {row["canonical_patch_id"]: row for row in resolved_req_rows}
    assert resolved_by_patch["PATCH_S2"]["required_sensor_family"] == "sentinel_2_optical_preferred"
    assert resolved_by_patch["PATCH_S2"]["original_required_sensor_family"] == "requires_manual_sensor_selection"
    assert resolved_by_patch["PATCH_DINO"]["is_requirement_actionable_after_sensor_resolution"] == "false"
    assert all(row["required_new_date_status"] == "UNSPECIFIED_REAL_ACQUISITION_NEEDED" for row in resolved_req_rows)
    assert metrics["slots_resolved_to_sentinel_2"] == 1
    assert metrics["slots_blocked_by_derived_dinov2_without_source_sensor"] == 1
    assert metrics["slots_blocked_by_unknown_asset_type"] == 1
    assert metrics["slots_blocked_by_ambiguous_sensor_family"] == 1
    assert metrics["patches_potentially_actionable_after_resolution"] == 1
    assert metrics["batches_potentially_recoverable"]["recoverable_1_patch"]["recoverable"] is True

    output_text = (
        resolution_path.read_text(encoding="utf-8")
        + resolved_req_path.read_text(encoding="utf-8")
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
    assert "2022-02-01" not in output_text
    assert "Guardrails preservados" in report_path.read_text(encoding="utf-8")
