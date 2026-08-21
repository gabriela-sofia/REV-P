from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_temporal_source_sensor_provenance_audit_mv1 import (
    BLOCKER_COLUMNS,
    PROVENANCE_COLUMNS,
    execute,
)


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

RESOLVED_REQ_COLUMNS = [
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
    "original_required_sensor_family",
    "resolved_required_sensor_family",
    "sensor_resolution_status",
    "sensor_resolution_confidence",
    "is_sensor_resolved",
    "is_requirement_actionable_after_sensor_resolution",
    "remaining_requirement_blocker",
    "resolution_lineage",
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


def asset(patch_id: str, asset_id: str, asset_type: str, source_file: str = "manifests/source_links.csv") -> dict[str, str]:
    return {
        "normalized_asset_id": f"{patch_id}::{asset_id}",
        "canonical_patch_id": patch_id,
        "asset_id": asset_id,
        "region": "Recife",
        "asset_type": asset_type,
        "acquisition_date": "unknown",
        "cloud_cover": "missing",
        "cloud_cover_required": "false",
        "cloud_cover_available": "false",
        "patch_asset_link_status": "LINKED",
        "temporal_usability_status": "CLEAN_TEMPORAL_ASSET" if asset_type == "dinov2_embedding" else "BLOCKED_MISSING_ASSET_TYPE",
        "is_clean_temporal_asset": "true" if asset_type == "dinov2_embedding" else "false",
        "source_files": source_file,
        "applied_repair_fields": "fixture",
        "applied_repair_rules": "fixture",
        "applied_repair_confidences": "HIGH_EXPLICIT_FIELD",
        "rejected_repair_count": "0",
        "lineage_status": "APPLIED_DETERMINISTIC_SHADOW",
        "guardrail_status": "GUARDRAILS_PRESERVED",
    }


def resolved_requirement(patch_id: str, actionable: str = "false") -> dict[str, str]:
    return {
        "canonical_patch_id": patch_id,
        "region": "Recife",
        "required_slot_index": "1",
        "existing_dates": "2022-01-01",
        "required_new_date_status": "UNSPECIFIED_REAL_ACQUISITION_NEEDED",
        "required_sensor_family": "blocked_derived_embedding_without_source_sensor",
        "required_minimum_metadata": "canonical_patch_id|region|asset_type|acquisition_date|patch_asset_link",
        "requires_same_patch_geometry": "true",
        "requires_same_preprocessing_contract": "true",
        "requires_dinov2_frozen_inference_later": "true",
        "requires_cloud_cover_if_s2": "false",
        "requires_no_label_creation": "true",
        "requires_no_event_assumption": "true",
        "requirement_status": "ACTIONABLE_REQUIREMENT",
        "blocking_reason": "NONE",
        "original_required_sensor_family": "requires_manual_sensor_selection",
        "resolved_required_sensor_family": "blocked_derived_embedding_without_source_sensor",
        "sensor_resolution_status": "BLOCKED_DERIVED_EMBEDDING_WITHOUT_SOURCE_SENSOR",
        "sensor_resolution_confidence": "REJECTED_DERIVED_EMBEDDING_ONLY",
        "is_sensor_resolved": "false",
        "is_requirement_actionable_after_sensor_resolution": actionable,
        "remaining_requirement_blocker": "SENSOR_NOT_RESOLVED_TO_SENTINEL_2_OPTICAL",
        "resolution_lineage": "fixture",
    }


def write_inputs(tmp_path: Path) -> tuple[str, str, str]:
    assets = [
        asset("PATCH_DINO_BLOCKED", "DINO_BLOCKED", "dinov2_embedding", ""),
        asset("PATCH_DINO_S2", "DINO_S2", "dinov2_embedding"),
        asset("PATCH_UNKNOWN", "UNKNOWN_1", "unknown", ""),
        asset("PATCH_AMBIG", "DINO_AMBIG", "dinov2_embedding"),
    ]
    resolved_requirements = [
        resolved_requirement("PATCH_DINO_BLOCKED"),
        resolved_requirement("PATCH_DINO_S2"),
        resolved_requirement("PATCH_UNKNOWN"),
        resolved_requirement("PATCH_AMBIG"),
    ]
    request_rows = [
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
        for index, row in enumerate(resolved_requirements, start=1)
    ]
    resolution_rows = [
        {
            "canonical_patch_id": row["canonical_patch_id"],
            "region": row["region"],
            "required_slot_index": row["required_slot_index"],
            "original_required_sensor_family": "requires_manual_sensor_selection",
            "resolved_required_sensor_family": row["resolved_required_sensor_family"],
            "sensor_resolution_status": row["sensor_resolution_status"],
            "sensor_resolution_rule": "fixture",
            "sensor_resolution_confidence": row["sensor_resolution_confidence"],
            "evidence_asset_ids": "fixture",
            "evidence_asset_types": "dinov2_embedding",
            "has_clean_s2_asset": "false",
            "has_clean_s1_asset": "false",
            "has_dinov2_embedding_only": "true",
            "has_unknown_asset_type": "false",
            "is_sensor_resolved": "false",
            "is_request_potentially_actionable_after_resolution": "false",
            "blocking_reason": row["remaining_requirement_blocker"],
            "guardrail_status": "GUARDRAILS_PRESERVED_SENSOR_FAMILY_RESOLUTION_ONLY",
        }
        for row in resolved_requirements
    ]
    source_rows = [
        {
            "canonical_patch_id": "PATCH_DINO_S2",
            "asset_id": "DINO_S2",
            "source_asset_id": "S2_SOURCE_PRODUCT",
            "source_sensor_family": "sentinel_2_optical",
            "source_acquisition_date": "2022-02-01",
            "source_cloud_cover": "7",
        },
        {
            "canonical_patch_id": "PATCH_AMBIG",
            "asset_id": "DINO_AMBIG",
            "source_asset_id": "AMBIG_SOURCE",
            "source_sensor_family": "sentinel_1_sar;sentinel_2_optical",
            "source_acquisition_date": "2022-02-01",
            "source_cloud_cover": "missing",
        },
    ]

    write_csv(tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv", ASSET_COLUMNS, assets)
    write_csv(tmp_path / "outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv", RESOLVED_REQ_COLUMNS, resolved_requirements)
    write_csv(tmp_path / "outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv", list(request_rows[0].keys()), request_rows)
    write_csv(tmp_path / "outputs_public/tables/revp_temporal_sensor_family_resolution_mv1.csv", list(resolution_rows[0].keys()), resolution_rows)
    write_csv(tmp_path / "manifests/source_links.csv", list(source_rows[0].keys()), source_rows)
    write_csv(tmp_path / "outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv", ["canonical_patch_id", "region"], [{"canonical_patch_id": row["canonical_patch_id"], "region": "Recife"} for row in resolved_requirements])
    write_csv(tmp_path / "outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv", ["canonical_patch_id", "asset_id"], [{"canonical_patch_id": row["canonical_patch_id"], "asset_id": "fixture"} for row in resolved_requirements])
    return (
        (tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv").read_text(encoding="utf-8"),
        (tmp_path / "outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv").read_text(encoding="utf-8"),
        (tmp_path / "outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv").read_text(encoding="utf-8"),
    )


def test_source_sensor_provenance_audit_outputs_rules_and_guardrails(tmp_path: Path) -> None:
    original_assets, original_requirements, original_requests = write_inputs(tmp_path)

    first = execute(tmp_path)
    second = execute(tmp_path)

    provenance_path = tmp_path / "outputs_public/tables/revp_temporal_source_sensor_provenance_mv1.csv"
    blockers_path = tmp_path / "outputs_public/tables/revp_temporal_source_sensor_blockers_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_source_sensor_provenance_mv1.json"
    report_path = tmp_path / "outputs_public/execution_reports/revp_temporal_source_sensor_provenance_mv1.md"

    assert first == second
    assert provenance_path.exists()
    assert blockers_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert (tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv").read_text(encoding="utf-8") == original_assets
    assert (tmp_path / "outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv").read_text(encoding="utf-8") == original_requirements
    assert (tmp_path / "outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv").read_text(encoding="utf-8") == original_requests

    provenance_rows = read_csv(provenance_path)
    blocker_rows = read_csv(blockers_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert list(provenance_rows[0].keys()) == PROVENANCE_COLUMNS
    assert list(blocker_rows[0].keys()) == BLOCKER_COLUMNS
    assert "step_a_source_sensor_provenance_status" in metrics

    by_patch = {row["canonical_patch_id"]: row for row in provenance_rows}
    assert by_patch["PATCH_DINO_BLOCKED"]["source_link_status"] == "BLOCKED_DINO_WITHOUT_EXPLICIT_SOURCE"
    assert by_patch["PATCH_DINO_S2"]["source_link_status"] == "SOURCE_CONFIRMED_SENTINEL_2"
    assert by_patch["PATCH_DINO_S2"]["source_sensor_family"] == "sentinel_2_optical"
    assert by_patch["PATCH_DINO_S2"]["is_dinov2_derived_asset"] == "true"
    assert by_patch["PATCH_UNKNOWN"]["source_link_status"] == "BLOCKED_UNKNOWN_ASSET_TYPE"
    assert by_patch["PATCH_AMBIG"]["source_link_status"] == "BLOCKED_AMBIGUOUS_SOURCE_SENSOR"
    assert metrics["dinov2_with_explicit_source"] == 2
    assert metrics["dinov2_without_explicit_source"] == 1
    assert metrics["unknowns_blocked"] == 1

    output_text = (
        provenance_path.read_text(encoding="utf-8")
        + blockers_path.read_text(encoding="utf-8")
        + metrics_path.read_text(encoding="utf-8")
        + report_path.read_text(encoding="utf-8")
    )
    for forbidden in [
        "C:\\Users\\gabriela",
        "final_label",
        "formal_negative",
        "class_0",
        "ground_truth_operational",
        "label final",
        "negativo formal",
        "classe 0",
        "ground truth operacional",
    ]:
        assert forbidden not in output_text
    assert "2022-03-01" not in output_text
    assert "Guardrails preservados" in report_path.read_text(encoding="utf-8")
