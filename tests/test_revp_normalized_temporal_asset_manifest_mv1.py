from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ground_truth"))

from revp_normalized_temporal_asset_manifest_mv1 import ASSET_COLUMNS, COVERAGE_COLUMNS, execute


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_placeholder_inputs(tmp_path: Path) -> None:
    for rel in [
        "outputs_public/tables/revp_temporal_asset_readiness_mv1.csv",
        "outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv",
    ]:
        write_csv(tmp_path / rel, ["canonical_patch_id"], [{"canonical_patch_id": "PATCH_READY"}])
    for rel in [
        "outputs_public/metrics/revp_temporal_asset_readiness_mv1.json",
        "outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json",
        "outputs_public/metrics/revp_temporal_metadata_repair_candidates_mv1.json",
    ]:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")


def candidate(
    patch: str,
    asset: str,
    field: str,
    value: str,
    *,
    rule: str = "EXPLICIT_FIELD_NORMALIZATION",
    confidence: str = "HIGH_EXPLICIT_FIELD",
    applicable: str = "true",
    deterministic: str = "true",
    rejection: str = "NONE",
) -> dict[str, str]:
    return {
        "canonical_patch_id": patch,
        "asset_id": asset,
        "source_file": "manifests/test/source.csv",
        "source_field": field,
        "repair_field": field,
        "original_value": value,
        "candidate_value": value,
        "normalized_value": value,
        "repair_rule": rule,
        "repair_confidence": confidence,
        "is_deterministic": deterministic,
        "is_applicable": applicable,
        "rejection_reason": rejection,
        "asset_type_before": "unknown_asset_type",
        "asset_type_after": value if field == "asset_type" else "unknown_asset_type",
        "region_before": "unknown",
        "region_after": value if field == "region" else "unknown",
        "acquisition_date_before": "unknown",
        "acquisition_date_after": value if field == "acquisition_date" else "unknown",
        "cloud_cover_before": "missing",
        "cloud_cover_after": value if field == "cloud_cover" else "missing",
        "would_increase_clean_asset_count": "true",
        "would_increase_clean_date_count": "true" if field == "acquisition_date" else "false",
        "would_reduce_blocker": applicable,
        "guardrail_status": "GUARDRAILS_PRESERVED_REPAIR_CANDIDATE_ONLY",
    }


def link_candidate(patch: str, asset: str) -> dict[str, str]:
    row = candidate(patch, asset, "patch_asset_link", f"{patch}:{asset}")
    row["source_field"] = "patch_id+asset_id"
    row["repair_rule"] = "PATCH_ASSET_LINK_FROM_MANIFEST"
    return row


def test_normalized_manifest_outputs_shadow_rules_and_determinism(tmp_path: Path) -> None:
    write_placeholder_inputs(tmp_path)
    source_manifest = tmp_path / "manifests/test/source.csv"
    write_csv(source_manifest, ["canonical_patch_id", "asset_id"], [{"canonical_patch_id": "PATCH_READY", "asset_id": "A1"}])
    before = source_manifest.read_text(encoding="utf-8")

    rows: list[dict[str, str]] = []
    for asset, date in [("A1", "2022-01-01"), ("A2", "2022-01-15"), ("A3", "2022-02-01")]:
        rows.extend(
            [
                link_candidate("PATCH_READY", asset),
                candidate("PATCH_READY", asset, "region", "Recife"),
                candidate("PATCH_READY", asset, "asset_type", "sentinel_2_optical"),
                candidate("PATCH_READY", asset, "acquisition_date", date),
                candidate("PATCH_READY", asset, "cloud_cover", "10.0", rule="S2_CLOUD_COVER_NUMERIC_FIELD"),
            ]
        )
    rows.extend(
        [
            link_candidate("PATCH_S1", "S1_A"),
            candidate("PATCH_S1", "S1_A", "region", "Curitiba"),
            candidate("PATCH_S1", "S1_A", "asset_type", "sentinel_1_sar"),
            candidate("PATCH_S1", "S1_A", "acquisition_date", "2022-03-01"),
            candidate("PATCH_S1", "S1_A", "cloud_cover", "missing", applicable="false", deterministic="false", rejection="NOT_REQUIRED_FOR_NON_S2_ASSET"),
            link_candidate("PATCH_REJECTED", "BAD_A"),
            candidate("PATCH_REJECTED", "BAD_A", "region", "unknown", applicable="false", deterministic="false", rejection="MISSING_OR_UNKNOWN"),
            candidate("PATCH_REJECTED", "BAD_A", "asset_type", "sentinel_2_optical"),
            candidate("PATCH_REJECTED", "BAD_A", "acquisition_date", "missing", applicable="false", deterministic="false", rejection="MISSING_OR_UNKNOWN"),
            candidate("PATCH_REJECTED", "BAD_A", "cloud_cover", "missing", applicable="false", deterministic="false", rejection="MISSING_OR_UNKNOWN"),
            link_candidate("PATCH_AMBIG", "AMB_A"),
            candidate("PATCH_AMBIG", "AMB_A", "region", "Recife", confidence="REJECTED_AMBIGUOUS", applicable="false", deterministic="false", rejection="AMBIGUOUS_CONFLICT"),
            candidate("PATCH_AMBIG", "AMB_A", "asset_type", "sentinel_2_optical"),
            candidate("PATCH_AMBIG", "AMB_A", "acquisition_date", "2022-04-01"),
        ]
    )
    write_csv(
        tmp_path / "outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv",
        list(rows[0].keys()),
        rows,
    )

    first = execute(tmp_path)
    second = execute(tmp_path)

    asset_path = tmp_path / "outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv"
    coverage_path = tmp_path / "outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv"
    metrics_path = tmp_path / "outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json"
    report_path = tmp_path / "outputs_public/execution_reports/revp_normalized_temporal_asset_manifest_mv1.md"

    assert first == second
    assert asset_path.exists()
    assert coverage_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert source_manifest.read_text(encoding="utf-8") == before

    asset_rows = read_csv(asset_path)
    coverage_rows = read_csv(coverage_path)
    assert list(asset_rows[0].keys()) == ASSET_COLUMNS
    assert list(coverage_rows[0].keys()) == COVERAGE_COLUMNS
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "step_a_normalized_manifest_status" in metrics

    by_asset = {row["normalized_asset_id"]: row for row in asset_rows}
    assert by_asset["PATCH_READY::A1"]["is_clean_temporal_asset"] == "true"
    assert by_asset["PATCH_S1::S1_A"]["cloud_cover_required"] == "false"
    assert by_asset["PATCH_S1::S1_A"]["is_clean_temporal_asset"] == "true"
    assert by_asset["PATCH_REJECTED::BAD_A"]["region"] == "unknown"
    assert by_asset["PATCH_REJECTED::BAD_A"]["is_clean_temporal_asset"] == "false"
    assert by_asset["PATCH_AMBIG::AMB_A"]["temporal_usability_status"] == "BLOCKED_AMBIGUOUS_REPAIR"

    by_patch = {row["canonical_patch_id"]: row for row in coverage_rows}
    assert by_patch["PATCH_READY"]["eligible_for_temporal_embedding_drift"] == "true"
    assert by_patch["PATCH_READY"]["unique_acquisition_dates_count"] == "3"
    assert by_patch["PATCH_REJECTED"]["eligible_for_temporal_embedding_drift"] == "false"

    output_text = asset_path.read_text(encoding="utf-8") + coverage_path.read_text(encoding="utf-8") + metrics_path.read_text(encoding="utf-8")
    for forbidden in ["final_label", "formal_negative", "class_0", "operational_ground_truth"]:
        assert forbidden not in output_text
    assert "Guardrails preservados" in report_path.read_text(encoding="utf-8")
