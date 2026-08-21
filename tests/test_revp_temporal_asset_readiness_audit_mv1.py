from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file()) / "scripts" / "ground_truth"))

from revp_temporal_asset_readiness_audit_mv1 import CSV_COLUMNS, execute


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_audit_generates_public_artifacts_columns_guardrails_and_counts(tmp_path: Path) -> None:
    write_csv(
        tmp_path / "manifests/dino_inputs/revp_v1fu_dino_sentinel_input_manifest/dino_sentinel_input_manifest_v1fu.csv",
        ["dino_input_id", "canonical_patch_id", "region", "source_asset_id", "acquisition_date", "cloud_cover"],
        [
            {
                "dino_input_id": "DINO_1A",
                "canonical_patch_id": "PATCH_1",
                "region": "Recife",
                "source_asset_id": "ASSET_1A",
                "acquisition_date": "2022-01-01",
                "cloud_cover": "10",
            },
            {
                "dino_input_id": "DINO_2A",
                "canonical_patch_id": "PATCH_2",
                "region": "Recife",
                "source_asset_id": "ASSET_2A",
                "acquisition_date": "2022-02-01",
                "cloud_cover": "20",
            },
            {
                "dino_input_id": "DINO_2B",
                "canonical_patch_id": "PATCH_2",
                "region": "Recife",
                "source_asset_id": "ASSET_2B",
                "acquisition_date": "2022-02-10",
                "cloud_cover": "30",
            },
            {
                "dino_input_id": "DINO_3A",
                "canonical_patch_id": "PATCH_3",
                "region": "Curitiba",
                "source_asset_id": "ASSET_3A",
                "acquisition_date": "2022-03-01",
                "cloud_cover": "5",
            },
            {
                "dino_input_id": "DINO_3B",
                "canonical_patch_id": "PATCH_3",
                "region": "Curitiba",
                "source_asset_id": "ASSET_3B",
                "acquisition_date": "2022-03-10",
                "cloud_cover": "15",
            },
            {
                "dino_input_id": "DINO_3C",
                "canonical_patch_id": "PATCH_3",
                "region": "Curitiba",
                "source_asset_id": "ASSET_3C",
                "acquisition_date": "2022-03-20",
                "cloud_cover": "25",
            },
            {
                "dino_input_id": "DINO_X",
                "canonical_patch_id": "PATCH_X",
                "region": "Petropolis",
                "source_asset_id": "ASSET_X",
                "acquisition_date": "unknown",
                "cloud_cover": "missing",
            },
        ],
    )

    first_metrics = execute(tmp_path)
    second_metrics = execute(tmp_path)

    table = tmp_path / "outputs_public/tables/revp_temporal_asset_readiness_mv1.csv"
    report = tmp_path / "outputs_public/execution_reports/revp_temporal_asset_readiness_mv1.md"
    metrics_path = tmp_path / "outputs_public/metrics/revp_temporal_asset_readiness_mv1.json"

    assert table.exists()
    assert report.exists()
    assert metrics_path.exists()
    assert first_metrics == second_metrics

    rows = read_csv(table)
    assert list(rows[0].keys()) == CSV_COLUMNS
    assert first_metrics["step_a_global_status"] == "STEP_A_BLOCKED_NEEDS_TEMPORAL_ASSET_BACKFILL"
    assert first_metrics["patches_with_1_date"] == 1
    assert first_metrics["patches_with_2_dates"] == 1
    assert first_metrics["patches_with_3plus_dates"] == 1
    assert first_metrics["eligible_patch_count_for_temporal_drift"] == 1

    patch_x = next(row for row in rows if row["canonical_patch_id"] == "PATCH_X")
    assert patch_x["acquisition_dates_count"] == "0"
    assert patch_x["acquisition_dates"] == "unknown"
    assert patch_x["cloud_cover_mean"] == "missing"

    persisted = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert persisted["step_a_global_status"] == first_metrics["step_a_global_status"]
    assert "step_a_global_status" in persisted
    assert "DINOv2 permanece frozen" in persisted["guardrails_preserved"]
    assert "label-free e review-only" in report.read_text(encoding="utf-8")
    assert "Guardrails preservados" in report.read_text(encoding="utf-8")

    forbidden_field_terms = ("label", "negative", "ground_truth", "training")
    assert not any(term in column for column in CSV_COLUMNS for term in forbidden_field_terms)
    assert not any(term in key for key in persisted for term in forbidden_field_terms)


def test_ready_status_requires_30_patches_with_3_clean_dates(tmp_path: Path) -> None:
    rows: list[dict[str, str]] = []
    for patch_index in range(30):
        for date_index in range(3):
            rows.append(
                {
                    "binding_id": f"B{patch_index}_{date_index}",
                    "patch_id": f"PATCH_{patch_index:03d}",
                    "region": "Curitiba",
                    "asset_id": f"ASSET_{patch_index}_{date_index}",
                    "acquisition_datetime": f"2022-01-{date_index + 1:02d}T00:00:00Z",
                    "cloud_cover": "0",
                }
            )
    write_csv(
        tmp_path / "outputs_public/mv2_sentinel_binding/mv2_13_binding_matrix.csv",
        ["binding_id", "patch_id", "region", "asset_id", "acquisition_datetime", "cloud_cover"],
        rows,
    )

    metrics = execute(tmp_path)

    assert metrics["eligible_patch_count_for_temporal_drift"] == 30
    assert metrics["step_a_global_status"] == "STEP_A_READY_TEMPORAL_EMBEDDING_DRIFT"
