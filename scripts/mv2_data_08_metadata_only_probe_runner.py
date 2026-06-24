"""MV2-DATA-08 metadata-only probe runner.

Default execution is offline and fail-closed. No provider call is made unless a
local config explicitly enables network metadata calls while keeping raster and
canary downloads disabled.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_probe"
TEMPORAL_PATH = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_promotion" / "mv2_data_06_temporal_window_promotion.csv"
LINEAGE_PATH = PROJECT_ROOT / "outputs_public" / "mv2_data_source_sensor_lineage_promotion" / "mv2_data_07_sensor_lineage_promotion.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "api_config.local.json"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def metadata_preflight(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        return {"status": "BLOCKED_NO_CONFIG", "config_present": False, "calls_allowed": False}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("allow_network") or not config.get("allow_metadata_calls"):
        return {"status": "BLOCKED_BY_FLAGS", "config_present": True, "calls_allowed": False}
    if config.get("allow_raster_download") or config.get("allow_canary_download"):
        return {"status": "BLOCKED_BY_FLAGS", "config_present": True, "calls_allowed": False}
    providers = config.get("providers", {})
    gee = providers.get("GEE", {})
    if gee.get("enabled") and gee.get("project_id_env") and not os.environ.get(gee["project_id_env"]):
        return {"status": "BLOCKED_BY_FLAGS", "config_present": True, "calls_allowed": False}
    return {"status": "READY_METADATA_ONLY", "config_present": True, "calls_allowed": True}


def eligible_targets() -> list[dict[str, str]]:
    temporal = {row.get("asset_id", ""): row for row in read_csv(TEMPORAL_PATH)}
    lineage = read_csv(LINEAGE_PATH)
    rows: list[dict[str, str]] = []
    for row in lineage:
        temp = temporal.get(row.get("asset_id", ""), {})
        if (
            temp.get("promotion_status") == "PROMOTED_METADATA_READY"
            and row.get("lineage_classification") == "SENTINEL_2_ELIGIBLE"
        ):
            merged = {**temp, **row}
            rows.append(merged)
    return rows


def provider_rows(targets: list[dict[str, str]], preflight: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    if not targets or not preflight.get("calls_allowed"):
        base = targets or read_csv(TEMPORAL_PATH)
        return [
            {
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "provider": provider,
                "scene_id": "",
                "product_id": "",
                "datetime_utc": "",
                "mgrs_tile": "",
                "datatake_identifier": "",
                "generation_time": "",
                "cloudy_pixel_percentage": "",
                "nodata_pixel_percentage": "",
                "thin_cirrus_percentage": "",
                "geometry": "",
                "bbox": "",
                "odata_id": "",
                "odata_name": "",
                "odata_geofootprint": "",
                "odata_s3path": "",
                "consensus_status": "NO_CALL",
                "blocked_reason": preflight["status"],
            }
            for row in base
        ]
    return []


def write_schema() -> None:
    write_json(
        PROJECT_ROOT / "datasets" / "schemas" / "schema_mv2_data_08_metadata_only_probe.json",
        {
            "schema_id": "schema_mv2_data_08_metadata_only_probe",
            "required_fields": ["patch_id", "asset_id", "provider", "consensus_status"],
            "consensus_statuses": ["STRONG", "MEDIUM_REVIEW", "WEAK_BLOCKED", "CONFLICT", "NO_CALL", "NO_MATCH"],
            "downloads_allowed": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--allow-metadata-calls", action="store_true")
    args = parser.parse_args(argv)
    preflight = metadata_preflight(Path(args.config))
    targets = eligible_targets()
    fields = [
        "patch_id",
        "asset_id",
        "provider",
        "scene_id",
        "product_id",
        "datetime_utc",
        "mgrs_tile",
        "datatake_identifier",
        "generation_time",
        "cloudy_pixel_percentage",
        "nodata_pixel_percentage",
        "thin_cirrus_percentage",
        "geometry",
        "bbox",
        "odata_id",
        "odata_name",
        "odata_geofootprint",
        "odata_s3path",
        "consensus_status",
        "blocked_reason",
    ]
    write_schema()
    all_rows: list[dict[str, Any]] = []
    for provider, filename in [("GEE", "mv2_data_08_gee_metadata.csv"), ("CDSE_STAC", "mv2_data_08_stac_metadata.csv"), ("CDSE_ODATA", "mv2_data_08_odata_metadata.csv")]:
        rows = provider_rows(targets, preflight, provider)
        all_rows.extend(rows)
        write_csv(OUT_DIR / filename, fields, rows)
    consensus = provider_rows(targets, preflight, "CONSENSUS")
    write_csv(OUT_DIR / "mv2_data_08_lineage_consensus.csv", fields, consensus)
    summary = {
        "stage": "DATA-08",
        "preflight_status": preflight["status"],
        "eligible_targets": len(targets),
        "calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
        "confirmed_lineage": 0,
    }
    write_json(OUT_DIR / "mv2_data_08_summary.json", summary)
    write_text(
        OUT_DIR / "mv2_data_08_report.md",
        f"# DATA-08 metadata-only probe\n\n- preflight: {summary['preflight_status']}\n- eligible targets: {summary['eligible_targets']}\n- calls/downloads/rasters/crops: 0/0/0/0",
    )
    write_text(OUT_DIR / "commands.txt", "python scripts/mv2_data_08_metadata_only_probe_runner.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
