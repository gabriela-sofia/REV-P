"""MV2 crop authorization policy.

Evaluates metadata-only crop authorization without creating or downloading any
raster/crop artifact.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_crop_policy"
TEMPORAL_PATH = PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_promotion" / "mv2_data_06_temporal_window_promotion.csv"
LINEAGE_PATH = PROJECT_ROOT / "outputs_public" / "mv2_data_source_sensor_lineage_promotion" / "mv2_data_07_sensor_lineage_promotion.csv"
CONSENSUS_PATH = PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_probe" / "mv2_data_08_lineage_consensus.csv"


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


def evaluate_crop_authorization(row: dict[str, str]) -> str:
    if row.get("temporal_status") != "PROMOTED_METADATA_READY":
        return "NOT_AUTHORIZED_NO_TEMPORAL_WINDOW"
    if row.get("lineage_classification") != "SENTINEL_2_ELIGIBLE":
        return "NOT_AUTHORIZED_NO_SENSOR"
    if not row.get("product_id") or not row.get("scene_id"):
        return "NOT_AUTHORIZED_NO_LINEAGE"
    if not row.get("bbox") and not row.get("geometry"):
        return "NOT_AUTHORIZED_NO_AOI"
    if row.get("consensus_status") == "CONFLICT":
        return "NOT_AUTHORIZED_CONFLICT"
    return "AUTHORIZED_METADATA_ONLY"


def build_rows() -> list[dict[str, Any]]:
    temporal = {row.get("asset_id", ""): row for row in read_csv(TEMPORAL_PATH)}
    lineage = {row.get("asset_id", ""): row for row in read_csv(LINEAGE_PATH)}
    consensus = {row.get("asset_id", ""): row for row in read_csv(CONSENSUS_PATH)}
    asset_ids = sorted(set(temporal) | set(lineage) | set(consensus))
    rows: list[dict[str, Any]] = []
    for asset_id in asset_ids:
        temp = temporal.get(asset_id, {})
        lin = lineage.get(asset_id, {})
        con = consensus.get(asset_id, {})
        merged = {
            "patch_id": temp.get("patch_id") or lin.get("patch_id") or con.get("patch_id", ""),
            "asset_id": asset_id,
            "temporal_status": temp.get("promotion_status", ""),
            "lineage_classification": lin.get("lineage_classification", ""),
            "product_id": con.get("product_id", ""),
            "scene_id": con.get("scene_id", ""),
            "bbox": con.get("bbox", ""),
            "geometry": con.get("geometry", ""),
            "consensus_status": con.get("consensus_status", "NO_CALL"),
        }
        status = evaluate_crop_authorization(merged)
        rows.append({**merged, "authorization_status": status, "downloads": 0, "rasters": 0, "crops": 0})
    return rows


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    rows = build_rows()
    fields = [
        "patch_id",
        "asset_id",
        "temporal_status",
        "lineage_classification",
        "product_id",
        "scene_id",
        "bbox",
        "geometry",
        "consensus_status",
        "authorization_status",
        "downloads",
        "rasters",
        "crops",
    ]
    write_csv(OUT_DIR / "revp_crop_authorization_candidates.csv", fields, rows)
    summary = {
        "targets": len(rows),
        "authorized_metadata_only": sum(1 for row in rows if row["authorization_status"] == "AUTHORIZED_METADATA_ONLY"),
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }
    write_json(OUT_DIR / "revp_crop_authorization_summary.json", summary)
    write_text(OUT_DIR / "revp_crop_authorization_report.md", f"# Crop authorization policy\n\n- authorized metadata-only: {summary['authorized_metadata_only']}\n- downloads/rasters/crops: 0/0/0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
