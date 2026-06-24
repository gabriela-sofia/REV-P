"""MV2-DATA-07 source sensor lineage promotion.

Promotes only traceable Sentinel-2 lineage to spectral eligibility. Unknown,
visual renders, DINO, NPZ, conflicts, and Sentinel-1 support remain blocked for
the optical Sentinel-2 baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_source_sensor_lineage_promotion"
SEED_PATH = PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_seed" / "revp_source_sensor_lineage_seed_10.csv"


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


def classify_sensor_lineage(sensor_family: str, source_asset_ref: str = "") -> tuple[str, str]:
    family = (sensor_family or "UNKNOWN").strip().upper()
    if family == "SENTINEL_2":
        if not source_asset_ref:
            return "UNKNOWN_BLOCKED", "sem_source_asset_ref_rastreavel"
        return "SENTINEL_2_ELIGIBLE", ""
    if family == "SENTINEL_1":
        return "SENTINEL_1_SUPPORT_ONLY", "suporte_sar_nao_baseline_optico_s2"
    if family == "DINO_DERIVED":
        return "DINO_DERIVED_BLOCKED", "dino_nao_e_raster_espectral"
    if family == "PNG_RENDER":
        return "PNG_RENDER_BLOCKED", "png_nao_e_raster_espectral"
    if family == "NPZ_EMBEDDING":
        return "NPZ_EMBEDDING_BLOCKED", "npz_embedding_nao_e_raster_espectral"
    if family == "CONFLICT":
        return "CONFLICT_BLOCKED", "lineage_em_conflito"
    return "UNKNOWN_BLOCKED", "sensor_lineage_ausente"


def build_rows(input_path: Path = SEED_PATH) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(read_csv(input_path), 1):
        status, reason = classify_sensor_lineage(row.get("sensor_family", ""), row.get("source_asset_ref", ""))
        rows.append(
            {
                "promotion_id": f"MV2_DATA_07_SENSOR_{idx:03d}",
                "patch_id": row.get("patch_id", ""),
                "asset_id": row.get("asset_id", ""),
                "slot_id": row.get("slot_id", ""),
                "evidence_id": row.get("evidence_id", ""),
                "asset_ref": row.get("asset_ref", ""),
                "source_asset_ref": row.get("source_asset_ref", ""),
                "sensor_family": row.get("sensor_family", "UNKNOWN") or "UNKNOWN",
                "lineage_classification": status,
                "spectral_s2_eligible": str(status == "SENTINEL_2_ELIGIBLE").lower(),
                "blocked_reason": reason,
            }
        )
    return rows


def write_schema() -> None:
    write_json(
        PROJECT_ROOT / "datasets" / "schemas" / "schema_mv2_data_07_source_sensor_lineage_promotion.json",
        {
            "schema_id": "schema_mv2_data_07_source_sensor_lineage_promotion",
            "required_fields": ["patch_id", "asset_id", "sensor_family", "lineage_classification"],
            "allowed_classifications": [
                "SENTINEL_2_ELIGIBLE",
                "SENTINEL_1_SUPPORT_ONLY",
                "DINO_DERIVED_BLOCKED",
                "PNG_RENDER_BLOCKED",
                "NPZ_EMBEDDING_BLOCKED",
                "UNKNOWN_BLOCKED",
                "CONFLICT_BLOCKED",
            ],
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(SEED_PATH))
    args = parser.parse_args(argv)
    rows = build_rows(Path(args.input))
    fields = [
        "promotion_id",
        "patch_id",
        "asset_id",
        "slot_id",
        "evidence_id",
        "asset_ref",
        "source_asset_ref",
        "sensor_family",
        "lineage_classification",
        "spectral_s2_eligible",
        "blocked_reason",
    ]
    write_schema()
    write_csv(OUT_DIR / "mv2_data_07_sensor_lineage_promotion.csv", fields, rows)
    write_csv(OUT_DIR / "mv2_data_07_s2_eligible_batch.csv", fields, [row for row in rows if row["lineage_classification"] == "SENTINEL_2_ELIGIBLE"])
    write_csv(OUT_DIR / "mv2_data_07_blocked_batch.csv", fields, [row for row in rows if row["lineage_classification"] != "SENTINEL_2_ELIGIBLE"])
    counts = Counter(row["lineage_classification"] for row in rows)
    summary = {
        "stage": "DATA-07",
        "total_targets": len(rows),
        "sentinel_2_eligible": counts.get("SENTINEL_2_ELIGIBLE", 0),
        "sentinel_1_support_only": counts.get("SENTINEL_1_SUPPORT_ONLY", 0),
        "unknown_blocked": counts.get("UNKNOWN_BLOCKED", 0),
        "blocked_total": sum(count for key, count in counts.items() if key != "SENTINEL_2_ELIGIBLE"),
        "api_calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }
    write_json(OUT_DIR / "mv2_data_07_summary.json", summary)
    write_text(
        OUT_DIR / "mv2_data_07_report.md",
        f"# DATA-07 source sensor lineage promotion\n\n- Sentinel-2 eligible: {summary['sentinel_2_eligible']}\n- unknown blocked: {summary['unknown_blocked']}\n- blocked total: {summary['blocked_total']}\n- calls/downloads/rasters/crops: 0/0/0/0",
    )
    write_text(OUT_DIR / "commands.txt", "python scripts/mv2_data_07_source_sensor_lineage_promotion.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
