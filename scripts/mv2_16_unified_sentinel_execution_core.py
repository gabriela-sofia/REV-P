"""MV2-16 unified Sentinel execution core dry-run.

Builds a unified gate matrix from DATA-06/07/08, crop policy, and SCL readiness.
It does not call external APIs and does not create/download raster or crop data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_16_unified_sentinel_execution_core"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_patch_bindings() -> list[dict[str, str]]:
    return read_csv(PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_seed" / "revp_temporal_window_seed_10.csv")


def load_temporal_seed() -> list[dict[str, str]]:
    return read_csv(PROJECT_ROOT / "outputs_public" / "mv2_data_temporal_window_promotion" / "mv2_data_06_temporal_window_promotion.csv")


def load_sensor_lineage_seed() -> list[dict[str, str]]:
    return read_csv(PROJECT_ROOT / "outputs_public" / "mv2_data_source_sensor_lineage_promotion" / "mv2_data_07_sensor_lineage_promotion.csv")


def load_scene_consensus_if_exists() -> list[dict[str, str]]:
    return read_csv(PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_probe" / "mv2_data_08_lineage_consensus.csv")


def load_crop_policy_if_exists() -> list[dict[str, str]]:
    return read_csv(PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_crop_policy" / "revp_crop_authorization_candidates.csv")


def compute_gate_a(row: dict[str, str]) -> str:
    if row.get("temporal_status") != "PROMOTED_METADATA_READY":
        return "BLOCKED_NO_TEMPORAL_WINDOW"
    if row.get("lineage_status") != "SENTINEL_2_ELIGIBLE":
        return "BLOCKED_NO_SENSOR_LINEAGE"
    if row.get("metadata_status") in {"BLOCKED_NO_CONFIG", "BLOCKED_BY_FLAGS", "NO_CALL"}:
        return "BLOCKED_NO_CONFIG"
    if not row.get("product_id") or not row.get("scene_id"):
        return "BLOCKED_NO_LINEAGE"
    if row.get("crop_status") != "AUTHORIZED_METADATA_ONLY":
        return "BLOCKED_NO_NATIVE_RASTER"
    if row.get("scl_qa_status") != "READY":
        return "BLOCKED_NO_SCL_QA"
    return "READY_METADATA_ONLY"


def compute_gate_b(row: dict[str, str]) -> str:
    return "GEOMETRY_BACKLOG_READY"


def compute_gate_c(row: dict[str, str]) -> str:
    return "POLICY_READY"


def compute_gate_d(row: dict[str, str]) -> str:
    return "POLICY_READY"


def compute_day10_status(row: dict[str, str]) -> str:
    return "BLOCKED" if compute_gate_a(row).startswith("BLOCKED") else "READY_REVIEW_ONLY"


def compute_mv2_16_readiness(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "NOT_READY_FOR_MV2_16"
    if any(row.get("gate_a") == "READY_METADATA_ONLY" for row in rows):
        return "READY_FOR_MV2_16_METADATA_ONLY"
    return "READY_FOR_MV2_16_DRY_RUN"


def write_schema() -> None:
    write_json(
        PROJECT_ROOT / "datasets" / "schemas" / "schema_mv2_16_unified_gate_status.json",
        {
            "schema_id": "schema_mv2_16_unified_gate_status",
            "required_fields": ["patch_id", "asset_id", "gate_a", "gate_b", "gate_c", "gate_d", "day10_status"],
            "allowed_readiness": [
                "NOT_READY_FOR_MV2_16",
                "READY_FOR_MV2_16_DRY_RUN",
                "READY_FOR_MV2_16_METADATA_ONLY",
                "READY_FOR_MV2_16_LOCAL_RASTER_CANARY",
            ],
        },
    )


def build_unified_rows() -> list[dict[str, str]]:
    temporal = {row.get("asset_id", ""): row for row in load_temporal_seed()}
    lineage = {row.get("asset_id", ""): row for row in load_sensor_lineage_seed()}
    consensus = {row.get("asset_id", ""): row for row in load_scene_consensus_if_exists()}
    crop = {row.get("asset_id", ""): row for row in load_crop_policy_if_exists()}
    scl = {row.get("asset_id", ""): row for row in read_csv(PROJECT_ROOT / "outputs_public" / "mv2_pre_unification_scl_qa" / "revp_scl_qa_readiness.csv")}
    base = load_patch_bindings()
    rows: list[dict[str, str]] = []
    metadata_summary = read_json(PROJECT_ROOT / "outputs_public" / "mv2_data_metadata_only_probe" / "mv2_data_08_summary.json")
    for item in base:
        asset_id = item.get("asset_id", "")
        temp = temporal.get(asset_id, {})
        lin = lineage.get(asset_id, {})
        con = consensus.get(asset_id, {})
        cr = crop.get(asset_id, {})
        sq = scl.get(asset_id, {})
        row = {
            "patch_id": item.get("patch_id", ""),
            "asset_id": asset_id,
            "temporal_status": temp.get("promotion_status", ""),
            "lineage_status": lin.get("lineage_classification", ""),
            "metadata_status": metadata_summary.get("preflight_status", "BLOCKED_NO_CONFIG"),
            "product_id": con.get("product_id", ""),
            "scene_id": con.get("scene_id", ""),
            "crop_status": cr.get("authorization_status", ""),
            "scl_qa_status": sq.get("scl_qa_status", ""),
        }
        row["gate_a"] = compute_gate_a(row)
        row["gate_b"] = compute_gate_b(row)
        row["gate_c"] = compute_gate_c(row)
        row["gate_d"] = compute_gate_d(row)
        row["day10_status"] = compute_day10_status(row)
        rows.append(row)
    return rows


def write_unified_gate_matrix(rows: list[dict[str, str]]) -> None:
    fields = ["patch_id", "asset_id", "temporal_status", "lineage_status", "metadata_status", "product_id", "scene_id", "crop_status", "scl_qa_status", "gate_a", "gate_b", "gate_c", "gate_d", "day10_status"]
    write_csv(OUT_DIR / "mv2_16_unified_gate_matrix.csv", fields, rows)


def write_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    summary = {
        "stage": "MV2-16",
        "targets": len(rows),
        "readiness": compute_mv2_16_readiness(rows),
        "gate_a_blocked": sum(1 for row in rows if row["gate_a"].startswith("BLOCKED")),
        "gate_b": "GEOMETRY_BACKLOG_READY",
        "gate_c": "POLICY_READY",
        "gate_d": "POLICY_READY",
        "day10_status": "BLOCKED",
        "calls": 0,
        "downloads": 0,
        "rasters": 0,
        "crops": 0,
    }
    write_json(OUT_DIR / "mv2_16_summary.json", summary)
    return summary


def write_report(summary: dict[str, Any]) -> None:
    write_text(
        OUT_DIR / "mv2_16_report.md",
        f"# MV2-16 unified Sentinel execution core\n\n- readiness: {summary['readiness']}\n- Gate A blocked targets: {summary['gate_a_blocked']}\n- Gate B/C/D: {summary['gate_b']} / {summary['gate_c']} / {summary['gate_d']}\n- calls/downloads/rasters/crops: 0/0/0/0",
    )


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser().parse_args(argv)
    write_schema()
    rows = build_unified_rows()
    write_unified_gate_matrix(rows)
    summary = write_summary(rows)
    write_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
