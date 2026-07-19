"""MV2-DATA-01 Tarefa 1 — Descobrir os 128 targets de lineage.

Lê o índice de bindings MEDIUM (MV2-14), seleciona os bindings médios, preserva
asset_id/patch_id/region/bbox/CRS/sensor/status, classifica prioridade por região e
completude e NÃO inventa scene_id/datetime/tile/cloud_cover.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_lineage_targets.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_01_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    build_targets,
    ensure_output_dir,
    write_csv,
)

COLUMNS = [
    "target_id",
    "binding_id",
    "anchor_id",
    "asset_id",
    "patch_id",
    "region",
    "sensor",
    "platform",
    "bbox",
    "crs",
    "has_asset_id",
    "has_patch_id",
    "has_bbox",
    "has_crs",
    "has_scene_id",
    "has_datetime",
    "has_tile",
    "has_cloud_cover",
    "lineage_status",
    "priority",
    "blocking_reason",
    "next_action",
]


def main() -> None:
    ensure_output_dir()
    targets = build_targets()
    out = OUTPUT_DIR / "mv2_data_01_lineage_targets.csv"
    write_csv(out, COLUMNS, targets)
    ready = sum(1 for t in targets if str(t["lineage_status"]).startswith("TARGET_READY"))
    print(
        f"[mv2_data_01_targets] {len(targets)} alvos | prontos(spatial)={ready} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
