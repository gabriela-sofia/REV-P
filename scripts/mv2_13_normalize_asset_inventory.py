"""MV2-13 Tarefa 4 — Normaliza inventário de assets (separa render de raster nativo)."""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_output_dir,
    normalize_assets,
    write_csv,
)

COLUMNS = [
    "asset_id", "patch_id", "region", "sensor", "platform", "scene_id",
    "acquisition_datetime", "cloud_cover", "tile_id", "asset_kind",
    "is_visual_render", "is_npz_visual", "is_dinov2_embedding",
    "is_native_sentinel_raster", "source_path", "source_file",
    "metadata_status", "notes",
]


def main() -> None:
    ensure_output_dir()
    rows = normalize_assets()
    out = OUTPUT_DIR / "mv2_13_asset_inventory_normalized.csv"
    write_csv(out, COLUMNS, rows)
    native = sum(1 for r in rows if r["is_native_sentinel_raster"] == "true")
    print(f"[mv2_13_assets] assets: {len(rows)} | raster nativo: {native}")
    print(f"[mv2_13_assets] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
