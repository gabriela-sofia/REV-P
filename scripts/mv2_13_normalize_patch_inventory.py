"""MV2-13 Tarefa 3 — Normaliza inventário de patches (geometria bounds, CRS, status)."""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_output_dir,
    normalize_patches,
    write_csv,
)

COLUMNS = [
    "patch_id", "region", "city", "canonical_status", "has_geometry",
    "geometry_path", "geometry_type", "bbox", "crs", "crs_status",
    "source_file", "notes",
]


def main() -> None:
    ensure_output_dir()
    rows = normalize_patches()
    out = OUTPUT_DIR / "mv2_13_patch_inventory_normalized.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_13_patches] patches: {len(rows)}")
    print(f"[mv2_13_patches] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
