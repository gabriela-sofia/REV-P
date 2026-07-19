"""MV2-13 Tarefa 2 — Normaliza âncoras Sentinel (sem inferir vínculo forte de texto)."""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    ensure_output_dir,
    normalize_anchors,
    write_csv,
)

COLUMNS = [
    "anchor_id", "source_file", "raw_anchor_key", "raw_anchor_text",
    "normalized_anchor_key", "extracted_sensor", "extracted_platform",
    "extracted_scene_id", "extracted_tile", "extracted_datetime",
    "extracted_region", "extracted_asset_id", "extracted_patch_id",
    "extraction_confidence", "extraction_notes",
]


def main() -> None:
    ensure_output_dir()
    rows = normalize_anchors()
    out = OUTPUT_DIR / "mv2_13_normalized_anchors.csv"
    write_csv(out, COLUMNS, rows)
    print(f"[mv2_13_anchors] normalizadas: {len(rows)}")
    print(f"[mv2_13_anchors] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
