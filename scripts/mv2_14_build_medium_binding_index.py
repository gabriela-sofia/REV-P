"""MV2-14 Tarefa 3 — Índice dos 128 bindings MEDIUM do MV2-13 (alvos da recuperação).

Lê mv2_13_binding_matrix.csv, seleciona os BINDING_MEDIUM e preserva o que já existe
(asset_id, patch_id, bbox, CRS, região, sensor, day10_status) marcando o que falta.

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_medium_binding_index.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_14_gee_lineage_common import (
    MV2_13_BINDING_MATRIX,
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "target_id", "binding_id", "anchor_id", "asset_id", "patch_id", "region",
    "sensor", "platform", "bbox", "crs", "day10_status", "missing_scene_id",
    "missing_datetime", "missing_tile", "missing_cloud_cover",
    "missing_export_provenance", "recovery_priority", "notes",
]


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(MV2_13_BINDING_MATRIX)
    medium = [b for b in binding if clean(b.get("binding_strength")) == "BINDING_MEDIUM"]
    rows: list[dict[str, object]] = []
    for i, b in enumerate(medium, start=1):
        rows.append({
            "target_id": f"MV2_14_TGT_{i:05d}",
            "binding_id": clean(b.get("binding_id")),
            "anchor_id": clean(b.get("anchor_id")),
            "asset_id": clean(b.get("asset_id")),
            "patch_id": clean(b.get("patch_id")),
            "region": clean(b.get("region")),
            "sensor": clean(b.get("sensor")),
            "platform": clean(b.get("platform")),
            "bbox": clean(b.get("bbox")),
            "crs": clean(b.get("crs")),
            "day10_status": clean(b.get("day10_status")),
            "missing_scene_id": str(not clean(b.get("scene_id"))).lower(),
            "missing_datetime": str(not clean(b.get("acquisition_datetime"))).lower(),
            "missing_tile": str(not clean(b.get("tile_id"))).lower(),
            "missing_cloud_cover": str(not clean(b.get("cloud_cover"))).lower(),
            "missing_export_provenance": "true",
            # Prioridade: Petrópolis tem cenas locais candidatas (revisão); zonas 22/25 não.
            "recovery_priority": "P1_REVIEW_CANDIDATE_SCENES_LOCAL" if clean(b.get("region")) == "Petropolis" else "P0_NO_LOCAL_SCENE",
            "notes": "alvo de recuperação de lineage temporal/cena; lado espacial ja pronto",
        })
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_medium_binding_index.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter
    reg = Counter(r["region"] for r in rows)
    print(f"[mv2_14_medium] bindings MEDIUM: {len(rows)} | por regiao: {dict(reg)}")
    print(f"[mv2_14_medium] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
