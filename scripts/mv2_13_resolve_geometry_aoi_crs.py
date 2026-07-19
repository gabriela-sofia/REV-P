"""MV2-13 Tarefa 7 — Resolve geometria/AOI/CRS por binding.

Lê a matriz de binding e, para cada linha, valida geometria de patch, bbox e CRS.
Não cria geometria nova por inferência e não usa município como AOI de patch.
Geometria BOUNDS_ONLY é AOI grosseira (aceitável para dry-run hipotético, nunca como
footprint de evento).

Saída: outputs_public/mv2_sentinel_binding/mv2_13_geometry_aoi_crs_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    crs_status,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "binding_id", "anchor_id", "patch_id", "has_patch_geometry", "geometry_path",
    "geometry_type", "bbox", "crs", "crs_status", "aoi_status",
    "geometry_blocking_reason", "next_action",
]


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    rows: list[dict[str, object]] = []
    for b in binding:
        bbox = clean(b.get("bbox"))
        crs = clean(b.get("crs"))
        geom_status = clean(b.get("geometry_status"))
        has_geom = bool(bbox) and geom_status not in {"", "ABSENT", "NONE"}
        cstat = crs_status(crs)
        if not has_geom:
            aoi = "AOI_ABSENT"
            block = "sem_geometria_de_patch_para_AOI"
            action = "recuperar_ou_digitalizar_geometria_de_patch_com_CRS"
        elif cstat == "ABSENT":
            aoi = "AOI_NO_CRS"
            block = "geometria_presente_mas_sem_CRS"
            action = "recuperar_CRS_da_geometria"
        else:
            # bounds de header = AOI grosseira coerente para query espacial hipotética
            aoi = "AOI_COARSE_BOUNDS_OK_FOR_DRY_RUN"
            block = "AOI_grosseira_bounds_only_nao_e_footprint_de_evento"
            action = "manter_para_dry_run_espacial_e_refinar_geometria_quando_disponivel"
        rows.append(
            {
                "binding_id": clean(b.get("binding_id")),
                "anchor_id": clean(b.get("anchor_id")),
                "patch_id": clean(b.get("patch_id")),
                "has_patch_geometry": str(has_geom).lower(),
                "geometry_path": clean(b.get("geometry_path")),
                "geometry_type": "bbox_bounds" if has_geom else "NONE",
                "bbox": bbox,
                "crs": crs,
                "crs_status": cstat,
                "aoi_status": aoi,
                "geometry_blocking_reason": block,
                "next_action": action,
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_13_geometry_aoi_crs_matrix.csv"
    write_csv(out, COLUMNS, rows)
    ok = sum(1 for r in rows if r["has_patch_geometry"] == "true")
    print(f"[mv2_13_geometry] linhas: {len(rows)} | com geometria de patch: {ok}")
    print(f"[mv2_13_geometry] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
