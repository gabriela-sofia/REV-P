"""MV2-13 Tarefa 10 — Gate do Dia 10 (baseline espectral Sentinel).

Decide, por binding/patch/asset/anchor, se o Dia 10 continua bloqueado. Regra dura:
can_unlock_day10_now=false se não houver raster Sentinel nativo já presente e validado.
Como este marco NÃO baixa raster, o desbloqueio imediato permanece false.

Saída: outputs_public/mv2_sentinel_binding/mv2_13_day10_spectral_gate.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean,
    ensure_output_dir,
    normalize_assets,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "day10_gate_id", "anchor_id", "asset_id", "patch_id", "region",
    "binding_strength", "stac_status", "has_native_raster", "has_valid_geometry",
    "has_crs", "has_scene_id", "has_datetime", "has_cloud_cover",
    "can_unlock_day10_now", "can_unlock_day10_next_step", "day10_status",
    "blocking_reason", "next_action",
]


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    # Mapa asset_id -> raster nativo? (todos false neste corpus)
    native = {a["asset_id"]: a["is_native_sentinel_raster"] == "true" for a in normalize_assets() if a["asset_id"]}
    rows: list[dict[str, object]] = []
    for i, b in enumerate(binding, start=1):
        asset_id = clean(b.get("asset_id"))
        has_native = native.get(asset_id, False)
        has_geom = clean(b.get("geometry_status")) == "BOUNDS_ONLY" and bool(clean(b.get("bbox")))
        has_crs = clean(b.get("crs_status")) in {"PRESENT_VALID", "PRESENT_UNVERIFIED"}
        has_scene = bool(clean(b.get("scene_id")))
        has_dt = clean(b.get("acquisition_datetime")) != ""
        has_cloud = clean(b.get("cloud_cover")) != ""

        # Desbloqueio imediato exige raster nativo validado — ausente.
        can_now = "true" if has_native else "false"
        # Próximo passo: só se binding forte + geometria + cena + datetime (e então
        # ainda assim depende de adquirir o raster). Aqui nunca é satisfeito.
        strong = clean(b.get("binding_strength")) == "BINDING_STRONG"
        can_next = "true" if (strong and has_geom and has_crs and has_scene and has_dt) else "false"

        day10 = clean(b.get("day10_status"))
        if has_native:
            day10 = "DAY10_NOT_READY"
            block = "raster_presente_mas_pipeline_espectral_nao_executado_neste_marco"
            action = "validar_bandas_e_planejar_extracao_em_marco_dedicado"
        else:
            block = "sem_raster_sentinel_nativo_validado;" + clean(b.get("blocking_reason"))
            if day10 == "DAY10_BLOCKED_NO_SCENE_ID":
                action = "recuperar_scene_id_datetime_e_depois_adquirir_raster_nativo"
            elif day10 == "DAY10_BLOCKED_NO_GEOMETRY":
                action = "vincular_cena_a_patch_com_geometria_e_CRS"
            else:
                action = "estabelecer_binding_forte_antes_de_qualquer_aquisicao"

        rows.append(
            {
                "day10_gate_id": f"MV2_13_DAY10_{i:05d}",
                "anchor_id": clean(b.get("anchor_id")),
                "asset_id": asset_id,
                "patch_id": clean(b.get("patch_id")),
                "region": clean(b.get("region")),
                "binding_strength": clean(b.get("binding_strength")),
                "stac_status": clean(b.get("stac_status")),
                "has_native_raster": str(has_native).lower(),
                "has_valid_geometry": str(has_geom).lower(),
                "has_crs": str(has_crs).lower(),
                "has_scene_id": str(has_scene).lower(),
                "has_datetime": str(has_dt).lower(),
                "has_cloud_cover": str(has_cloud).lower(),
                "can_unlock_day10_now": can_now,
                "can_unlock_day10_next_step": can_next,
                "day10_status": day10,
                "blocking_reason": block,
                "next_action": action,
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_13_day10_spectral_gate.csv"
    write_csv(out, COLUMNS, rows)
    now = sum(1 for r in rows if r["can_unlock_day10_now"] == "true")
    nxt = sum(1 for r in rows if r["can_unlock_day10_next_step"] == "true")
    print(f"[mv2_13_day10] linhas: {len(rows)} | unlock now: {now} | unlock next: {nxt}")
    print(f"[mv2_13_day10] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
