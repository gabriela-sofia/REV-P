"""MV2-13 Tarefa 8 — Resolve temporalidade e scene lineage por binding.

Lê a matriz de binding e classifica scene_id, acquisition_datetime, cloud_cover, tile
e sensor. Não infere data por nome fraco. Diz se serve para Trilha A e para STAC.

Saída: outputs_public/mv2_sentinel_binding/mv2_13_temporal_scene_lineage_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    classify_datetime,
    clean,
    ensure_output_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "binding_id", "anchor_id", "asset_id", "patch_id", "sensor", "platform",
    "scene_id", "acquisition_datetime", "datetime_status", "cloud_cover",
    "cloud_cover_status", "tile_id", "scene_lineage_status",
    "can_support_temporal_track", "can_support_stac", "temporal_blocking_reason",
    "next_action",
]


def build_rows() -> list[dict[str, object]]:
    binding = read_csv_rows(OUTPUT_DIR / "mv2_13_binding_matrix.csv")
    rows: list[dict[str, object]] = []
    for b in binding:
        scene = clean(b.get("scene_id"))
        tile = clean(b.get("tile_id"))
        sensor = clean(b.get("sensor"))
        dt_norm, dt_status = classify_datetime(clean(b.get("acquisition_datetime")))
        cloud = clean(b.get("cloud_cover"))
        cloud_status = "PRESENT" if cloud else "ABSENT"

        if scene:
            lineage = "SCENE_ID_PRESENT_UNLINKED_TO_PATCH"
        elif tile and dt_status == "VALID_SINGLE" and sensor:
            lineage = "TILE_DATE_SENSOR_PARTIAL"
        else:
            lineage = "NO_SCENE_NO_TEMPORAL"

        # Trilha A exige data de aquisição válida por patch (ausente em todo o corpus).
        can_temporal = "false"
        # STAC exige cena OU tile+data+sensor; nada disso fecha aqui.
        can_stac = "false"

        if not scene and dt_status != "VALID_SINGLE":
            block = "sem_scene_id_e_sem_datetime_valido"
            action = "recuperar_scene_id_ou_datetime_via_historico_GEE"
        elif scene and not clean(b.get("patch_id")):
            block = "scene_id_presente_mas_sem_patch_para_vincular"
            action = "vincular_cena_a_patch_oficial_sem_inferencia"
        elif dt_status == "INVALID_FUTURE":
            block = "datetime_futuro_invalido_descartado"
            action = "recuperar_datetime_real_de_aquisicao"
        else:
            block = "temporalidade_incompleta"
            action = "recuperar_campos_temporais_faltantes"

        rows.append(
            {
                "binding_id": clean(b.get("binding_id")),
                "anchor_id": clean(b.get("anchor_id")),
                "asset_id": clean(b.get("asset_id")),
                "patch_id": clean(b.get("patch_id")),
                "sensor": sensor,
                "platform": clean(b.get("platform")),
                "scene_id": scene,
                "acquisition_datetime": dt_norm,
                "datetime_status": dt_status,
                "cloud_cover": cloud,
                "cloud_cover_status": cloud_status,
                "tile_id": tile,
                "scene_lineage_status": lineage,
                "can_support_temporal_track": can_temporal,
                "can_support_stac": can_stac,
                "temporal_blocking_reason": block,
                "next_action": action,
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_13_temporal_scene_lineage_matrix.csv"
    write_csv(out, COLUMNS, rows)
    with_scene = sum(1 for r in rows if r["scene_id"])
    with_dt = sum(1 for r in rows if r["datetime_status"] == "VALID_SINGLE")
    print(f"[mv2_13_temporal] linhas: {len(rows)} | com scene_id: {with_scene} | datetime valido: {with_dt}")
    print(f"[mv2_13_temporal] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
