"""MV2-14 Tarefa 7 — Gate Dia 10 pós-lineage.

Mesmo com lineage completo, o Dia 10 não pode ser desbloqueado agora: esta etapa não
baixa raster nem cria feature espectral. O máximo é sinalizar "próximo passo".

Saída: outputs_public/mv2_gee_lineage_recovery/mv2_14_day10_post_lineage_gate.csv
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
    "day10_post_lineage_id", "asset_id", "patch_id", "region", "scene_id",
    "datetime", "tile", "cloud_cover", "has_bbox", "has_crs", "has_native_raster",
    "has_spectral_features", "can_unlock_day10_now", "can_unlock_day10_next_step",
    "day10_status", "blocking_reason", "next_action",
]


def build_rows() -> list[dict[str, object]]:
    lineage = read_csv_rows(OUTPUT_DIR / "mv2_14_lineage_binding_matrix.csv")
    binding = {clean(b.get("binding_id")): b for b in read_csv_rows(MV2_13_BINDING_MATRIX)}
    rows: list[dict[str, object]] = []
    for i, lr in enumerate(lineage, start=1):
        b = binding.get(clean(lr.get("binding_id")), {})
        has_bbox = bool(clean(b.get("bbox")))
        has_crs = clean(b.get("crs_status")) in {"PRESENT_VALID", "PRESENT_UNVERIFIED"}
        scene = clean(lr.get("recovered_scene_id"))
        dt = clean(lr.get("recovered_datetime"))
        status = clean(lr.get("lineage_status"))

        # Nunca há raster nativo nesta etapa.
        has_native = False
        has_spectral = False
        can_now = "false"  # sem raster nativo validado

        if status == "LINEAGE_RECOVERED_STRONG" and has_bbox and has_crs and scene and dt:
            can_next = "true"
            day10 = "DAY10_READY_FOR_RASTER_DOWNLOAD_REVIEW"
            block = "lineage_forte;falta_apenas_adquirir_raster_em_etapa_revisada"
            action = "revisar_e_planejar_download_de_raster_nativo_sem_executar_agora"
        elif status == "LINEAGE_CANDIDATE_REVIEW":
            can_next = "false"
            day10 = "DAY10_BLOCKED_NO_NATIVE_RASTER"
            block = "cena_nao_recuperada_por_patch;hipotese_requer_revisao"
            action = "fechar_lineage_por_patch_antes_de_pensar_em_raster"
        else:
            can_next = "false"
            day10 = "DAY10_BLOCKED_NO_NATIVE_RASTER"
            block = "sem_lineage;sem_raster_nativo"
            action = "recuperar_lineage_GEE_deste_patch"

        rows.append({
            "day10_post_lineage_id": f"MV2_14_D10_{i:05d}",
            "asset_id": clean(lr.get("asset_id")),
            "patch_id": clean(lr.get("patch_id")),
            "region": clean(lr.get("region")),
            "scene_id": scene,
            "datetime": dt,
            "tile": clean(lr.get("recovered_tile")),
            "cloud_cover": clean(lr.get("recovered_cloud_cover")),
            "has_bbox": str(has_bbox).lower(),
            "has_crs": str(has_crs).lower(),
            "has_native_raster": str(has_native).lower(),
            "has_spectral_features": str(has_spectral).lower(),
            "can_unlock_day10_now": can_now,
            "can_unlock_day10_next_step": can_next,
            "day10_status": day10,
            "blocking_reason": block,
            "next_action": action,
        })
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_14_day10_post_lineage_gate.csv"
    write_csv(out, COLUMNS, rows)
    now = sum(1 for r in rows if r["can_unlock_day10_now"] == "true")
    nxt = sum(1 for r in rows if r["can_unlock_day10_next_step"] == "true")
    print(f"[mv2_14_day10] linhas: {len(rows)} | unlock now: {now} | unlock next_step: {nxt}")
    print(f"[mv2_14_day10] saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
