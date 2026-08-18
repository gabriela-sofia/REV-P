"""MV2-DATA-01 Tarefa 3 — Plano de consulta GEE metadata-only.

Para cada target com bbox/CRS válido, gera a INSTRUÇÃO programática de uma futura consulta
de metadados no Google Earth Engine. NÃO executa GEE API, NÃO exporta raster, NÃO baixa
nada. Todas as linhas têm would_call_gee=false, would_export_raster=false e
execution_status=NOT_EXECUTED_PLAN_ONLY.

A janela de datas sugerida é um LIMITE do arquivo Sentinel-2 (não uma data de aquisição
do alvo) — serve só para acotar a consulta futura.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_gee_metadata_query_plan.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_01_common import (
    DEFAULT_S2_COLLECTION,
    METADATA_FIELDS,
    OUTPUT_DIR,
    PROJECT_ROOT,
    S2_ARCHIVE_WINDOW_END,
    S2_ARCHIVE_WINDOW_START,
    build_targets,
    ensure_output_dir,
    is_spatial_ready,
    write_csv,
)

COLUMNS = [
    "query_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "source_collection",
    "aoi_source",
    "bbox",
    "crs",
    "suggested_date_window_start",
    "suggested_date_window_end",
    "metadata_fields_to_collect",
    "would_call_gee",
    "would_export_raster",
    "execution_status",
    "reason_not_executed",
    "next_action",
]


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fields = "|".join(METADATA_FIELDS)
    n = 0
    for t in build_targets():
        # Só planejamos consulta para alvos com bbox/CRS válido (lado espacial pronto).
        if not is_spatial_ready(t):
            continue
        n += 1
        rows.append(
            {
                "query_id": f"MV2_DATA_01_GEEQ_{n:05d}",
                "target_id": t["target_id"],
                "asset_id": t["asset_id"],
                "patch_id": t["patch_id"],
                "region": t["region"],
                "source_collection": DEFAULT_S2_COLLECTION,
                "aoi_source": "patch_bbox_reprojected_from_crs",
                "bbox": t["bbox"],
                "crs": t["crs"],
                "suggested_date_window_start": S2_ARCHIVE_WINDOW_START,
                "suggested_date_window_end": S2_ARCHIVE_WINDOW_END,
                "metadata_fields_to_collect": fields,
                "would_call_gee": "false",
                "would_export_raster": "false",
                "execution_status": "NOT_EXECUTED_PLAN_ONLY",
                "reason_not_executed": (
                    "marco_metadata_only;recuperacao_de_lineage_sem_chamada_GEE_real"
                ),
                "next_action": (
                    "executar_consulta_metadata_only_em_revisao_humana_e_registrar_"
                    "evidencia_no_template"
                ),
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_data_01_gee_metadata_query_plan.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_01_gee_plan] {len(rows)} planos metadata-only (would_call_gee=false) | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
