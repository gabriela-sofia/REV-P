"""MV2-DATA-01 Tarefa 4 — Plano de consulta STAC/OData metadata-only.

Para os mesmos targets espacialmente prontos, registra o PAYLOAD ESPERADO de uma futura
consulta de metadados em CDSE STAC, CDSE OData e GEE metadata. NÃO executa HTTP, NÃO baixa
nada. Todas as linhas têm would_execute_http=false, would_download=false e
execution_status=NOT_EXECUTED_PLAN_ONLY.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_stac_odata_metadata_plan.csv
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
    STAC_PROVIDERS,
    build_targets,
    ensure_output_dir,
    is_spatial_ready,
    write_csv,
)

COLUMNS = [
    "query_id",
    "target_id",
    "provider",
    "endpoint_type",
    "collection",
    "bbox_or_intersects_source",
    "datetime_or_range",
    "query_payload_or_filter",
    "expected_metadata_fields",
    "would_execute_http",
    "would_download",
    "execution_status",
    "reason_not_executed",
    "next_action",
]

# Tipo de endpoint e descrição do payload por provider (apenas referência, nunca chamada).
PROVIDER_SPEC = {
    "CDSE_STAC": (
        "STAC_ITEM_SEARCH",
        "POST /search com {collections, bbox(WGS84), datetime, limit}; campos em properties",
    ),
    "CDSE_ODATA": (
        "ODATA_PRODUCTS_FILTER",
        "$filter=Collection eq 'SENTINEL-2' and OData.CSC.Intersects(area=geography'..') "
        "and ContentDate/Start ge .. and ContentDate/Start le ..",
    ),
    "GEE_METADATA": (
        "EE_IMAGECOLLECTION_FILTER_METADATA_ONLY",
        "ImageCollection(collection).filterBounds(aoi).filterDate(start,end)"
        ".aggregate_array(property)  // sem getDownloadURL/Export",
    ),
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fields = "|".join(METADATA_FIELDS)
    date_range = f"{S2_ARCHIVE_WINDOW_START}/{S2_ARCHIVE_WINDOW_END}"
    n = 0
    for t in build_targets():
        if not is_spatial_ready(t):
            continue
        for provider in STAC_PROVIDERS:
            endpoint_type, payload = PROVIDER_SPEC[provider]
            n += 1
            rows.append(
                {
                    "query_id": f"MV2_DATA_01_SQ_{n:05d}",
                    "target_id": t["target_id"],
                    "provider": provider,
                    "endpoint_type": endpoint_type,
                    "collection": DEFAULT_S2_COLLECTION,
                    "bbox_or_intersects_source": "patch_bbox_reprojected_to_WGS84",
                    "datetime_or_range": date_range,
                    "query_payload_or_filter": payload,
                    "expected_metadata_fields": fields,
                    "would_execute_http": "false",
                    "would_download": "false",
                    "execution_status": "NOT_EXECUTED_PLAN_ONLY",
                    "reason_not_executed": (
                        "marco_metadata_only;sem_HTTP_real_ate_lineage_revisado"
                    ),
                    "next_action": (
                        "executar_consulta_metadata_only_em_revisao_humana_apos_"
                        "recuperar_scene_id"
                    ),
                }
            )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_data_01_stac_odata_metadata_plan.csv"
    write_csv(out, COLUMNS, rows)
    print(
        f"[mv2_data_01_stac_plan] {len(rows)} planos STAC/OData (would_execute_http=false) | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
