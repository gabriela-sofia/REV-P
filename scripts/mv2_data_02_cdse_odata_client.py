"""MV2-DATA-02 Tarefa 3 — Cliente CDSE OData (metadata + resolvedor de download).

Resolve produtos por product_id/scene_id quando houver, e PREPARA o endpoint de download
sem baixar por padrão. Executa metadados só se a config permitir; nunca baixa sem
allow_raster_download=true E o target explicitamente aprovado num plano canary.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_cdse_odata_metadata_results.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import config_blocks_metadata, load_config
from mv2_data_02_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_targets,
    read_csv_rows,
    write_csv,
)

CDSE_ODATA_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

COLUMNS = [
    "result_id",
    "target_id",
    "provider",
    "executed",
    "http_status",
    "product_name",
    "product_id",
    "odata_id",
    "content_length",
    "online_status",
    "publication_date",
    "checksum_available",
    "download_url_resolved",
    "download_executed",
    "blocking_reason",
    "notes",
]


def _stac_product_ids() -> dict[str, str]:
    """product_id/scene_id já resolvidos pelo STAC (se houver), por target_id."""
    out: dict[str, str] = {}
    rows = read_csv_rows(PUBLIC_DIR / "mv2_data_02_cdse_stac_metadata_results.csv")
    for r in rows:
        pid = clean(r.get("product_id")) or clean(r.get("scene_id"))
        if pid:
            out[clean(r.get("target_id"))] = pid
    return out


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    targets = load_targets()
    resolved = _stac_product_ids()
    metadata_blocked = config_blocks_metadata(cfg) or not cfg.get("cdse_odata_enabled")

    rows: list[dict[str, object]] = []
    for i, t in enumerate(targets, start=1):
        tid = clean(t.get("target_id"))
        product_id = resolved.get(tid, "")
        base: dict[str, object] = {
            "result_id": f"MV2_DATA_02_ODATA_{i:05d}",
            "target_id": tid,
            "provider": "CDSE_ODATA",
            "product_name": "",
            "product_id": product_id,
            "odata_id": "",
            "content_length": "",
            "online_status": "",
            "publication_date": "",
            "checksum_available": "",
            "download_url_resolved": "",
            # Download JAMAIS por padrão neste cliente; só o executor canary baixa.
            "download_executed": "false",
        }
        if not product_id:
            base.update(
                {
                    "executed": "false",
                    "http_status": "",
                    "blocking_reason": "NO_PRODUCT_ID_OR_SCENE_ID_TO_RESOLVE",
                    "notes": "OData precisa de product_id/scene_id; lineage ainda nao recuperado",
                }
            )
        elif metadata_blocked:
            base.update(
                {
                    "executed": "false",
                    "http_status": "",
                    "blocking_reason": "allow_network/allow_metadata_calls/cdse_odata_enabled=false",
                    "notes": f"filtro OData pronto p/ '{product_id}' (NOT_EXECUTED_CONFIG_BLOCKED)",
                }
            )
        else:
            # Mesmo com metadados permitidos, download segue bloqueado aqui.
            base.update(
                {
                    "executed": "false",
                    "http_status": "",
                    "blocking_reason": "metadata_resolver_only;download_no_executor_canary",
                    "notes": f"resolveria via {CDSE_ODATA_PRODUCTS_URL}?$filter=Name eq '{product_id}'",
                }
            )
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_cdse_odata_metadata_results.csv"
    write_csv(out, COLUMNS, rows)
    downloads = sum(1 for r in rows if r["download_executed"] == "true")
    print(
        f"[mv2_data_02_odata] {len(rows)} alvos | downloads={downloads} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
