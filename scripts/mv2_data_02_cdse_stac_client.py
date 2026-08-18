"""MV2-DATA-02 Tarefa 2 — Cliente CDSE STAC (metadata-only).

Monta payloads STAC a partir dos targets DATA-01 e executa busca de metadados APENAS se a
config permitir (allow_network=true e allow_metadata_calls=true). Nunca baixa asset; só
salva uma resposta leve normalizada. Se a config bloquear, gera um registro
NOT_EXECUTED_CONFIG_BLOCKED (simulação auditável, sem rede).

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_cdse_stac_metadata_results.csv
"""

from __future__ import annotations

import json
from pathlib import Path

from mv2_data_02_api_config import config_blocks_metadata, load_config
from mv2_data_02_common import (
    DEFAULT_S2_COLLECTION,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_targets,
    write_csv,
)

CDSE_STAC_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"

COLUMNS = [
    "result_id",
    "target_id",
    "provider",
    "request_mode",
    "executed",
    "http_status",
    "matched_items",
    "selected_item_id",
    "product_id",
    "scene_id",
    "datetime",
    "mgrs_tile",
    "cloud_cover",
    "collection",
    "bbox_match_status",
    "metadata_status",
    "blocking_reason",
    "notes",
]


def build_payload(target: dict[str, str]) -> dict[str, object]:
    """Payload STAC item-search (bbox em WGS84 seria reprojetado do CRS do patch)."""
    return {
        "collections": ["SENTINEL-2"],
        "bbox_source_crs": clean(target.get("crs")),
        "bbox_native": clean(target.get("bbox")),
        "datetime": None,  # nunca inventado; janela viria do plano/lineage
        "limit": 50,
        "query": {"productType": {"eq": "S2MSI2A"}},
        "endpoint": CDSE_STAC_SEARCH_URL,
    }


def _try_network_search(payload: dict[str, object]) -> tuple[bool, str, str]:
    """Executa a busca real só se requests existir. Retorna (executou, http_status, erro)."""
    try:
        import requests  # noqa: F401  (import tardio, opcional)
    except ImportError:
        return False, "", "requests_indisponivel"
    try:
        resp = requests.post(
            str(payload["endpoint"]),
            json={k: v for k, v in payload.items() if k != "endpoint"},
            timeout=30,
        )
        return True, str(resp.status_code), ""
    except Exception as exc:  # rede/proxy/timeout — fail-closed, sem propagar
        return False, "", f"erro_rede:{type(exc).__name__}"


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    targets = load_targets()
    blocked = config_blocks_metadata(cfg) or not cfg.get("cdse_stac_enabled")

    rows: list[dict[str, object]] = []
    for i, t in enumerate(targets, start=1):
        payload = build_payload(t)
        base: dict[str, object] = {
            "result_id": f"MV2_DATA_02_STAC_{i:05d}",
            "target_id": clean(t.get("target_id")),
            "provider": "CDSE_STAC",
            "collection": DEFAULT_S2_COLLECTION,
            "product_id": "",
            "scene_id": "",
            "datetime": "",
            "mgrs_tile": "",
            "cloud_cover": "",
            "selected_item_id": "",
            "matched_items": "",
            "bbox_match_status": "NOT_EVALUATED",
        }
        if blocked:
            base.update(
                {
                    "request_mode": "metadata_only",
                    "executed": "false",
                    "http_status": "",
                    "metadata_status": "NOT_EXECUTED_CONFIG_BLOCKED",
                    "blocking_reason": "allow_network/allow_metadata_calls/cdse_stac_enabled=false",
                    "notes": "payload_pronto:" + json.dumps(
                        {k: payload[k] for k in ("collections", "bbox_native", "bbox_source_crs")},
                        ensure_ascii=False,
                    ),
                }
            )
        else:
            executed, status, err = _try_network_search(payload)
            base.update(
                {
                    "request_mode": "metadata_only",
                    "executed": "true" if executed else "false",
                    "http_status": status,
                    "metadata_status": "EXECUTED_NO_MATCH_PARSED" if executed else "NOT_EXECUTED_NETWORK_ERROR",
                    "blocking_reason": "" if executed else err,
                    "notes": "busca metadata-only; parsing de itens fica para revisao humana",
                }
            )
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_cdse_stac_metadata_results.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    print(
        f"[mv2_data_02_stac] {len(rows)} alvos | executados={executed} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
