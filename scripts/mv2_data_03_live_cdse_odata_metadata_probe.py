"""MV2-DATA-03 Tarefa 5 — Live CDSE OData metadata probe (resolução leve).

Resolve produto por product_id/scene_id SÓ se a config permitir. Não baixa arquivo; apenas
resolve metadados leves e a possível URL de download — NUNCA salvando URL com token.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_live_cdse_odata_metadata_probe.csv
"""

from __future__ import annotations

import re
from pathlib import Path

from mv2_data_03_common import (
    MAX_CANARY_CANDIDATES,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    effective_flags,
    ensure_public_dir,
    lib_available,
    metadata_blocked,
    read_csv_rows,
    write_csv,
)

CDSE_ODATA_PRODUCTS_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TIMEOUT_S = 20
TOKEN_RE = re.compile(r"(?i)(token|access_token|key|password)=[^&\s]+")

COLUMNS = [
    "probe_id",
    "canary_candidate_id",
    "executed",
    "http_status",
    "product_name",
    "product_id",
    "odata_id",
    "content_length",
    "online_status",
    "checksum_available",
    "download_url_resolved",
    "download_executed",
    "metadata_match_status",
    "blocking_reason",
    "notes",
]


def _sanitize_url(url: str) -> str:
    """Remove qualquer token/credencial de uma URL antes de registrar."""
    return TOKEN_RE.sub(r"\1=__REDACTED__", clean(url))


def _try_resolve(product_id: str) -> tuple[bool, str, str]:
    if not lib_available("requests"):
        return False, "", "requests_indisponivel"
    try:
        import requests

        flt = f"{CDSE_ODATA_PRODUCTS_URL}?$filter=Name eq '{product_id}'"
        resp = requests.get(flt, timeout=TIMEOUT_S)
        return True, str(resp.status_code), ""
    except Exception as exc:
        return False, "", f"erro_rede:{type(exc).__name__}"


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    candidates = read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")[:MAX_CANARY_CANDIDATES]
    blocked = metadata_blocked(flags) or not flags.get("cdse_odata_enabled")

    rows: list[dict[str, object]] = []
    for i, c in enumerate(candidates, start=1):
        product_id = clean(c.get("product_id"))
        base: dict[str, object] = {
            "probe_id": f"MV2_DATA_03_ODATAPROBE_{i:02d}",
            "canary_candidate_id": clean(c.get("canary_candidate_id")),
            "product_name": "",
            "product_id": product_id,
            "odata_id": "",
            "content_length": "",
            "online_status": "",
            "checksum_available": "",
            "download_url_resolved": "",
            "download_executed": "false",  # OData probe NUNCA baixa
            "executed": "false",
            "http_status": "",
        }
        if blocked:
            base.update({
                "metadata_match_status": "NOT_EXECUTED",
                "blocking_reason": "config_local_ausente_ou_flags_odata_metadata_desabilitadas",
                "notes": f"filtro OData pronto p/ '{product_id}' (sem download, sem token salvo)",
            })
        else:
            executed, status, err = _try_resolve(product_id)
            base.update({
                "executed": "true" if executed else "false",
                "http_status": status,
                "metadata_match_status": "EXECUTED_NO_PARSE" if executed else "NOT_EXECUTED",
                "blocking_reason": "" if executed else err,
                "notes": "resolucao leve; URL sanitizada sem token; arquivo nunca baixado aqui",
            })
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_live_cdse_odata_metadata_probe.csv"
    # Defesa em profundidade: nenhuma URL registrada pode conter token.
    for r in rows:
        r["download_url_resolved"] = _sanitize_url(str(r.get("download_url_resolved", "")))
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    downloads = sum(1 for r in rows if r["download_executed"] == "true")
    print(
        f"[mv2_data_03_odata_probe] {len(rows)} probes | executados={executed} downloads={downloads} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
