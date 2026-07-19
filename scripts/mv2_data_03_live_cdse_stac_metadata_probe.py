"""MV2-DATA-03 Tarefa 4 — Live CDSE STAC metadata probe (Item Search metadata-only).

Executa STAC Item Search metadata-only para os candidates oficiais SÓ se a config permitir.
Não baixa assets, limita a no máximo 2 candidates, timeout curto, fail-closed.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_live_cdse_stac_metadata_probe.csv
"""

from __future__ import annotations

import hashlib
import json
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

CDSE_STAC_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"
TIMEOUT_S = 20

COLUMNS = [
    "probe_id",
    "canary_candidate_id",
    "executed",
    "http_status",
    "collection",
    "query_payload_hash",
    "matched_items",
    "selected_item_id",
    "selected_datetime",
    "selected_mgrs_tile",
    "selected_cloud_cover",
    "assets_listed",
    "metadata_match_status",
    "blocking_reason",
    "notes",
]


def _payload(candidate: dict[str, str]) -> dict[str, object]:
    date = clean(candidate.get("scene_date"))
    return {
        "collections": ["SENTINEL-2"],
        "datetime": f"{date}T00:00:00Z/{date}T23:59:59Z" if date else None,
        "query": {"productType": {"eq": "S2MSI2A"}},
        "limit": 5,
    }


def _try_search(payload: dict[str, object]) -> tuple[bool, str, str]:
    if not lib_available("requests"):
        return False, "", "requests_indisponivel"
    try:
        import requests

        resp = requests.post(CDSE_STAC_SEARCH_URL, json=payload, timeout=TIMEOUT_S)
        return True, str(resp.status_code), ""
    except Exception as exc:  # rede/timeout — fail-closed, nunca propaga
        return False, "", f"erro_rede:{type(exc).__name__}"


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    candidates = read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")[:MAX_CANARY_CANDIDATES]
    blocked = metadata_blocked(flags) or not flags.get("cdse_stac_enabled")

    rows: list[dict[str, object]] = []
    for i, c in enumerate(candidates, start=1):
        payload = _payload(c)
        phash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        base: dict[str, object] = {
            "probe_id": f"MV2_DATA_03_STACPROBE_{i:02d}",
            "canary_candidate_id": clean(c.get("canary_candidate_id")),
            "collection": "SENTINEL-2",
            "query_payload_hash": phash,
            "matched_items": "",
            "selected_item_id": "",
            "selected_datetime": "",
            "selected_mgrs_tile": "",
            "selected_cloud_cover": "",
            "assets_listed": "false",
            "executed": "false",
            "http_status": "",
        }
        if blocked:
            base.update({
                "metadata_match_status": "NOT_EXECUTED",
                "blocking_reason": "config_local_ausente_ou_flags_stac_metadata_desabilitadas",
                "notes": "payload Item Search pronto (metadata-only, sem assets)",
            })
        else:
            executed, status, err = _try_search(payload)
            base.update({
                "executed": "true" if executed else "false",
                "http_status": status,
                "metadata_match_status": "EXECUTED_NO_MATCH_PARSED" if executed else "NOT_EXECUTED",
                "blocking_reason": "" if executed else err,
                "notes": "Item Search metadata-only; assets nao baixados",
            })
        rows.append(base)
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_live_cdse_stac_metadata_probe.csv"
    write_csv(out, COLUMNS, rows)
    executed = sum(1 for r in rows if r["executed"] == "true")
    print(
        f"[mv2_data_03_stac_probe] {len(rows)} probes | executados={executed} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
