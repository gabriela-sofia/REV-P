"""MV2-DATA-03 Tarefa 6 — Resolver consenso de metadata canary.

Cruza registry oficial + probes GEE/STAC/OData e classifica o consenso. Não exige que todos
os providers executem: se só o registry oficial tem lineage, status REGISTRY_ONLY_STRONG; se
a API confirma, API_CONFIRMED_STRONG; se diverge, CONFLICT.

Saída: outputs_public/mv2_data_live_api_canary/mv2_data_03_canary_metadata_consensus.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_03_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    classify_datetime,
    clean,
    crs_status,
    effective_flags,
    ensure_public_dir,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "consensus_id",
    "canary_candidate_id",
    "scene_id",
    "product_id",
    "datetime",
    "mgrs_tile",
    "cloud_cover",
    "registry_status",
    "gee_status",
    "stac_status",
    "odata_status",
    "provider_agreement",
    "consensus_status",
    "canary_download_data_ready",
    "canary_download_config_ready",
    "canary_download_allowed",
    "can_use_for_corpus_day10",
    "blocking_reason",
    "next_action",
]


def _by_candidate(name: str) -> dict[str, dict[str, str]]:
    return {clean(r.get("canary_candidate_id")): r for r in read_csv_rows(PUBLIC_DIR / name)}


def build_rows() -> list[dict[str, object]]:
    flags = effective_flags()
    candidates = read_csv_rows(PUBLIC_DIR / "mv2_data_03_official_canary_candidates.csv")
    gee = _by_candidate("mv2_data_03_live_gee_metadata_probe.csv")
    stac = _by_candidate("mv2_data_03_live_cdse_stac_metadata_probe.csv")
    odata = _by_candidate("mv2_data_03_live_cdse_odata_metadata_probe.csv")

    config_ready = bool(flags.get("allow_network")) and bool(flags.get("allow_raster_download")) and bool(flags.get("allow_canary"))

    rows: list[dict[str, object]] = []
    for i, c in enumerate(candidates, start=1):
        cid = clean(c.get("canary_candidate_id"))
        scene = clean(c.get("scene_id"))
        scene_date = clean(c.get("scene_date"))
        crs = clean(c.get("crs"))
        collection = clean(c.get("gee_collection"))

        # Registry forte: scene_id granule + data válida não-futura + CRS + collection.
        registry_strong = (
            bool(scene)
            and classify_datetime(scene_date[:10])[1] == "VALID_SINGLE"
            and crs_status(crs) != "ABSENT"
            and bool(collection)
        )
        registry_status = "STRONG" if registry_strong else "WEAK"

        def pstatus(d: dict[str, str]) -> str:
            return clean(d.get("metadata_match_status")) or "NOT_EXECUTED"

        gee_st = pstatus(gee.get(cid, {}))
        stac_st = pstatus(stac.get(cid, {}))
        odata_st = pstatus(odata.get(cid, {}))
        api_executed = any(
            clean(d.get(cid, {}).get("executed")) == "true" for d in (gee, stac, odata)
        )

        # scene_id divergente entre API e registry -> CONFLICT (nenhum diverge no modo offline).
        api_scene_vals = {
            clean(gee.get(cid, {}).get("selected_product_id")),
            clean(stac.get(cid, {}).get("selected_item_id")),
        }
        api_scene_vals.discard("")
        conflict = bool(api_scene_vals) and scene and api_scene_vals != {scene}

        if not registry_strong and not api_executed:
            consensus = "NOT_EXECUTED"
            agreement = "NONE"
            blocking = "sem_lineage_registry_e_sem_api"
        elif conflict:
            consensus = "CONFLICT"
            agreement = "CONFLICT"
            blocking = "scene_id_api_diverge_do_registry"
        elif registry_strong and api_executed:
            consensus = "API_CONFIRMED_STRONG"
            agreement = "REGISTRY_PLUS_API"
            blocking = ""
        elif registry_strong:
            consensus = "REGISTRY_ONLY_STRONG"
            agreement = "REGISTRY_ONLY"
            blocking = "api_nao_executada_lineage_so_do_registry_oficial"
        else:
            consensus = "PARTIAL_CONFIRMATION"
            agreement = "PARTIAL"
            blocking = "metadado_parcial"

        data_ready = consensus in {"API_CONFIRMED_STRONG", "REGISTRY_ONLY_STRONG"}
        download_allowed = data_ready and config_ready

        rows.append(
            {
                "consensus_id": f"MV2_DATA_03_CONS_{i:02d}",
                "canary_candidate_id": cid,
                "scene_id": scene,
                "product_id": clean(c.get("product_id")),
                "datetime": scene_date,
                "mgrs_tile": clean(c.get("mgrs_tile")),
                "cloud_cover": clean(c.get("cloud_cover")),
                "registry_status": registry_status,
                "gee_status": gee_st,
                "stac_status": stac_st,
                "odata_status": odata_st,
                "provider_agreement": agreement,
                "consensus_status": consensus,
                "canary_download_data_ready": "true" if data_ready else "false",
                "canary_download_config_ready": "true" if config_ready else "false",
                "canary_download_allowed": "true" if download_allowed else "false",
                # Regra absoluta deste marco.
                "can_use_for_corpus_day10": "false",
                "blocking_reason": blocking or ("config_bloqueia_download" if not config_ready else ""),
                "next_action": (
                    "habilitar config+env e rodar executor canary (1 max)"
                    if data_ready and not download_allowed
                    else ("executar 1 canary privado" if download_allowed else "recuperar lineage")
                ),
            }
        )
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_03_canary_metadata_consensus.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["consensus_status"]) for r in rows)
    print(
        f"[mv2_data_03_consensus] {len(rows)} | "
        + " ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
