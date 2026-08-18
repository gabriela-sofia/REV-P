"""MV2-DATA-04 Tarefa 6 — Resolver lineage de corpus por API.

Cruza os probes GEE/STAC/OData por target do batch e classifica o status de lineage. Sem
janela temporal, não promove (BLOCKED_NO_TEMPORAL_WINDOW). Sem product_id, OData não promove.
can_unlock_day10_now=false sempre. Nunca usa o canary Protocolo-C.

Saída: outputs_public/mv2_data_corpus_metadata_probe/mv2_data_04_corpus_api_lineage_resolution.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_04_common import (
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    crs_status,
    ensure_public_dir,
    load_batch,
    read_csv_rows,
    write_csv,
)

COLUMNS = [
    "resolution_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "product_id",
    "scene_id",
    "datetime",
    "mgrs_tile",
    "cloud_cover",
    "source_providers",
    "lineage_status",
    "can_stac_dry_run",
    "can_raster_download_next_step",
    "can_unlock_day10_now",
    "blocking_reason",
    "next_action",
]


def _index(name: str) -> dict[str, dict[str, str]]:
    return {clean(r.get("target_id")): r for r in read_csv_rows(PUBLIC_DIR / name)}


def build_rows() -> list[dict[str, object]]:
    batch = load_batch()
    gee = _index("mv2_data_04_gee_corpus_metadata_probe.csv")
    stac = _index("mv2_data_04_cdse_stac_corpus_metadata_probe.csv")
    odata = _index("mv2_data_04_cdse_odata_metadata_probe.csv")

    rows: list[dict[str, object]] = []
    for i, t in enumerate(batch, start=1):
        tid = clean(t.get("target_id"))
        g, s, o = gee.get(tid, {}), stac.get(tid, {}), odata.get(tid, {})

        executed = [n for n, r in (("GEE", g), ("CDSE_STAC", s), ("CDSE_ODATA", o))
                    if clean(r.get("executed")) == "true"]
        gee_status = clean(g.get("metadata_status"))
        stac_status = clean(s.get("metadata_status"))

        scene_vals = {clean(g.get("selected_product_id")), clean(s.get("selected_item_id"))}
        scene_vals.discard("")
        dt_vals = {clean(g.get("selected_datetime")), clean(s.get("selected_datetime"))}
        dt_vals.discard("")
        scene_id = next(iter(scene_vals)) if len(scene_vals) == 1 else ""
        datetime_val = next(iter(dt_vals)) if len(dt_vals) == 1 else ""

        no_temporal = (gee_status == "BLOCKED_NO_TEMPORAL_WINDOW"
                       and stac_status == "BLOCKED_NO_TEMPORAL_WINDOW")

        if no_temporal:
            status = "CORPUS_LINEAGE_BLOCKED_NO_TEMPORAL_WINDOW"
            blocking = "sem_janela_temporal_por_target"
            nxt = "preencher_temporal_window_template_a_partir_de_evento_ou_export"
        elif not executed:
            status = "CORPUS_LINEAGE_NOT_EXECUTED"
            blocking = "probes_nao_executaram_config_bloqueada"
            nxt = "habilitar_config_metadata_e_credencial"
        elif len(scene_vals) > 1:
            status = "CORPUS_LINEAGE_CONFLICT"
            blocking = "scene_id_divergente_entre_provedores"
            nxt = "revisao_humana_de_qual_cena_corresponde"
        elif scene_id and datetime_val and len(executed) >= 2:
            status = "CORPUS_LINEAGE_RESOLVED_STRONG"
            blocking = ""
            nxt = "registrar_lineage_e_planejar_proximo_marco_de_raster_controlado"
        elif scene_id or datetime_val:
            status = "CORPUS_LINEAGE_RESOLVED_PARTIAL"
            blocking = "metadado_parcial_ou_fonte_unica"
            nxt = "confirmar_em_segunda_fonte"
        else:
            status = "CORPUS_LINEAGE_REVIEW_REQUIRED"
            blocking = "executou_mas_sem_metadado_explicito"
            nxt = "revisao_humana"

        bbox_crs_ok = bool(clean(t.get("bbox"))) and crs_status(clean(t.get("crs"))) != "ABSENT"
        can_stac_dry = "true" if (bbox_crs_ok and bool(datetime_val) and not no_temporal
                                  and status.startswith("CORPUS_LINEAGE_RESOLVED")) else "false"
        rows.append({
            "resolution_id": f"MV2_DATA_04_RES_{i:02d}",
            "target_id": tid,
            "asset_id": clean(t.get("asset_id")),
            "patch_id": clean(t.get("patch_id")),
            "region": clean(t.get("region")),
            "product_id": clean(g.get("selected_product_id")) or clean(o.get("product_id")),
            "scene_id": scene_id,
            "datetime": datetime_val,
            "mgrs_tile": clean(g.get("selected_mgrs_tile")) or clean(s.get("selected_mgrs_tile")),
            "cloud_cover": clean(g.get("selected_cloudy_pixel_percentage")) or clean(s.get("selected_cloud_cover")),
            "source_providers": "|".join(executed) or "none",
            "lineage_status": status,
            "can_stac_dry_run": can_stac_dry,
            # marco metadata-only: raster sempre como passo futuro bloqueado
            "can_raster_download_next_step": "false",
            "can_unlock_day10_now": "false",
            "blocking_reason": blocking,
            "next_action": nxt,
        })
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_04_corpus_api_lineage_resolution.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["lineage_status"]) for r in rows)
    print(
        f"[mv2_data_04_lineage] {len(rows)} | "
        + " ".join(f"{k.replace('CORPUS_LINEAGE_', '')}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
