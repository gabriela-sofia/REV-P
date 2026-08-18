"""MV2-DATA-02 Tarefa 5 — Resolvedor unificado de lineage via API.

Cruza os resultados GEE / CDSE STAC / CDSE OData por target e promove APENAS metadados
consistentes. Não aceita fonte única fraca em conflito, não aceita datetime futuro, não
aceita match só por cidade/tile e não promove sem target_id/asset_id/patch_id.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_api_lineage_resolution.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import load_config
from mv2_data_02_common import (
    DEFAULT_S2_COLLECTION,
    PROJECT_ROOT,
    PUBLIC_DIR,
    classify_datetime,
    clean,
    crs_status,
    ensure_public_dir,
    load_targets,
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
    "provider_agreement",
    "lineage_status",
    "can_promote_to_stac_dry_run",
    "can_promote_to_raster_canary",
    "can_promote_to_bulk_download",
    "blocking_reason",
    "next_action",
]


def _index(name: str, key: str = "target_id") -> dict[str, dict[str, str]]:
    return {clean(r.get(key)): r for r in read_csv_rows(PUBLIC_DIR / name)}


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    targets = load_targets()
    gee = _index("mv2_data_02_gee_metadata_results.csv")
    stac = _index("mv2_data_02_cdse_stac_metadata_results.csv")
    odata = _index("mv2_data_02_cdse_odata_metadata_results.csv")

    rows: list[dict[str, object]] = []
    for i, t in enumerate(targets, start=1):
        tid = clean(t.get("target_id"))
        asset_id = clean(t.get("asset_id"))
        patch_id = clean(t.get("patch_id"))
        bbox = clean(t.get("bbox"))
        crs = clean(t.get("crs"))

        g, s, o = gee.get(tid, {}), stac.get(tid, {}), odata.get(tid, {})
        executed = [
            name for name, r in (("GEE", g), ("CDSE_STAC", s), ("CDSE_ODATA", o))
            if clean(r.get("executed")) == "true"
        ]

        # Coleta valores explícitos (scene_id) das fontes que executaram.
        scene_vals = {
            clean(g.get("selected_scene_id")),
            clean(s.get("scene_id")),
        }
        scene_vals.discard("")
        dt_vals = {clean(g.get("selected_datetime")), clean(s.get("datetime"))}
        dt_vals.discard("")

        scene_id = next(iter(scene_vals)) if len(scene_vals) == 1 else ""
        datetime_val = next(iter(dt_vals)) if len(dt_vals) == 1 else ""
        dt_future = bool(datetime_val) and classify_datetime(datetime_val[:10])[1] == "INVALID_FUTURE"

        agreement = "NONE"
        if not executed:
            status = "API_LINEAGE_NOT_EXECUTED"
            blocking = "nenhuma_fonte_executou_consulta_metadata_only"
            next_action = "habilitar_config_metadata_e_credencial_para_consultar"
        elif dt_future:
            status = "API_LINEAGE_INVALID"
            blocking = "datetime_futuro_nao_e_aquisicao"
            next_action = "descartar_datetime_futuro_e_reconsultar"
        elif len(scene_vals) > 1:
            status = "API_LINEAGE_CONFLICT"
            agreement = "CONFLICT"
            blocking = "scene_id_divergente_entre_fontes"
            next_action = "revisar_manualmente_qual_cena_corresponde_ao_patch"
        elif not scene_vals and not dt_vals:
            status = "API_LINEAGE_NOT_FOUND"
            blocking = "consulta_executou_mas_nenhum_metadado_explicito_retornado"
            next_action = "refinar_AOI_janela_ou_revisar_export_original"
        elif scene_id and datetime_val and len(executed) >= 2:
            status = "API_LINEAGE_RESOLVED_STRONG"
            agreement = "MULTI_PROVIDER_AGREE"
            blocking = ""
            next_action = "elegivel_para_stac_dry_run_e_canary_se_config_permitir"
        elif scene_id or datetime_val:
            status = "API_LINEAGE_RESOLVED_PARTIAL"
            agreement = "SINGLE_PROVIDER"
            blocking = "metadado_parcial_ou_fonte_unica"
            next_action = "confirmar_em_segunda_fonte_antes_de_promover"
        else:
            status = "API_LINEAGE_REVIEW_REQUIRED"
            blocking = "evidencia_insuficiente_para_promover"
            next_action = "revisao_humana"

        has_ids = bool(tid and asset_id and patch_id)
        bbox_crs_ok = bool(bbox) and crs_status(crs) != "ABSENT"

        can_stac_dry = (
            "true"
            if (has_ids and bbox_crs_ok and bool(datetime_val) and not dt_future)
            else "false"
        )
        can_canary = (
            "true"
            if (
                status == "API_LINEAGE_RESOLVED_STRONG"
                and has_ids
                and bool(cfg.get("allow_raster_download"))
                and bool(cfg.get("allow_canary_download"))
            )
            else "false"
        )
        rows.append(
            {
                "resolution_id": f"MV2_DATA_02_RES_{i:05d}",
                "target_id": tid,
                "asset_id": asset_id,
                "patch_id": patch_id,
                "region": clean(t.get("region")),
                "product_id": clean(g.get("selected_product_id")) or clean(o.get("product_id")),
                "scene_id": scene_id,
                "datetime": datetime_val,
                "mgrs_tile": clean(g.get("selected_mgrs_tile")) or clean(s.get("mgrs_tile")),
                "cloud_cover": clean(g.get("selected_cloudy_pixel_percentage")) or clean(s.get("cloud_cover")),
                "source_providers": "|".join(executed) or "none",
                "provider_agreement": agreement,
                "lineage_status": status,
                "can_promote_to_stac_dry_run": can_stac_dry,
                "can_promote_to_raster_canary": can_canary,
                # Bulk download JAMAIS neste marco.
                "can_promote_to_bulk_download": "false",
                "blocking_reason": blocking,
                "next_action": next_action,
                "_collection": DEFAULT_S2_COLLECTION,
            }
        )
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_api_lineage_resolution.csv"
    write_csv(out, COLUMNS, rows)
    from collections import Counter

    dist = Counter(str(r["lineage_status"]) for r in rows)
    print(
        f"[mv2_data_02_lineage] {len(rows)} | "
        + " ".join(f"{k.replace('API_LINEAGE_', '').lower()}={v}" for k, v in sorted(dist.items()))
        + f" | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
