"""MV2-DATA-02 Tarefa 6 — Plano de raster canary (1 a 3 targets, fail-closed).

Monta um plano de download/crop canary para no máximo 3 alvos com lineage FORTE, preferindo
âncoras com scene_id documentado no registry oficial (e, se houver, resoluções API fortes).
Se nenhum alvo forte existir, gera um plano vazio bloqueado. NÃO baixa raster aqui.

Saída: outputs_public/mv2_data_api_raster_harness/mv2_data_02_raster_canary_plan.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_02_api_config import load_config
from mv2_data_02_common import (
    CANARY_BANDS,
    CANARY_SCL,
    PROJECT_ROOT,
    PUBLIC_DIR,
    clean,
    ensure_public_dir,
    load_official_strong_anchors,
    read_csv_rows,
    write_csv,
)

MAX_CANARY = 3
# Estimativa conservadora de um crop multibanda pequeno (6 bandas + SCL, ~512px @ 10-20m).
ESTIMATED_CROP_MB = 18

COLUMNS = [
    "canary_id",
    "target_id",
    "asset_id",
    "patch_id",
    "region",
    "product_id",
    "scene_id",
    "datetime",
    "bbox",
    "crs",
    "bands_requested",
    "scl_requested",
    "private_output_dir",
    "estimated_size_mb",
    "canary_allowed_by_data",
    "canary_allowed_by_config",
    "download_allowed_now",
    "crop_allowed_now",
    "blocking_reason",
    "next_action",
]


def _api_strong_candidates() -> list[dict[str, str]]:
    rows = read_csv_rows(PUBLIC_DIR / "mv2_data_02_api_lineage_resolution.csv")
    return [r for r in rows if clean(r.get("lineage_status")) == "API_LINEAGE_RESOLVED_STRONG"]


def build_rows() -> list[dict[str, object]]:
    cfg = load_config()
    private_dir = str(cfg.get("private_output_dir"))
    bands = ",".join(CANARY_BANDS)
    config_ok = bool(cfg.get("allow_canary_download")) and bool(cfg.get("allow_raster_download"))
    config_block = (
        "config_permite_canary"
        if config_ok
        else "config_bloqueia_canary(allow_canary_download/allow_raster_download=false)"
    )

    candidates: list[dict[str, object]] = []

    # Prioridade 1: âncoras oficiais com scene_id documentado (lineage forte de registry).
    for a in load_official_strong_anchors():
        if len(candidates) >= MAX_CANARY:
            break
        candidates.append(
            {
                "target_id": clean(a.get("reference_patch_id")),
                "asset_id": "",
                "patch_id": clean(a.get("reference_patch_id")),
                "region": clean(a.get("region")),
                "product_id": clean(a.get("scene_id_sanitized")),
                "scene_id": clean(a.get("scene_id_sanitized")),
                "datetime": clean(a.get("scene_date")),
                "bbox": "",  # AOI derivada de centro+tamanho na execução; nunca inventada aqui
                "crs": clean(a.get("crs")),
                "lineage_source": "OFFICIAL_REGISTRY_DOCUMENTED_SCENE",
                "aoi_note": "aoi_derivar_de_anchor_center_e_patch_size_m_na_execucao",
            }
        )

    # Prioridade 2: resoluções API fortes (nenhuma em modo fail-closed offline).
    for r in _api_strong_candidates():
        if len(candidates) >= MAX_CANARY:
            break
        candidates.append(
            {
                "target_id": clean(r.get("target_id")),
                "asset_id": clean(r.get("asset_id")),
                "patch_id": clean(r.get("patch_id")),
                "region": clean(r.get("region")),
                "product_id": clean(r.get("product_id")),
                "scene_id": clean(r.get("scene_id")),
                "datetime": clean(r.get("datetime")),
                "bbox": "",
                "crs": "",
                "lineage_source": "API_RESOLVED_STRONG",
                "aoi_note": "aoi_do_patch_target",
            }
        )

    rows: list[dict[str, object]] = []
    for i, c in enumerate(candidates, start=1):
        data_ok = bool(c["scene_id"]) and bool(c["datetime"]) and bool(c["crs"])
        rows.append(
            {
                "canary_id": f"MV2_DATA_02_CANARY_{i:02d}",
                "target_id": c["target_id"],
                "asset_id": c["asset_id"],
                "patch_id": c["patch_id"],
                "region": c["region"],
                "product_id": c["product_id"],
                "scene_id": c["scene_id"],
                "datetime": c["datetime"],
                "bbox": c["bbox"],
                "crs": c["crs"],
                "bands_requested": bands,
                "scl_requested": CANARY_SCL,
                "private_output_dir": private_dir,
                "estimated_size_mb": ESTIMATED_CROP_MB,
                "canary_allowed_by_data": "true" if data_ok else "false",
                "canary_allowed_by_config": "true" if config_ok else "false",
                # Fail-closed: nada é baixado/croppado por padrão.
                "download_allowed_now": "false",
                "crop_allowed_now": "false",
                "blocking_reason": (
                    config_block
                    if data_ok
                    else "lineage_insuficiente_para_canary"
                ),
                "next_action": (
                    "habilitar config canary + REV_P_ALLOW_RASTER_CANARY=YES p/ executor"
                    if data_ok
                    else "recuperar_scene_id_datetime_crs_antes_do_canary"
                ),
            }
        )
    return rows


def main() -> None:
    ensure_public_dir()
    rows = build_rows()
    out = PUBLIC_DIR / "mv2_data_02_raster_canary_plan.csv"
    write_csv(out, COLUMNS, rows)
    data_ok = sum(1 for r in rows if r["canary_allowed_by_data"] == "true")
    print(
        f"[mv2_data_02_canary_plan] {len(rows)} candidatos (data_ok={data_ok}) | "
        f"download_allowed_now=false | saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
