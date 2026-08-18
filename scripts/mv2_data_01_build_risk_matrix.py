"""MV2-DATA-01 Tarefa 7 — Matriz de risco DATA-01.

Registra os riscos metodológicos do pacote de recuperação de lineage: riscos intrínsecos
por target (patch sem cena, auto-join de tile, proxy cidade/região) e riscos de guardrail
globais (download sem binding, STAC antes de lineage, falso desbloqueio do Dia 10, etc.).
Garante que todos os RISK_TYPES obrigatórios estejam presentes.

Saída: outputs_public/mv2_data_gee_lineage_targets/mv2_data_01_risk_matrix.csv
"""

from __future__ import annotations

from pathlib import Path

from mv2_data_01_common import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    REGION_ZONE_HINT,
    RISK_TYPES,
    build_targets,
    clean,
    ensure_output_dir,
    is_spatial_ready,
    write_csv,
)

COLUMNS = [
    "risk_id",
    "target_id",
    "asset_id",
    "patch_id",
    "severity",
    "risk_type",
    "evidence",
    "mitigation",
    "blocks_next_step",
]

# Riscos globais de guardrail (target_id=GLOBAL). (severity, evidence, mitigation, blocks)
GLOBAL_RISKS = {
    "RISK_SCENE_WITHOUT_PATCH": (
        "MEDIA",
        "uma cena recuperada poderia ser anexada sem patch correspondente",
        "exigir patch_id valido antes de aceitar qualquer scene_id",
        "true",
    ),
    "RISK_FILENAME_INFERENCE": (
        "ALTA",
        "nome de arquivo de patch/asset pode sugerir cena/data/tile",
        "proibir inferencia de lineage por filename; so evidencia GEE explicita",
        "true",
    ),
    "RISK_FUTURE_DATETIME": (
        "ALTA",
        "datetime de processamento/futuro pode ser confundido com aquisicao",
        "rejeitar datetime fora de 2015..2024 (classify_datetime INVALID_FUTURE)",
        "true",
    ),
    "RISK_GEE_METADATA_MISSING": (
        "MEDIA",
        "export original pode nao existir mais ou nao ser localizavel",
        "marcar target como nao recuperavel sem inventar metadados",
        "true",
    ),
    "RISK_STAC_BEFORE_LINEAGE": (
        "ALTA",
        "executar STAC/OData real antes de ter scene_id/datetime confiavel",
        "manter would_execute_http=false ate lineage forte revisado",
        "true",
    ),
    "RISK_RASTER_DOWNLOAD_WITHOUT_BINDING": (
        "ALTA",
        "baixar Sentinel sem binding cena-patch confirmado",
        "nenhum download neste marco; downloads_executed deve ser 0",
        "true",
    ),
    "RISK_DAY10_FALSE_UNLOCK": (
        "ALTA",
        "tratar pacote de planos como se o Dia 10 estivesse desbloqueado",
        "can_unlock_day10_now=false; so raster nativo validado desbloqueia",
        "true",
    ),
    "RISK_PNG_AS_RASTER": (
        "ALTA",
        "PNG/NPZ/render/DINOv2 tratado como raster espectral nativo",
        "nenhum raster nativo criado; native_rasters_created deve ser 0",
        "true",
    ),
    "RISK_UNKNOWN_AS_NEGATIVE": (
        "ALTA",
        "ausencia de lineage interpretada como evidencia negativa",
        "UNKNOWN/NOT_FOUND nunca vira negativo nem label",
        "true",
    ),
}


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    n = 0

    # 1) Riscos intrínsecos por target.
    for t in build_targets():
        tid = str(t["target_id"])
        region = clean(t["region"])
        zone_crs, tile_hint = REGION_ZONE_HINT.get(region, ("", ""))

        # RISK_PATCH_WITHOUT_SCENE: tem patch, falta cena (todos os alvos prontos).
        if clean(t["has_patch_id"]) == "true" and clean(t["has_scene_id"]) != "true":
            n += 1
            rows.append(
                {
                    "risk_id": f"MV2_DATA_01_RISK_{n:05d}",
                    "target_id": tid,
                    "asset_id": t["asset_id"],
                    "patch_id": t["patch_id"],
                    "severity": "MEDIA",
                    "risk_type": "RISK_PATCH_WITHOUT_SCENE",
                    "evidence": "patch_id presente, scene_id ausente (lineage nao recuperado)",
                    "mitigation": "recuperar scene_id via revisao GEE metadata-only; nao inferir",
                    "blocks_next_step": "true",
                }
            )

        # RISK_TILE_AUTO_JOIN: zona UTM conhecida -> tentacao de join por tile.
        if zone_crs:
            n += 1
            rows.append(
                {
                    "risk_id": f"MV2_DATA_01_RISK_{n:05d}",
                    "target_id": tid,
                    "asset_id": t["asset_id"],
                    "patch_id": t["patch_id"],
                    "severity": "ALTA",
                    "risk_type": "RISK_TILE_AUTO_JOIN",
                    "evidence": (
                        f"regiao {region} usa {zone_crs} (~MGRS {tile_hint}); "
                        "tile NAO pode ser preenchido por zona"
                    ),
                    "mitigation": "tile so por evidencia GEE da imagem; nunca T23KPR auto",
                    "blocks_next_step": "true",
                }
            )

        # RISK_CITY_REGION_PROXY: cidade/região não estabelece lineage.
        if region:
            n += 1
            rows.append(
                {
                    "risk_id": f"MV2_DATA_01_RISK_{n:05d}",
                    "target_id": tid,
                    "asset_id": t["asset_id"],
                    "patch_id": t["patch_id"],
                    "severity": "ALTA",
                    "risk_type": "RISK_CITY_REGION_PROXY",
                    "evidence": f"regiao {region} conhecida; nao basta para lineage forte",
                    "mitigation": "cidade/regiao nunca vira vinculo cena-patch nem label",
                    "blocks_next_step": (
                        "true" if not is_spatial_ready(t) else "false"
                    ),
                }
            )

    # 2) Riscos globais de guardrail (um por tipo, target_id=GLOBAL).
    for risk_type, (severity, evidence, mitigation, blocks) in GLOBAL_RISKS.items():
        n += 1
        rows.append(
            {
                "risk_id": f"MV2_DATA_01_RISK_{n:05d}",
                "target_id": "GLOBAL",
                "asset_id": "",
                "patch_id": "",
                "severity": severity,
                "risk_type": risk_type,
                "evidence": evidence,
                "mitigation": mitigation,
                "blocks_next_step": blocks,
            }
        )
    return rows


def main() -> None:
    ensure_output_dir()
    rows = build_rows()
    out = OUTPUT_DIR / "mv2_data_01_risk_matrix.csv"
    write_csv(out, COLUMNS, rows)
    present = {str(r["risk_type"]) for r in rows}
    missing = [rt for rt in RISK_TYPES if rt not in present]
    assert not missing, f"risk_types obrigatorios ausentes: {missing}"
    print(
        f"[mv2_data_01_risk] {len(rows)} riscos | tipos={len(present)}/{len(RISK_TYPES)} | "
        f"saida: {Path(out).relative_to(PROJECT_ROOT).as_posix()}"
    )


if __name__ == "__main__":
    main()
