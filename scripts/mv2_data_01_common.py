"""Núcleo compartilhado do marco MV2-DATA-01 — GEE Lineage Target Pack (Vertente B).

Transforma os 128 bindings MEDIUM identificados por MV2-13/MV2-14 numa fila objetiva e
validada de recuperação de lineage Sentinel/GEE *metadata-only*: alvos, template manual,
planos de consulta (GEE metadata / STAC / OData), checklist e matriz de risco.

Esta é a primeira infraestrutura programática da Vertente B (desbloqueio científico de
dados). Ela NÃO baixa raster, NÃO executa GEE/STAC/OData real, NÃO cria crop, NÃO calcula
feature espectral, NÃO cria label/silver/gold/negativo e NÃO desbloqueia o Dia 10.

Fail-closed:
- scene_id / datetime / tile / cloud_cover nunca são inferidos; só entram com evidência
  explícita já recuperada na matriz de lineage MV2-14;
- datetime futuro nunca é aceito como válido;
- cidade/região/zona UTM sozinhas nunca estabelecem lineage (no máximo geram hipótese de
  revisão manual);
- nenhuma cena T23KPR (ou qualquer tile MGRS) é auto-vinculada a um patch;
- toda consulta é PLANO; would_call_gee / would_execute_http / would_download = false.

Reusa os utilitários de IO e classificadores do MV2-13 (uma só fonte de verdade).
"""

from __future__ import annotations

from pathlib import Path

from mv2_13_sentinel_binding_common import (  # noqa: F401  (reuso/reexport intencional)
    PROJECT_ROOT,
    _branch_of,
    _worktree_roots,
    classify_datetime,
    clean,
    crs_status,
    read_csv_rows,
    write_csv,
    write_json,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs_public" / "mv2_data_gee_lineage_targets"
SCHEMA_DIR = PROJECT_ROOT / "datasets" / "schemas"

BRANCH_EXECUTADA = "marco/validacao-label-free-evidencia-estrutural-mv1"
BRANCH_ESPERADA = "marco/mv2-data-vertente-b-desbloqueio-dados"

# Entradas obrigatórias (matriz de bindings MEDIUM do MV2-13/MV2-14). Resolvidas na
# working tree LOCAL ou, se ausentes, read-only em outro worktree (sem checkout).
DATA_INPUT_RELPATHS = {
    "medium_binding_index": "outputs_public/mv2_gee_lineage_recovery/mv2_14_medium_binding_index.csv",
    "lineage_binding_matrix": "outputs_public/mv2_gee_lineage_recovery/mv2_14_lineage_binding_matrix.csv",
    "mv2_13_binding_matrix": "outputs_public/mv2_sentinel_binding/mv2_13_binding_matrix.csv",
}

# ---------------------------------------------------------------------------
# Taxonomias obrigatórias
# ---------------------------------------------------------------------------

TARGET_STATUSES = {
    "TARGET_READY_FOR_MANUAL_GEE_LOOKUP",
    "TARGET_READY_FOR_METADATA_QUERY_PLAN",
    "TARGET_BLOCKED_MISSING_PATCH",
    "TARGET_BLOCKED_MISSING_ASSET",
    "TARGET_BLOCKED_MISSING_GEOMETRY",
    "TARGET_BLOCKED_MISSING_CRS",
    "TARGET_INVALID",
}

PRIORITIES = {
    "P0_HIGH_SENSOR_CONFIRMED_QUERY",
    "P1_MEDIUM_LOCAL_CANDIDATE_REVIEW",
    "P2_LOW_NO_LOCAL_SCENE_QUERY",
    "P3_BLOCKED_INCOMPLETE",
}

RISK_TYPES = [
    "RISK_SCENE_WITHOUT_PATCH",
    "RISK_PATCH_WITHOUT_SCENE",
    "RISK_TILE_AUTO_JOIN",
    "RISK_CITY_REGION_PROXY",
    "RISK_FILENAME_INFERENCE",
    "RISK_FUTURE_DATETIME",
    "RISK_GEE_METADATA_MISSING",
    "RISK_STAC_BEFORE_LINEAGE",
    "RISK_RASTER_DOWNLOAD_WITHOUT_BINDING",
    "RISK_DAY10_FALSE_UNLOCK",
    "RISK_PNG_AS_RASTER",
    "RISK_UNKNOWN_AS_NEGATIVE",
]

# Campos de metadados a coletar numa consulta metadata-only (nunca executada aqui).
METADATA_FIELDS = [
    "PRODUCT_ID",
    "MGRS_TILE",
    "system:time_start",
    "CLOUDY_PIXEL_PERCENTAGE",
    "CLOUD_COVERAGE_ASSESSMENT",
    "NODATA_PIXEL_PERCENTAGE",
    "DATATAKE_IDENTIFIER",
    "GENERATION_TIME",
    "SPACECRAFT_NAME",
]

# Coleção padrão Sentinel-2 (referência de consulta; nada é baixado).
DEFAULT_S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"

STAC_PROVIDERS = ["CDSE_STAC", "CDSE_ODATA", "GEE_METADATA"]

# Janela de consulta é um LIMITE de arquivo Sentinel-2, NÃO uma data de aquisição do
# alvo. Mantida histórica (sem futuro): início ~lançamento S2A, fim em 2024-12-31.
S2_ARCHIVE_WINDOW_START = "2015-06-23"
S2_ARCHIVE_WINDOW_END = "2024-12-31"

# Mapa zona UTM -> prefixo MGRS por região. APENAS para registrar o risco de auto-join;
# NUNCA é usado para preencher tile de um alvo.
REGION_ZONE_HINT = {
    "Curitiba": ("EPSG:32722", "22J"),
    "Petropolis": ("EPSG:32723", "23K"),
    "Recife": ("EPSG:32725", "25M"),
}

_WORKTREE_ROOTS_CACHE: list[Path] | None = None


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _worktree_roots_cached() -> list[Path]:
    global _WORKTREE_ROOTS_CACHE
    if _WORKTREE_ROOTS_CACHE is None:
        _WORKTREE_ROOTS_CACHE = _worktree_roots()
    return _WORKTREE_ROOTS_CACHE


def resolve_data_input(key: str) -> tuple[Path | None, str]:
    """Resolve uma entrada por chave: (path_absoluto_ou_None, origem).

    origem = "LOCAL" na working tree atual ou
    "READ_ONLY_FROM_OTHER_WORKTREE:<branch>" se vier de outro worktree.
    """
    rel = DATA_INPUT_RELPATHS[key]
    local = PROJECT_ROOT / rel
    if local.exists():
        return local, "LOCAL"
    for root in _worktree_roots_cached():
        if root == PROJECT_ROOT:
            continue
        cand = root / rel
        if cand.exists():
            return cand, f"READ_ONLY_FROM_OTHER_WORKTREE:{_branch_of(root)}"
    return None, "MISSING"


# ---------------------------------------------------------------------------
# Núcleo: derivação dos alvos de lineage (fonte única para todos os scripts)
# ---------------------------------------------------------------------------


def _bbox_is_valid(bbox: str) -> bool:
    """bbox precisa ter 4 números reais separados por vírgula; sem isso é inválido."""
    parts = [p.strip() for p in clean(bbox).split(",")]
    if len(parts) != 4:
        return False
    try:
        [float(p) for p in parts]
    except ValueError:
        return False
    return True


def _sensor_confirmed(sensor: str, platform: str) -> bool:
    s = clean(sensor).upper()
    p = clean(platform).upper()
    return s in {"S2_PROBABLE", "S2_MSI"} or p.startswith("SENTINEL-2")


def build_targets() -> list[dict[str, object]]:
    """Constrói os alvos de lineage a partir do índice de bindings MEDIUM (MV2-14).

    Preserva asset_id/patch_id/region/bbox/CRS/sensor/status; classifica prioridade por
    região e completude; NUNCA inventa scene_id/datetime/tile/cloud_cover (só aceita
    evidência já recuperada na matriz de lineage, e datetime futuro nunca é válido).
    """
    src_path, src_origin = resolve_data_input("medium_binding_index")
    rows = read_csv_rows(src_path)

    lineage_path, _ = resolve_data_input("lineage_binding_matrix")
    lineage = {clean(r.get("target_id")): r for r in read_csv_rows(lineage_path)}

    targets: list[dict[str, object]] = []
    for i, r in enumerate(rows, start=1):
        asset_id = clean(r.get("asset_id"))
        patch_id = clean(r.get("patch_id"))
        bbox = clean(r.get("bbox"))
        crs = clean(r.get("crs"))
        region = clean(r.get("region"))
        sensor = clean(r.get("sensor")) or "UNKNOWN"
        platform = clean(r.get("platform")) or "UNKNOWN"
        binding_id = clean(r.get("binding_id"))
        anchor_id = clean(r.get("anchor_id"))
        src_target_id = clean(r.get("target_id"))
        src_priority = clean(r.get("recovery_priority"))

        lm = lineage.get(src_target_id, {})
        # Lineage só conta se houver evidência explícita já recuperada no MV2-14.
        rec_scene = clean(lm.get("recovered_scene_id"))
        rec_dt = clean(lm.get("recovered_datetime"))
        rec_tile = clean(lm.get("recovered_tile"))
        rec_cloud = clean(lm.get("recovered_cloud_cover"))
        lineage_status_src = clean(lm.get("lineage_status")) or "LINEAGE_NOT_FOUND"

        has_asset = bool(asset_id)
        has_patch = bool(patch_id)
        bbox_valid = _bbox_is_valid(bbox)
        has_bbox = bbox_valid
        crs_present = crs_status(crs) in {"PRESENT_VALID", "PRESENT_UNVERIFIED"}
        has_crs = crs_present

        has_scene_id = bool(rec_scene)
        # datetime futuro nunca é aceito como presente/válido.
        dt_ok = bool(rec_dt) and classify_datetime(rec_dt[:10])[1] == "VALID_SINGLE"
        has_datetime = dt_ok
        has_tile = bool(rec_tile)
        has_cloud_cover = bool(rec_cloud)

        # bbox presente porém malformado é contradição estrutural -> INVALID.
        bbox_present_but_bad = bool(bbox) and not bbox_valid

        if bbox_present_but_bad:
            status = "TARGET_INVALID"
            blocking = "bbox_presente_mas_malformado;geometria_nao_confiavel"
            next_action = "corrigir_geometria_do_patch_antes_de_qualquer_consulta"
            priority = "P3_BLOCKED_INCOMPLETE"
        elif not has_patch:
            status = "TARGET_BLOCKED_MISSING_PATCH"
            blocking = "sem_patch_id_para_AOI_e_overlay"
            next_action = "recuperar_patch_id_oficial_antes_de_planejar_consulta"
            priority = "P3_BLOCKED_INCOMPLETE"
        elif not has_asset:
            status = "TARGET_BLOCKED_MISSING_ASSET"
            blocking = "sem_asset_id_oficial"
            next_action = "recuperar_asset_id_oficial_antes_de_planejar_consulta"
            priority = "P3_BLOCKED_INCOMPLETE"
        elif not has_bbox:
            status = "TARGET_BLOCKED_MISSING_GEOMETRY"
            blocking = "sem_bbox_para_AOI"
            next_action = "recuperar_geometria_bbox_do_patch_antes_de_planejar_consulta"
            priority = "P3_BLOCKED_INCOMPLETE"
        elif not has_crs:
            status = "TARGET_BLOCKED_MISSING_CRS"
            blocking = "sem_CRS_para_reprojetar_AOI"
            next_action = "recuperar_CRS_do_patch_antes_de_planejar_consulta"
            priority = "P3_BLOCKED_INCOMPLETE"
        else:
            # Lado espacial pronto. Faltam scene_id/datetime/tile/cloud e proveniência.
            blocking = (
                "lado_espacial_pronto;faltam_scene_id_datetime_tile_cloud_cover_"
                "e_proveniencia_de_export_GEE"
            )
            local_candidate = (
                lineage_status_src == "LINEAGE_CANDIDATE_REVIEW"
                or src_priority.startswith("P1")
            )
            if local_candidate:
                status = "TARGET_READY_FOR_MANUAL_GEE_LOOKUP"
                priority = "P1_MEDIUM_LOCAL_CANDIDATE_REVIEW"
                next_action = (
                    "revisar_cenas_candidatas_locais_e_historico_de_export_GEE_deste_"
                    "patch_sem_inferir_tile_nem_cidade"
                )
            else:
                status = "TARGET_READY_FOR_METADATA_QUERY_PLAN"
                next_action = (
                    "executar_plano_metadata_only_GEE_STAC_por_AOI_e_janela_para_"
                    "recuperar_scene_id_datetime_tile_cloud_sem_baixar_raster"
                )
                if _sensor_confirmed(sensor, platform):
                    priority = "P0_HIGH_SENSOR_CONFIRMED_QUERY"
                else:
                    priority = "P2_LOW_NO_LOCAL_SCENE_QUERY"

        targets.append(
            {
                "target_id": f"MV2_DATA_01_TGT_{i:05d}",
                "binding_id": binding_id,
                "anchor_id": anchor_id,
                "source_target_id": src_target_id,
                "asset_id": asset_id,
                "patch_id": patch_id,
                "region": region,
                "sensor": sensor,
                "platform": platform,
                "bbox": bbox,
                "crs": crs,
                "has_asset_id": str(has_asset).lower(),
                "has_patch_id": str(has_patch).lower(),
                "has_bbox": str(has_bbox).lower(),
                "has_crs": str(has_crs).lower(),
                "has_scene_id": str(has_scene_id).lower(),
                "has_datetime": str(has_datetime).lower(),
                "has_tile": str(has_tile).lower(),
                "has_cloud_cover": str(has_cloud_cover).lower(),
                "lineage_status": status,
                "priority": priority,
                "blocking_reason": blocking,
                "next_action": next_action,
                # campos auxiliares para consumidores (não vão à CSV de alvos):
                "_src_origin": src_origin,
                "_lineage_status_src": lineage_status_src,
                "_spatial_ready": status.startswith("TARGET_READY"),
            }
        )
    return targets


def is_spatial_ready(t: dict[str, object]) -> bool:
    return bool(t.get("_spatial_ready"))
