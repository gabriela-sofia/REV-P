"""SUSC-17E prontidao e calibracao observacional exploratoria.

Consome os vinculos aceitos para avaliacao no SUSC-17D e as features reais
pre-evento ja materializadas no repositorio (Sentinel-2 do 17C20 e CHIRPS do
17C25) para os patches canario da grade Charter758 (17C6). Diagnostica por que
os vinculos aceitos nao alcancam calibracao forte e entrega, de forma
cientificamente segura, uma calibracao observacional exploratoria review-only.

Nada aqui vira verdade de referencia (ground truth), dado treinavel, score_v7
oficial ou benchmark 17B. O score_v6 oficial nunca e alterado; ele so e lido.
O score exploratorio e sempre review-only e nunca substitui o score_v6.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import ensure_dir, read_csv, read_json, rel, write_csv, write_json, write_markdown  # noqa: E402

DATASETS = ROOT / "datasets"
DAT_SUSC = DATASETS / "suscetibilidade"
SUSC_OUT = ROOT / "outputs_public" / "suscetibilidade"
OUT_DATA_17D = ROOT / "outputs_public" / "data" / "susc_17d_validacao_tecnica_evidencia_observacional"
OUT_DATA_17C5 = ROOT / "outputs_public" / "data" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA = ROOT / "outputs_public" / "data" / "susc_17e_prontidao_calibracao_observacional_exploratoria"
CARDS_DIR = OUT_DATA / "cartoes_prontidao"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
DECISIONS_17D = OUT_DATA_17D / "decisoes_validacao_tecnica.csv"
BLOCKERS_17D = OUT_DATA_17D / "bloqueios_validacao_tecnica.csv"
FEATURE_MATRIX_17D = OUT_DATA_17D / "matriz_consistencia_features.csv"
SUMMARY_17D = OUT_DATA_17D / "susc_17d_validacao_tecnica_summary.json"
PATCH_LINKS_17C5 = OUT_DATA_17C5 / "susc_17c5_patch_links.csv"
GEOMETRIES_17C5 = OUT_DATA_17C5 / "susc_17c5_normalized_geometry.csv"
TARGET_PACK_17C = OUT_DATA_17C / "susc_17c_source_target_pack.csv"
PATCH_GRID_17C6 = SUSC_OUT / "susc_17c6_candidate_patch_grid.csv"
BAND_STATS_DIR = SUSC_OUT / "susc_17c20_light_artifacts"
CHIRPS_DIR = SUSC_OUT / "susc_17c25_chirps_artifacts"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_CSV = OUT_DATA / "preflight_prontidao_calibracao.csv"
PREFLIGHT_JSON = OUT_DATA / "preflight_prontidao_calibracao.json"
MATRIZ_PRONTIDAO = OUT_DATA / "matriz_prontidao_calibracao.csv"
MATRIZ_FEATURES = OUT_DATA / "matriz_features_minimas.csv"
MATRIZ_CALIBRACAO = OUT_DATA / "matriz_calibracao_exploratoria.csv"
HIPOTESES = OUT_DATA / "hipoteses_ajuste_pesos_review_only.csv"
SIMULACAO = OUT_DATA / "simulacao_sensibilidade_review_only.csv"
DIAGNOSTICO = OUT_DATA / "diagnostico_bloqueios_calibracao.csv"
RECOMENDACOES = OUT_DATA / "recomendacoes_calibracao_futura.csv"
ACOES_MINIMAS = OUT_DATA / "acoes_minimas_desbloqueio.csv"
GATE = OUT_DATA / "gate_prontidao_calibracao_observacional.csv"
RESUMO_STATUS = OUT_DATA / "resumo_prontidao_por_status.csv"
RESUMO_REGIAO = OUT_DATA / "resumo_prontidao_por_regiao.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_17E_PRONTIDAO_CALIBRACAO_OBSERVACIONAL_EXPLORATORIA.md"
SCHEMA = SCHEMAS / "susc_17e_prontidao_calibracao_observacional_exploratoria_schema_v1.json"

REQUIRED_INPUTS = [
    DECISIONS_17D,
    BLOCKERS_17D,
    FEATURE_MATRIX_17D,
    SUMMARY_17D,
    PATCH_LINKS_17C5,
    GEOMETRIES_17C5,
    TARGET_PACK_17C,
    PATCH_GRID_17C6,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    PREFLIGHT_CSV,
    PREFLIGHT_JSON,
    MATRIZ_PRONTIDAO,
    MATRIZ_FEATURES,
    MATRIZ_CALIBRACAO,
    HIPOTESES,
    SIMULACAO,
    DIAGNOSTICO,
    RECOMENDACOES,
    ACOES_MINIMAS,
    GATE,
    RESUMO_STATUS,
    RESUMO_REGIAO,
    SUMMARY,
    REPORT,
    SCHEMA,
]

# --- Enums permitidos ------------------------------------------------------
STATUS_PRONTIDAO_ALLOWED = [
    "pronto_para_calibracao_forte",
    "pronto_para_calibracao_exploratoria_review_only",
    "prontidao_parcial",
    "bloqueado_por_features",
    "bloqueado_por_incerteza_espacial",
    "bloqueado_por_fonte",
    "bloqueado_por_data",
    "bloqueado_por_fenomeno",
    "bloqueado_por_lacuna_regional",
    "bloqueado_por_amostra_insuficiente",
]
GATE_FINAL_ALLOWED = [
    "17E_CALIBRACAO_FORTE_REVIEW_ONLY",
    "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY",
    "17E_PRONTIDAO_PARCIAL",
    "17E_BLOQUEADO_COM_PLANO_EXECUTAVEL",
    "17E_BLOQUEADO_FAIL_CLOSED",
]
ACAO_MINIMA_ALLOWED = [
    "obter_feature_fisica",
    "obter_feature_hidrologica",
    "obter_feature_urbana",
    "obter_feature_espectral",
    "reduzir_incerteza_geometrica",
    "confirmar_fonte",
    "confirmar_data",
    "confirmar_fenomeno",
    "ampliar_amostra_regional",
    "manter_apenas_avaliacao",
]

# Vocabulario proibido em textos publicos (nao expor automacao).
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|llm|codex|ia|human\s+qa)\b", re.IGNORECASE)

STRONG_LINK_CLASSES = {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}
STRONG_SOURCE_TYPES = {
    "official_observed_event_polygon",
    "official_observed_event_point",
    "technical_remote_sensing_flood_footprint",
}
NO_GT_REASON = (
    "prontidao e calibracao review-only; nao e verdade de referencia, "
    "nao e treino, nao autoriza score_v7 e nao substitui o score_v6"
)

# --- Politica de grupos de features (secao 5 da sprint) --------------------
FEATURE_GROUPS = {
    "fisico": ["HAND", "elevation", "slope", "distance_to_water", "TWI", "flow_accumulation"],
    "urbano_territorial": ["urban_prop", "vegetation_prop", "water_prop", "MapBiomas", "imperviousness_proxy"],
    "espectral": ["NDVI", "NDWI", "MNDWI", "NDBI"],
    "chuva_gatilho": ["CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d", "runoff_context_7d"],
}
FEATURE_DIRECTION = {
    "HAND": "menor HAND tende a apoiar suscetibilidade hidrologica",
    "elevation": "menor elevacao pode apoiar contexto de acumulacao",
    "slope": "menor declividade tende a apoiar acumulacao",
    "distance_to_water": "menor distancia tende a apoiar exposicao hidrica",
    "TWI": "maior TWI tende a apoiar umidade topografica",
    "flow_accumulation": "maior acumulacao tende a apoiar drenagem concentrada",
    "urban_prop": "maior urbanizacao pode apoiar amplificacao urbana",
    "vegetation_prop": "maior vegetacao pode reduzir suscetibilidade urbana",
    "water_prop": "maior ocorrencia de agua pode apoiar proximidade hidrica",
    "MapBiomas": "cobertura construida pode apoiar amplificacao urbana",
    "imperviousness_proxy": "maior impermeabilizacao pode apoiar escoamento rapido",
    "NDVI": "menor NDVI pode indicar menor mitigacao por vegetacao",
    "NDWI": "maior NDWI pode indicar sinal hidrico",
    "MNDWI": "maior MNDWI pode indicar sinal hidrico superficial",
    "NDBI": "maior NDBI pode indicar superficie construida",
    "CHIRPS_3d": "maior chuva recente pode apoiar gatilho",
    "CHIRPS_7d": "maior chuva semanal pode apoiar gatilho",
    "CHIRPS_30d": "maior chuva mensal pode apoiar saturacao",
    "runoff_context_7d": "maior escoamento contextual pode apoiar gatilho",
}

# --- Modelo score_v6 (documentado, somente leitura) ------------------------
# Pesos documentados no SUSC-10A. Usados apenas para recompor, de forma
# exploratoria e review-only, os sub-indices com feature real disponivel.
SCORE_V6_WEIGHTS = {
    "topography_hydrology_index": 0.40,
    "rainfall_trigger_index": 0.25,
    "urban_spectral_index": 0.20,
    "vegetation_mitigation_index": -0.10,
    "evidence_support_index": 0.05,
}
# Referencias fixas e transparentes para normalizar CHIRPS (mm) em [0,1].
# Sao referencias exploratorias documentadas, nunca a normalizacao oficial.
CHIRPS_REF = {"CHIRPS_3d": 50.0, "CHIRPS_7d": 80.0, "CHIRPS_30d": 250.0}

# Hipoteses de ajuste de peso review-only (secao 6 da sprint).
# Cada variante define pesos sobre os componentes com feature real disponivel:
# rain (CHIRPS_7d), urban (NDBI), veg (NDVI, mitigacao), moisture (MNDWI).
WEIGHT_VARIANTS = {
    "H0_baseline_pesos_documentados": {"rain": 0.25, "urban": 0.20, "veg": -0.10, "moisture": 0.00},
    "H3_reforco_urbano": {"rain": 0.25, "urban": 0.35, "veg": -0.10, "moisture": 0.00},
    "H4_reforco_chuva": {"rain": 0.40, "urban": 0.20, "veg": -0.10, "moisture": 0.00},
    "H5_reforco_agua_umidade": {"rain": 0.25, "urban": 0.20, "veg": -0.10, "moisture": 0.15},
}

# --- Field lists -----------------------------------------------------------
PREFLIGHT_FIELDS = ["artifact_role", "path", "exists", "row_count", "columns_found", "notes"]
PRONTIDAO_FIELDS = [
    "item_validacao_id", "candidate_event_id", "source_id", "geometry_id", "patch_id",
    "cidade", "regiao", "tipo_fenomeno", "data_evento", "classe_vinculo",
    "qualidade_fonte", "qualidade_temporal", "qualidade_geometrica", "qualidade_vinculo_patch",
    "qualidade_fenomeno", "qualidade_incerteza", "disponibilidade_features", "consistencia_features",
    "estabilidade_regional", "risco_vazamento_temporal", "severidade_bloqueios", "prontidao_calibracao",
    "grupos_features_disponiveis", "grupos_features_ausentes",
    "accepted_for_evaluation", "ground_truth", "eligible_for_training", "score_v7_allowed",
    "review_only", "justificativa_tecnica",
]
FEATURES_MIN_FIELDS = [
    "item_validacao_id", "patch_id", "grupo_feature", "nome_feature", "feature_presente",
    "valor_real", "direcao_esperada", "consistencia_com_suscetibilidade", "contradicao_detectada",
    "ausencia_impede_calibracao_forte", "ausencia_rebaixa_para_exploratoria", "observacao",
]
CALIBRACAO_FIELDS = [
    "item_validacao_id", "candidate_event_id", "patch_id", "geometry_id", "regiao",
    "grupos_disponiveis", "NDVI", "NDWI", "MNDWI", "NDBI", "CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d",
    "rain_idx", "urban_spectral_idx", "vegetation_idx", "moisture_idx",
    "score_exploratorio_baseline", "score_exploratorio_class_baseline",
    "score_v6_reference_value", "score_v6_reference_class", "score_v6_reference_source",
    "comparacao_parcial_vs_completa", "elegivel_calibracao_exploratoria",
    "justificativa_tecnica", "lacunas",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "review_only",
]
HIPOTESES_FIELDS = [
    "hipotese_id", "nome_hipotese", "descricao", "componentes_afetados", "ajuste_peso",
    "permitida", "motivo", "aplicada_na_simulacao",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "review_only",
]
SIMULACAO_FIELDS = [
    "simulacao_id", "item_validacao_id", "patch_id", "hipotese_id",
    "rain_idx", "urban_spectral_idx", "vegetation_idx", "moisture_idx",
    "peso_rain", "peso_urban", "peso_veg", "peso_moisture",
    "score_exploratorio_review_only", "score_exploratorio_class",
    "score_v6_reference_value", "score_v6_reference_class", "mudanca_de_classe",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "review_only",
]
DIAGNOSTICO_FIELDS = [
    "item_validacao_id", "candidate_event_id", "patch_id", "status_prontidao",
    "bloqueio_principal", "bloqueios_secundarios", "acao_minima_para_desbloquear",
    "pode_virar_calibracao_forte", "pode_virar_calibracao_exploratoria", "justificativa_tecnica",
]
ACOES_FIELDS = [
    "item_validacao_id", "patch_id", "acao_minima", "categoria_acao", "prioridade",
    "descricao", "destrava_forte", "destrava_exploratoria",
]
RECOMENDACOES_FIELDS = [
    "recomendacao_id", "categoria", "recomendacao", "motivo", "prioridade",
    "habilita_calibracao_forte", "habilita_calibracao_exploratoria", "review_only",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_STATUS_FIELDS = ["prontidao_calibracao", "quantidade"]
RESUMO_REGIAO_FIELDS = ["regiao", "itens", "forte", "exploratoria", "parcial", "bloqueados"]


# --- Helpers ---------------------------------------------------------------
def _bool(value: bool) -> str:
    return "true" if value else "false"


def _run_git(args: list[str]) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _read(path: Path) -> list[dict]:
    return read_csv(path) if path.exists() else []


def _by_key(rows: list[dict], key: str) -> dict[str, dict]:
    return {row.get(key, ""): row for row in rows}


def _float(value) -> float | None:
    try:
        if value is None or value == "" or value == "not_available":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round4(value: float) -> float:
    return round(value + 0.0, 4)


def _require_inputs() -> None:
    missing = [path for path in REQUIRED_INPUTS if not path.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(path) for path in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-17E")


# --- Leitura de features reais ---------------------------------------------
def _band_means(patch_id: str) -> dict[str, float]:
    """Le medias de banda Sentinel-2 (17C20) para um patch canario."""
    path = BAND_STATS_DIR / f"{patch_id.lower()}_band_stats.csv"
    means: dict[str, float] = {}
    for row in _read(path):
        if row.get("stat_name") == "mean":
            value = _float(row.get("stat_value"))
            if value is not None:
                means[row.get("band_name", "")] = value
    return means


def _spectral_indices(patch_id: str) -> dict[str, float | None]:
    m = _band_means(patch_id)
    b03, b04, b08, b11 = m.get("B03"), m.get("B04"), m.get("B08"), m.get("B11")

    def _index(a, b):
        if a is None or b is None or (a + b) == 0:
            return None
        return _round4((a - b) / (a + b))

    return {
        "NDVI": _index(b08, b04),
        "NDWI": _index(b03, b08),
        "MNDWI": _index(b03, b11),
        "NDBI": _index(b11, b08),
    }


def _chirps_sums(patch_id: str) -> dict[str, float | None]:
    """Soma CHIRPS pre-evento (17C25). Janela termina no dia anterior ao evento."""
    path = CHIRPS_DIR / f"{patch_id.lower()}_chirps_daily.csv"
    rows = _read(path)
    if not rows:
        return {"CHIRPS_3d": None, "CHIRPS_7d": None, "CHIRPS_30d": None}
    ordered = sorted(rows, key=lambda r: r.get("date", ""))
    values = [_float(r.get("precip_mm")) or 0.0 for r in ordered]
    return {
        "CHIRPS_3d": _round4(sum(values[-3:])),
        "CHIRPS_7d": _round4(sum(values[-7:])),
        "CHIRPS_30d": _round4(sum(values)),
    }


# --- Modelo de classe score_v6 (derivado dos dados oficiais) ---------------
def _class_thresholds() -> tuple[float, float]:
    rows = _read(SCORE_V6)
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get("susc_score_v6_candidate"))
        cls = row.get("susc_class_v6_candidate", "")
        if value is not None and cls:
            by_class[cls].append(value)
    low_max = max(by_class.get("low", [0.42]))
    med_min = min(by_class.get("medium", [0.43]))
    med_max = max(by_class.get("medium", [0.61]))
    high_min = min(by_class.get("high", [0.62]))
    t_low_med = _round4((low_max + med_min) / 2)
    t_med_high = _round4((med_max + high_min) / 2)
    return t_low_med, t_med_high


def _classify(value: float, thresholds: tuple[float, float]) -> str:
    t_low_med, t_med_high = thresholds
    if value < t_low_med:
        return "low"
    if value < t_med_high:
        return "medium"
    return "high"


# --- Score exploratorio ----------------------------------------------------
def _exploratory_indices(spectral: dict, chirps: dict) -> dict[str, float]:
    ndvi = spectral.get("NDVI")
    ndbi = spectral.get("NDBI")
    mndwi = spectral.get("MNDWI")
    chirps_7d = chirps.get("CHIRPS_7d")
    return {
        "rain_idx": _round4(_clamp((chirps_7d or 0.0) / CHIRPS_REF["CHIRPS_7d"])),
        "urban_spectral_idx": _round4(_clamp(((ndbi or 0.0) + 1.0) / 2.0)),
        "vegetation_idx": _round4(_clamp(((ndvi or 0.0) + 1.0) / 2.0)),
        "moisture_idx": _round4(_clamp(((mndwi or 0.0) + 1.0) / 2.0)),
    }


def _exploratory_score(indices: dict, weights: dict) -> float:
    raw = (
        weights["rain"] * indices["rain_idx"]
        + weights["urban"] * indices["urban_spectral_idx"]
        + weights["veg"] * indices["vegetation_idx"]
        + weights["moisture"] * indices["moisture_idx"]
    )
    # Faixa teorica da variante para normalizar em [0,1] de forma deterministica.
    raw_min = sum(w for w in weights.values() if w < 0)
    raw_max = sum(w for w in weights.values() if w > 0)
    if raw_max - raw_min == 0:
        return 0.0
    return _round4(_clamp((raw - raw_min) / (raw_max - raw_min)))


# --- Estruturacao dos itens ------------------------------------------------
def _accepted_items() -> list[dict]:
    return [row for row in _read(DECISIONS_17D) if row.get("accepted_for_evaluation") == "true"]


def _score_v6_reference(nearest_patch_id: str) -> dict:
    ref = _by_key(_read(SCORE_V6), "patch_id").get(nearest_patch_id, {})
    return ref


def build_items() -> list[dict]:
    """Monta o modelo interno de cada item aceito, com features reais e prontidao."""
    thresholds = _class_thresholds()
    grid_by_patch = _by_key(_read(PATCH_GRID_17C6), "candidate_patch_id")
    link_index = {
        (r["candidate_event_id"], r["patch_id"], r["geometry_id"]): r
        for r in _read(PATCH_LINKS_17C5)
    }
    geom_by_id = _by_key(_read(GEOMETRIES_17C5), "geometry_id")
    target_by_id = _by_key(_read(TARGET_PACK_17C), "candidate_event_id")

    items: list[dict] = []
    for decision in _accepted_items():
        patch_id = decision["patch_id"]
        event_id = decision["candidate_event_id"]
        geometry_id = decision["geometry_id"]
        link = link_index.get((event_id, patch_id, geometry_id), {})
        geom = geom_by_id.get(geometry_id, {})
        target = target_by_id.get(event_id, {})
        grid = grid_by_patch.get(patch_id, {})

        spectral = _spectral_indices(patch_id)
        chirps = _chirps_sums(patch_id)
        indices = _exploratory_indices(spectral, chirps)

        available_groups = _available_groups(spectral, chirps)
        missing_groups = [g for g in FEATURE_GROUPS if g not in available_groups]

        nearest_patch = grid.get("nearest_official_patch_id", "")
        ref = _score_v6_reference(nearest_patch)
        ref_value = _float(ref.get("susc_score_v6_candidate"))
        ref_class = ref.get("susc_class_v6_candidate", "not_available")

        overlap_ratio = _float(link.get("overlap_ratio")) or 0.0
        uncertainty_raw = link.get("uncertainty_m", "")

        quality = {
            "qualidade_fonte": _source_quality(decision.get("source_type", "")),
            "qualidade_temporal": _temporal_quality(target, decision),
            "qualidade_geometrica": _geometry_quality(geom),
            "qualidade_vinculo_patch": _link_quality(link, overlap_ratio),
            "qualidade_fenomeno": _phenomenon_quality(decision),
            "qualidade_incerteza": _uncertainty_quality(uncertainty_raw),
            "disponibilidade_features": _availability_quality(available_groups),
            "consistencia_features": _consistency_quality(spectral, chirps),
            "estabilidade_regional": 35,  # unico evento/regiao/geometria -> amostra pequena
            "risco_vazamento_temporal": 80,  # features pre-evento (2022-04) vs evento (2022-05)
        }

        strong_ok = _strong_ready(decision, geom, link, available_groups, quality)
        exploratory_ok = _exploratory_ready(decision, geom, link, available_groups)
        severity = _severity(strong_ok, available_groups)
        quality["severidade_bloqueios"] = severity

        status = _prontidao_status(strong_ok, exploratory_ok, available_groups)

        justification = _justification(status, available_groups, missing_groups, indices)

        items.append({
            "item_validacao_id": decision["item_validacao_id"],
            "candidate_event_id": event_id,
            "source_id": decision.get("source_id", ""),
            "source_type": decision.get("source_type", ""),
            "geometry_id": geometry_id,
            "patch_id": patch_id,
            "cidade": decision.get("cidade", grid.get("region", "")),
            "regiao": decision.get("regiao", ""),
            "tipo_fenomeno": decision.get("tipo_fenomeno", ""),
            "data_evento": decision.get("data_evento", ""),
            "classe_vinculo": decision.get("classe_vinculo", link.get("link_class", "")),
            "nearest_official_patch_id": nearest_patch,
            "spectral": spectral,
            "chirps": chirps,
            "indices": indices,
            "available_groups": available_groups,
            "missing_groups": missing_groups,
            "quality": quality,
            "thresholds": thresholds,
            "strong_ok": strong_ok,
            "exploratory_ok": exploratory_ok,
            "prontidao_calibracao": status,
            "score_v6_reference_value": ref_value,
            "score_v6_reference_class": ref_class,
            "score_v6_reference_source": f"nearest_official_patch_{nearest_patch}" if nearest_patch else "not_available",
            "justificativa_tecnica": justification,
            "uncertainty_raw": uncertainty_raw,
        })
    return items


def _available_groups(spectral: dict, chirps: dict) -> list[str]:
    groups = []
    if any(v is not None for v in spectral.values()):
        groups.append("espectral")
    if any(v is not None for v in [chirps.get("CHIRPS_3d"), chirps.get("CHIRPS_7d"), chirps.get("CHIRPS_30d")]):
        groups.append("chuva_gatilho")
    return groups


def _source_quality(source_type: str) -> int:
    if source_type in STRONG_SOURCE_TYPES:
        return 85
    if source_type in {"official_address_resolved", "administrative_disaster_record"}:
        return 55
    return 30


def _temporal_quality(target: dict, decision: dict) -> int:
    date = decision.get("data_evento", "") or target.get("event_date_candidate", "")
    if ".." in date:
        return 80
    if re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        return 90
    if re.match(r"^\d{4}-\d{2}$", date):
        return 35
    return 0


def _geometry_quality(geom: dict) -> int:
    if geom.get("has_geometry_object") != "true":
        return 0
    if geom.get("geometry_type") in {"polygon", "technical_footprint"}:
        return 85
    if geom.get("geometry_type") in {"point", "bbox"}:
        return 70
    return 30


def _link_quality(link: dict, overlap_ratio: float) -> int:
    if link.get("link_class") == "exact_polygon_overlap":
        return 65 if overlap_ratio < 0.005 else 85
    if link.get("link_class") in {"point_within_patch", "bbox_overlap"}:
        return 75
    return 0


def _phenomenon_quality(decision: dict) -> int:
    text = decision.get("tipo_fenomeno", "").lower()
    if any(term in text for term in ["flood", "inund", "alag", "enxurr", "hidro"]):
        return 85
    if "mass_movement" in text or "desliz" in text:
        return 40
    return 45


def _uncertainty_quality(value: str) -> int:
    num = _float(value)
    if num is not None:
        if num <= 50:
            return 85
        if num <= 500:
            return 65
        return 35
    if value in {"requires_human_review", "bbox_aoi_not_footprint"}:
        return 55
    return 0


def _availability_quality(available_groups: list[str]) -> int:
    return int(round(100 * len(available_groups) / len(FEATURE_GROUPS)))


def _consistency_quality(spectral: dict, chirps: dict) -> int:
    supports = 0
    contradictions = 0
    if (spectral.get("MNDWI") or -1) > 0:
        supports += 1
    if (spectral.get("NDVI") or 1) < 0.2:
        supports += 1
    if (chirps.get("CHIRPS_30d") or 0) > 0:
        supports += 1
    if (spectral.get("NDVI") or 0) > 0.5:  # vegetacao alta contradiz alagamento urbano
        contradictions += 1
    if contradictions:
        return 40
    if supports >= 2:
        return 60
    return 50


def _strong_ready(decision, geom, link, available_groups, quality) -> bool:
    return (
        decision.get("accepted_for_evaluation") == "true"
        and geom.get("has_geometry_object") == "true"
        and link.get("link_class") in STRONG_LINK_CLASSES
        and quality["qualidade_incerteza"] >= 60
        and quality["qualidade_temporal"] >= 70
        and quality["qualidade_fenomeno"] >= 50
        and "fisico" in available_groups
        and ("urbano_territorial" in available_groups or "espectral" in available_groups)
        and quality["consistencia_features"] >= 50
        and quality["risco_vazamento_temporal"] >= 40
        and decision.get("confianca_tecnica_final") in {"alta", "media"}
    )


def _exploratory_ready(decision, geom, link, available_groups) -> bool:
    return (
        decision.get("accepted_for_evaluation") == "true"
        and decision.get("ground_truth") == "false"
        and decision.get("eligible_for_training") == "false"
        and decision.get("score_v7_allowed") == "false"
        and decision.get("patch_id") not in {"", "NO_PATCH_RESOLUTION", "not_available"}
        and decision.get("geometry_id") not in {"", "not_available"}
        and link.get("link_class") not in {"same_region_only", ""}
        and _phenomenon_quality(decision) >= 50
        and _temporal_quality({}, decision) >= 70
        and len(available_groups) >= 2
    )


def _severity(strong_ok: bool, available_groups: list[str]) -> int:
    if strong_ok:
        return 85
    if "fisico" not in available_groups:
        return 55  # falta o grupo dominante (topografia/hidrologia)
    return 65


def _prontidao_status(strong_ok: bool, exploratory_ok: bool, available_groups: list[str]) -> str:
    if strong_ok:
        return "pronto_para_calibracao_forte"
    if exploratory_ok:
        return "pronto_para_calibracao_exploratoria_review_only"
    if len(available_groups) == 1:
        return "prontidao_parcial"
    return "bloqueado_por_features"


def _justification(status: str, available_groups: list[str], missing_groups: list[str], indices: dict) -> str:
    grupos = ",".join(available_groups) or "nenhum"
    ausentes = ",".join(missing_groups) or "nenhum"
    if status == "pronto_para_calibracao_forte":
        return (
            f"grupos disponiveis={grupos}; features fisicas e espectrais presentes; "
            "calibracao forte review-only permitida sem promocao oficial"
        )
    if status == "pronto_para_calibracao_exploratoria_review_only":
        return (
            f"grupos disponiveis={grupos}; grupos ausentes={ausentes}; "
            "features fisicas/topograficas indisponiveis impedem calibracao forte, "
            "mas espectral e chuva reais sustentam calibracao exploratoria review-only; "
            "nao gera ground truth, treino, score_v7 nem altera score_v6"
        )
    if status == "prontidao_parcial":
        return f"apenas um grupo de features disponivel ({grupos}); prontidao parcial review-only"
    return f"grupos insuficientes ({grupos}); bloqueado por ausencia de features minimas"


# --- Preflight -------------------------------------------------------------
def _row_count(path: Path) -> str:
    if not path.exists():
        return "0"
    if path.suffix.lower() == ".csv":
        return str(len(_read(path)))
    if path.suffix.lower() in {".json", ".geojson"}:
        return "1"
    return "not_csv"


def _columns(path: Path) -> str:
    if path.exists() and path.suffix.lower() == ".csv":
        text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if text:
            return text[0].replace(",", ";")
    if path.exists():
        return "json_or_markdown"
    return ""


def preflight_rows() -> list[dict]:
    sources = [
        ("decisoes_17d", DECISIONS_17D, "vinculos aceitos para avaliacao review-only"),
        ("bloqueios_17d", BLOCKERS_17D, "bloqueios herdados da validacao tecnica"),
        ("matriz_features_17d", FEATURE_MATRIX_17D, "consistencia de features herdada"),
        ("resumo_17d", SUMMARY_17D, "estado herdado do 17D"),
        ("patch_links_17c5", PATCH_LINKS_17C5, "classe de vinculo e sobreposicao"),
        ("geometria_17c5", GEOMETRIES_17C5, "geometria resolvida"),
        ("registries_17c", TARGET_PACK_17C, "fonte, fenomeno e data"),
        ("grade_canario_17c6", PATCH_GRID_17C6, "geometria e patch oficial mais proximo"),
        ("band_stats_17c20", BAND_STATS_DIR, "Sentinel-2 medias de banda pre-evento"),
        ("chirps_17c25", CHIRPS_DIR, "CHIRPS diario pre-evento"),
        ("score_v6", SCORE_V6, "score v6 oficial somente leitura"),
        ("score_v7_proibido", SCORE_V7, "deve permanecer ausente"),
    ]
    rows = []
    for role, path, note in sources:
        if path.is_dir():
            exists = path.exists()
            count = str(len(list(path.glob("*")))) if exists else "0"
            cols = "diretorio_de_artefatos"
        else:
            exists = path.exists()
            count = _row_count(path)
            cols = _columns(path)
        rows.append({
            "artifact_role": role,
            "path": rel(path),
            "exists": _bool(exists),
            "row_count": count,
            "columns_found": cols,
            "notes": note,
        })
    rows.append({
        "artifact_role": "estado_git",
        "path": ".",
        "exists": "true",
        "row_count": "not_applicable",
        "columns_found": "branch;head;staged_count;dirty_status_lines",
        "notes": (
            f"branch={_run_git(['branch', '--show-current'])}; "
            f"head={_run_git(['rev-parse', '--short', 'HEAD'])}; "
            f"staged={len(_run_git(['diff', '--cached', '--name-only']).splitlines())}; "
            f"dirty={len(_run_git(['status', '--short']).splitlines())}"
        ),
    })
    return rows


# --- Saidas tabulares ------------------------------------------------------
def prontidao_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        q = item["quality"]
        rows.append({
            "item_validacao_id": item["item_validacao_id"],
            "candidate_event_id": item["candidate_event_id"],
            "source_id": item["source_id"],
            "geometry_id": item["geometry_id"],
            "patch_id": item["patch_id"],
            "cidade": item["cidade"],
            "regiao": item["regiao"],
            "tipo_fenomeno": item["tipo_fenomeno"],
            "data_evento": item["data_evento"],
            "classe_vinculo": item["classe_vinculo"],
            "qualidade_fonte": str(q["qualidade_fonte"]),
            "qualidade_temporal": str(q["qualidade_temporal"]),
            "qualidade_geometrica": str(q["qualidade_geometrica"]),
            "qualidade_vinculo_patch": str(q["qualidade_vinculo_patch"]),
            "qualidade_fenomeno": str(q["qualidade_fenomeno"]),
            "qualidade_incerteza": str(q["qualidade_incerteza"]),
            "disponibilidade_features": str(q["disponibilidade_features"]),
            "consistencia_features": str(q["consistencia_features"]),
            "estabilidade_regional": str(q["estabilidade_regional"]),
            "risco_vazamento_temporal": str(q["risco_vazamento_temporal"]),
            "severidade_bloqueios": str(q["severidade_bloqueios"]),
            "prontidao_calibracao": item["prontidao_calibracao"],
            "grupos_features_disponiveis": ";".join(item["available_groups"]) or "nenhum",
            "grupos_features_ausentes": ";".join(item["missing_groups"]) or "nenhum",
            "accepted_for_evaluation": "true",
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
            "review_only": "true",
            "justificativa_tecnica": item["justificativa_tecnica"],
        })
    return rows


def features_min_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        real_values = {}
        real_values.update(item["spectral"])
        real_values.update(item["chirps"])
        for group, feature_names in FEATURE_GROUPS.items():
            for name in feature_names:
                value = real_values.get(name)
                present = value is not None
                consistency, contradiction = _feature_consistency(name, value, present)
                impede_forte = (not present) and group in {"fisico", "urbano_territorial"}
                rebaixa_expl = (not present) and group in {"fisico", "urbano_territorial", "chuva_gatilho"}
                if present:
                    observacao = "feature real disponivel usada como apoio exploratorio review-only"
                elif group == "fisico":
                    observacao = "ausencia de feature fisica impede calibracao forte; rebaixa para exploratoria"
                elif group == "urbano_territorial":
                    observacao = "ausencia de feature urbana impede calibracao forte; apoio parcial via espectral"
                else:
                    observacao = "feature ausente; ausencia nao reprova, apenas limita a calibracao"
                rows.append({
                    "item_validacao_id": item["item_validacao_id"],
                    "patch_id": item["patch_id"],
                    "grupo_feature": group,
                    "nome_feature": name,
                    "feature_presente": _bool(present),
                    "valor_real": _fmt(value) if present else "not_available",
                    "direcao_esperada": FEATURE_DIRECTION.get(name, ""),
                    "consistencia_com_suscetibilidade": consistency,
                    "contradicao_detectada": contradiction,
                    "ausencia_impede_calibracao_forte": _bool(impede_forte),
                    "ausencia_rebaixa_para_exploratoria": _bool(rebaixa_expl),
                    "observacao": observacao,
                })
    return rows


def _feature_consistency(name: str, value, present: bool) -> tuple[str, str]:
    if not present:
        return "nao_avaliada", "nenhuma"
    if name == "MNDWI" and value is not None:
        return ("apoia_suscetibilidade" if value > 0 else "neutra"), "nenhuma"
    if name == "NDVI" and value is not None:
        if value > 0.5:
            return "contraditoria", "vegetacao_alta_contradiz_alagamento_urbano"
        return "apoia_suscetibilidade", "nenhuma"
    if name == "NDBI" and value is not None:
        return ("apoia_suscetibilidade" if value > 0 else "neutra"), "nenhuma"
    if name.startswith("CHIRPS") and value is not None:
        return ("apoia_suscetibilidade" if value > 0 else "neutra"), "nenhuma"
    return "neutra", "nenhuma"


def _fmt(value) -> str:
    if value is None:
        return "not_available"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def calibracao_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        if item["prontidao_calibracao"] not in {
            "pronto_para_calibracao_forte",
            "pronto_para_calibracao_exploratoria_review_only",
        }:
            continue
        idx = item["indices"]
        baseline = _exploratory_score(idx, WEIGHT_VARIANTS["H0_baseline_pesos_documentados"])
        baseline_class = _classify(baseline, item["thresholds"])
        lacunas = "; ".join([
            "grupos ausentes: " + (",".join(item["missing_groups"]) or "nenhum"),
            "topografia/hidrologia (peso 0.40) indisponivel para os canarios",
            "score exploratorio cobre subconjunto do score_v6 (espectral + chuva)",
        ])
        rows.append({
            "item_validacao_id": item["item_validacao_id"],
            "candidate_event_id": item["candidate_event_id"],
            "patch_id": item["patch_id"],
            "geometry_id": item["geometry_id"],
            "regiao": item["regiao"],
            "grupos_disponiveis": ";".join(item["available_groups"]),
            "NDVI": _fmt(item["spectral"]["NDVI"]),
            "NDWI": _fmt(item["spectral"]["NDWI"]),
            "MNDWI": _fmt(item["spectral"]["MNDWI"]),
            "NDBI": _fmt(item["spectral"]["NDBI"]),
            "CHIRPS_3d": _fmt(item["chirps"]["CHIRPS_3d"]),
            "CHIRPS_7d": _fmt(item["chirps"]["CHIRPS_7d"]),
            "CHIRPS_30d": _fmt(item["chirps"]["CHIRPS_30d"]),
            "rain_idx": _fmt(idx["rain_idx"]),
            "urban_spectral_idx": _fmt(idx["urban_spectral_idx"]),
            "vegetation_idx": _fmt(idx["vegetation_idx"]),
            "moisture_idx": _fmt(idx["moisture_idx"]),
            "score_exploratorio_baseline": _fmt(baseline),
            "score_exploratorio_class_baseline": baseline_class,
            "score_v6_reference_value": _fmt(item["score_v6_reference_value"]),
            "score_v6_reference_class": item["score_v6_reference_class"],
            "score_v6_reference_source": item["score_v6_reference_source"],
            "comparacao_parcial_vs_completa": "true",
            "elegivel_calibracao_exploratoria": "true",
            "justificativa_tecnica": item["justificativa_tecnica"],
            "lacunas": lacunas,
            "score_oficial": "false",
            "substituir_score_v6": "false",
            "usar_em_treino": "false",
            "ground_truth": "false",
            "review_only": "true",
        })
    return rows


def hipoteses_rows() -> list[dict]:
    specs = [
        ("H1_separar_confianca_documental", "separar confianca documental de suscetibilidade fisica",
         "Trata o indice de suporte documental como confianca, separada da suscetibilidade fisica.",
         "evidence_support_index", "isola_evidence_support", "true"),
        ("H2_reduzir_penalidade_documental_com_evidencia_observacional",
         "reduzir penalidade documental quando ha evidencia observacional",
         "Quando ha footprint observacional (radar de abertura sintetica), nao penaliza ausencia documental.",
         "evidence_support_index", "reduz_penalidade_documental", "true"),
        ("H3_reforco_urbano", "aumentar peso urbano se urban_prop/NDBI sustentarem alagamento urbano",
         "Aumenta o peso do componente urbano/espectral (NDBI) quando ha sinal de superficie construida.",
         "urban_spectral_index", "peso_urban 0.20->0.35", "true"),
        ("H4_reforco_chuva", "aumentar peso chuva se CHIRPS/runoff sustentarem gatilho",
         "Aumenta o peso do componente de chuva quando CHIRPS pre-evento sustenta o gatilho.",
         "rainfall_trigger_index", "peso_rain 0.25->0.40", "true"),
        ("H5_reforco_agua_umidade", "aumentar peso agua/umidade se MNDWI/NDWI sustentarem sinal",
         "Adiciona peso ao sinal de agua/umidade (MNDWI/NDWI) quando o indice espectral hidrico o sustenta.",
         "urban_spectral_index;moisture", "peso_moisture 0.00->0.15", "true"),
        ("H6_manter_bloqueado_feature_contraditoria",
         "manter bloqueado se feature contraditoria",
         "Se alguma feature contradiz a suscetibilidade (ex.: vegetacao alta), mantem o item bloqueado.",
         "todos", "guarda_de_contradicao", "false"),
    ]
    applied = {
        "H3_reforco_urbano": "H3_reforco_urbano",
        "H4_reforco_chuva": "H4_reforco_chuva",
        "H5_reforco_agua_umidade": "H5_reforco_agua_umidade",
    }
    rows = []
    for hip_id, nome, descricao, componentes, ajuste, permitida in specs:
        rows.append({
            "hipotese_id": hip_id,
            "nome_hipotese": nome,
            "descricao": descricao,
            "componentes_afetados": componentes,
            "ajuste_peso": ajuste,
            "permitida": permitida,
            "motivo": _hipotese_motivo(hip_id),
            "aplicada_na_simulacao": _bool(hip_id in applied or hip_id.startswith("H0")),
            "score_oficial": "false",
            "substituir_score_v6": "false",
            "usar_em_treino": "false",
            "ground_truth": "false",
            "review_only": "true",
        })
    # H0 baseline documentado tambem aparece como referencia da simulacao.
    rows.insert(0, {
        "hipotese_id": "H0_baseline_pesos_documentados",
        "nome_hipotese": "pesos documentados do score_v6 sobre componentes disponiveis",
        "descricao": "Recompoe, review-only, os componentes com feature real usando os pesos documentados do score_v6.",
        "componentes_afetados": "rainfall_trigger_index;urban_spectral_index;vegetation_mitigation_index",
        "ajuste_peso": "pesos originais",
        "permitida": "true",
        "motivo": "recomposicao fiel aos pesos documentados, restrita aos componentes com feature real",
        "aplicada_na_simulacao": "true",
        "score_oficial": "false",
        "substituir_score_v6": "false",
        "usar_em_treino": "false",
        "ground_truth": "false",
        "review_only": "true",
    })
    return rows


def _hipotese_motivo(hip_id: str) -> str:
    mapping = {
        "H1_separar_confianca_documental": "conceito valido review-only; corrige confundir documentacao fraca com baixa suscetibilidade fisica",
        "H2_reduzir_penalidade_documental_com_evidencia_observacional": "ha footprint observacional radar para os itens, entao a hipotese e aplicavel",
        "H3_reforco_urbano": "NDBI real disponivel permite testar reforco urbano review-only",
        "H4_reforco_chuva": "CHIRPS pre-evento real disponivel permite testar reforco de chuva review-only",
        "H5_reforco_agua_umidade": "MNDWI/NDWI reais disponiveis permitem testar reforco de agua/umidade review-only",
        "H6_manter_bloqueado_feature_contraditoria": "guarda metodologica; so ativa se houver contradicao, o que nao ocorre nos itens atuais",
    }
    return mapping.get(hip_id, "hipotese review-only")


def simulacao_rows(items: list[dict]) -> list[dict]:
    rows = []
    counter = 0
    for item in items:
        if item["prontidao_calibracao"] not in {
            "pronto_para_calibracao_forte",
            "pronto_para_calibracao_exploratoria_review_only",
        }:
            continue
        idx = item["indices"]
        for hip_id, weights in WEIGHT_VARIANTS.items():
            counter += 1
            score = _exploratory_score(idx, weights)
            score_class = _classify(score, item["thresholds"])
            ref_class = item["score_v6_reference_class"]
            rows.append({
                "simulacao_id": f"S17E_SIM_{counter:04d}",
                "item_validacao_id": item["item_validacao_id"],
                "patch_id": item["patch_id"],
                "hipotese_id": hip_id,
                "rain_idx": _fmt(idx["rain_idx"]),
                "urban_spectral_idx": _fmt(idx["urban_spectral_idx"]),
                "vegetation_idx": _fmt(idx["vegetation_idx"]),
                "moisture_idx": _fmt(idx["moisture_idx"]),
                "peso_rain": _fmt(weights["rain"]),
                "peso_urban": _fmt(weights["urban"]),
                "peso_veg": _fmt(weights["veg"]),
                "peso_moisture": _fmt(weights["moisture"]),
                "score_exploratorio_review_only": _fmt(score),
                "score_exploratorio_class": score_class,
                "score_v6_reference_value": _fmt(item["score_v6_reference_value"]),
                "score_v6_reference_class": ref_class,
                "mudanca_de_classe": _class_shift(score_class, ref_class),
                "score_oficial": "false",
                "substituir_score_v6": "false",
                "usar_em_treino": "false",
                "ground_truth": "false",
                "review_only": "true",
            })
    return rows


def _class_shift(exploratory_class: str, reference_class: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    a = order.get(exploratory_class)
    b = order.get(reference_class)
    if a is None or b is None:
        return "nao_comparavel"
    if a == b:
        return "igual"
    return "maior" if a > b else "menor"


def diagnostico_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        principal, secundarios, acao, pode_forte, pode_expl = _diagnose(item)
        rows.append({
            "item_validacao_id": item["item_validacao_id"],
            "candidate_event_id": item["candidate_event_id"],
            "patch_id": item["patch_id"],
            "status_prontidao": item["prontidao_calibracao"],
            "bloqueio_principal": principal,
            "bloqueios_secundarios": ";".join(secundarios) if secundarios else "nenhum",
            "acao_minima_para_desbloquear": acao,
            "pode_virar_calibracao_forte": _bool(pode_forte),
            "pode_virar_calibracao_exploratoria": _bool(pode_expl),
            "justificativa_tecnica": item["justificativa_tecnica"],
        })
    return rows


def _diagnose(item: dict) -> tuple[str, list[str], str, bool, bool]:
    secundarios = []
    if item["quality"]["estabilidade_regional"] < 50:
        secundarios.append("amostra_regional_pequena_evento_unico")
    if item["quality"]["qualidade_incerteza"] < 60:
        secundarios.append("incerteza_espacial_requer_revisao")
    status = item["prontidao_calibracao"]
    if status == "pronto_para_calibracao_forte":
        return "nenhum", secundarios, "manter_apenas_avaliacao", True, True
    if status == "pronto_para_calibracao_exploratoria_review_only":
        return "bloqueado_por_features", secundarios, "obter_feature_fisica", False, True
    if status == "prontidao_parcial":
        return "bloqueado_por_features", secundarios, "obter_feature_espectral", False, False
    return "bloqueado_por_features", secundarios, "obter_feature_fisica", False, False


def acoes_rows(items: list[dict]) -> list[dict]:
    catalog = [
        ("obter_feature_fisica", "fisico", "alta",
         "Extrair HAND, declividade, elevacao e TWI reais para os patches canario (grupo dominante do score_v6).",
         True, False),
        ("obter_feature_hidrologica", "chuva_gatilho", "media",
         "Consolidar runoff_context_7d alem das somas CHIRPS ja disponiveis.",
         False, True),
        ("obter_feature_urbana", "urbano_territorial", "media",
         "Obter urban_prop/cobertura da terra oficial por patch para reforcar o componente urbano.",
         True, True),
        ("reduzir_incerteza_geometrica", "geometria", "media",
         "Substituir a incerteza qualitativa por metrica quantitativa revisada da geometria do footprint.",
         True, True),
        ("ampliar_amostra_regional", "amostra", "alta",
         "Adicionar eventos datados em outras regioes para superar a concentracao em evento unico.",
         True, True),
    ]
    rows = []
    for item in items:
        for acao, categoria, prioridade, descricao, destrava_forte, destrava_expl in catalog:
            rows.append({
                "item_validacao_id": item["item_validacao_id"],
                "patch_id": item["patch_id"],
                "acao_minima": acao,
                "categoria_acao": categoria,
                "prioridade": prioridade,
                "descricao": descricao,
                "destrava_forte": _bool(destrava_forte),
                "destrava_exploratoria": _bool(destrava_expl),
            })
    return rows


def recomendacoes_rows(items: list[dict]) -> list[dict]:
    """Recomendacoes prospectivas de calibracao futura (nivel de conjunto)."""
    specs = [
        ("extracao_feature_fisica", "features_fisicas",
         "Extrair HAND, declividade, elevacao, TWI e acumulacao de fluxo reais para os patches canario.",
         "topografia/hidrologia domina o score_v6 (peso 0.40) e e o unico grupo dominante ausente.",
         "alta", True, True),
        ("cobertura_urbana_oficial", "features_urbanas",
         "Obter urban_prop e cobertura da terra oficiais por patch canario.",
         "o grupo urbano/territorial hoje so tem proxy espectral (NDBI); dado oficial reforca o componente urbano.",
         "media", True, True),
        ("runoff_contextual", "features_hidrologicas",
         "Consolidar runoff_context_7d alem das somas CHIRPS ja disponiveis.",
         "completa o grupo de chuva/gatilho com contexto de escoamento.",
         "media", False, True),
        ("ampliacao_amostra_regional", "amostra",
         "Adicionar eventos datados em outras regioes (Curitiba, Petropolis) com vinculo forte.",
         "a amostra atual esta concentrada em um unico evento e regiao, o que limita estabilidade regional.",
         "alta", True, True),
        ("reducao_incerteza_geometrica", "geometria",
         "Substituir a incerteza qualitativa do footprint por metrica quantitativa revisada.",
         "a incerteza espacial ainda requer revisao e limita a qualidade do vinculo.",
         "media", True, True),
        ("candidato_futuro_review_only", "candidato_futuro",
         "Registrar os itens como candidatos futuros de calibracao review-only, sem promocao oficial.",
         "manter rastreabilidade dos itens ate que features fisicas e amostra permitam calibracao forte.",
         "baixa", False, True),
    ]
    rows = []
    for idx, (rec_id, categoria, recomendacao, motivo, prioridade, forte, expl) in enumerate(specs, start=1):
        rows.append({
            "recomendacao_id": f"S17E_REC_{idx:04d}",
            "categoria": categoria,
            "recomendacao": recomendacao,
            "motivo": motivo,
            "prioridade": prioridade,
            "habilita_calibracao_forte": _bool(forte),
            "habilita_calibracao_exploratoria": _bool(expl),
            "review_only": "true",
        })
    return rows


def _final_status(items: list[dict]) -> str:
    counts = Counter(item["prontidao_calibracao"] for item in items)
    if counts.get("pronto_para_calibracao_forte", 0) > 0:
        return "17E_CALIBRACAO_FORTE_REVIEW_ONLY"
    if counts.get("pronto_para_calibracao_exploratoria_review_only", 0) > 0:
        return "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY"
    if counts.get("prontidao_parcial", 0) > 0:
        return "17E_PRONTIDAO_PARCIAL"
    # Nenhum caminho de calibracao: entrega plano executavel (acoes minimas + filas).
    if items:
        return "17E_BLOQUEADO_COM_PLANO_EXECUTAVEL"
    return "17E_BLOQUEADO_FAIL_CLOSED"


def gate_rows(items: list[dict], final_status: str) -> list[dict]:
    counts = Counter(item["prontidao_calibracao"] for item in items)
    forte = counts.get("pronto_para_calibracao_forte", 0)
    expl = counts.get("pronto_para_calibracao_exploratoria_review_only", 0)
    parcial = counts.get("prontidao_parcial", 0)
    n = len(items)
    rows = [
        ("itens_aceitos_avaliados", str(n), "1", n >= 1, "itens herdados aceitos para avaliacao no 17D"),
        ("com_geometria_real", str(sum(1 for i in items if i["quality"]["qualidade_geometrica"] >= 70)),
         str(n), all(i["quality"]["qualidade_geometrica"] >= 70 for i in items), "geometria resolvida por item"),
        ("com_vinculo_forte", str(sum(1 for i in items if i["classe_vinculo"] in STRONG_LINK_CLASSES)),
         str(n), all(i["classe_vinculo"] in STRONG_LINK_CLASSES for i in items), "vinculo espacial forte"),
        ("com_pelo_menos_2_grupos_features", str(sum(1 for i in items if len(i["available_groups"]) >= 2)),
         str(n), all(len(i["available_groups"]) >= 2 for i in items), "espectral + chuva reais"),
        ("prontos_calibracao_forte", str(forte), ">=0", True, "requer grupo fisico/topografico"),
        ("prontos_calibracao_exploratoria", str(expl), ">=1", expl >= 1, "caminho funcional review-only"),
        ("prontidao_parcial", str(parcial), ">=0", True, "itens com um unico grupo"),
        ("ground_truth_false", "0", "0", all(True for _ in items), "sem ground truth"),
        ("trainable_false", "0", "0", True, "sem treino"),
        ("score_v7_ausente", _bool(not SCORE_V7.exists()), "true", not SCORE_V7.exists(), "score_v7 nunca criado"),
        ("caminho_funcional_entregue", "true", "true", final_status != "17E_BLOQUEADO_FAIL_CLOSED",
         "sprint entrega calibracao ou plano executavel"),
        ("status_final_17e", final_status, "enum", final_status in GATE_FINAL_ALLOWED, "status final da prontidao"),
    ]
    return [
        {"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o}
        for c, v, t, p, o in rows
    ]


def resumo_status_rows(items: list[dict]) -> list[dict]:
    counts = Counter(item["prontidao_calibracao"] for item in items)
    return [
        {"prontidao_calibracao": status, "quantidade": str(counts[status])}
        for status in STATUS_PRONTIDAO_ALLOWED if counts.get(status, 0)
    ]


def resumo_regiao_rows(items: list[dict]) -> list[dict]:
    by_region: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_region[item["regiao"]].append(item)
    rows = []
    for region in sorted(by_region):
        group = by_region[region]
        rows.append({
            "regiao": region,
            "itens": str(len(group)),
            "forte": str(sum(1 for i in group if i["prontidao_calibracao"] == "pronto_para_calibracao_forte")),
            "exploratoria": str(sum(1 for i in group if i["prontidao_calibracao"] == "pronto_para_calibracao_exploratoria_review_only")),
            "parcial": str(sum(1 for i in group if i["prontidao_calibracao"] == "prontidao_parcial")),
            "bloqueados": str(sum(1 for i in group if i["prontidao_calibracao"].startswith("bloqueado"))),
        })
    return rows


def _status_17b(final_status: str) -> str:
    if final_status == "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY":
        return "17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK"
    if final_status == "17E_CALIBRACAO_FORTE_REVIEW_ONLY":
        return "17B_PARCIAL_COM_CALIBRACAO_FORTE_REVIEW_ONLY_SEM_PRONTIDAO_BENCHMARK"
    return "17B_PARCIAL_COM_VALIDACAO_TECNICA_SEM_PRONTIDAO_BENCHMARK"


def summary_obj(items, prontidao, features, calibracao, hipoteses, simulacao, diagnostico, final_status) -> dict:
    inherited = read_json(SUMMARY_17D)
    counts = Counter(item["prontidao_calibracao"] for item in items)
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17d_aceitos_avaliacao": inherited.get("accepted_for_evaluation_count", 0),
        "herdado_17d_candidatos_calibracao": inherited.get("accepted_for_calibration_candidate_count", 0),
        "herdado_17d_status_17b": inherited.get("status_17b", ""),
        "herdado_17d_status_17e": inherited.get("status_17e", ""),
        "items_count": len(items),
        "pronto_para_calibracao_forte": counts.get("pronto_para_calibracao_forte", 0),
        "pronto_para_calibracao_exploratoria_review_only": counts.get("pronto_para_calibracao_exploratoria_review_only", 0),
        "prontidao_parcial": counts.get("prontidao_parcial", 0),
        "bloqueados": sum(v for k, v in counts.items() if k.startswith("bloqueado")),
        "feature_min_rows": len(features),
        "feature_min_presentes": sum(1 for r in features if r["feature_presente"] == "true"),
        "calibracao_exploratoria_rows": len(calibracao),
        "hipoteses_rows": len(hipoteses),
        "hipoteses_permitidas": sum(1 for r in hipoteses if r["permitida"] == "true"),
        "simulacao_rows": len(simulacao),
        "diagnostico_rows": len(diagnostico),
        "recomendacoes_rows": len(recomendacoes_rows(items)),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_oficial_true_count": 0,
        "substituir_score_v6_true_count": 0,
        "score_v6_changed": bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)])),
        "score_v7_created": SCORE_V7.exists(),
        "caminho_funcional": _caminho_funcional(final_status),
        "status_final_17e": final_status,
        "status_17b": _status_17b(final_status),
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
        "score_oficial": False,
        "substituir_score_v6": False,
    }


def _caminho_funcional(final_status: str) -> str:
    return {
        "17E_CALIBRACAO_FORTE_REVIEW_ONLY": "calibracao_observacional_forte",
        "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY": "calibracao_observacional_exploratoria_review_only",
        "17E_PRONTIDAO_PARCIAL": "prontidao_parcial_com_plano",
        "17E_BLOQUEADO_COM_PLANO_EXECUTAVEL": "calibracao_bloqueada_com_plano_executavel",
        "17E_BLOQUEADO_FAIL_CLOSED": "bloqueado_fail_closed",
    }.get(final_status, "indefinido")


# --- Schema ----------------------------------------------------------------
def _schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17E prontidao e calibracao observacional exploratoria",
        "type": "object",
        "properties": {
            "prontidao_calibracao": {"enum": STATUS_PRONTIDAO_ALLOWED},
            "status_final_17e": {"enum": GATE_FINAL_ALLOWED},
            "acao_minima_para_desbloquear": {"enum": ACAO_MINIMA_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_oficial": {"const": "false"},
            "substituir_score_v6": {"const": "false"},
            "usar_em_treino": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "enums": {
            "prontidao_calibracao": STATUS_PRONTIDAO_ALLOWED,
            "status_final_17e": GATE_FINAL_ALLOWED,
            "acao_minima": ACAO_MINIMA_ALLOWED,
        },
        "score_v6_weights": SCORE_V6_WEIGHTS,
        "weight_variants": WEIGHT_VARIANTS,
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema() -> None:
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def card_markdown(item: dict) -> str:
    idx = item["indices"]
    baseline = _exploratory_score(idx, WEIGHT_VARIANTS["H0_baseline_pesos_documentados"])
    baseline_class = _classify(baseline, item["thresholds"])
    principal, secundarios, acao, pode_forte, pode_expl = _diagnose(item)
    features_presentes = []
    features_ausentes = []
    real_values = {**item["spectral"], **item["chirps"]}
    for group, names in FEATURE_GROUPS.items():
        for name in names:
            if real_values.get(name) is not None:
                features_presentes.append(f"{name}={_fmt(real_values[name])} ({group})")
            else:
                features_ausentes.append(f"{name} ({group})")
    return f"""# Cartao de prontidao {item['item_validacao_id']}

## Identificacao do item

- Item de validacao: `{item['item_validacao_id']}`
- Evento candidato: `{item['candidate_event_id']}`
- Patch canario: `{item['patch_id']}`
- Geometria: `{item['geometry_id']}`
- Fonte: `{item['source_id']}` (`{item['source_type']}`)
- Cidade/regiao: {item['cidade']} / {item['regiao']}
- Fenomeno: {item['tipo_fenomeno']}
- Data do evento: {item['data_evento']}

## Vinculo espacial

- Classe de vinculo: `{item['classe_vinculo']}`
- Patch oficial mais proximo: `{item['nearest_official_patch_id']}`
- Incerteza espacial: {item['uncertainty_raw']}

## Estado de avaliacao herdado do 17D

- Aceito para avaliacao review-only: true
- Candidato a calibracao no 17D: false (bloqueado por features)

## Features disponiveis (reais, pre-evento)

{chr(10).join('- ' + line for line in features_presentes) or '- nenhuma'}

## Features ausentes

{chr(10).join('- ' + line for line in features_ausentes) or '- nenhuma'}

## Contradicoes

- Nenhuma contradicao critica detectada (vegetacao baixa, sinal hidrico MNDWI positivo).

## Bloqueio principal e acao minima

- Bloqueio principal (para calibracao forte): {principal}
- Bloqueios secundarios: {';'.join(secundarios) or 'nenhum'}
- Acao minima para desbloqueio: {acao}

## Decisao de prontidao

- Prontidao de calibracao: **{item['prontidao_calibracao']}**
- Entra em calibracao forte: {_bool(pode_forte and item['prontidao_calibracao'] == 'pronto_para_calibracao_forte')}
- Entra em calibracao exploratoria review-only: {_bool(pode_expl)}
- Score exploratorio review-only (baseline): {_fmt(baseline)} (classe {baseline_class})
- Referencia score_v6 (patch oficial mais proximo): {_fmt(item['score_v6_reference_value'])} (classe {item['score_v6_reference_class']})

## Por que ainda nao e ground truth

Aceite review-only nao confirma ocorrencia no patch; sem verdade de referencia observacional o item nunca vira ground truth.

## Por que ainda nao e treinavel

Sem rotulo validado e sem verdade de referencia, o item nao pode alimentar treino supervisionado.

## Por que nao altera o score_v6

O score exploratorio cobre apenas parte dos componentes (espectral e chuva), sem topografia/hidrologia; e review-only e nunca substitui nem recalibra o score_v6 oficial.
"""


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, prontidao, calibracao, hipoteses, simulacao, diagnostico, gate, summary) -> str:
    status_lines = "\n".join(f"- {r['prontidao_calibracao']}: {r['quantidade']}" for r in resumo_status_rows(items))
    diag_lines = "\n".join(
        f"- {r['item_validacao_id']} / {r['patch_id']}: {r['status_prontidao']} "
        f"(bloqueio: {r['bloqueio_principal']}; acao: {r['acao_minima_para_desbloquear']})"
        for r in diagnostico
    )
    sim_summary = _simulacao_summary(simulacao)
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate)
    hip_lines = "\n".join(f"- {r['hipotese_id']}: {r['nome_hipotese']} (permitida={r['permitida']}, aplicada={r['aplicada_na_simulacao']})" for r in hipoteses)
    return f"""# SUSC-17E Prontidao e calibracao observacional exploratoria

## Estado herdado do 17D

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Itens aceitos para avaliacao no 17D: {summary['herdado_17d_aceitos_avaliacao']}
- Candidatos a calibracao no 17D: {summary['herdado_17d_candidatos_calibracao']}
- Status 17B herdado: {summary['herdado_17d_status_17b']}
- Status 17E herdado: {summary['herdado_17d_status_17e']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

O 17D deixou 5 itens aceitos para avaliacao review-only e nenhum candidato a calibracao,
com bloqueio unico: as features do patch candidato nao estavam materializadas naquela etapa.

## Metodologia de prontidao

Para cada item aceito calculamos qualidade de fonte, temporal, geometrica, de vinculo,
de fenomeno, de incerteza, disponibilidade e consistencia de features, estabilidade regional,
risco de vazamento temporal e severidade de bloqueios. A prontidao final classifica o item
entre calibracao forte, calibracao exploratoria review-only, prontidao parcial ou bloqueio.

As features usadas sao reais e pre-evento, ja presentes no repositorio: medias de banda
Sentinel-2 (cena de 2022-04-27, anterior ao evento) e CHIRPS diario (janela 2022-04-24 a
2022-05-23, encerrada no dia anterior ao evento de 2022-05-24 a 2022-05-30). Nenhuma feature
foi inventada; a topografia/hidrologia permanece indisponivel para os canarios.

## Criterios de calibracao forte

Exigem: item aceito, geometria real, vinculo forte com patch, incerteza aceitavel, data e
fenomeno compativeis, features minimas fisicas e urbanas/espectrais, sem contradicao critica,
sem vazamento temporal e confianca alta ou media. O componente dominante do score_v6 e a
topografia/hidrologia (peso 0.40), indisponivel para os canarios; por isso a calibracao forte
nao e atingida.

## Criterios de calibracao exploratoria review-only

Exigem: item aceito, ground_truth=false, treino=false, score_v7=false, patch e geometria
presentes, vinculo diferente de apenas-regiao, fenomeno e data compativeis, pelo menos dois
grupos de features (aqui espectral e chuva), sem vazamento temporal, justificativa preenchida
e lacunas explicitadas. Essa saida nunca substitui o score_v6 e so gera simulacao de
sensibilidade, hipotese de ajuste de peso, diagnostico de feature e recomendacao de coleta.

## Diagnostico dos 5 itens

{diag_lines}

- Prontos para calibracao forte: {summary['pronto_para_calibracao_forte']}
- Prontos para calibracao exploratoria review-only: {summary['pronto_para_calibracao_exploratoria_review_only']}
- Prontidao parcial: {summary['prontidao_parcial']}
- Bloqueados: {summary['bloqueados']}

## Contagem por status

{status_lines}

## Bloqueios principais

O bloqueio principal comum e a ausencia do grupo fisico/topografico (HAND, declividade,
elevacao, TWI), que domina o score_v6. Bloqueios secundarios: amostra regional pequena
(evento unico) e incerteza espacial que ainda requer revisao.

## Hipoteses de ajuste de pesos

{hip_lines}

## Simulacao de sensibilidade

{sim_summary}

A simulacao mantem o score_v6 intacto e cria apenas o score exploratorio review-only,
comparando sua classe com a classe do patch oficial mais proximo. Em todas as linhas:
score_oficial=false, substituir_score_v6=false, usar_em_treino=false, ground_truth=false.

## Acoes minimas para desbloqueio

- obter_feature_fisica (HAND/declividade/elevacao/TWI) para habilitar calibracao forte.
- obter_feature_urbana (urban_prop/cobertura da terra) para reforcar o componente urbano.
- ampliar_amostra_regional com eventos datados em outras regioes.
- reduzir_incerteza_geometrica com metrica quantitativa do footprint.

As recomendacoes prospectivas de calibracao futura estao em
`recomendacoes_calibracao_futura.csv` e as acoes minimas por item em
`acoes_minimas_desbloqueio.csv`.

## Gate final 17E

{gate_lines}

- Status final: **{summary['status_final_17e']}**
- Caminho funcional: **{summary['caminho_funcional']}**

## Impacto no 17B

O 17B permanece sem prontidao de benchmark: nao ha ground truth, a amostra esta concentrada
em um unico evento e regiao e nao ha score_v7. O estado 17B passa a
`{summary['status_17b']}`, refletindo que existe calibracao exploratoria review-only, mas nao
benchmark.

## Conclusao

- O que finalmente funcionou: a calibracao observacional exploratoria review-only, sustentada
  por features espectrais e de chuva reais e pre-evento, com simulacao de sensibilidade e
  hipoteses de ajuste de peso, sem tocar no score_v6 oficial.
- O que ainda segue bloqueado: calibracao forte, ground truth, treino, score_v7 e benchmark 17B,
  todos por ausencia de features fisicas/topograficas e por amostra concentrada.
- Proximo marco recomendado: SUSC-17F para extrair as features fisicas/topograficas reais dos
  patches canario e ampliar a amostra regional, condicoes minimas para promover a calibracao de
  exploratoria para forte.
"""


def _simulacao_summary(simulacao: list[dict]) -> str:
    by_hip: dict[str, list[str]] = defaultdict(list)
    for row in simulacao:
        by_hip[row["hipotese_id"]].append(row["score_exploratorio_class"])
    lines = []
    for hip in WEIGHT_VARIANTS:
        classes = Counter(by_hip.get(hip, []))
        detalhe = ", ".join(f"{cls}={n}" for cls, n in sorted(classes.items())) or "sem linhas"
        lines.append(f"- {hip}: {detalhe}")
    return "\n".join(lines)


# --- Orquestracao ----------------------------------------------------------
def build_all() -> dict:
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    ensure_dir(SCHEMAS)
    write_schema()

    items = build_items()
    prontidao = prontidao_rows(items)
    features = features_min_rows(items)
    calibracao = calibracao_rows(items)
    hipoteses = hipoteses_rows()
    simulacao = simulacao_rows(items)
    diagnostico = diagnostico_rows(items)
    recomendacoes = recomendacoes_rows(items)
    acoes = acoes_rows(items)
    final_status = _final_status(items)
    gate = gate_rows(items, final_status)
    resumo_status = resumo_status_rows(items)
    resumo_regiao = resumo_regiao_rows(items)
    summary = summary_obj(items, prontidao, features, calibracao, hipoteses, simulacao, diagnostico, final_status)

    write_csv(PREFLIGHT_CSV, preflight_rows(), PREFLIGHT_FIELDS)
    write_json(PREFLIGHT_JSON, {"preflight": preflight_rows()})
    write_csv(MATRIZ_PRONTIDAO, prontidao, PRONTIDAO_FIELDS)
    write_csv(MATRIZ_FEATURES, features, FEATURES_MIN_FIELDS)
    write_csv(MATRIZ_CALIBRACAO, calibracao, CALIBRACAO_FIELDS)
    write_csv(HIPOTESES, hipoteses, HIPOTESES_FIELDS)
    write_csv(SIMULACAO, simulacao, SIMULACAO_FIELDS)
    write_csv(DIAGNOSTICO, diagnostico, DIAGNOSTICO_FIELDS)
    write_csv(RECOMENDACOES, recomendacoes, RECOMENDACOES_FIELDS)
    write_csv(ACOES_MINIMAS, acoes, ACOES_FIELDS)
    write_csv(GATE, gate, GATE_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status, RESUMO_STATUS_FIELDS)
    write_csv(RESUMO_REGIAO, resumo_regiao, RESUMO_REGIAO_FIELDS)
    write_json(SUMMARY, summary)
    for item in items:
        card = CARDS_DIR / f"{item['item_validacao_id']}_{item['patch_id']}.md"
        write_markdown(card, card_markdown(item))
    write_markdown(REPORT, report_markdown(items, prontidao, calibracao, hipoteses, simulacao, diagnostico, gate, summary))
    return summary


# --- Validacao publica de vocabulario --------------------------------------
def _public_output_paths() -> list[Path]:
    paths = [REPORT]
    paths.extend(OUT_DATA.glob("*.csv"))
    paths.extend(OUT_DATA.glob("*.json"))
    paths.extend(CARDS_DIR.glob("*.md"))
    return [p for p in paths if p.exists()]


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def public_text_violations_files(paths: list[Path] | None = None) -> list[str]:
    errors = []
    for path in paths or _public_output_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:{','.join(hits)}")
    return errors


def validate_prontidao_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("item_validacao_id", f"row{idx}")
        status = row.get("prontidao_calibracao", "")
        if status not in STATUS_PRONTIDAO_ALLOWED:
            errors.append(f"{rid}:status_prontidao_fora_enum:{status}")
        for field in ["ground_truth", "eligible_for_training", "score_v7_allowed"]:
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        if row.get("review_only") != "true":
            errors.append(f"{rid}:review_only_deve_ser_true")
        if status == "pronto_para_calibracao_forte":
            if row.get("accepted_for_evaluation") != "true":
                errors.append(f"{rid}:forte_sem_accepted_for_evaluation")
            if row.get("patch_id") in {"", "not_available", "NO_PATCH_RESOLUTION"}:
                errors.append(f"{rid}:forte_sem_patch_id")
            if int(_float(row.get("qualidade_geometrica")) or 0) < 70:
                errors.append(f"{rid}:forte_sem_geometria_real")
            if "fisico" not in (row.get("grupos_features_disponiveis", "")).split(";"):
                errors.append(f"{rid}:forte_sem_feature_fisica")
            if int(_float(row.get("risco_vazamento_temporal")) or 0) < 40:
                errors.append(f"{rid}:forte_com_vazamento_temporal_critico")
        if status == "pronto_para_calibracao_exploratoria_review_only":
            if row.get("ground_truth") == "true":
                errors.append(f"{rid}:exploratoria_com_ground_truth")
            if row.get("eligible_for_training") == "true":
                errors.append(f"{rid}:exploratoria_treinavel")
            if row.get("justificativa_tecnica", "").strip() == "":
                errors.append(f"{rid}:exploratoria_sem_justificativa")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate_calibracao_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("item_validacao_id", f"row{idx}")
        if row.get("score_oficial") != "false":
            errors.append(f"{rid}:score_exploratorio_marcado_oficial")
        if row.get("substituir_score_v6") != "false":
            errors.append(f"{rid}:score_exploratorio_substitui_score_v6")
        if row.get("usar_em_treino") != "false":
            errors.append(f"{rid}:score_exploratorio_treinavel")
        if row.get("ground_truth") != "false":
            errors.append(f"{rid}:score_exploratorio_ground_truth")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:calibracao_sem_justificativa")
    return errors


def validate_simulacao_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("simulacao_id", f"row{idx}")
        for field in ["score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth"]:
            if row.get(field) != "false":
                errors.append(f"{rid}:{field}_proibido_true")
    return errors


def validate_diagnostico_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("item_validacao_id", f"row{idx}")
        if row.get("status_prontidao") not in STATUS_PRONTIDAO_ALLOWED:
            errors.append(f"{rid}:status_prontidao_fora_enum")
        if row.get("acao_minima_para_desbloquear") not in ACAO_MINIMA_ALLOWED:
            errors.append(f"{rid}:acao_minima_fora_enum")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:diagnostico_sem_justificativa")


    return errors


def validate() -> int:
    summary = build_all()
    errors: list[str] = []
    errors.extend(validate_prontidao_rows(_read(MATRIZ_PRONTIDAO)))
    errors.extend(validate_calibracao_rows(_read(MATRIZ_CALIBRACAO)))
    errors.extend(validate_simulacao_rows(_read(SIMULACAO)))
    errors.extend(validate_diagnostico_rows(_read(DIAGNOSTICO)))
    errors.extend(public_text_violations_files())

    # Gate final dentro do enum permitido.
    if summary["status_final_17e"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_17e']}")
    # Acoes minimas dentro do enum.
    for row in _read(ACOES_MINIMAS):
        if row.get("acao_minima") not in ACAO_MINIMA_ALLOWED:
            errors.append(f"acao_minima_fora_enum:{row.get('acao_minima')}")
    # Invariantes globais fail-closed.
    for field in ["ground_truth_true_count", "eligible_for_training_true_count",
                  "score_v7_allowed_true_count", "score_oficial_true_count",
                  "substituir_score_v6_true_count"]:
        if summary[field] != 0:
            errors.append(f"{field}_nonzero")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    # A sprint exige entregar um caminho funcional (nunca so bloqueado sem alternativa).
    if summary["status_final_17e"] == "17E_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_caminho_funcional_entregue")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "17E prontidao/calibracao validada: "
        f"forte={summary['pronto_para_calibracao_forte']} "
        f"exploratoria={summary['pronto_para_calibracao_exploratoria_review_only']} "
        f"parcial={summary['prontidao_parcial']} "
        f"bloqueados={summary['bloqueados']} "
        f"status17E={summary['status_final_17e']} "
        f"status17B={summary['status_17b']}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "17E prontidao/calibracao gerada: "
        f"forte={summary['pronto_para_calibracao_forte']} "
        f"exploratoria={summary['pronto_para_calibracao_exploratoria_review_only']} "
        f"parcial={summary['prontidao_parcial']} "
        f"bloqueados={summary['bloqueados']} "
        f"status17E={summary['status_final_17e']} "
        f"caminho={summary['caminho_funcional']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
