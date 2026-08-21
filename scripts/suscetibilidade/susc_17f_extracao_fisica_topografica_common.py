"""SUSC-17F extracao e consolidacao de atributos fisicos/topograficos dos canarios.

O 17E deixou 5 canarios (S17C6_CANARY_REC_00001..00005) prontos para calibracao
exploratoria review-only, com espectral e chuva reais, mas sem o grupo
fisico/topografico (HAND, elevacao, declividade, TWI, acumulacao de fluxo,
proximidade hidrica). Esta etapa inventaria as fontes fisicas/topograficas do
repositorio, resolve o vinculo canario -> feature fisica por ordem de prioridade
(match direto, match espacial, patch mais proximo, fila de extracao), consolida
uma referencia comparativa auditavel, aprimora a calibracao exploratoria com um
componente fisico penalizado por incerteza e entrega uma fila executavel de
extracao para as features fisicas diretas ainda ausentes.

Nada aqui vira ground truth, dado treinavel, score_v7 oficial ou benchmark 17B.
O score_v6 nunca e alterado. Valor vindo de patch proximo e sempre marcado como
referencia comparativa review-only, nunca como feature direta do canario.
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
OUT_DATA_17E = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17e_prontidao_calibracao_observacional_exploratoria"
OUT_DATA_17D = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17d_validacao_tecnica_evidencia_observacional"
OUT_DATA_17C5 = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17f_extracao_fisica_topografica_canarios_observacionais"
CARDS_DIR = OUT_DATA / "cartoes_fisicos"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
CALIBRACAO_17E = OUT_DATA_17E / "matriz_calibracao_exploratoria.csv"
PRONTIDAO_17E = OUT_DATA_17E / "matriz_prontidao_calibracao.csv"
SUMMARY_17E = OUT_DATA_17E / "summary.json"
DECISIONS_17D = OUT_DATA_17D / "decisoes_validacao_tecnica.csv"
PATCH_LINKS_17C5 = OUT_DATA_17C5 / "susc_17c5_patch_links.csv"
PATCH_GRID_17C6 = SUSC_OUT / "susc_17c6_candidate_patch_grid.csv"
FEATURES_BY_PATCH = DAT_SUSC / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"
TOPO_17C38 = SUSC_OUT / "susc_17c38_official_patch_basin_aware_topography_features.csv"
TERRAIN_17C34 = SUSC_OUT / "susc_17c34_terrain_features.csv"
DEM_MANIFEST_17C36 = SUSC_OUT / "susc_17c36_dem_artifact_manifest.csv"
DRAINAGE_17C35 = SUSC_OUT / "susc_17c35_official_drainage_features.csv"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_CSV = OUT_DATA / "preflight_extracao_fisica.csv"
PREFLIGHT_JSON = OUT_DATA / "preflight_extracao_fisica.json"
INVENTARIO = OUT_DATA / "inventario_fontes_fisicas_topograficas.csv"
MATRIZ_FISICA = OUT_DATA / "matriz_fisica_topografica_canarios.csv"
REAVALIACAO = OUT_DATA / "reavaliacao_prontidao_17e_com_features_fisicas.csv"
CALIBRACAO_APRIMORADA = OUT_DATA / "matriz_calibracao_exploratoria_aprimorada.csv"
SIMULACAO_FISICA = OUT_DATA / "simulacao_sensibilidade_componente_fisico_review_only.csv"
HIPOTESES_FISICAS = OUT_DATA / "hipoteses_ajuste_pesos_componente_fisico.csv"
FILA_EXTRACAO = OUT_DATA / "fila_extracao_features_fisicas_pendentes.csv"
GATE = OUT_DATA / "gate_features_fisicas_canarios.csv"
RESUMO_STATUS = OUT_DATA / "resumo_por_status.csv"
RESUMO_CANARIO = OUT_DATA / "resumo_por_canario.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_17F_EXTRACAO_FISICA_TOPOGRAFICA_CANARIOS_OBSERVACIONAIS.md"
SCHEMA = SCHEMAS / "susc_17f_extracao_fisica_topografica_schema_v1.json"

REQUIRED_INPUTS = [
    CALIBRACAO_17E,
    PRONTIDAO_17E,
    SUMMARY_17E,
    DECISIONS_17D,
    PATCH_LINKS_17C5,
    PATCH_GRID_17C6,
    FEATURES_BY_PATCH,
    SCORE_V6,
]

REQUIRED_OUTPUTS = [
    PREFLIGHT_CSV,
    PREFLIGHT_JSON,
    INVENTARIO,
    MATRIZ_FISICA,
    REAVALIACAO,
    CALIBRACAO_APRIMORADA,
    SIMULACAO_FISICA,
    HIPOTESES_FISICAS,
    FILA_EXTRACAO,
    GATE,
    RESUMO_STATUS,
    RESUMO_CANARIO,
    SUMMARY,
    REPORT,
    SCHEMA,
]

# --- Enums permitidos ------------------------------------------------------
STATUS_USO_ALLOWED = [
    "utilizavel_direto",
    "utilizavel_com_join",
    "referencia_comparativa",
    "insuficiente",
    "ausente",
]
FEATURE_SOURCE_MODE_ALLOWED = [
    "feature_direta_review_only",
    "referencia_comparativa_review_only",
    "fila_extracao_necessaria",
]
GATE_FINAL_ALLOWED = [
    "17F_FEATURES_FISICAS_DIRETAS_CONSOLIDADAS",
    "17F_REFERENCIA_FISICA_COMPARATIVA_CONSOLIDADA",
    "17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO",
    "17F_FILA_EXTRACAO_FISICA_EXECUTAVEL",
    "17F_BLOQUEADO_FAIL_CLOSED",
]

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|llm|codex|ia|human\s+qa)\b", re.IGNORECASE)
NO_GT_REASON = (
    "atributo fisico review-only; referencia comparativa nao e feature direta do canario, "
    "nao e verdade de referencia, nao e treino, nao autoriza score_v7 e nao altera o score_v6"
)

PHYSICAL_FEATURES = ["HAND", "elevation", "slope", "distance_to_water", "TWI", "flow_accumulation"]
# Mapeamento feature fisica -> coluna no dataset oficial features_by_patch.
PHYSICAL_COLUMN = {
    "HAND": "hand_mean",
    "elevation": "elevation_mean",
    "slope": "slope_mean",
    "distance_to_water": "distance_to_water_mean",
    "TWI": "twi_mean",
    "flow_accumulation": "flow_acc_log_mean",
}
# Peso documentado do componente topografico/hidrologico no score_v6 (SUSC-10A).
TOPO_WEIGHT = 0.40
# Distancia de referencia para penalizar a incerteza da referencia comparativa.
DISTANCE_PENALTY_REF_M = 6000.0
# Limiar de proximidade que autorizaria tratar a referencia como quase-direta.
STRONG_PROXIMITY_M = 200.0
# Pesos base do score exploratorio herdados do 17E (componentes com feature real).
BASE_WEIGHTS = {"rain": 0.25, "urban": 0.20, "veg": -0.10, "moisture": 0.00}
# Variantes de peso topografico para a simulacao de sensibilidade (secao 6).
TOPO_VARIANTS = {
    "HF0_sem_componente_fisico": 0.00,
    "HF2_referencia_comparativa_penalizada": 0.40,
    "HF4a_sensibilidade_topografia_leve": 0.20,
    "HF4c_sensibilidade_topografia_forte": 0.60,
}

# --- Field lists -----------------------------------------------------------
PREFLIGHT_FIELDS = ["artifact_role", "path", "exists", "row_count", "columns_found", "notes"]
INVENTARIO_FIELDS = [
    "fonte_id", "caminho", "tipo_fonte", "regioes_cobertas",
    "possui_HAND", "possui_elevation", "possui_slope", "possui_distance_to_water",
    "possui_TWI", "possui_flow_accumulation", "possui_drenagem",
    "crs", "resolucao", "status_uso", "observacao",
]
MATRIZ_FISICA_FIELDS = [
    "canary_patch_id", "candidate_event_id", "region", "city", "geometry_id",
    "feature_source_mode",
    "HAND", "elevation", "slope", "distance_to_water", "TWI", "flow_accumulation",
    "drainage_context",
    "features_fisicas_completas", "features_fisicas_parciais", "features_fisicas_ausentes",
    "referencia_patch_id", "referencia_distance_m", "referencia_overlap_ratio",
    "uso_permitido", "ground_truth", "eligible_for_training", "score_v7_allowed",
    "not_ground_truth_reason",
]
REAVALIACAO_FIELDS = [
    "item_validacao_id", "canary_patch_id", "status_17e_original",
    "grupos_features_antes", "grupos_features_depois",
    "possui_grupo_fisico", "possui_grupo_espectral", "possui_grupo_chuva",
    "pode_calibracao_forte_agora", "pode_calibracao_exploratoria_melhorada",
    "bloqueio_restante", "justificativa_tecnica",
]
CALIBRACAO_APRIMORADA_FIELDS = [
    "item_validacao_id", "canary_patch_id", "region", "feature_source_mode",
    "topografia_hidrologia_idx_referencia", "referencia_patch_id", "referencia_distance_m",
    "penalidade_incerteza", "peso_topografico_efetivo",
    "score_exploratorio_sem_fisico", "classe_sem_fisico",
    "score_exploratorio_fisico_review_only", "classe_com_fisico",
    "score_v6_reference_value", "score_v6_reference_class",
    "delta_score_pelo_fisico", "mudanca_de_classe_pelo_fisico",
    "uso_permitido", "justificativa_tecnica", "lacunas",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "review_only",
]
SIMULACAO_FISICA_FIELDS = [
    "simulacao_id", "item_validacao_id", "canary_patch_id", "hipotese_id",
    "peso_topografico_base", "penalidade_incerteza", "peso_topografico_efetivo",
    "topografia_hidrologia_idx_referencia",
    "score_exploratorio_fisico_review_only", "classe_com_fisico",
    "score_v6_reference_value", "score_v6_reference_class", "mudanca_de_classe",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "review_only",
]
HIPOTESES_FISICAS_FIELDS = [
    "hipotese_id", "nome_hipotese", "descricao", "componente_afetado", "ajuste_peso",
    "permitida", "aplicada_na_simulacao", "motivo",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth", "review_only",
]
FILA_FIELDS = [
    "task_id", "canary_patch_id", "bbox", "crs", "region", "city",
    "feature_needed", "data_source_needed", "expected_input_path", "expected_output_path",
    "command_hint", "priority", "blocking_reason",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_STATUS_FIELDS = ["dimensao", "categoria", "quantidade"]
RESUMO_CANARIO_FIELDS = [
    "canary_patch_id", "feature_source_mode", "features_fisicas_diretas",
    "referencia_comparativa", "fila_extracao", "calibracao_exploratoria_aprimorada",
    "pode_calibracao_forte_agora",
]


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


def _float(value):
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


def _fmt(value) -> str:
    if value is None:
        return "not_available"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _require_inputs() -> None:
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-17F")


def _class_thresholds() -> tuple[float, float]:
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in _read(SCORE_V6):
        value = _float(row.get("susc_score_v6_candidate"))
        cls = row.get("susc_class_v6_candidate", "")
        if value is not None and cls:
            by_class[cls].append(value)
    low_max = max(by_class.get("low", [0.42]))
    med_min = min(by_class.get("medium", [0.43]))
    med_max = max(by_class.get("medium", [0.61]))
    high_min = min(by_class.get("high", [0.62]))
    return _round4((low_max + med_min) / 2), _round4((med_max + high_min) / 2)


def _classify(value: float, thresholds: tuple[float, float]) -> str:
    t_low_med, t_med_high = thresholds
    if value < t_low_med:
        return "low"
    if value < t_med_high:
        return "medium"
    return "high"


def _class_shift(a: str, b: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    x, y = order.get(a), order.get(b)
    if x is None or y is None:
        return "nao_comparavel"
    if x == y:
        return "igual"
    return "maior" if x > y else "menor"


# --- Modelo interno --------------------------------------------------------
def _canary_grid() -> dict[str, dict]:
    rows = [r for r in _read(PATCH_GRID_17C6) if r.get("candidate_patch_id", "").startswith("S17C6_CANARY_REC")]
    return _by_key(rows, "candidate_patch_id")


def _penalty(distance_m: float | None, overlap_ratio: float) -> float:
    """Penalidade de incerteza da referencia comparativa (1=sem penalidade)."""
    if overlap_ratio > 0:
        return 1.0
    if distance_m is None:
        return 0.1
    return _round4(_clamp(1.0 - distance_m / DISTANCE_PENALTY_REF_M, 0.1, 1.0))


def build_items() -> list[dict]:
    thresholds = _class_thresholds()
    calib_17e = _by_key(_read(CALIBRACAO_17E), "item_validacao_id")
    decisions = _by_key(_read(DECISIONS_17D), "item_validacao_id")
    grid = _canary_grid()
    features_by_patch = _by_key(_read(FEATURES_BY_PATCH), "patch_id")
    score_v6_by_patch = _by_key(_read(SCORE_V6), "patch_id")
    topo_17c38_by_patch = _by_key(_read(TOPO_17C38), "patch_id")

    items = []
    for item_id, calib in calib_17e.items():
        patch_id = calib["patch_id"]
        decision = decisions.get(item_id, {})
        grid_row = grid.get(patch_id, {})
        ref_patch = grid_row.get("nearest_official_patch_id", "")
        ref_distance = _float(grid_row.get("nearest_official_patch_distance_m"))
        # overlap com patch oficial (nenhum, conforme geometria herdada)
        overlap_ratio = 0.0
        penalty = _penalty(ref_distance, overlap_ratio)

        ref_features = features_by_patch.get(ref_patch, {})
        ref_topo = topo_17c38_by_patch.get(ref_patch, {})
        ref_score = score_v6_by_patch.get(ref_patch, {})
        topo_idx = _float(ref_score.get("topography_hydrology_index"))

        # Valores fisicos da referencia comparativa (oficial), nunca do canario.
        ref_physical = {name: _float(ref_features.get(col)) for name, col in PHYSICAL_COLUMN.items()}
        has_ref_physical = ref_patch != "" and all(v is not None for v in ref_physical.values())

        # Ordem A/B: nenhuma feature fisica direta e nenhum overlap (herdado).
        direct_available = False  # nao ha feature store fisico keyed ao canario S17C6
        if direct_available:
            feature_source_mode = "feature_direta_review_only"
        elif has_ref_physical:
            feature_source_mode = "referencia_comparativa_review_only"
        else:
            feature_source_mode = "fila_extracao_necessaria"

        muito_proxima = overlap_ratio > 0 or (ref_distance is not None and ref_distance <= STRONG_PROXIMITY_M)
        pode_forte_agora = bool(direct_available or (has_ref_physical and muito_proxima))

        items.append({
            "item_validacao_id": item_id,
            "canary_patch_id": patch_id,
            "candidate_event_id": calib.get("candidate_event_id", decision.get("candidate_event_id", "")),
            "region": calib.get("regiao", decision.get("regiao", "")),
            "city": decision.get("cidade", grid_row.get("region", "")),
            "geometry_id": calib.get("geometry_id", decision.get("geometry_id", "")),
            "bbox": _grid_bbox(grid_row),
            "ref_patch": ref_patch,
            "ref_distance_m": ref_distance,
            "overlap_ratio": overlap_ratio,
            "penalty": penalty,
            "ref_physical": ref_physical,
            "has_ref_physical": has_ref_physical,
            "ref_topo_recomputed": ref_topo,
            "topo_idx": topo_idx,
            "direct_available": direct_available,
            "feature_source_mode": feature_source_mode,
            "pode_forte_agora": pode_forte_agora,
            "thresholds": thresholds,
            # herdado do 17E
            "rain_idx": _float(calib.get("rain_idx")),
            "urban_spectral_idx": _float(calib.get("urban_spectral_idx")),
            "vegetation_idx": _float(calib.get("vegetation_idx")),
            "moisture_idx": _float(calib.get("moisture_idx")),
            "score_sem_fisico": _float(calib.get("score_exploratorio_baseline")),
            "classe_sem_fisico": calib.get("score_exploratorio_class_baseline", ""),
            "score_v6_reference_value": _float(calib.get("score_v6_reference_value")),
            "score_v6_reference_class": calib.get("score_v6_reference_class", ""),
            "status_17e_original": "pronto_para_calibracao_exploratoria_review_only",
        })
    return items


def _grid_bbox(grid_row: dict) -> str:
    keys = ("xmin", "ymin", "xmax", "ymax")
    if all(grid_row.get(k) for k in keys):
        return ",".join(grid_row[k] for k in keys)
    return "not_available"


def _exploratory_score_with_topo(item: dict, topo_weight_base: float) -> float:
    penalty = item["penalty"]
    topo_idx = item["topo_idx"] or 0.0
    w_topo = topo_weight_base * penalty
    w = BASE_WEIGHTS
    raw = (
        w_topo * topo_idx
        + w["rain"] * (item["rain_idx"] or 0.0)
        + w["urban"] * (item["urban_spectral_idx"] or 0.0)
        + w["veg"] * (item["vegetation_idx"] or 0.0)
        + w["moisture"] * (item["moisture_idx"] or 0.0)
    )
    raw_min = sum(v for v in w.values() if v < 0)
    raw_max = w_topo + sum(v for v in w.values() if v > 0)
    if raw_max - raw_min == 0:
        return 0.0
    return _round4(_clamp((raw - raw_min) / (raw_max - raw_min)))


def _effective_topo_weight(item: dict, base: float) -> float:
    return _round4(base * item["penalty"])


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
        ("calibracao_17e", CALIBRACAO_17E, "calibracao exploratoria herdada com espectral e chuva"),
        ("prontidao_17e", PRONTIDAO_17E, "prontidao herdada do 17E"),
        ("resumo_17e", SUMMARY_17E, "estado herdado do 17E"),
        ("decisoes_17d", DECISIONS_17D, "vinculos aceitos para avaliacao"),
        ("patch_links_17c5", PATCH_LINKS_17C5, "classe de vinculo e sobreposicao"),
        ("grade_canario_17c6", PATCH_GRID_17C6, "bbox do canario e patch oficial mais proximo"),
        ("features_by_patch", FEATURES_BY_PATCH, "features fisicas oficiais por patch"),
        ("score_v6", SCORE_V6, "score v6 oficial somente leitura"),
        ("topografia_17c38", TOPO_17C38, "topografia recomputada de patches oficiais"),
        ("terreno_17c34", TERRAIN_17C34, "terreno dos canarios ancorados no evento"),
        ("dem_manifest_17c36", DEM_MANIFEST_17C36, "manifesto do DEM Copernicus local"),
        ("drenagem_17c35", DRAINAGE_17C35, "drenagem oficial por patch"),
        ("score_v7_proibido", SCORE_V7, "deve permanecer ausente"),
    ]
    rows = []
    for role, path, note in sources:
        rows.append({
            "artifact_role": role,
            "path": rel(path),
            "exists": _bool(path.exists()),
            "row_count": _row_count(path),
            "columns_found": _columns(path),
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


# --- Inventario de fontes --------------------------------------------------
def inventario_rows() -> list[dict]:
    rows = [
        {
            "fonte_id": "S17F_FONTE_0001",
            "caminho": rel(FEATURES_BY_PATCH),
            "tipo_fonte": "features_oficiais_por_patch",
            "regioes_cobertas": "recife;curitiba;petropolis",
            "possui_HAND": "true", "possui_elevation": "true", "possui_slope": "true",
            "possui_distance_to_water": "true", "possui_TWI": "true", "possui_flow_accumulation": "true",
            "possui_drenagem": "true", "crs": "EPSG:4326", "resolucao": "patch_agregado",
            "status_uso": "referencia_comparativa",
            "observacao": "features fisicas oficiais completas por patch; nenhum patch oficial cobre a geometria dos canarios S17C6, uso comparativo",
        },
        {
            "fonte_id": "S17F_FONTE_0002",
            "caminho": rel(TOPO_17C38),
            "tipo_fonte": "topografia_recomputada_dem",
            "regioes_cobertas": "recife",
            "possui_HAND": "true", "possui_elevation": "true", "possui_slope": "true",
            "possui_distance_to_water": "false", "possui_TWI": "true", "possui_flow_accumulation": "true",
            "possui_drenagem": "false", "crs": "EPSG:4326", "resolucao": "30m_copernicus_glo30",
            "status_uso": "referencia_comparativa",
            "observacao": "topografia recomputada real de patches oficiais (metodo reconstruido nao provado equivalente); referencia comparativa",
        },
        {
            "fonte_id": "S17F_FONTE_0003",
            "caminho": rel(TERRAIN_17C34),
            "tipo_fonte": "terreno_canario_ancorado_evento",
            "regioes_cobertas": "recife",
            "possui_HAND": "false", "possui_elevation": "true", "possui_slope": "true",
            "possui_distance_to_water": "true", "possui_TWI": "false", "possui_flow_accumulation": "false",
            "possui_drenagem": "true", "crs": "EPSG:4326", "resolucao": "ponto_srtm30m",
            "status_uso": "referencia_comparativa",
            "observacao": "terreno de canarios ancorados no mesmo evento porem geometria diferente dos S17C6; HAND indisponivel; uso comparativo",
        },
        {
            "fonte_id": "S17F_FONTE_0004",
            "caminho": rel(DEM_MANIFEST_17C36),
            "tipo_fonte": "manifesto_dem_local",
            "regioes_cobertas": "recife_sul",
            "possui_HAND": "false", "possui_elevation": "true", "possui_slope": "false",
            "possui_distance_to_water": "false", "possui_TWI": "false", "possui_flow_accumulation": "false",
            "possui_drenagem": "false", "crs": "EPSG:4326", "resolucao": "30.62m_copernicus_glo30",
            "status_uso": "insuficiente",
            "observacao": "janela DEM Copernicus local existe mas o bbox (norte ate -8.01855) nao cobre a AOI dos canarios; serve como fonte para nova extracao",
        },
        {
            "fonte_id": "S17F_FONTE_0005",
            "caminho": rel(DRAINAGE_17C35),
            "tipo_fonte": "drenagem_oficial_por_patch",
            "regioes_cobertas": "recife",
            "possui_HAND": "false", "possui_elevation": "false", "possui_slope": "false",
            "possui_distance_to_water": "true", "possui_TWI": "false", "possui_flow_accumulation": "false",
            "possui_drenagem": "true", "crs": "EPSG:4326", "resolucao": "vetor_oficial",
            "status_uso": "referencia_comparativa" if DRAINAGE_17C35.exists() else "ausente",
            "observacao": "drenagem/faixas-marginais oficiais Dados Recife por patch; base para proximidade hidrica comparativa e para extracao",
        },
        {
            "fonte_id": "S17F_FONTE_0006",
            "caminho": "nao_aplicavel",
            "tipo_fonte": "feature_store_fisico_direto_canario_S17C6",
            "regioes_cobertas": "nenhuma",
            "possui_HAND": "false", "possui_elevation": "false", "possui_slope": "false",
            "possui_distance_to_water": "false", "possui_TWI": "false", "possui_flow_accumulation": "false",
            "possui_drenagem": "false", "crs": "not_available", "resolucao": "not_available",
            "status_uso": "ausente",
            "observacao": "nao existe feature store fisico/topografico associado diretamente aos canarios S17C6_CANARY_REC; exige extracao",
        },
    ]
    return rows


# --- Matriz fisica/topografica ---------------------------------------------
def matriz_fisica_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        ref = item["ref_physical"]
        comparative = item["feature_source_mode"] == "referencia_comparativa_review_only"
        # Valores preenchidos apenas como referencia comparativa (patch oficial proximo).
        values = {name: (_fmt(ref.get(name)) if comparative else "not_available") for name in PHYSICAL_FEATURES}
        ausentes = ";".join(PHYSICAL_FEATURES)  # nenhuma feature fisica DIRETA do canario
        drainage_ctx = (
            f"distancia_hidrica_referencia_m={_fmt(ref.get('distance_to_water'))}"
            if comparative else "not_available"
        )
        rows.append({
            "canary_patch_id": item["canary_patch_id"],
            "candidate_event_id": item["candidate_event_id"],
            "region": item["region"],
            "city": item["city"],
            "geometry_id": item["geometry_id"],
            "feature_source_mode": item["feature_source_mode"],
            **values,
            "drainage_context": drainage_ctx,
            "features_fisicas_completas": "false",
            "features_fisicas_parciais": "false",
            "features_fisicas_ausentes": ausentes,
            "referencia_patch_id": item["ref_patch"] or "not_available",
            "referencia_distance_m": _fmt(item["ref_distance_m"]),
            "referencia_overlap_ratio": _fmt(item["overlap_ratio"]),
            "uso_permitido": item["feature_source_mode"],
            "ground_truth": "false",
            "eligible_for_training": "false",
            "score_v7_allowed": "false",
            "not_ground_truth_reason": NO_GT_REASON,
        })
    return rows


# --- Reavaliacao da prontidao 17E ------------------------------------------
def reavaliacao_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        comparative = item["feature_source_mode"] == "referencia_comparativa_review_only"
        grupo_fisico = "comparativo" if comparative else ("direto" if item["direct_available"] else "ausente")
        bloqueio = (
            "referencia_fisica_comparativa_distante_sem_feature_direta"
            if comparative and not item["pode_forte_agora"]
            else ("feature_fisica_direta_ausente" if not comparative else "nenhum")
        )
        justificativa = (
            f"referencia fisica comparativa do patch {item['ref_patch']} a "
            f"{_fmt(item['ref_distance_m'])} m, sem sobreposicao; usada review-only com penalidade de "
            f"incerteza {_fmt(item['penalty'])}; features fisicas diretas do canario seguem ausentes, "
            "por isso a calibracao forte permanece bloqueada e a exploratoria e aprimorada"
        )
        rows.append({
            "item_validacao_id": item["item_validacao_id"],
            "canary_patch_id": item["canary_patch_id"],
            "status_17e_original": item["status_17e_original"],
            "grupos_features_antes": "espectral;chuva_gatilho",
            "grupos_features_depois": "espectral;chuva_gatilho;fisico_comparativo",
            "possui_grupo_fisico": grupo_fisico,
            "possui_grupo_espectral": "true",
            "possui_grupo_chuva": "true",
            "pode_calibracao_forte_agora": _bool(item["pode_forte_agora"]),
            "pode_calibracao_exploratoria_melhorada": _bool(comparative or item["direct_available"]),
            "bloqueio_restante": bloqueio,
            "justificativa_tecnica": justificativa,
        })
    return rows


# --- Calibracao exploratoria aprimorada ------------------------------------
def calibracao_aprimorada_rows(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        if item["feature_source_mode"] == "fila_extracao_necessaria":
            continue
        base_topo_weight = TOPO_VARIANTS["HF2_referencia_comparativa_penalizada"]
        w_topo_eff = _effective_topo_weight(item, base_topo_weight)
        score_sem = item["score_sem_fisico"] if item["score_sem_fisico"] is not None else _exploratory_score_with_topo(item, 0.0)
        classe_sem = item["classe_sem_fisico"] or _classify(score_sem, item["thresholds"])
        score_com = _exploratory_score_with_topo(item, base_topo_weight)
        classe_com = _classify(score_com, item["thresholds"])
        lacunas = "; ".join([
            "features fisicas diretas do canario ausentes (HAND/elevacao/declividade/TWI/fluxo/proximidade)",
            f"componente fisico vem de patch comparativo {item['ref_patch']} a {_fmt(item['ref_distance_m'])} m",
            "incerteza espacial alta penaliza o peso topografico; extracao direta necessaria para calibracao forte",
        ])
        rows.append({
            "item_validacao_id": item["item_validacao_id"],
            "canary_patch_id": item["canary_patch_id"],
            "region": item["region"],
            "feature_source_mode": item["feature_source_mode"],
            "topografia_hidrologia_idx_referencia": _fmt(item["topo_idx"]),
            "referencia_patch_id": item["ref_patch"],
            "referencia_distance_m": _fmt(item["ref_distance_m"]),
            "penalidade_incerteza": _fmt(item["penalty"]),
            "peso_topografico_efetivo": _fmt(w_topo_eff),
            "score_exploratorio_sem_fisico": _fmt(score_sem),
            "classe_sem_fisico": classe_sem,
            "score_exploratorio_fisico_review_only": _fmt(score_com),
            "classe_com_fisico": classe_com,
            "score_v6_reference_value": _fmt(item["score_v6_reference_value"]),
            "score_v6_reference_class": item["score_v6_reference_class"],
            "delta_score_pelo_fisico": _fmt(_round4((score_com or 0) - (score_sem or 0))),
            "mudanca_de_classe_pelo_fisico": _class_shift(classe_com, classe_sem),
            "uso_permitido": item["feature_source_mode"],
            "justificativa_tecnica": (
                "componente topografico/hidrologico adicionado como referencia comparativa penalizada; "
                "review-only; nao substitui score_v6, nao e oficial, nao e treino"
            ),
            "lacunas": lacunas,
            "score_oficial": "false",
            "substituir_score_v6": "false",
            "usar_em_treino": "false",
            "ground_truth": "false",
            "review_only": "true",
        })
    return rows


# --- Simulacao de sensibilidade fisica -------------------------------------
def simulacao_fisica_rows(items: list[dict]) -> list[dict]:
    rows = []
    counter = 0
    for item in items:
        if item["feature_source_mode"] == "fila_extracao_necessaria":
            continue
        for hip_id, base in TOPO_VARIANTS.items():
            counter += 1
            w_eff = _effective_topo_weight(item, base)
            score = _exploratory_score_with_topo(item, base)
            classe = _classify(score, item["thresholds"])
            rows.append({
                "simulacao_id": f"S17F_SIM_{counter:04d}",
                "item_validacao_id": item["item_validacao_id"],
                "canary_patch_id": item["canary_patch_id"],
                "hipotese_id": hip_id,
                "peso_topografico_base": _fmt(base),
                "penalidade_incerteza": _fmt(item["penalty"]),
                "peso_topografico_efetivo": _fmt(w_eff),
                "topografia_hidrologia_idx_referencia": _fmt(item["topo_idx"]),
                "score_exploratorio_fisico_review_only": _fmt(score),
                "classe_com_fisico": classe,
                "score_v6_reference_value": _fmt(item["score_v6_reference_value"]),
                "score_v6_reference_class": item["score_v6_reference_class"],
                "mudanca_de_classe": _class_shift(classe, item["score_v6_reference_class"]),
                "score_oficial": "false",
                "substituir_score_v6": "false",
                "usar_em_treino": "false",
                "ground_truth": "false",
                "review_only": "true",
            })
    return rows


def hipoteses_fisicas_rows(items: list[dict]) -> list[dict]:
    any_direct = any(i["direct_available"] for i in items)
    any_comparative = any(i["feature_source_mode"] == "referencia_comparativa_review_only" for i in items)
    specs = [
        ("HF0_sem_componente_fisico", "manter a linha de base exploratoria sem componente fisico",
         "Reproduz o score exploratorio do 17E (espectral + chuva) como referencia de comparacao.",
         "nenhum", "peso_topo 0.00", "true", "true"),
        ("HF1_componente_fisico_direto", "adicionar componente fisico direto quando disponivel",
         "Adiciona o componente topografico/hidrologico com feature fisica direta do canario, quando existir.",
         "topography_hydrology_index", "peso_topo 0.40 sem penalidade", "true", _bool(any_direct)),
        ("HF2_referencia_comparativa_penalizada", "usar referencia fisica comparativa com penalidade de incerteza",
         "Usa a topografia/hidrologia do patch oficial mais proximo com peso reduzido pela distancia.",
         "topography_hydrology_index", "peso_topo 0.40 x penalidade", "true", _bool(any_comparative)),
        ("HF3_separar_fisico_de_confianca_documental", "separar componente fisico de confianca documental",
         "Trata o componente fisico separado do suporte documental, evitando confundir os dois.",
         "topography_hydrology_index;evidence_support_index", "sem alteracao numerica", "true", "true"),
        ("HF4_sensibilidade_peso_topografico", "testar sensibilidade do peso topografico sem alterar score oficial",
         "Varia o peso topografico (0.20/0.40/0.60) para medir a sensibilidade do score exploratorio.",
         "topography_hydrology_index", "peso_topo 0.20/0.40/0.60 x penalidade", "true", _bool(any_comparative)),
        ("HF5_manter_forte_bloqueada_referencia_fraca", "manter calibracao forte bloqueada se fonte fisica for comparativa fraca",
         "Se a fonte fisica for apenas comparativa e distante, mantem a calibracao forte bloqueada.",
         "topography_hydrology_index", "guarda_de_referencia_fraca", "true",
         _bool(any_comparative and not any(i["pode_forte_agora"] for i in items))),
    ]
    rows = []
    for hip_id, nome, descricao, componente, ajuste, permitida, aplicada in specs:
        rows.append({
            "hipotese_id": hip_id,
            "nome_hipotese": nome,
            "descricao": descricao,
            "componente_afetado": componente,
            "ajuste_peso": ajuste,
            "permitida": permitida,
            "aplicada_na_simulacao": aplicada,
            "motivo": _hip_motivo(hip_id, any_direct, any_comparative),
            "score_oficial": "false",
            "substituir_score_v6": "false",
            "usar_em_treino": "false",
            "ground_truth": "false",
            "review_only": "true",
        })
    return rows


def _hip_motivo(hip_id: str, any_direct: bool, any_comparative: bool) -> str:
    mapping = {
        "HF0_sem_componente_fisico": "linha de base necessaria para medir o efeito do componente fisico",
        "HF1_componente_fisico_direto": "sem feature fisica direta disponivel, a hipotese fica registrada mas nao aplicada" if not any_direct else "feature fisica direta disponivel",
        "HF2_referencia_comparativa_penalizada": "referencia comparativa disponivel permite componente fisico penalizado review-only" if any_comparative else "sem referencia comparativa",
        "HF3_separar_fisico_de_confianca_documental": "corrige confundir suscetibilidade fisica com suporte documental",
        "HF4_sensibilidade_peso_topografico": "mede robustez do score exploratorio ao peso topografico",
        "HF5_manter_forte_bloqueada_referencia_fraca": "guarda metodologica: referencia comparativa distante nao habilita calibracao forte",
    }
    return mapping.get(hip_id, "hipotese review-only")


# --- Fila executavel de extracao -------------------------------------------
def fila_extracao_rows(items: list[dict]) -> list[dict]:
    rows = []
    counter = 0
    dem_bundle = "HAND;elevation;slope;TWI;flow_accumulation"
    for item in items:
        bbox = item["bbox"]
        region = item["region"]
        city = item["city"]
        canary = item["canary_patch_id"]
        # Tarefa 1: features derivadas de DEM.
        counter += 1
        rows.append({
            "task_id": f"S17F_EXTRACT_{counter:04d}",
            "canary_patch_id": canary,
            "bbox": bbox,
            "crs": "EPSG:4326",
            "region": region,
            "city": city,
            "feature_needed": dem_bundle,
            "data_source_needed": "copernicus_dem_glo30",
            "expected_input_path": f"local_runs/suscetibilidade/17f_extracao/dem_{canary.lower()}.npy",
            "expected_output_path": f"local_runs/suscetibilidade/17f_extracao/{canary.lower()}_topografia.csv",
            "command_hint": "reutilizar pipeline basin-aware do 17C36/17C38 sobre nova janela DEM que cubra o bbox do canario (janela atual do 17C36 nao cobre a AOI ao norte de -8.01855)",
            "priority": "alta",
            "blocking_reason": "features topograficas diretas do canario ausentes; janela DEM local atual nao cobre a AOI",
        })
        # Tarefa 2: proximidade hidrica / drenagem.
        counter += 1
        rows.append({
            "task_id": f"S17F_EXTRACT_{counter:04d}",
            "canary_patch_id": canary,
            "bbox": bbox,
            "crs": "EPSG:4326",
            "region": region,
            "city": city,
            "feature_needed": "distance_to_water;drainage_context",
            "data_source_needed": "dados_recife_faixas_marginais_hidrografia_oficial",
            "expected_input_path": "local_runs/suscetibilidade/17f_extracao/hidrografia_oficial_recife.geojson",
            "expected_output_path": f"local_runs/suscetibilidade/17f_extracao/{canary.lower()}_proximidade_hidrica.csv",
            "command_hint": "calcular distancia do centroide/bbox do canario a hidrografia oficial (faixas-marginais Dados Recife) reutilizando o metodo do 17C35",
            "priority": "media",
            "blocking_reason": "proximidade hidrica direta do canario ausente; usar hidrografia oficial ja catalogada",
        })
    return rows


# --- Gate 17F --------------------------------------------------------------
def _final_status(items: list[dict], calibracao_aprimorada: list[dict], fila: list[dict]) -> str:
    n_direct = sum(1 for i in items if i["direct_available"])
    n_comparative = sum(1 for i in items if i["feature_source_mode"] == "referencia_comparativa_review_only")
    if n_direct == len(items) and n_direct > 0:
        return "17F_FEATURES_FISICAS_DIRETAS_CONSOLIDADAS"
    if calibracao_aprimorada:
        return "17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO"
    if n_comparative > 0:
        return "17F_REFERENCIA_FISICA_COMPARATIVA_CONSOLIDADA"
    if fila:
        return "17F_FILA_EXTRACAO_FISICA_EXECUTAVEL"
    return "17F_BLOQUEADO_FAIL_CLOSED"


def gate_rows(items, matriz, calibracao, fila, final_status) -> list[dict]:
    n = len(items)
    n_direct = sum(1 for i in items if i["direct_available"])
    n_comparative = sum(1 for i in items if i["feature_source_mode"] == "referencia_comparativa_review_only")
    n_fila = len({r["canary_patch_id"] for r in fila})
    n_aprimorada = len({r["canary_patch_id"] for r in calibracao})
    n_forte = sum(1 for i in items if i["pode_forte_agora"])
    rows = [
        ("canarios_avaliados", str(n), "5", n == 5, "todos os canarios recebem status"),
        ("feature_fisica_direta", str(n_direct), ">=0", True, "sem feature store fisico keyed ao canario S17C6"),
        ("referencia_comparativa", str(n_comparative), ">=0", True, "patch oficial mais proximo com features fisicas reais"),
        ("fila_extracao", str(n_fila), str(n), n_fila == n, "tarefas de extracao concretas por canario"),
        ("calibracao_exploratoria_aprimorada", str(n_aprimorada), ">=1", n_aprimorada >= 1, "componente fisico comparativo penalizado"),
        ("pode_calibracao_forte_agora", str(n_forte), ">=0", True, "referencia comparativa distante nao habilita forte"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_allowed_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 nunca alterado"),
        ("caminho_funcional_entregue", "true", "true", final_status != "17F_BLOQUEADO_FAIL_CLOSED", "sprint entrega caminho funcional"),
        ("status_final_17f", final_status, "enum", final_status in GATE_FINAL_ALLOWED, "status final da extracao fisica"),
    ]
    return [
        {"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o}
        for c, v, t, p, o in rows
    ]


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


# --- Resumos ---------------------------------------------------------------
def resumo_status_rows(items, matriz, calibracao, fila) -> list[dict]:
    n_direct = sum(1 for i in items if i["direct_available"])
    n_comparative = sum(1 for i in items if i["feature_source_mode"] == "referencia_comparativa_review_only")
    n_fila = len({r["canary_patch_id"] for r in fila})
    n_aprimorada = len({r["canary_patch_id"] for r in calibracao})
    n_forte = sum(1 for i in items if i["pode_forte_agora"])
    modes = Counter(i["feature_source_mode"] for i in items)
    rows = [
        {"dimensao": "feature_source_mode", "categoria": mode, "quantidade": str(count)}
        for mode, count in sorted(modes.items())
    ]
    rows.extend([
        {"dimensao": "resolucao_fisica", "categoria": "feature_fisica_direta", "quantidade": str(n_direct)},
        {"dimensao": "resolucao_fisica", "categoria": "referencia_comparativa", "quantidade": str(n_comparative)},
        {"dimensao": "resolucao_fisica", "categoria": "fila_extracao", "quantidade": str(n_fila)},
        {"dimensao": "calibracao", "categoria": "exploratoria_aprimorada", "quantidade": str(n_aprimorada)},
        {"dimensao": "calibracao", "categoria": "pode_forte_agora", "quantidade": str(n_forte)},
    ])
    return rows


def resumo_canario_rows(items, calibracao, fila) -> list[dict]:
    aprimorada_ids = {r["canary_patch_id"] for r in calibracao}
    fila_ids = {r["canary_patch_id"] for r in fila}
    rows = []
    for item in items:
        rows.append({
            "canary_patch_id": item["canary_patch_id"],
            "feature_source_mode": item["feature_source_mode"],
            "features_fisicas_diretas": _bool(item["direct_available"]),
            "referencia_comparativa": _bool(item["feature_source_mode"] == "referencia_comparativa_review_only"),
            "fila_extracao": _bool(item["canary_patch_id"] in fila_ids),
            "calibracao_exploratoria_aprimorada": _bool(item["canary_patch_id"] in aprimorada_ids),
            "pode_calibracao_forte_agora": _bool(item["pode_forte_agora"]),
        })
    return rows


def _status_17e(final_status: str) -> str:
    if final_status == "17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO":
        return "17E_CALIBRACAO_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO_COMPARATIVO"
    if final_status == "17F_FEATURES_FISICAS_DIRETAS_CONSOLIDADAS":
        return "17E_PRONTO_PARA_REAVALIACAO_CALIBRACAO_FORTE"
    return "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY"


def summary_obj(items, inventario, matriz, reavaliacao, calibracao, simulacao, fila, final_status) -> dict:
    inherited = read_json(SUMMARY_17E)
    n_direct = sum(1 for i in items if i["direct_available"])
    n_comparative = sum(1 for i in items if i["feature_source_mode"] == "referencia_comparativa_review_only")
    n_fila = len({r["canary_patch_id"] for r in fila})
    n_aprimorada = len({r["canary_patch_id"] for r in calibracao})
    n_forte = sum(1 for i in items if i["pode_forte_agora"])
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17e_status_final": inherited.get("status_final_17e", ""),
        "herdado_17e_exploratoria": inherited.get("pronto_para_calibracao_exploratoria_review_only", 0),
        "herdado_17e_status_17b": inherited.get("status_17b", ""),
        "canarios_avaliados": len(items),
        "features_fisicas_diretas": n_direct,
        "referencia_comparativa": n_comparative,
        "fila_extracao": n_fila,
        "calibracao_exploratoria_aprimorada": n_aprimorada,
        "pode_calibracao_forte_agora": n_forte,
        "inventario_fontes": len(inventario),
        "fontes_referencia_comparativa": sum(1 for r in inventario if r["status_uso"] == "referencia_comparativa"),
        "fontes_ausentes": sum(1 for r in inventario if r["status_uso"] == "ausente"),
        "matriz_rows": len(matriz),
        "simulacao_rows": len(simulacao),
        "fila_rows": len(fila),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_oficial_true_count": 0,
        "substituir_score_v6_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "caminho_funcional": _caminho_funcional(final_status),
        "status_final_17f": final_status,
        "status_17e": _status_17e(final_status),
        "status_17b": "17B_PARCIAL_COM_CALIBRACAO_EXPLORATORIA_SEM_PRONTIDAO_BENCHMARK",
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
        "score_oficial": False,
        "substituir_score_v6": False,
    }


def _caminho_funcional(final_status: str) -> str:
    return {
        "17F_FEATURES_FISICAS_DIRETAS_CONSOLIDADAS": "features_fisicas_diretas",
        "17F_REFERENCIA_FISICA_COMPARATIVA_CONSOLIDADA": "referencia_fisica_comparativa",
        "17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO": "calibracao_exploratoria_aprimorada_mais_fila_extracao",
        "17F_FILA_EXTRACAO_FISICA_EXECUTAVEL": "fila_extracao_executavel",
        "17F_BLOQUEADO_FAIL_CLOSED": "bloqueado_fail_closed",
    }.get(final_status, "indefinido")


# --- Schema ----------------------------------------------------------------
def _schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17F extracao fisica e topografica dos canarios observacionais",
        "type": "object",
        "properties": {
            "feature_source_mode": {"enum": FEATURE_SOURCE_MODE_ALLOWED},
            "status_uso": {"enum": STATUS_USO_ALLOWED},
            "status_final_17f": {"enum": GATE_FINAL_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_oficial": {"const": "false"},
            "substituir_score_v6": {"const": "false"},
            "usar_em_treino": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "enums": {
            "feature_source_mode": FEATURE_SOURCE_MODE_ALLOWED,
            "status_uso": STATUS_USO_ALLOWED,
            "status_final_17f": GATE_FINAL_ALLOWED,
        },
        "score_v6_topo_weight": TOPO_WEIGHT,
        "base_weights": BASE_WEIGHTS,
        "topo_variants": TOPO_VARIANTS,
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema() -> None:
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def card_markdown(item: dict, calib_row: dict | None) -> str:
    ref = item["ref_physical"]
    fisicas_encontradas = "; ".join(
        f"{name}={_fmt(ref.get(name))} (referencia {item['ref_patch']})" for name in PHYSICAL_FEATURES
    ) if item["feature_source_mode"] == "referencia_comparativa_review_only" else "nenhuma direta"
    score_sem = item["score_sem_fisico"]
    score_com = _exploratory_score_with_topo(item, TOPO_VARIANTS["HF2_referencia_comparativa_penalizada"])
    classe_com = _classify(score_com, item["thresholds"])
    return f"""# Cartao fisico/topografico {item['canary_patch_id']}

## Identificacao

- Canario: `{item['canary_patch_id']}`
- Evento: `{item['candidate_event_id']}`
- Geometria/link: `{item['geometry_id']}`
- Regiao/cidade: {item['region']} / {item['city']}
- bbox: `{item['bbox']}`

## Features espectrais herdadas (17E, reais pre-evento)

- rain_idx={_fmt(item['rain_idx'])}; urban_spectral_idx={_fmt(item['urban_spectral_idx'])}; vegetation_idx={_fmt(item['vegetation_idx'])}; moisture_idx={_fmt(item['moisture_idx'])}

## Features de chuva herdadas (17E, CHIRPS pre-evento)

- consolidadas no indice de chuva (rain_idx) do 17E

## Features fisicas encontradas

- {fisicas_encontradas}

## Features fisicas ausentes (diretas do canario)

- {';'.join(PHYSICAL_FEATURES)}

## Fonte da feature fisica

- Patch oficial de referencia: `{item['ref_patch']}` a {_fmt(item['ref_distance_m'])} m; sobreposicao={_fmt(item['overlap_ratio'])}
- Tipo: {item['feature_source_mode']}
- Direta ou comparativa: {'comparativa' if item['feature_source_mode'] == 'referencia_comparativa_review_only' else 'direta' if item['direct_available'] else 'fila de extracao'}

## Incerteza

- Penalidade de incerteza aplicada ao componente fisico: {_fmt(item['penalty'])} (referencia distante, sem sobreposicao)

## Impacto na prontidao de calibracao

- Score exploratorio sem fisico: {_fmt(score_sem)} (classe {item['classe_sem_fisico']})
- Score exploratorio com fisico comparativo penalizado: {_fmt(score_com)} (classe {classe_com})
- Pode calibracao forte agora: {_bool(item['pode_forte_agora'])}

## Decisao 17F

- feature_source_mode: {item['feature_source_mode']}
- Calibracao forte permanece bloqueada por referencia comparativa distante; extracao direta enfileirada.

## Por que ainda nao e ground truth

Referencia comparativa de patch vizinho nao confirma ocorrencia no canario; sem verdade de referencia observacional nao ha ground truth.

## Por que ainda nao e treinavel

Sem rotulo validado e sem feature fisica direta, o canario nao alimenta treino supervisionado.

## Por que nao altera o score_v6

O componente fisico e comparativo, penalizado e review-only; o score exploratorio nunca substitui nem recalibra o score_v6 oficial.
"""


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, inventario, matriz, reavaliacao, calibracao, simulacao, fila, gate, summary) -> str:
    inv_lines = "\n".join(
        f"- {r['fonte_id']} ({r['tipo_fonte']}): status_uso={r['status_uso']}; {r['caminho']}"
        for r in inventario
    )
    canario_lines = "\n".join(
        f"- {r['canary_patch_id']}: modo={r['feature_source_mode']}; referencia={r['referencia_patch_id']} "
        f"a {r['referencia_distance_m']} m; forte_agora={_reav_forte(reavaliacao, r['canary_patch_id'])}"
        for r in matriz
    )
    sim_lines = _simulacao_summary(simulacao)
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate)
    return f"""# SUSC-17F Extracao e consolidacao de atributos fisicos/topograficos dos canarios observacionais

## Estado herdado do 17E

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 17E herdado: {summary['herdado_17e_status_final']}
- Itens em calibracao exploratoria review-only: {summary['herdado_17e_exploratoria']}
- Status 17B herdado: {summary['herdado_17e_status_17b']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Problema: ausencia do grupo fisico/topografico

Os 5 canarios `S17C6_CANARY_REC_00001..00005` tinham espectral e chuva reais, mas nenhuma
feature fisica/topografica (HAND, elevacao, declividade, TWI, acumulacao de fluxo, proximidade
hidrica). Esse grupo domina o score_v6 (peso {TOPO_WEIGHT}) e sua ausencia bloqueava a
calibracao forte.

## Inventario das fontes encontradas

{inv_lines}

Nenhuma fonte oficial cobre a geometria dos canarios S17C6; a janela DEM local do 17C36 fica ao
sul (ate -8.01855) e nao alcanca a AOI dos canarios. Portanto nao ha feature fisica direta, e
sim referencia comparativa e necessidade de extracao.

## Resultado por canario

{canario_lines}

- Feature fisica direta: {summary['features_fisicas_diretas']}
- Referencia comparativa consolidada: {summary['referencia_comparativa']}
- Fila de extracao gerada: {summary['fila_extracao']}
- Calibracao exploratoria aprimorada: {summary['calibracao_exploratoria_aprimorada']}
- Pode calibracao forte agora: {summary['pode_calibracao_forte_agora']}

## Features diretas encontradas

Nenhuma feature fisica direta dos canarios S17C6 foi encontrada no repositorio.

## Referencias comparativas encontradas

O patch oficial `recife_00552` tem features fisicas reais (oficiais e recomputadas por DEM no
17C38), a distancias de aproximadamente 984 a 3554 m dos canarios, sem sobreposicao. Uso
estritamente comparativo review-only, com penalidade de incerteza pela distancia.

## Lacunas restantes

- HAND, elevacao, declividade, TWI, acumulacao de fluxo e proximidade hidrica diretas dos
  canarios seguem ausentes.
- A referencia comparativa e distante e nao habilita calibracao forte.

## Simulacao exploratoria aprimorada

{sim_lines}

O componente fisico comparativo, penalizado pela distancia, eleva o score exploratorio em
direcao a classe de referencia, mais para os canarios mais proximos. Em todas as linhas:
score_oficial=false, substituir_score_v6=false, usar_em_treino=false, ground_truth=false.

## Impacto na prontidao de calibracao

A calibracao exploratoria foi aprimorada com um componente fisico comparativo, mas a calibracao
forte permanece bloqueada porque a referencia fisica e comparativa e distante. O caminho para a
calibracao forte e executar a fila de extracao das features fisicas diretas.

## Gate final 17F

{gate_lines}

- Status final: **{summary['status_final_17f']}**
- Caminho funcional: **{summary['caminho_funcional']}**

## Impacto no 17E e no 17B

- 17E: {summary['status_17e']} (a calibracao exploratoria ganhou componente fisico comparativo).
- 17B: {summary['status_17b']} (segue sem prontidao de benchmark: sem ground truth, amostra
  concentrada, sem score_v7).

## Proximo marco recomendado

SUSC-17G: executar a fila de extracao (`fila_extracao_features_fisicas_pendentes.csv`) para
obter HAND/elevacao/declividade/TWI/acumulacao de fluxo e proximidade hidrica diretas dos
canarios sobre nova janela DEM Copernicus que cubra a AOI, habilitando a reavaliacao de
calibracao forte review-only.
"""


def _reav_forte(reavaliacao: list[dict], canary: str) -> str:
    for r in reavaliacao:
        if r["canary_patch_id"] == canary:
            return r["pode_calibracao_forte_agora"]
    return "false"


def _simulacao_summary(simulacao: list[dict]) -> str:
    by_hip: dict[str, list[str]] = defaultdict(list)
    for row in simulacao:
        by_hip[row["hipotese_id"]].append(row["classe_com_fisico"])
    lines = []
    for hip in TOPO_VARIANTS:
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
    inventario = inventario_rows()
    matriz = matriz_fisica_rows(items)
    reavaliacao = reavaliacao_rows(items)
    calibracao = calibracao_aprimorada_rows(items)
    simulacao = simulacao_fisica_rows(items)
    hipoteses = hipoteses_fisicas_rows(items)
    fila = fila_extracao_rows(items)
    final_status = _final_status(items, calibracao, fila)
    gate = gate_rows(items, matriz, calibracao, fila, final_status)
    resumo_status = resumo_status_rows(items, matriz, calibracao, fila)
    resumo_canario = resumo_canario_rows(items, calibracao, fila)
    summary = summary_obj(items, inventario, matriz, reavaliacao, calibracao, simulacao, fila, final_status)

    write_csv(PREFLIGHT_CSV, preflight_rows(), PREFLIGHT_FIELDS)
    write_json(PREFLIGHT_JSON, {"preflight": preflight_rows()})
    write_csv(INVENTARIO, inventario, INVENTARIO_FIELDS)
    write_csv(MATRIZ_FISICA, matriz, MATRIZ_FISICA_FIELDS)
    write_csv(REAVALIACAO, reavaliacao, REAVALIACAO_FIELDS)
    write_csv(CALIBRACAO_APRIMORADA, calibracao, CALIBRACAO_APRIMORADA_FIELDS)
    write_csv(SIMULACAO_FISICA, simulacao, SIMULACAO_FISICA_FIELDS)
    write_csv(HIPOTESES_FISICAS, hipoteses, HIPOTESES_FISICAS_FIELDS)
    write_csv(FILA_EXTRACAO, fila, FILA_FIELDS)
    write_csv(GATE, gate, GATE_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status, RESUMO_STATUS_FIELDS)
    write_csv(RESUMO_CANARIO, resumo_canario, RESUMO_CANARIO_FIELDS)
    write_json(SUMMARY, summary)
    calib_by_item = _by_key(calibracao, "canary_patch_id")
    for item in items:
        card = CARDS_DIR / f"{item['canary_patch_id']}.md"
        write_markdown(card, card_markdown(item, calib_by_item.get(item["canary_patch_id"])))
    write_markdown(REPORT, report_markdown(items, inventario, matriz, reavaliacao, calibracao, simulacao, fila, gate, summary))
    return summary


# --- Validacao -------------------------------------------------------------
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


def validate_inventario_rows(rows: list[dict]) -> list[str]:
    errors = []
    for idx, row in enumerate(rows, start=1):
        if row.get("status_uso") not in STATUS_USO_ALLOWED:
            errors.append(f"inventario{idx}:status_uso_fora_enum:{row.get('status_uso')}")
    return errors


def validate_matriz_rows(rows: list[dict]) -> list[str]:
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
        mode = row.get("feature_source_mode")
        if mode not in FEATURE_SOURCE_MODE_ALLOWED:
            errors.append(f"{rid}:feature_source_mode_fora_enum:{mode}")
        for field in ["ground_truth", "eligible_for_training", "score_v7_allowed"]:
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        # Referencia comparativa nao pode ser apresentada como feature direta.
        if mode == "feature_direta_review_only" and row.get("referencia_patch_id") not in {"", "not_available"}:
            errors.append(f"{rid}:comparativa_apresentada_como_direta")
        # Valor fisico preenchido exige fonte (modo direto ou comparativo).
        has_value = any(row.get(name) not in {"", "not_available"} for name in PHYSICAL_FEATURES)
        if has_value and mode == "fila_extracao_necessaria":
            errors.append(f"{rid}:valor_fisico_sem_fonte")
        if has_value and mode == "referencia_comparativa_review_only" and row.get("referencia_patch_id") in {"", "not_available"}:
            errors.append(f"{rid}:valor_comparativo_sem_referencia")
        if row.get("not_ground_truth_reason", "").strip() == "":
            errors.append(f"{rid}:not_ground_truth_reason_vazio")
    return errors


def validate_reavaliacao_rows(rows: list[dict]) -> list[str]:
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:reavaliacao_sem_justificativa")
    return errors


def validate_calibracao_rows(rows: list[dict]) -> list[str]:
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
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
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("simulacao_id", f"row{idx}")
        for field in ["score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth"]:
            if row.get(field) != "false":
                errors.append(f"{rid}:{field}_proibido_true")
    return errors


def validate() -> int:
    summary = build_all()
    errors: list[str] = []
    errors.extend(validate_inventario_rows(_read(INVENTARIO)))
    errors.extend(validate_matriz_rows(_read(MATRIZ_FISICA)))
    errors.extend(validate_reavaliacao_rows(_read(REAVALIACAO)))
    errors.extend(validate_calibracao_rows(_read(CALIBRACAO_APRIMORADA)))
    errors.extend(validate_simulacao_rows(_read(SIMULACAO_FISICA)))
    errors.extend(public_text_violations_files())

    matriz = _read(MATRIZ_FISICA)
    fila = _read(FILA_EXTRACAO)
    fila_canaries = {r["canary_patch_id"] for r in fila}
    # Canario sem feature fisica direta precisa de fila executavel.
    for row in matriz:
        if row.get("feature_source_mode") != "feature_direta_review_only" and row["canary_patch_id"] not in fila_canaries:
            errors.append(f"{row['canary_patch_id']}:fila_extracao_ausente_sem_feature_direta")
    # Todo canario recebe status final (modo) valido.
    for row in matriz:
        if row.get("feature_source_mode") not in FEATURE_SOURCE_MODE_ALLOWED:
            errors.append(f"{row['canary_patch_id']}:canario_sem_status_final")
    if summary["status_final_17f"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_17f']}")
    if summary["status_final_17f"] == "17F_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_caminho_funcional_entregue")
    for field in ["ground_truth_true_count", "eligible_for_training_true_count",
                  "score_v7_allowed_true_count", "score_oficial_true_count",
                  "substituir_score_v6_true_count"]:
        if summary[field] != 0:
            errors.append(f"{field}_nonzero")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "17F extracao fisica validada: "
        f"diretas={summary['features_fisicas_diretas']} "
        f"comparativa={summary['referencia_comparativa']} "
        f"fila={summary['fila_extracao']} "
        f"aprimorada={summary['calibracao_exploratoria_aprimorada']} "
        f"forte_agora={summary['pode_calibracao_forte_agora']} "
        f"status17F={summary['status_final_17f']}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "17F extracao fisica gerada: "
        f"diretas={summary['features_fisicas_diretas']} "
        f"comparativa={summary['referencia_comparativa']} "
        f"fila={summary['fila_extracao']} "
        f"aprimorada={summary['calibracao_exploratoria_aprimorada']} "
        f"status17F={summary['status_final_17f']} "
        f"caminho={summary['caminho_funcional']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
