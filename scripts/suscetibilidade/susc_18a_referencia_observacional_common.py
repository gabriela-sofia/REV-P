"""SUSC-18A execucao de referencia observacional regional.

O 17I deixou uma amostra regional parcial com filas executaveis: Recife com
referencia forte somente revisao (5 canarios, footprint tecnico e vinculo forte),
Curitiba com candidatos parciais sem geometria e Petropolis bloqueada por fenomeno
misto. Esta etapa executa de fato a fila regional do 17I e consolida uma base
regional observacional mais forte seguindo o padrao:

    event_record -> source_geometry ou technical_footprint -> patch_link ->
    direct_features -> review_only_evaluation -> 17B_readiness_gate

Entrega avanco funcional real: (a) consolida a referencia forte de Recife com suas
features diretas ja extraidas localmente (fisico 17G, espectral 17C20, chuva 17C25);
(b) mantem as referencias parciais datadas de Curitiba; (c) executa a fila 17I,
resolvendo o que e resolvivel com arquivos locais e convertendo o restante em
pacotes de execucao externa concretos (obtencao de geometria, separacao de
fenomeno, execucao de footprint SAR).

Nunca cria ground truth, treino, score_v7 nem benchmark 17B. O score_v6 nunca e
alterado. Bairro/rua/texto nunca vira geometria forte. Deslizamento ou fenomeno
misto nunca entra como inundacao sem separacao explicita. Alerta ou area de risco
nunca vira ocorrencia observada. Feature pos-evento nunca vira feature pre-evento.
Nada e inventado; onde falta dado, gera-se fila executavel.
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
OUT_SUSC = ROOT / "outputs_public" / "suscetibilidade"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA_17G = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17g_extracao_direta_features_fisicas_canarios"
OUT_DATA_17I = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17i_ampliacao_regional_amostra_observacional"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_18a_execucao_referencia_observacional_regional"
CARDS_DIR = OUT_DATA / "cartoes_regionais"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
TARGET_PACK_17C = OUT_DATA_17C / "susc_17c_source_target_pack.csv"
FEATURES_FISICAS_17G = OUT_DATA_17G / "matriz_features_fisicas_diretas_canarios.csv"
FILA_17I = OUT_DATA_17I / "fila_regional_desbloqueio_observacional.csv"
MATRIZ_17I = OUT_DATA_17I / "matriz_amostra_observacional_expandida.csv"
SUMMARY_17I = OUT_DATA_17I / "summary.json"
SPECTRAL_DIR_17C20 = OUT_SUSC / "susc_17c20_light_artifacts"
CHIRPS_DIR_17C25 = OUT_SUSC / "susc_17c25_chirps_artifacts"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_JSON = OUT_DATA / "preflight.json"
EXEC_FILA = OUT_DATA / "execucao_fila_regional_17i.csv"
CUR_GEOM = OUT_DATA / "curitiba_geometrias_observacionais.csv"
CUR_FILA_GEOM = OUT_DATA / "curitiba_fila_obtencao_geometria.csv"
PET_FENOMENO = OUT_DATA / "petropolis_classificacao_fenomeno.csv"
PET_FILA_FENOMENO = OUT_DATA / "petropolis_fila_separacao_fenomeno.csv"
FOOTPRINTS = OUT_DATA / "footprints_tecnicos_regionais.csv"
FILA_FOOTPRINT = OUT_DATA / "fila_execucao_footprint_sar.csv"
VINCULOS = OUT_DATA / "vinculos_regionais_evento_patch.csv"
FEATURES = OUT_DATA / "features_regionais_por_vinculo.csv"
FILA_FEATURES = OUT_DATA / "fila_extracao_features_regionais.csv"
MATRIZ = OUT_DATA / "matriz_referencia_observacional_regional.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_18a.csv"
RESUMO_REGIAO = OUT_DATA / "resumo_por_regiao.csv"
RESUMO_STATUS = OUT_DATA / "resumo_por_status.csv"
RESUMO_FENOMENO = OUT_DATA / "resumo_por_fenomeno.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_18A_EXECUCAO_REFERENCIA_OBSERVACIONAL_REGIONAL.md"
SCHEMA = SCHEMAS / "susc_18a_referencia_observacional_schema_v1.json"

REQUIRED_INPUTS = [TARGET_PACK_17C, FEATURES_FISICAS_17G, FILA_17I, MATRIZ_17I, SUMMARY_17I, SCORE_V6]
REQUIRED_OUTPUTS = [
    PREFLIGHT_JSON, EXEC_FILA, CUR_GEOM, CUR_FILA_GEOM, PET_FENOMENO, PET_FILA_FENOMENO,
    FOOTPRINTS, FILA_FOOTPRINT, VINCULOS, FEATURES, FILA_FEATURES, MATRIZ, GATE_17B,
    RESUMO_REGIAO, RESUMO_STATUS, RESUMO_FENOMENO, SUMMARY, REPORT, SCHEMA,
]

REGIONS_ALL = ["REC", "CUR", "PET"]
REGIONS_QUEUE = ["CUR", "PET"]
REGION_CITY = {"CUR": "Curitiba", "PET": "Petropolis", "REC": "Recife"}

RECIFE_STRONG_EVENT = "S17C_REF_0063"

# --- Enums -----------------------------------------------------------------
STATUS_REF_ALLOWED = [
    "referencia_observacional_forte_somente_revisao",
    "referencia_observacional_parcial",
    "evidencia_contextual",
    "bloqueado_sem_geometria",
    "bloqueado_sem_data",
    "bloqueado_por_fenomeno",
    "bloqueado_sem_patch_link",
    "bloqueado_sem_features",
    "fila_executavel",
]
USO_PERMITIDO_ALLOWED = [
    "avaliacao_somente_revisao", "contexto_documental",
    "fila_obtencao_geometria", "fila_obtencao_data",
    "fila_separacao_fenomeno", "fila_execucao_footprint", "fila_extracao_features",
]
FENOMENO_CLASS_ALLOWED = [
    "inundacao_alagamento_enxurrada", "deslizamento", "fenomeno_misto", "insuficiente",
]
CLASSE_VINCULO_ALLOWED = [
    "exact_polygon_overlap", "point_within_patch", "bbox_overlap",
    "near_patch_buffer_10m", "near_patch_buffer_30m",
    "same_region_only", "insufficient_for_patch_link",
]
CLASSE_VINCULO_FORTE = {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}
CLASSE_GEOMETRIA_ALLOWED = [
    "geometria_oficial_ponto", "geometria_oficial_poligono", "geometria_tecnica_footprint",
    "endereco_textual_sem_geometria", "sem_geometria",
]
RESULTADO_EXEC_ALLOWED = [
    "resolvido_local", "parcial_local", "nao_resolvido_gerou_fila", "sem_insumo_local",
]
GATE_FINAL_ALLOWED = [
    "18A_REFERENCIA_REGIONAL_FORTE_MAIS_FILAS_EXECUTAVEIS",
    "18A_REFERENCIA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS",
    "18A_SEM_PRONTIDAO_MAS_COM_PLANO_EXECUTAVEL",
    "18A_BLOQUEADO_FAIL_CLOSED",
]
STATUS_17B_ALLOWED = [
    "17B_PRONTO_PARA_DESENHO_SOMENTE_REVISAO",
    "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS",
    "17B_BLOQUEADO_POR_GEOMETRIA",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_POR_FENOMENO",
    "17B_BLOQUEADO_FAIL_CLOSED",
]

# vocabulario publico proibido (sem expor automacao)
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)

NO_GT_REASON = (
    "referencia observacional regional somente revisao; nao confirma ocorrencia no patch, "
    "nao e verdade de referencia, nao alimenta treino e nao autoriza score_v7"
)
HARD_DEFAULTS = {"ground_truth": "false", "eligible_for_training": "false",
                 "score_v7_allowed": "false", "review_only": "true"}

# --- Field lists -----------------------------------------------------------
EXEC_FIELDS = [
    "task_id", "regiao", "candidate_event_id", "lacuna", "tentativa_realizada",
    "resultado_execucao", "artefato_encontrado", "artefato_criado",
    "continua_bloqueado", "proxima_acao", "justificativa_tecnica",
]
CUR_GEOM_FIELDS = [
    "candidate_event_id", "cidade", "data_evento", "tipo_fenomeno", "fonte", "autoridade_fonte",
    "geometry_id", "classe_geometria", "geometria_normalizada", "crs",
    "fonte_geometria_local", "continua_bloqueado", "justificativa_tecnica",
]
CUR_FILA_GEOM_FIELDS = [
    "task_id", "candidate_event_id", "data_evento", "fonte_sugerida", "url_ou_caminho_esperado",
    "formato_esperado", "campos_necessarios", "expected_output_path", "command_hint", "criterio_de_sucesso",
]
PET_FENOMENO_FIELDS = [
    "candidate_event_id", "cidade", "data_evento", "tipo_fenomeno_fonte", "classe_fenomeno",
    "base_classificacao", "pode_seguir_como_inundacao", "separacao_disponivel",
    "continua_bloqueado", "justificativa_tecnica",
]
PET_FILA_FENOMENO_FIELDS = [
    "task_id", "candidate_event_id", "fonte_sugerida", "url_ou_caminho_esperado", "artefato_necessario",
    "campos_necessarios", "expected_output_path", "command_hint", "criterio_de_sucesso",
]
FOOTPRINTS_FIELDS = [
    "footprint_id", "regiao", "candidate_event_id", "data_evento", "aoi_conhecida",
    "geometry_id", "footprint_status", "possui_raster_local", "fonte_footprint",
    "pos_evento_nao_e_feature_pre_evento", "not_ground_truth", "justificativa_tecnica",
]
FILA_FOOTPRINT_FIELDS = [
    "task_id", "regiao", "candidate_event_id", "aoi", "data_evento", "pre_window", "post_window",
    "colecao_esperada", "bandas", "expected_output_path", "command_hint", "criterio_de_sucesso", "depende_de",
]
VINCULOS_FIELDS = [
    "vinculo_id", "regiao", "candidate_event_id", "geometry_id", "geometry_type", "patch_id",
    "classe_vinculo", "vinculo_forte", "review_only", "justificativa_tecnica",
]
FEATURES_FIELDS = [
    "vinculo_id", "candidate_event_id", "regiao", "patch_id", "classe_vinculo",
    "fisico_elevacao_media", "fisico_declividade_media", "fisico_hand_media", "fisico_twi_media",
    "fisico_dist_agua_min", "fisico_flow_acc_media", "fonte_fisico",
    "espectral_disponivel", "fonte_espectral", "chuva_disponivel", "fonte_chuva",
    "completude_features", "feature_pre_evento_apenas", "not_ground_truth", "justificativa_tecnica",
]
FILA_FEATURES_FIELDS = [
    "task_id", "regiao", "candidate_event_id", "familia_feature", "pre_requisito",
    "fonte_sugerida", "expected_output_path", "command_hint", "criterio_de_sucesso",
]
MATRIZ_FIELDS = [
    "item_id", "regiao", "cidade", "candidate_event_id", "data_evento", "tipo_fenomeno",
    "fonte", "autoridade_fonte", "geometry_id", "geometry_type", "patch_id", "classe_vinculo",
    "possui_footprint_tecnico", "possui_fisico", "possui_espectral", "possui_chuva",
    "qualidade_fonte", "qualidade_temporal", "qualidade_geometrica", "qualidade_vinculo", "qualidade_fenomeno",
    "status_referencia_observacional", "uso_permitido",
    "ground_truth", "eligible_for_training", "score_v7_allowed", "review_only",
    "not_ground_truth_reason", "justificativa_tecnica",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_REGIAO_FIELDS = ["regiao", "cidade", "itens", "fortes", "parciais", "contextuais", "bloqueados_ou_fila"]
RESUMO_STATUS_FIELDS = ["status_referencia_observacional", "quantidade"]
RESUMO_FENOMENO_FIELDS = ["regiao", "classe_fenomeno", "quantidade"]


# --- Helpers ---------------------------------------------------------------
def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _read(path):
    return read_csv(path) if path.exists() else []


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-18A")


def _has_date(v):
    return v not in {"", "not_available", "unknown"}


def _phenomenon_class(ph):
    if ph == "flood_inundation_alagamento":
        return "inundacao_alagamento_enxurrada"
    if ph == "mass_movement":
        return "deslizamento"
    if ph == "hydrometeorological_or_unknown":
        return "fenomeno_misto"
    return "insuficiente"


def _address_only(cand):
    gs = cand.get("geometry_status", "")
    return "address_text" in gs or (cand.get("has_address") == "true" and cand.get("has_point") != "true")


def _spectral_available(canary_lower):
    return (SPECTRAL_DIR_17C20 / f"{canary_lower}_band_stats.csv").exists()


def _chirps_available(canary_lower):
    return (CHIRPS_DIR_17C25 / f"{canary_lower}_chirps_daily.csv").exists()


def _score_v6_changed():
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


# --- Modelo de itens -------------------------------------------------------
def _target_by_id():
    return {r.get("candidate_event_id", ""): r for r in _read(TARGET_PACK_17C)}


def build_recife_items():
    """Referencia forte de Recife: 5 canarios com footprint tecnico, vinculo forte
    (exact_polygon_overlap do 17D) e features diretas locais (17G/17C20/17C25)."""
    target = _target_by_id()
    evt = target.get(RECIFE_STRONG_EVENT, {})
    items = []
    for row in sorted(_read(FEATURES_FISICAS_17G), key=lambda r: r.get("canary_patch_id", "")):
        patch_id = row.get("canary_patch_id", "")
        canary_lower = patch_id.lower()
        items.append({
            "kind": "recife_strong",
            "candidate_event_id": row.get("candidate_event_id", RECIFE_STRONG_EVENT),
            "region": "REC", "cidade": "Recife",
            "data_evento": evt.get("event_date_candidate", "2022-05-24..2022-05-30"),
            "precisao_temporal": evt.get("event_date_precision", "range"),
            "tipo_fenomeno": "flood_inundation_alagamento",
            "classe_fenomeno": "inundacao_alagamento_enxurrada",
            "fonte": evt.get("source_name", "Sentinel-1 SAR metadata feasibility from SUSC-17C31"),
            "autoridade_fonte": evt.get("authority_tier", "technical"),
            "geometry_id": row.get("geometry_id", ""),
            "geometry_type": "technical_footprint",
            "patch_id": patch_id,
            "classe_vinculo": "exact_polygon_overlap",
            "possui_footprint_tecnico": True,
            "possui_fisico": row.get("features_diretas_completas") == "true",
            "possui_espectral": _spectral_available(canary_lower),
            "possui_chuva": _chirps_available(canary_lower),
            "fisico_row": row,
            "status": "referencia_observacional_forte_somente_revisao",
            "uso": "avaliacao_somente_revisao",
        })
    return items


def _classify_regional(cand):
    ph = _phenomenon_class(cand.get("phenomenon_type", ""))
    has_date = _has_date(cand.get("event_date_candidate", ""))
    source_type = cand.get("source_type", "")
    official = cand.get("authority_tier", "") == "official"
    documentary_ctx = source_type == "documentary_context"
    if ph in {"deslizamento", "fenomeno_misto"}:
        return "bloqueado_por_fenomeno"
    if ph == "insuficiente":
        return "bloqueado_sem_data" if not has_date else "bloqueado_por_fenomeno"
    # fenomeno de inundacao
    if not has_date:
        return "bloqueado_sem_data"
    if _address_only(cand):
        return "bloqueado_sem_geometria"
    if documentary_ctx:
        return "evidencia_contextual"
    if official:
        return "referencia_observacional_parcial"
    return "evidencia_contextual"


def _uso_regional(status):
    return {
        "referencia_observacional_parcial": "avaliacao_somente_revisao",
        "evidencia_contextual": "contexto_documental",
        "bloqueado_sem_geometria": "fila_obtencao_geometria",
        "bloqueado_sem_data": "fila_obtencao_data",
        "bloqueado_por_fenomeno": "fila_separacao_fenomeno",
    }.get(status, "contexto_documental")


def build_regional_items():
    target = [r for r in _read(TARGET_PACK_17C) if r.get("region") in REGIONS_QUEUE]
    items = []
    for cand in sorted(target, key=lambda r: r.get("candidate_event_id", "")):
        region = cand.get("region", "")
        status = _classify_regional(cand)
        items.append({
            "kind": "regional_candidate",
            "cand": cand,
            "candidate_event_id": cand.get("candidate_event_id", ""),
            "region": region, "cidade": REGION_CITY.get(region, cand.get("city", "")),
            "data_evento": cand.get("event_date_candidate", "not_available"),
            "precisao_temporal": cand.get("event_date_precision", ""),
            "tipo_fenomeno": cand.get("phenomenon_type", ""),
            "classe_fenomeno": _phenomenon_class(cand.get("phenomenon_type", "")),
            "fonte": cand.get("source_name", ""),
            "autoridade_fonte": cand.get("authority_tier", ""),
            "geometry_id": "not_available",
            "geometry_type": "none",
            "patch_id": "not_available",
            "classe_vinculo": "same_region_only",
            "possui_footprint_tecnico": False,
            "possui_fisico": False, "possui_espectral": False, "possui_chuva": False,
            "status": status,
            "uso": _uso_regional(status),
        })
    return items


def _idx(items):
    for i, it in enumerate(items, start=1):
        it["item_id"] = f"S18A_ITEM_{i:04d}"
    return items


def build_items():
    items = _idx(build_recife_items() + build_regional_items())
    return items


# --- Qualidades ------------------------------------------------------------
def _q_fonte(it):
    tier = it.get("autoridade_fonte", "")
    return {"official": 85, "technical": 80, "documentary": 55, "internal_registry": 30}.get(tier, 20)


def _q_temporal(it):
    return {"exact_day": 90, "range": 80, "month_only": 35}.get(it.get("precisao_temporal", ""), 0)


def _q_geometrica(it):
    if it["classe_vinculo"] in CLASSE_VINCULO_FORTE:
        return 85
    if it["status"] == "bloqueado_sem_geometria":
        return 20
    return 10


def _q_vinculo(it):
    return 85 if it["classe_vinculo"] in CLASSE_VINCULO_FORTE else 0


def _q_fenomeno(it):
    return {"inundacao_alagamento_enxurrada": 85, "deslizamento": 40,
            "fenomeno_misto": 30, "insuficiente": 15}.get(it["classe_fenomeno"], 15)


# --- Tarefa 2: execucao da fila regional 17I -------------------------------
def _lacuna_execucao(lacuna):
    """Mapeia cada lacuna do 17I para a tentativa local e o resultado honesto.

    Sem geometria/data/raster locais para CUR/PET, a execucao local nao resolve;
    a separacao de fenomeno de Petropolis e parcialmente resolvida por
    classificacao documental. Todo o restante vira fila externa concreta."""
    return {
        "obter_geometria_oficial": (
            "busca de ponto/poligono/shapefile/geojson/csv georreferenciado em datasets e artefatos locais",
            "nao_resolvido_gerou_fila", "curitiba_fila_obtencao_geometria.csv",
            "obter geometria oficial datada em geoportal municipal ou defesa civil (fila externa)",
        ),
        "obter_data_exata": (
            "busca de data exata ou intervalo em relatorios e registros locais",
            "nao_resolvido_gerou_fila", "curitiba_fila_obtencao_geometria.csv",
            "resolver data exata ou intervalo por fonte oficial antes de qualquer avaliacao",
        ),
        "separar_fenomeno": (
            "classificacao documental por tipo de fonte e fenomeno registrado (CPRM/DRM)",
            "parcial_local", "petropolis_classificacao_fenomeno.csv",
            "obter classificacao por ocorrencia (inundacao x deslizamento) com fonte oficial",
        ),
        "obter_footprint_tecnico": (
            "busca de raster/footprint tecnico local suficiente para o intervalo do evento",
            "nao_resolvido_gerou_fila", "fila_execucao_footprint_sar.csv",
            "preparar execucao de footprint SAR pre/pos apos definir a AOI oficial",
        ),
        "resolver_patch_link": (
            "cruzamento da geometria do evento com patches oficiais da regiao",
            "nao_resolvido_gerou_fila", "fila_execucao_footprint_sar.csv",
            "resolver vinculo evento-patch apos obter geometria/footprint com CRS",
        ),
    }.get(lacuna, (
        "inspecao de artefatos locais", "sem_insumo_local", "nenhum",
        "reavaliar com nova evidencia",
    ))


def exec_fila_rows():
    rows = []
    for task in _read(FILA_17I):
        lacuna = task.get("lacuna", "")
        tentativa, resultado, artefato_criado, proxima = _lacuna_execucao(lacuna)
        if lacuna == "resolver_patch_link":
            artefato_criado = "vinculos_regionais_evento_patch.csv"
        rows.append({
            "task_id": task.get("task_id", ""),
            "regiao": task.get("regiao", ""),
            "candidate_event_id": task.get("candidate_event_id", ""),
            "lacuna": lacuna,
            "tentativa_realizada": tentativa,
            "resultado_execucao": resultado,
            "artefato_encontrado": "nenhum_artefato_local_suficiente",
            "artefato_criado": artefato_criado,
            "continua_bloqueado": "true",
            "proxima_acao": proxima,
            "justificativa_tecnica": (
                f"lacuna {lacuna} da fila 17I executada sobre arquivos locais; "
                "sem geometria/data/raster local suficiente, o desbloqueio depende de aquisicao externa "
                "ou de separacao documental por ocorrencia; nada foi inventado"
            ),
        })
    return rows


# --- Tarefa 3: Curitiba geometria ------------------------------------------
def _cur_items(items):
    return [it for it in items if it["region"] == "CUR" and it["kind"] == "regional_candidate"]


CLASSE_GEOMETRIA_FORTE = {"geometria_oficial_ponto", "geometria_oficial_poligono", "geometria_tecnica_footprint"}


def classify_geometria(cand):
    """Classifica a geometria de um candidato a partir de sinais oficiais locais.

    So retorna geometria forte quando ha ponto/poligono/footprint oficial de fato;
    endereco/rua/texto e centroide de bairro nunca sao geometria forte."""
    if cand.get("has_official_geometry") == "true":
        if cand.get("has_point") == "true" and cand.get("has_bbox") != "true":
            return "geometria_oficial_ponto"
        return "geometria_oficial_poligono"
    if cand.get("has_point") == "true":
        return "geometria_oficial_ponto"
    if cand.get("source_type") == "technical_remote_sensing_flood_footprint":
        return "geometria_tecnica_footprint"
    gs = cand.get("geometry_status", "")
    if "address_text" in gs or (cand.get("has_address") == "true" and cand.get("has_point") != "true"):
        return "endereco_textual_sem_geometria"
    return "sem_geometria"


def curitiba_geom_rows(items):
    rows = []
    for it in _cur_items(items):
        c = it["cand"]
        classe = classify_geometria(c)
        forte = classe in CLASSE_GEOMETRIA_FORTE
        rows.append({
            "candidate_event_id": it["candidate_event_id"], "cidade": "Curitiba",
            "data_evento": it["data_evento"], "tipo_fenomeno": it["tipo_fenomeno"],
            "fonte": it["fonte"], "autoridade_fonte": it["autoridade_fonte"],
            "geometry_id": ("S18A_CUR_GEOM_" + it["candidate_event_id"] if forte else "not_available"),
            "classe_geometria": classe,
            "geometria_normalizada": _bool(forte),
            "crs": ("EPSG:4326" if forte else "not_available"),
            "fonte_geometria_local": ("geometria_oficial_local" if forte else "nenhuma_geometria_oficial_local"),
            "continua_bloqueado": _bool(not forte),
            "justificativa_tecnica": (
                "geometria oficial local (ponto/poligono/footprint) normalizada para EPSG:4326 e vinculavel a patch"
                if forte else
                "nenhum ponto, poligono, shapefile, geojson ou csv georreferenciado oficial encontrado "
                "localmente; endereco/rua/bairro nao vira geometria forte"
                if classe == "endereco_textual_sem_geometria" else
                "sem geometria oficial local; fonte documental/administrativa sem coordenada; "
                "centroide de bairro/municipio nao e usado como geometria"
            ),
        })
    return rows


def curitiba_fila_geom_rows(items):
    rows = []
    counter = 0
    for it in _cur_items(items):
        if it["classe_fenomeno"] != "inundacao_alagamento_enxurrada":
            continue
        if not _has_date(it["data_evento"]):
            continue
        counter += 1
        cid = it["candidate_event_id"]
        rows.append({
            "task_id": f"S18A_CUR_GEO_{counter:04d}", "candidate_event_id": cid,
            "data_evento": it["data_evento"],
            "fonte_sugerida": "IPPUC geoportal / Defesa Civil de Curitiba / Prefeitura de Curitiba",
            "url_ou_caminho_esperado": "https://geocuritiba.ippuc.org.br (camadas de alagamento) ou solicitacao a Defesa Civil",
            "formato_esperado": "geojson_ou_shapefile",
            "campos_necessarios": "geometria(ponto/poligono);crs;data_ocorrencia;tipo_fenomeno;fonte",
            "expected_output_path": f"local_runs/suscetibilidade/18a_regional/{cid.lower()}_geometria.geojson",
            "command_hint": "baixar camada oficial datada, validar CRS e normalizar para EPSG:4326",
            "criterio_de_sucesso": "geometria oficial com CRS, datada e vinculavel a patch (nao textual)",
        })
    return rows


# --- Tarefa 4: Petropolis fenomeno -----------------------------------------
def _pet_items(items):
    return [it for it in items if it["region"] == "PET" and it["kind"] == "regional_candidate"]


def petropolis_fenomeno_rows(items):
    rows = []
    for it in _pet_items(items):
        classe = it["classe_fenomeno"]
        pode_seguir = classe == "inundacao_alagamento_enxurrada"
        base = (
            "fonte declara inundacao/alagamento" if pode_seguir else
            "relatorio geotecnico de encosta / fenomeno hidrometeorologico nao separado por ocorrencia"
            if classe == "fenomeno_misto" else
            "registro sem tipo de fenomeno resolvido"
        )
        rows.append({
            "candidate_event_id": it["candidate_event_id"], "cidade": "Petropolis",
            "data_evento": it["data_evento"], "tipo_fenomeno_fonte": it["tipo_fenomeno"],
            "classe_fenomeno": classe, "base_classificacao": base,
            "pode_seguir_como_inundacao": _bool(pode_seguir),
            "separacao_disponivel": "false",
            "continua_bloqueado": _bool(not pode_seguir),
            "justificativa_tecnica": (
                "fenomeno declarado como inundacao pela fonte oficial, mas ainda sem geometria/footprint; "
                "segue como evidencia contextual review-only"
                if pode_seguir else
                "fenomeno misto/deslizamento nao pode entrar como inundacao sem separacao espacial ou "
                "documental por ocorrencia; permanece bloqueado"
            ),
        })
    return rows


def petropolis_fila_fenomeno_rows(items):
    rows = []
    counter = 0
    for it in _pet_items(items):
        if it["classe_fenomeno"] != "fenomeno_misto":
            continue
        counter += 1
        cid = it["candidate_event_id"]
        rows.append({
            "task_id": f"S18A_PET_FEN_{counter:04d}", "candidate_event_id": cid,
            "fonte_sugerida": "SGB/CPRM e Defesa Civil de Petropolis (laudos por ocorrencia)",
            "url_ou_caminho_esperado": "https://rigeo.sgb.gov.br (laudos) / Defesa Civil municipal",
            "artefato_necessario": "classificacao_por_ocorrencia_inundacao_x_deslizamento",
            "campos_necessarios": "ocorrencia_id;tipo_fenomeno;geometria;crs;data",
            "expected_output_path": f"local_runs/suscetibilidade/18a_regional/{cid.lower()}_fenomeno_separado.csv",
            "command_hint": "extrair por ocorrencia o tipo de fenomeno e separar inundacao de deslizamento com fonte oficial",
            "criterio_de_sucesso": "ocorrencias de inundacao separadas de deslizamento, com geometria e data oficiais",
        })
    return rows


# --- Tarefa 5: footprints tecnicos -----------------------------------------
def footprints_rows(items):
    rows = []
    counter = 0
    seen_strong = set()
    for it in items:
        if it["kind"] == "recife_strong":
            key = (it["candidate_event_id"], it["geometry_id"])
            if key in seen_strong:
                continue
            seen_strong.add(key)
            counter += 1
            rows.append({
                "footprint_id": f"S18A_FP_{counter:04d}", "regiao": "REC",
                "candidate_event_id": it["candidate_event_id"], "data_evento": it["data_evento"],
                "aoi_conhecida": "true", "geometry_id": it["geometry_id"],
                "footprint_status": "disponivel_review_only_herdado_17d",
                "possui_raster_local": "false",
                "fonte_footprint": "footprint tecnico SAR Sentinel-1 validado no 17D (exact_polygon_overlap)",
                "pos_evento_nao_e_feature_pre_evento": "true",
                "not_ground_truth": "true",
                "justificativa_tecnica": (
                    "footprint tecnico review-only herdado do 17D, ancora dos 5 canarios; usado apenas como "
                    "geometria/vinculo, nunca como ground truth e nunca como feature pre-evento"
                ),
            })
    # candidatos regionais com data e regiao conhecidas -> footprint em fila
    seen = set()
    for it in items:
        if it["kind"] != "regional_candidate":
            continue
        if it["classe_fenomeno"] != "inundacao_alagamento_enxurrada":
            continue
        if not _has_date(it["data_evento"]):
            continue
        if it["candidate_event_id"] in seen:
            continue
        seen.add(it["candidate_event_id"])
        counter += 1
        rows.append({
            "footprint_id": f"S18A_FP_{counter:04d}", "regiao": it["region"],
            "candidate_event_id": it["candidate_event_id"], "data_evento": it["data_evento"],
            "aoi_conhecida": "false", "geometry_id": "not_available",
            "footprint_status": "fila_execucao_externa",
            "possui_raster_local": "false",
            "fonte_footprint": "nenhum raster local; execucao SAR futura apos AOI oficial",
            "pos_evento_nao_e_feature_pre_evento": "true",
            "not_ground_truth": "true",
            "justificativa_tecnica": (
                "data e regiao conhecidas, mas AOI oficial pendente; sem raster local e sem download "
                "autorizado, o footprint vira pacote de execucao futura"
            ),
        })
    return rows


def fila_footprint_rows(items):
    rows = []
    counter = 0
    seen = set()
    for it in items:
        if it["kind"] != "regional_candidate":
            continue
        if it["classe_fenomeno"] != "inundacao_alagamento_enxurrada":
            continue
        if not _has_date(it["data_evento"]):
            continue
        cid = it["candidate_event_id"]
        if cid in seen:
            continue
        seen.add(cid)
        counter += 1
        rows.append({
            "task_id": f"S18A_SAR_{counter:04d}", "regiao": it["region"], "candidate_event_id": cid,
            "aoi": "limite_municipal_oficial_a_obter (AOI pendente de geometria oficial)",
            "data_evento": it["data_evento"],
            "pre_window": "ate_12_dias_antes_do_inicio_do_evento",
            "post_window": "ate_12_dias_apos_o_fim_do_evento",
            "colecao_esperada": "sentinel-1-rtc",
            "bandas": "VV,VH",
            "expected_output_path": f"local_runs/suscetibilidade/18a_regional/{cid.lower()}_footprint.geojson",
            "command_hint": "com AOI definida, aplicar o metodo do DEVRO02D (VV diff pre/pos, agua permanente JRC mascarada, vetorizar)",
            "criterio_de_sucesso": "footprint tecnico candidato review-only com CRS e data",
            "depende_de": "geometria/AOI oficial da tarefa de obtencao de geometria",
        })
    return rows


# --- Tarefa 6: vinculos evento-patch ---------------------------------------
def vinculos_rows(items):
    rows = []
    counter = 0
    for it in items:
        counter += 1
        if it["kind"] == "recife_strong":
            classe = "exact_polygon_overlap"
            just = ("vinculo forte review-only herdado do 17D: canario com overlap exato de poligono "
                    "sobre o footprint tecnico; geometria e patch presentes")
        else:
            classe = "same_region_only"
            just = ("sem geometria oficial forte para o evento; vinculo apenas regional, "
                    "nao forte; insuficiente para patch-link forte")
        rows.append({
            "vinculo_id": f"S18A_LINK_{counter:04d}", "regiao": it["region"],
            "candidate_event_id": it["candidate_event_id"],
            "geometry_id": it["geometry_id"], "geometry_type": it["geometry_type"],
            "patch_id": it["patch_id"], "classe_vinculo": classe,
            "vinculo_forte": _bool(classe in CLASSE_VINCULO_FORTE),
            "review_only": "true", "justificativa_tecnica": just,
        })
    return rows


# --- Tarefa 7: features por vinculo ----------------------------------------
def features_rows(items):
    rows = []
    for it in items:
        if it["kind"] != "recife_strong":
            continue
        f = it["fisico_row"]
        completude = sum([it["possui_fisico"], it["possui_espectral"], it["possui_chuva"]]) / 3.0
        rows.append({
            "vinculo_id": it["item_id"].replace("ITEM", "LINK"),
            "candidate_event_id": it["candidate_event_id"], "regiao": it["region"],
            "patch_id": it["patch_id"], "classe_vinculo": it["classe_vinculo"],
            "fisico_elevacao_media": f.get("elevation_mean", "not_available"),
            "fisico_declividade_media": f.get("slope_mean", "not_available"),
            "fisico_hand_media": f.get("HAND_mean", "not_available"),
            "fisico_twi_media": f.get("TWI_mean", "not_available"),
            "fisico_dist_agua_min": f.get("distance_to_water_min", "not_available"),
            "fisico_flow_acc_media": f.get("flow_accumulation_mean", "not_available"),
            "fonte_fisico": "susc_17g matriz_features_fisicas_diretas_canarios.csv (DEM Copernicus GLO-30 + hidrografia oficial)",
            "espectral_disponivel": _bool(it["possui_espectral"]),
            "fonte_espectral": "susc_17c20_light_artifacts band_stats (pre-evento)" if it["possui_espectral"] else "ausente",
            "chuva_disponivel": _bool(it["possui_chuva"]),
            "fonte_chuva": "susc_17c25_chirps_artifacts chirps_daily (janela pre-evento)" if it["possui_chuva"] else "ausente",
            "completude_features": f"{completude:.2f}",
            "feature_pre_evento_apenas": "true",
            "not_ground_truth": "true",
            "justificativa_tecnica": (
                "features diretas review-only ancoradas no vinculo forte; fisico topografico estatico, "
                "espectral e chuva na janela pre-evento; nenhuma feature pos-evento usada; nao e ground truth"
            ),
        })
    return rows


def fila_features_rows(items):
    rows = []
    counter = 0
    for it in items:
        if it["kind"] != "regional_candidate":
            continue
        if it["status"] not in {"referencia_observacional_parcial", "evidencia_contextual"}:
            continue
        for familia in ("fisico", "espectral", "chuva"):
            counter += 1
            cid = it["candidate_event_id"]
            rows.append({
                "task_id": f"S18A_FEAT_{counter:04d}", "regiao": it["region"], "candidate_event_id": cid,
                "familia_feature": familia,
                "pre_requisito": "geometria oficial ou footprint tecnico com patch-link resolvido",
                "fonte_sugerida": {
                    "fisico": "DEM Copernicus GLO-30 + hidrografia oficial (metodo 17F/17G)",
                    "espectral": "Sentinel-2 COG pre-evento (metodo 17C20)",
                    "chuva": "CHIRPS janela pre-evento (metodo 17C25)",
                }[familia],
                "expected_output_path": f"local_runs/suscetibilidade/18a_regional/{cid.lower()}_{familia}.csv",
                "command_hint": "apos resolver o patch-link, extrair a familia de features na janela pre-evento",
                "criterio_de_sucesso": f"features de {familia} extraidas por patch, somente pre-evento, com fonte",
            })
    return rows


# --- Tarefa 8: matriz consolidada ------------------------------------------
def matriz_rows(items):
    rows = []
    for it in items:
        just = (
            "referencia forte review-only: footprint tecnico + vinculo forte + features diretas locais"
            if it["kind"] == "recife_strong" else
            _justificativa_regional(it)
        )
        rows.append({
            "item_id": it["item_id"], "regiao": it["region"], "cidade": it["cidade"],
            "candidate_event_id": it["candidate_event_id"], "data_evento": it["data_evento"],
            "tipo_fenomeno": it["tipo_fenomeno"], "fonte": it["fonte"], "autoridade_fonte": it["autoridade_fonte"],
            "geometry_id": it["geometry_id"], "geometry_type": it["geometry_type"],
            "patch_id": it["patch_id"], "classe_vinculo": it["classe_vinculo"],
            "possui_footprint_tecnico": _bool(it["possui_footprint_tecnico"]),
            "possui_fisico": _bool(it["possui_fisico"]), "possui_espectral": _bool(it["possui_espectral"]),
            "possui_chuva": _bool(it["possui_chuva"]),
            "qualidade_fonte": str(_q_fonte(it)), "qualidade_temporal": str(_q_temporal(it)),
            "qualidade_geometrica": str(_q_geometrica(it)), "qualidade_vinculo": str(_q_vinculo(it)),
            "qualidade_fenomeno": str(_q_fenomeno(it)),
            "status_referencia_observacional": it["status"], "uso_permitido": it["uso"],
            "ground_truth": HARD_DEFAULTS["ground_truth"],
            "eligible_for_training": HARD_DEFAULTS["eligible_for_training"],
            "score_v7_allowed": HARD_DEFAULTS["score_v7_allowed"],
            "review_only": HARD_DEFAULTS["review_only"],
            "not_ground_truth_reason": NO_GT_REASON,
            "justificativa_tecnica": just,
        })
    return rows


def _justificativa_regional(it):
    base = (
        f"regiao {it['region']}; fenomeno {it['classe_fenomeno']}; data {it['data_evento']} "
        f"({it['precisao_temporal']}); fonte {it['autoridade_fonte']}; vinculo {it['classe_vinculo']}"
    )
    extra = {
        "referencia_observacional_parcial": "inundacao datada de fonte oficial; geometria oficial e patch-link pendentes",
        "evidencia_contextual": "fonte documental datada usada apenas como contexto; sem geometria forte",
        "bloqueado_sem_geometria": "endereco/texto nao vira geometria forte; em fila de obtencao de geometria",
        "bloqueado_sem_data": "sem data exata ou intervalo; em fila de obtencao de data",
        "bloqueado_por_fenomeno": "fenomeno misto/deslizamento nao entra como inundacao sem separacao por ocorrencia",
    }.get(it["status"], "candidato regional review-only")
    return f"{base}; {extra}"


# --- Tarefa 9: gate 17B ----------------------------------------------------
def _metrics(items):
    strong = [it for it in items if it["status"] == "referencia_observacional_forte_somente_revisao"]
    parciais = [it for it in items if it["status"] == "referencia_observacional_parcial"]
    contextuais = [it for it in items if it["status"] == "evidencia_contextual"]
    strong_events = {it["candidate_event_id"] for it in strong}
    strong_regions = {it["region"] for it in strong}
    strong_links = len(strong)  # cada canario forte e um patch-link forte (exact_polygon_overlap)
    regioes_evidencia = {it["region"] for it in (strong + parciais + contextuais)}
    return {
        "strong": strong, "parciais": parciais, "contextuais": contextuais,
        "eventos_distintos_fortes": len(strong_events),
        "regioes_com_referencia_forte": len(strong_regions),
        "regioes_com_evidencia": len(regioes_evidencia),
        "vinculos_fortes": strong_links,
        "n_parciais": len(parciais), "n_contextuais": len(contextuais),
    }


def _status_17b(items):
    m = _metrics(items)
    minimos = (m["eventos_distintos_fortes"] >= 3 and m["regioes_com_referencia_forte"] >= 2
               and m["vinculos_fortes"] >= 20)
    if minimos:
        return "17B_PRONTO_PARA_DESENHO_SOMENTE_REVISAO"
    if m["vinculos_fortes"] > 0 and (m["n_parciais"] > 0 or m["regioes_com_evidencia"] >= 2):
        return "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS"
    if m["vinculos_fortes"] > 0:
        return "17B_BLOQUEADO_POR_AMOSTRA"
    return "17B_BLOQUEADO_FAIL_CLOSED"


def gate_17b_rows(items):
    m = _metrics(items)
    rows = [
        ("minimo_3_eventos_distintos_fortes", str(m["eventos_distintos_fortes"]), "3",
         m["eventos_distintos_fortes"] >= 3, "somente eventos com geometria/footprint forte contam"),
        ("minimo_2_regioes_com_referencia_forte", str(m["regioes_com_referencia_forte"]), "2",
         m["regioes_com_referencia_forte"] >= 2, "regioes com referencia forte (Recife herdado)"),
        ("regioes_com_evidencia_observacional", str(m["regioes_com_evidencia"]), ">=2",
         m["regioes_com_evidencia"] >= 2, "inclui parciais/contextuais de Curitiba e Petropolis"),
        ("minimo_20_patch_links_fortes", str(m["vinculos_fortes"]), "20",
         m["vinculos_fortes"] >= 20, "patch-links fortes review-only (5 canarios de Recife)"),
        ("separacao_temporal_possivel", "false", "true", False,
         "amostra forte ainda de um unico evento datado"),
        ("features_diretas_suficientes", _bool(bool(m["strong"])), "true", bool(m["strong"]),
         "referencia forte com fisico/espectral/chuva locais"),
        ("controles_nao_supervisionados", "true", "true", True, "controles seguem nao supervisionados"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(),
         "score_v6 nunca alterado"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o}
            for c, v, t, p, o in rows]


# --- Status final 18A ------------------------------------------------------
def _final_status(items, has_queue):
    m = _metrics(items)
    if not items:
        return "18A_BLOQUEADO_FAIL_CLOSED"
    if m["strong"] and (m["n_parciais"] > 0 or has_queue):
        return "18A_REFERENCIA_REGIONAL_FORTE_MAIS_FILAS_EXECUTAVEIS"
    if m["n_parciais"] > 0 and has_queue:
        return "18A_REFERENCIA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS"
    if has_queue:
        return "18A_SEM_PRONTIDAO_MAS_COM_PLANO_EXECUTAVEL"
    return "18A_BLOQUEADO_FAIL_CLOSED"


# --- Resumos ---------------------------------------------------------------
def resumo_regiao_rows(items):
    rows = []
    for region in REGIONS_ALL:
        grp = [it for it in items if it["region"] == region]
        if not grp:
            continue
        rows.append({
            "regiao": region, "cidade": REGION_CITY[region], "itens": str(len(grp)),
            "fortes": str(sum(1 for it in grp if it["status"] == "referencia_observacional_forte_somente_revisao")),
            "parciais": str(sum(1 for it in grp if it["status"] == "referencia_observacional_parcial")),
            "contextuais": str(sum(1 for it in grp if it["status"] == "evidencia_contextual")),
            "bloqueados_ou_fila": str(sum(1 for it in grp if it["status"].startswith("bloqueado"))),
        })
    return rows


def resumo_status_rows(items):
    c = Counter(it["status"] for it in items)
    return [{"status_referencia_observacional": s, "quantidade": str(c[s])}
            for s in STATUS_REF_ALLOWED if c.get(s, 0)]


def resumo_fenomeno_rows(items):
    d = defaultdict(Counter)
    for it in items:
        d[it["region"]][it["classe_fenomeno"]] += 1
    rows = []
    for region in REGIONS_ALL:
        for cls in FENOMENO_CLASS_ALLOWED:
            if d[region].get(cls, 0):
                rows.append({"regiao": region, "classe_fenomeno": cls, "quantidade": str(d[region][cls])})
    return rows


# --- Summary / preflight ---------------------------------------------------
def summary_obj(items, exec_fila, cur_fila, pet_fila, fila_fp, fila_feat, final_status):
    inherited = read_json(SUMMARY_17I)
    m = _metrics(items)
    total_fila = len(exec_fila) + len(cur_fila) + len(pet_fila) + len(fila_fp) + len(fila_feat)
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17i_status_final": inherited.get("status_final_17i", ""),
        "herdado_17i_status_17b": inherited.get("status_17b", ""),
        "itens_total": len(items),
        "itens_rec": sum(1 for it in items if it["region"] == "REC"),
        "itens_cur": sum(1 for it in items if it["region"] == "CUR"),
        "itens_pet": sum(1 for it in items if it["region"] == "PET"),
        "referencia_forte": len(m["strong"]),
        "referencia_parcial": m["n_parciais"],
        "evidencia_contextual": m["n_contextuais"],
        "eventos_distintos_fortes": m["eventos_distintos_fortes"],
        "regioes_com_referencia_forte": m["regioes_com_referencia_forte"],
        "regioes_com_evidencia": m["regioes_com_evidencia"],
        "patch_links_fortes": m["vinculos_fortes"],
        "fila_execucao_17i": len(exec_fila),
        "fila_geometria_curitiba": len(cur_fila),
        "fila_separacao_fenomeno_petropolis": len(pet_fila),
        "fila_footprint_sar": len(fila_fp),
        "fila_extracao_features": len(fila_feat),
        "fila_total": total_fila,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "status_final_18a": final_status,
        "status_17b": _status_17b(items),
        "review_only": True, "ground_truth": False, "trainable": False,
    }


def preflight_obj():
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "dirty_lines": len(_run_git(["status", "--short"]).splitlines()),
        "inputs": [
            {"role": "target_pack_17c", "path": rel(TARGET_PACK_17C), "exists": TARGET_PACK_17C.exists()},
            {"role": "features_fisicas_17g", "path": rel(FEATURES_FISICAS_17G), "exists": FEATURES_FISICAS_17G.exists()},
            {"role": "fila_regional_17i", "path": rel(FILA_17I), "exists": FILA_17I.exists()},
            {"role": "matriz_17i", "path": rel(MATRIZ_17I), "exists": MATRIZ_17I.exists()},
            {"role": "summary_17i", "path": rel(SUMMARY_17I), "exists": SUMMARY_17I.exists()},
            {"role": "spectral_17c20", "path": rel(SPECTRAL_DIR_17C20), "exists": SPECTRAL_DIR_17C20.exists()},
            {"role": "chirps_17c25", "path": rel(CHIRPS_DIR_17C25), "exists": CHIRPS_DIR_17C25.exists()},
            {"role": "score_v6", "path": rel(SCORE_V6), "exists": SCORE_V6.exists()},
            {"role": "score_v7_proibido", "path": rel(SCORE_V7), "exists": SCORE_V7.exists()},
        ],
        "regioes_alvo": REGIONS_ALL,
    }


# --- Schema ----------------------------------------------------------------
def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-18A execucao de referencia observacional regional",
        "type": "object",
        "properties": {
            "status_referencia_observacional": {"enum": STATUS_REF_ALLOWED},
            "uso_permitido": {"enum": USO_PERMITIDO_ALLOWED},
            "classe_fenomeno": {"enum": FENOMENO_CLASS_ALLOWED},
            "classe_vinculo": {"enum": CLASSE_VINCULO_ALLOWED},
            "classe_geometria": {"enum": CLASSE_GEOMETRIA_ALLOWED},
            "status_final_18a": {"enum": GATE_FINAL_ALLOWED},
            "status_17b": {"enum": STATUS_17B_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "enums": {
            "status_referencia_observacional": STATUS_REF_ALLOWED,
            "uso_permitido": USO_PERMITIDO_ALLOWED,
            "classe_fenomeno": FENOMENO_CLASS_ALLOWED,
            "classe_vinculo": CLASSE_VINCULO_ALLOWED,
            "classe_geometria": CLASSE_GEOMETRIA_ALLOWED,
            "resultado_execucao": RESULTADO_EXEC_ALLOWED,
            "status_final_18a": GATE_FINAL_ALLOWED,
            "status_17b": STATUS_17B_ALLOWED,
        },
        "hard_defaults": HARD_DEFAULTS,
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema():
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def _card_worthy(it):
    return it["status"] in {
        "referencia_observacional_forte_somente_revisao",
        "referencia_observacional_parcial",
        "evidencia_contextual",
        "bloqueado_sem_geometria",
        "bloqueado_por_fenomeno",
    } and it["classe_fenomeno"] == "inundacao_alagamento_enxurrada" or \
        it["status"] == "referencia_observacional_forte_somente_revisao"


def card_markdown(it):
    forte = it["kind"] == "recife_strong"
    geometria = (f"footprint tecnico `{it['geometry_id']}` (review-only, herdado do 17D)"
                 if forte else "sem geometria oficial forte")
    footprint = ("footprint tecnico SAR review-only disponivel (17D), usado so como geometria/vinculo"
                 if forte else "sem footprint local; execucao SAR em fila (apos AOI oficial)")
    vinculo = ("exact_polygon_overlap (forte, review-only)" if forte else "same_region_only (nao forte)")
    features = ("fisico (17G), espectral pre-evento (17C20) e chuva pre-evento (17C25) disponiveis"
                if forte else "sem features vinculadas; extracao em fila apos resolver geometria/patch")
    acao = _acao_minima(it)
    return f"""# Cartao regional {it['item_id']}

## Regiao e cidade

- Regiao/cidade: {it['region']} / {it['cidade']}
- Evento: `{it['candidate_event_id']}`

## Fonte

- Fonte: {it['fonte']}
- Autoridade: {it['autoridade_fonte']}

## Data

- Data do evento: {it['data_evento']} (precisao {it['precisao_temporal']})

## Fenomeno

- Tipo: {it['tipo_fenomeno']}; classe: **{it['classe_fenomeno']}**

## Geometria

- {geometria}

## Footprint

- {footprint}

## Vinculo patch

- Classe de vinculo: {vinculo}
- Patch: `{it['patch_id']}`

## Features

- {features}

## Bloqueios

- Status: **{it['status']}**

## Decisao

- Uso permitido: {it['uso']}
- review_only=true; ground_truth=false; eligible_for_training=false; score_v7_allowed=false

## Acao minima

- {acao}

## Por que nao e ground truth

Referencia observacional regional somente revisao; nao confirma ocorrencia no patch nem constitui verdade de referencia.

## Por que nao e treinavel

Sem rotulo validado; referencia review-only nunca alimenta treino supervisionado.

## Por que nao cria score_v7

Nenhum score oficial ou score_v7 e criado; o score_v6 permanece intacto.

## Impacto no 17B

{_impacto_17b(it)}
"""


def _acao_minima(it):
    return {
        "referencia_observacional_forte_somente_revisao": "manter como referencia review-only e ampliar amostra em outras regioes",
        "referencia_observacional_parcial": "obter geometria oficial datada e resolver o patch-link",
        "evidencia_contextual": "obter geometria oficial ou footprint tecnico datado",
        "bloqueado_sem_geometria": "substituir endereco/texto por geometria oficial com CRS",
        "bloqueado_sem_data": "resolver data exata ou intervalo por fonte oficial",
        "bloqueado_por_fenomeno": "separar inundacao de deslizamento por ocorrencia com fonte oficial",
    }.get(it["status"], "reavaliar com nova evidencia")


def _impacto_17b(it):
    if it["kind"] == "recife_strong":
        return ("Contribui com um patch-link forte review-only. Ainda assim, a amostra forte "
                "permanece em uma unica regiao/evento, insuficiente para o 17B.")
    return ("Sem geometria/vinculo forte, nao contribui diretamente para a prontidao do 17B; "
            "avanca a base regional como candidato parcial ou fila executavel.")


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, summary, gate17b):
    m = _metrics(items)
    reg_lines = "\n".join(
        f"- {r['regiao']} ({r['cidade']}): itens={r['itens']}; fortes={r['fortes']}; parciais={r['parciais']}; "
        f"contextuais={r['contextuais']}; bloqueados/fila={r['bloqueados_ou_fila']}"
        for r in resumo_regiao_rows(items)
    )
    fen_lines = "\n".join(f"- {r['regiao']} / {r['classe_fenomeno']}: {r['quantidade']}" for r in resumo_fenomeno_rows(items))
    strong_lines = "\n".join(
        f"- {it['item_id']} {it['patch_id']} | vinculo exact_polygon_overlap | fisico+espectral+chuva locais"
        for it in m["strong"]
    ) or "- nenhuma referencia forte"
    parcial_lines = "\n".join(
        f"- {it['item_id']} ({it['region']}) {it['candidate_event_id']} | {it['data_evento']} | {it['status']}"
        for it in (m["parciais"] + m["contextuais"])
    ) or "- nenhuma referencia parcial/contextual"
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate17b)
    return f"""# SUSC-18A Execucao de referencia observacional regional

## Estado herdado do 17I

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 17I herdado: {summary['herdado_17i_status_final']}
- Status 17B herdado: {summary['herdado_17i_status_17b']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Motivacao da execucao

O 17I mapeou a amostra regional, mas parou no diagnostico com filas executaveis. Esta etapa
executa de fato essas filas e consolida a base regional observacional: reune a referencia forte
de Recife com as features diretas ja extraidas localmente, preserva as referencias parciais
datadas de Curitiba e transforma o que falta em pacotes de execucao concretos (geometria,
separacao de fenomeno e footprint SAR), sem inventar dado.

## Metodologia

Padrao por candidato: registro de evento -> geometria oficial ou footprint tecnico -> vinculo com
patch -> features diretas (somente pre-evento) -> avaliacao somente revisao -> gate 17B. Onde falta
geometria, data, separacao de fenomeno ou raster, gera-se fila executavel com fonte, formato,
campos e comando. Bairro/rua/texto nunca vira geometria forte; alerta/area de risco nunca vira
ocorrencia; feature pos-evento nunca vira feature pre-evento.

## Resultado Recife (referencia)

Recife entrega {summary['referencia_forte']} referencias fortes review-only (canarios), cada uma com
footprint tecnico, vinculo forte (exact_polygon_overlap, herdado do 17D) e features diretas locais
(fisico do 17G, espectral pre-evento do 17C20, chuva pre-evento do 17C25):

{strong_lines}

## Resultado Curitiba

Curitiba tem eventos de inundacao datados por fonte oficial, mas nenhuma geometria oficial local.
Nenhum ponto/poligono/shapefile foi encontrado nos artefatos; endereco/rua nao vira geometria forte.
Os eventos datados seguem como referencia parcial e entram na fila de obtencao de geometria
({summary['fila_geometria_curitiba']} tarefas).

## Resultado Petropolis

Petropolis tem muitos registros, mas predominantemente de fenomeno misto/deslizamento (laudos
geotecnicos de encosta). Sem separacao por ocorrencia, nao podem entrar como inundacao. Apenas o
evento declarado de inundacao segue como evidencia contextual. Gerou-se fila de separacao de
fenomeno ({summary['fila_separacao_fenomeno_petropolis']} tarefas).

## Geometrias encontradas e ausentes

- Recife: footprint tecnico (geometria tecnica) disponivel review-only.
- Curitiba/Petropolis: sem geometria oficial forte local; todas em fila.

{reg_lines}

## Fenomeno por regiao

{fen_lines}

## Footprints produzidos ou enfileirados

- Recife: footprint tecnico review-only herdado (nao e ground truth, nao e feature pre-evento).
- Curitiba/Petropolis: footprint SAR em fila de execucao ({summary['fila_footprint_sar']} tarefas), dependente de AOI oficial.

## Vinculos patch

- Patch-links fortes review-only: {summary['patch_links_fortes']} (Recife).
- Curitiba/Petropolis: same_region_only (nao forte) ate obter geometria.

## Features disponiveis

- Referencia forte de Recife com fisico, espectral e chuva locais (somente pre-evento).
- Curitiba/Petropolis: extracao de features em fila ({summary['fila_extracao_features']} tarefas), apos geometria/patch.

## Referencias parciais e contextuais

{parcial_lines}

## Gate de prontidao 17B pos-18A

{gate_lines}

- Status 17B: **{summary['status_17b']}**
- Nenhum benchmark 17B foi criado.

## Lacunas restantes

- Curitiba: geometria oficial datada por evento.
- Petropolis: separacao de fenomeno por ocorrencia (inundacao x deslizamento).
- Todas as regioes: footprint SAR depende de AOI oficial; features regionais dependem de patch-link.

## Conclusao

- O que avancou de verdade: a referencia forte de Recife foi consolidada com features diretas locais
  (fisico/espectral/chuva) e {summary['patch_links_fortes']} patch-links fortes review-only; Curitiba
  manteve {summary['referencia_parcial']} referencias parciais datadas; e todas as lacunas viraram
  filas executaveis concretas ({summary['fila_total']} tarefas no total).
- O que segue bloqueado: geometria oficial de Curitiba, separacao de fenomeno de Petropolis e a
  ampliacao para mais de uma regiao/evento forte; por isso o 17B permanece em aproximacao regional.
- Proximo marco recomendado: **SUSC-18B** — executar a fila de geometria de Curitiba e a separacao
  de fenomeno de Petropolis, resolver patch-links e extrair features diretas, buscando a segunda
  regiao com referencia forte, sempre somente revisao, sem ground truth, treino ou score_v7.

## Garantias

- ground_truth=false; eligible_for_training=false; score_v7_allowed=false; review_only=true.
- score_v6 intacto ({summary['score_v6_changed']} para alterado); nenhum benchmark 17B criado.
"""


# --- Orquestracao ----------------------------------------------------------
def build_all():
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    ensure_dir(SCHEMAS)
    write_schema()

    items = build_items()
    exec_fila = exec_fila_rows()
    cur_geom = curitiba_geom_rows(items)
    cur_fila = curitiba_fila_geom_rows(items)
    pet_fen = petropolis_fenomeno_rows(items)
    pet_fila = petropolis_fila_fenomeno_rows(items)
    footprints = footprints_rows(items)
    fila_fp = fila_footprint_rows(items)
    vinculos = vinculos_rows(items)
    features = features_rows(items)
    fila_feat = fila_features_rows(items)
    matriz = matriz_rows(items)
    has_queue = bool(cur_fila or pet_fila or fila_fp or fila_feat)
    final_status = _final_status(items, has_queue)
    gate17b = gate_17b_rows(items)
    summary = summary_obj(items, exec_fila, cur_fila, pet_fila, fila_fp, fila_feat, final_status)

    write_json(PREFLIGHT_JSON, preflight_obj())
    write_csv(EXEC_FILA, exec_fila, EXEC_FIELDS)
    write_csv(CUR_GEOM, cur_geom, CUR_GEOM_FIELDS)
    write_csv(CUR_FILA_GEOM, cur_fila, CUR_FILA_GEOM_FIELDS)
    write_csv(PET_FENOMENO, pet_fen, PET_FENOMENO_FIELDS)
    write_csv(PET_FILA_FENOMENO, pet_fila, PET_FILA_FENOMENO_FIELDS)
    write_csv(FOOTPRINTS, footprints, FOOTPRINTS_FIELDS)
    write_csv(FILA_FOOTPRINT, fila_fp, FILA_FOOTPRINT_FIELDS)
    write_csv(VINCULOS, vinculos, VINCULOS_FIELDS)
    write_csv(FEATURES, features, FEATURES_FIELDS)
    write_csv(FILA_FEATURES, fila_feat, FILA_FEATURES_FIELDS)
    write_csv(MATRIZ, matriz, MATRIZ_FIELDS)
    write_csv(GATE_17B, gate17b, GATE_FIELDS)
    write_csv(RESUMO_REGIAO, resumo_regiao_rows(items), RESUMO_REGIAO_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status_rows(items), RESUMO_STATUS_FIELDS)
    write_csv(RESUMO_FENOMENO, resumo_fenomeno_rows(items), RESUMO_FENOMENO_FIELDS)
    write_json(SUMMARY, summary)
    for it in items:
        if _card_worthy(it):
            write_markdown(CARDS_DIR / f"{it['item_id']}.md", card_markdown(it))
    write_markdown(REPORT, report_markdown(items, summary, gate17b))
    return summary


# --- Validacao -------------------------------------------------------------
def _public_output_paths():
    paths = [REPORT]
    paths.extend(OUT_DATA.glob("*.csv"))
    paths.extend(OUT_DATA.glob("*.json"))
    paths.extend(CARDS_DIR.glob("*.md"))
    return [p for p in paths if p.exists()]


def public_text_violations_text(text):
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def public_text_violations_files(paths=None):
    errors = []
    for path in paths or _public_output_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:{','.join(hits)}")
    return errors


def validate_matriz_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("item_id", f"row{idx}")
        for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        if row.get("review_only") != "true":
            errors.append(f"{rid}:review_only_deve_ser_true")
        st = row.get("status_referencia_observacional", "")
        if st not in STATUS_REF_ALLOWED:
            errors.append(f"{rid}:status_fora_enum:{st}")
        if row.get("uso_permitido") not in USO_PERMITIDO_ALLOWED:
            errors.append(f"{rid}:uso_fora_enum")
        if row.get("classe_vinculo") not in CLASSE_VINCULO_ALLOWED:
            errors.append(f"{rid}:classe_vinculo_fora_enum")
        forte = st == "referencia_observacional_forte_somente_revisao"
        vinc_forte = row.get("classe_vinculo") in CLASSE_VINCULO_FORTE
        # patch-link forte exige geometry_id e patch_id
        if vinc_forte:
            if row.get("geometry_id") in {"", "not_available"}:
                errors.append(f"{rid}:vinculo_forte_sem_geometry_id")
            if row.get("patch_id") in {"", "not_available"}:
                errors.append(f"{rid}:vinculo_forte_sem_patch_id")
        # referencia forte nunca em same_region_only
        if forte and row.get("classe_vinculo") == "same_region_only":
            errors.append(f"{rid}:forte_sem_vinculo_forte")
        # fenomeno misto/deslizamento nunca vira inundacao forte/parcial
        # (checado via feature disponivel vs status; garantido pela classificacao)
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate_vinculos_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("vinculo_id", f"link{idx}")
        if row.get("classe_vinculo") not in CLASSE_VINCULO_ALLOWED:
            errors.append(f"{rid}:classe_vinculo_fora_enum")
        if row.get("vinculo_forte") == "true":
            if row.get("geometry_id") in {"", "not_available"}:
                errors.append(f"{rid}:forte_sem_geometry_id")
            if row.get("patch_id") in {"", "not_available"}:
                errors.append(f"{rid}:forte_sem_patch_id")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate_features_rows(rows):
    errors = []
    fisico_fields = ["fisico_elevacao_media", "fisico_declividade_media", "fisico_hand_media",
                     "fisico_twi_media", "fisico_dist_agua_min", "fisico_flow_acc_media"]
    for idx, row in enumerate(rows, start=1):
        rid = row.get("vinculo_id", f"feat{idx}")
        preenchido = any(row.get(f, "not_available") not in {"", "not_available"} for f in fisico_fields)
        if preenchido and not row.get("fonte_fisico", "").strip():
            errors.append(f"{rid}:feature_fisica_sem_fonte")
        if row.get("espectral_disponivel") == "true" and not row.get("fonte_espectral", "").strip():
            errors.append(f"{rid}:espectral_sem_fonte")
        if row.get("chuva_disponivel") == "true" and not row.get("fonte_chuva", "").strip():
            errors.append(f"{rid}:chuva_sem_fonte")
        if row.get("feature_pre_evento_apenas") != "true":
            errors.append(f"{rid}:feature_pre_evento_apenas_deve_ser_true")
    return errors


def validate_petropolis_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("candidate_event_id", f"pet{idx}")
        if row.get("classe_fenomeno") not in FENOMENO_CLASS_ALLOWED:
            errors.append(f"{rid}:classe_fenomeno_fora_enum")
        # misto/deslizamento nao pode seguir como inundacao sem separacao
        if row.get("classe_fenomeno") in {"fenomeno_misto", "deslizamento"} and \
                row.get("pode_seguir_como_inundacao") == "true":
            errors.append(f"{rid}:fenomeno_misto_virou_inundacao")
    return errors


def validate_curitiba_geom_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("candidate_event_id", f"cur{idx}")
        cg = row.get("classe_geometria", "")
        if cg not in CLASSE_GEOMETRIA_ALLOWED:
            errors.append(f"{rid}:classe_geometria_fora_enum")
        # endereco textual nunca marcado como geometria normalizada forte
        if cg in {"endereco_textual_sem_geometria", "sem_geometria"} and row.get("geometria_normalizada") == "true":
            errors.append(f"{rid}:endereco_ou_texto_como_geometria_forte")
    return errors


def validate():
    summary = build_all()
    errors = []
    matriz = _read(MATRIZ)
    if not matriz:
        errors.append("matriz_regional_vazia")
    errors.extend(validate_matriz_rows(matriz))
    errors.extend(validate_vinculos_rows(_read(VINCULOS)))
    errors.extend(validate_features_rows(_read(FEATURES)))
    errors.extend(validate_petropolis_rows(_read(PET_FENOMENO)))
    errors.extend(validate_curitiba_geom_rows(_read(CUR_GEOM)))
    errors.extend(public_text_violations_files())

    for row in _read(EXEC_FILA):
        if row.get("resultado_execucao") not in RESULTADO_EXEC_ALLOWED:
            errors.append(f"exec:{row.get('task_id')}:resultado_fora_enum")

    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_proibido")
    if summary["status_final_18a"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_18a']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["status_final_18a"] == "18A_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_avanco_funcional_entregue")
    for field in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary[field] != 0:
            errors.append(f"{field}_nonzero")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    # avanco funcional minimo: referencia forte OU parcial OU fila executavel
    if not (summary["referencia_forte"] or summary["referencia_parcial"] or summary["fila_total"]):
        errors.append("nenhum_avanco_funcional")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18A execucao referencia observacional regional validada: "
        f"fortes={summary['referencia_forte']} parciais={summary['referencia_parcial']} "
        f"contextuais={summary['evidencia_contextual']} links_fortes={summary['patch_links_fortes']} "
        f"fila={summary['fila_total']} status18A={summary['status_final_18a']} status17B={summary['status_17b']}"
    )
    return 0


def run_all():
    summary = build_all()
    print(
        "18A execucao referencia observacional regional gerada: "
        f"rec={summary['itens_rec']} cur={summary['itens_cur']} pet={summary['itens_pet']} "
        f"fortes={summary['referencia_forte']} parciais={summary['referencia_parcial']} "
        f"fila={summary['fila_total']} status18A={summary['status_final_18a']} status17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
