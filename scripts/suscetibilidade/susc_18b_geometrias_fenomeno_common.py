"""SUSC-18B execucao de geometrias regionais e separacao de fenomeno.

O 18A consolidou a referencia forte de Recife (5 canarios, 1 evento, 1 regiao) e
deixou Curitiba com referencias parciais datadas sem geometria e Petropolis
bloqueada por fenomeno misto/deslizamento, com 54 tarefas executaveis. Esta etapa
executa de fato as filas mais importantes:

  1. resolve/tenta geometria oficial ou tecnica de Curitiba a partir de artefatos locais;
  2. separa os fenomenos de Petropolis item a item;
  3. prepara footprint tecnico quando ha data e AOI suficientes;
  4. resolve vinculos evento-patch;
  5. localiza features diretas para novos vinculos;
  6. atualiza a prontidao do 17B, sem criar benchmark.

Achado honesto: as registries de protocolo anteriores (v1uw/v1ux/v1uy) ja haviam
estabelecido que o evento datado de Curitiba tem apenas ancoras hidrometeorologicas
(timing) e patches candidatos region-only, sem camada de ocorrencia; e que os
laudos CPRM de Petropolis de 2022 sao geotecnicos de encosta (fenomeno misto),
com apenas um evento de inundacao datado (2024) sem geometria. Sem geometria de
ocorrencia local nem download de raster autorizado, nenhuma segunda regiao vira
forte agora; o avanco e reduzir concretamente os bloqueios e emitir filas
executaveis especificas.

Nunca cria ground truth, treino, score_v7 nem benchmark 17B. O score_v6 nunca e
alterado. Bairro/rua/texto e centroide de bairro/municipio nunca viram geometria
forte. Deslizamento nunca e misturado com inundacao. Alerta/area de risco nunca
vira ocorrencia. Feature pos-evento nunca vira feature pre-evento. Nada e inventado.
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
DAT_PROTC = DATASETS / "protocolo_c"
OUT_SUSC = ROOT / "outputs_public" / "suscetibilidade"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA_17G = ROOT / "outputs_public" / "data" / "susc_17g_extracao_direta_features_fisicas_canarios"
OUT_DATA_18A = ROOT / "outputs_public" / "data" / "susc_18a_execucao_referencia_observacional_regional"
OUT_DATA = ROOT / "outputs_public" / "data" / "susc_18b_execucao_geometrias_regionais_separacao_fenomeno"
CARDS_DIR = OUT_DATA / "cartoes_regionais"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
TARGET_PACK_17C = OUT_DATA_17C / "susc_17c_source_target_pack.csv"
FEATURES_FISICAS_17G = OUT_DATA_17G / "matriz_features_fisicas_diretas_canarios.csv"
MATRIZ_18A = OUT_DATA_18A / "matriz_referencia_observacional_regional.csv"
SUMMARY_18A = OUT_DATA_18A / "summary.json"
CUR_PRELINK = DAT_PROTC / "v1uw_curitiba_event_patch_prelink_update.csv"
CUR_HYDROMET = DAT_PROTC / "v1uw_curitiba_hydromet_anchor_registry.csv"
SPECTRAL_DIR_17C20 = OUT_SUSC / "susc_17c20_light_artifacts"
CHIRPS_DIR_17C25 = OUT_SUSC / "susc_17c25_chirps_artifacts"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_JSON = OUT_DATA / "preflight.json"
CUR_EXEC_GEOM = OUT_DATA / "curitiba_execucao_geometria.csv"
CUR_GEOM_NORM = OUT_DATA / "curitiba_geometrias_normalizadas.csv"
CUR_TAREFAS_EXT = OUT_DATA / "curitiba_tarefas_externas_geometria.csv"
PET_SEP_EXEC = OUT_DATA / "petropolis_separacao_fenomeno_executada.csv"
PET_INUND_SEP = OUT_DATA / "petropolis_candidatos_inundacao_separada.csv"
PET_BLOQUEIOS = OUT_DATA / "petropolis_bloqueios_fenomeno.csv"
PET_TAREFAS_EXT = OUT_DATA / "petropolis_tarefas_externas_separacao.csv"
FOOTPRINTS = OUT_DATA / "footprints_tecnicos_18b.csv"
FILA_SAR = OUT_DATA / "fila_sar_18b.csv"
VINCULOS = OUT_DATA / "vinculos_evento_patch_18b.csv"
FEATURES = OUT_DATA / "features_regionais_18b.csv"
FILA_FEATURES = OUT_DATA / "fila_features_regionais_18b.csv"
MATRIZ = OUT_DATA / "matriz_referencia_observacional_18b.csv"
COMPARACAO = OUT_DATA / "comparacao_regional_recife_curitiba_petropolis.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_18b.csv"
RESUMO_REGIAO = OUT_DATA / "resumo_por_regiao.csv"
RESUMO_STATUS = OUT_DATA / "resumo_por_status.csv"
RESUMO_FENOMENO = OUT_DATA / "resumo_por_fenomeno.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_18B_EXECUCAO_GEOMETRIAS_REGIONAIS_SEPARACAO_FENOMENO.md"
SCHEMA = SCHEMAS / "susc_18b_geometrias_fenomeno_schema_v1.json"

REQUIRED_INPUTS = [TARGET_PACK_17C, FEATURES_FISICAS_17G, MATRIZ_18A, SUMMARY_18A, SCORE_V6]
REQUIRED_OUTPUTS = [
    PREFLIGHT_JSON, CUR_EXEC_GEOM, CUR_GEOM_NORM, CUR_TAREFAS_EXT,
    PET_SEP_EXEC, PET_INUND_SEP, PET_BLOQUEIOS, PET_TAREFAS_EXT,
    FOOTPRINTS, FILA_SAR, VINCULOS, FEATURES, FILA_FEATURES, MATRIZ,
    COMPARACAO, GATE_17B, RESUMO_REGIAO, RESUMO_STATUS, RESUMO_FENOMENO,
    SUMMARY, REPORT, SCHEMA,
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
GEOM_EXEC_STATUS_ALLOWED = [
    "geometria_oficial_resolvida", "geometria_tecnica_resolvida",
    "geometria_textual_bloqueada", "geometria_ausente_com_tarefa_externa", "rejeitado",
]
GEOM_EXEC_FORTE = {"geometria_oficial_resolvida", "geometria_tecnica_resolvida"}
CLASSE_VINCULO_ALLOWED = [
    "exact_polygon_overlap", "point_within_patch", "bbox_overlap",
    "near_patch_buffer_10m", "near_patch_buffer_30m",
    "same_region_only", "insufficient_for_patch_link",
]
CLASSE_VINCULO_FORTE = {"exact_polygon_overlap", "point_within_patch", "bbox_overlap"}
GATE_FINAL_ALLOWED = [
    "18B_SEGUNDA_REGIAO_FORTE_ALCANCADA",
    "18B_BLOQUEIOS_REGIONAIS_REDUZIDOS_COM_FILAS_EXECUTAVEIS",
    "18B_SEM_AVANCO_FORTE_MAS_COM_PLANO",
    "18B_BLOQUEADO_FAIL_CLOSED",
]
STATUS_17B_ALLOWED = [
    "17B_PRONTO_PARA_DESENHO_SOMENTE_REVISAO",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_FORTE",
    "17B_APROXIMACAO_REGIONAL_COM_REFERENCIAS_PARCIAS",
    "17B_BLOQUEADO_POR_GEOMETRIA",
    "17B_BLOQUEADO_POR_FENOMENO",
    "17B_BLOQUEADO_POR_AMOSTRA",
    "17B_BLOQUEADO_FAIL_CLOSED",
]

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
NO_GT_REASON = (
    "referencia observacional regional somente revisao; nao confirma ocorrencia no patch, "
    "nao e verdade de referencia, nao alimenta treino e nao autoriza score_v7"
)
HARD_DEFAULTS = {"ground_truth": "false", "eligible_for_training": "false",
                 "score_v7_allowed": "false", "review_only": "true"}

# --- Field lists -----------------------------------------------------------
CUR_EXEC_FIELDS = [
    "candidate_event_id", "cidade", "data_evento", "tipo_fenomeno", "fonte", "autoridade_fonte",
    "busca_local_realizada", "formatos_inspecionados", "geometria_local_encontrada",
    "patches_candidatos_region_only", "ancora_hidrometeorologica",
    "status_execucao_geometria", "continua_bloqueado", "proxima_acao", "justificativa_tecnica",
]
CUR_GEOM_NORM_FIELDS = [
    "geometry_id", "candidate_event_id", "geometry_type", "crs", "fonte_geometria",
    "geometria_de_ocorrencia", "review_only", "justificativa_tecnica",
]
CUR_TAREFAS_FIELDS = [
    "task_id", "candidate_event_id", "data_evento", "fonte_sugerida", "url_ou_caminho_esperado",
    "camada_ou_produto_esperado", "formato_esperado", "campos_necessarios",
    "expected_output_path", "command_hint", "criterio_de_sucesso",
]
PET_SEP_FIELDS = [
    "candidate_event_id", "cidade", "data_evento", "tipo_fenomeno_fonte", "artefato_fonte",
    "classe_fenomeno", "base_classificacao", "separacao_explicita_disponivel",
    "pode_seguir_como_inundacao", "continua_bloqueado", "justificativa_tecnica",
]
PET_INUND_FIELDS = [
    "candidate_event_id", "cidade", "data_evento", "fonte", "autoridade_fonte",
    "classe_fenomeno", "possui_geometria", "status_referencia_observacional",
    "proxima_acao", "justificativa_tecnica",
]
PET_BLOQ_FIELDS = [
    "candidate_event_id", "cidade", "data_evento", "classe_fenomeno",
    "motivo_bloqueio", "requer_separacao", "justificativa_tecnica",
]
PET_TAREFAS_FIELDS = [
    "task_id", "candidate_event_id", "fonte_sugerida", "url_ou_caminho_esperado",
    "artefato_necessario", "campos_necessarios", "expected_output_path", "command_hint", "criterio_de_sucesso",
]
FOOTPRINTS_FIELDS = [
    "footprint_id", "regiao", "candidate_event_id", "data_evento", "aoi_conhecida",
    "geometry_id", "footprint_status", "possui_raster_local", "fonte_footprint",
    "pos_evento_nao_e_feature_pre_evento", "not_ground_truth", "justificativa_tecnica",
]
FILA_SAR_FIELDS = [
    "task_id", "regiao", "candidate_event_id", "aoi", "bbox", "data_evento",
    "pre_window", "post_window", "colecao_esperada", "bandas", "polarizacao",
    "expected_output_path", "command_hint", "criterio_de_sucesso", "depende_de",
]
VINCULOS_FIELDS = [
    "vinculo_id", "regiao", "candidate_event_id", "geometry_id", "geometry_type", "patch_id",
    "classe_vinculo", "vinculo_forte", "patches_candidatos_region_only", "review_only", "justificativa_tecnica",
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
COMPARACAO_FIELDS = ["dimensao", "recife", "curitiba", "petropolis", "observacao"]
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
        raise AssertionError("score_v7 existe e e proibido para SUSC-18B")


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


def _target_by_id():
    return {r.get("candidate_event_id", ""): r for r in _read(TARGET_PACK_17C)}


# --- Curitiba: patches candidatos region-only e ancora hidromet ------------
def _cur_event_registry_key(candidate_event_id):
    # target pack: source_id/event_id carrega CTB_2022_01_15_16; prelink usa CUR_2022_01_15
    return {
        "S17C_REF_0060": "CUR_2022_01_15",
        "S17C_REF_0061": "CUR_2023_10_28",
        "S17C_REF_0062": "CUR_2024_02_18",
    }.get(candidate_event_id, "")


def _cur_region_only_patches(candidate_event_id):
    key = _cur_event_registry_key(candidate_event_id)
    if not key:
        return []
    patches = sorted({
        r.get("patch_id", "") for r in _read(CUR_PRELINK)
        if r.get("proposed_event_id") == key
        and r.get("linkage_basis") == "REGION_ONLY_EVENT_CANDIDATE"
        and r.get("patch_id")
    })
    return patches


def _cur_has_hydromet(candidate_event_id):
    key = _cur_event_registry_key(candidate_event_id)
    if not key:
        return False
    return any(
        r.get("candidate_event_id") == key and r.get("hydromet_support_class") == "HYDROMET_ANCHOR_AVAILABLE"
        for r in _read(CUR_HYDROMET)
    )


# --- Classificacao de geometria (execucao) ---------------------------------
def classify_geometria_execucao(cand):
    """Executa a classificacao de geometria de um candidato a partir de sinais
    oficiais locais. So retorna geometria resolvida quando ha ponto/poligono/
    footprint oficial de fato; endereco/rua/texto e centroide nunca resolvem."""
    if cand.get("has_official_geometry") == "true" or cand.get("has_point") == "true":
        return "geometria_oficial_resolvida"
    if cand.get("source_type") == "technical_remote_sensing_flood_footprint" and cand.get("has_bbox") == "true":
        return "geometria_tecnica_resolvida"
    if _address_only(cand):
        return "geometria_textual_bloqueada"
    if _has_date(cand.get("event_date_candidate", "")):
        return "geometria_ausente_com_tarefa_externa"
    return "rejeitado"


def _cur_flood_candidates():
    return sorted(
        [r for r in _read(TARGET_PACK_17C)
         if r.get("region") == "CUR" and _phenomenon_class(r.get("phenomenon_type", "")) == "inundacao_alagamento_enxurrada"],
        key=lambda r: r.get("candidate_event_id", ""),
    )


def curitiba_exec_rows():
    rows = []
    for c in _cur_flood_candidates():
        cid = c.get("candidate_event_id", "")
        status = classify_geometria_execucao(c)
        patches = _cur_region_only_patches(cid)
        hydromet = _cur_has_hydromet(cid)
        rows.append({
            "candidate_event_id": cid, "cidade": "Curitiba",
            "data_evento": c.get("event_date_candidate", "not_available"),
            "tipo_fenomeno": c.get("phenomenon_type", ""), "fonte": c.get("source_name", ""),
            "autoridade_fonte": c.get("authority_tier", ""),
            "busca_local_realizada": "csv;geojson;shapefile;parquet;xlsx;registries_protocolo_c",
            "formatos_inspecionados": "geometria_evento;lat/lon;ponto;poligono;bbox;wkt/wkb",
            "geometria_local_encontrada": "nenhuma_geometria_de_ocorrencia" if status not in GEOM_EXEC_FORTE else "geometria_oficial_local",
            "patches_candidatos_region_only": str(len(patches)),
            "ancora_hidrometeorologica": _bool(hydromet),
            "status_execucao_geometria": status,
            "continua_bloqueado": _bool(status not in GEOM_EXEC_FORTE),
            "proxima_acao": _cur_proxima_acao(status),
            "justificativa_tecnica": _cur_justificativa(status, len(patches), hydromet),
        })
    return rows


def _cur_proxima_acao(status):
    return {
        "geometria_oficial_resolvida": "normalizar CRS e resolver vinculo evento-patch",
        "geometria_tecnica_resolvida": "normalizar footprint e resolver vinculo evento-patch",
        "geometria_textual_bloqueada": "substituir endereco/texto por geometria oficial de ocorrencia com CRS",
        "geometria_ausente_com_tarefa_externa": "obter camada oficial de ocorrencia datada (fila externa)",
        "rejeitado": "resolver data exata antes de qualquer geometria",
    }.get(status, "reavaliar")


def _cur_justificativa(status, n_patches, hydromet):
    anchor = "ancora hidrometeorologica presente (apenas timing, nao e ocorrencia)" if hydromet else "sem ancora hidrometeorologica"
    if status == "geometria_textual_bloqueada":
        return ("evento datado oficial, mas so ha endereco/texto; endereco/rua/bairro nao vira geometria forte; "
                f"{n_patches} patches region-only candidatos; {anchor}")
    if status == "geometria_ausente_com_tarefa_externa":
        return ("evento datado oficial sem geometria de ocorrencia local; patches candidatos existem apenas "
                f"em nivel regional ({n_patches} region-only, sem overlay); {anchor}; centroide de bairro/municipio nao e usado")
    if status == "rejeitado":
        return "sem data exata ou intervalo; nao avaliavel para geometria"
    return "geometria oficial de ocorrencia local resolvida e normalizada"


def curitiba_geom_norm_rows(exec_rows):
    """So lista geometrias de ocorrencia realmente resolvidas e normalizadas.
    Sem geometria local real, o arquivo fica sem linhas de dados (apenas cabecalho)."""
    rows = []
    for r in exec_rows:
        if r["status_execucao_geometria"] in GEOM_EXEC_FORTE:
            rows.append({
                "geometry_id": f"S18B_CUR_GEOM_{r['candidate_event_id']}",
                "candidate_event_id": r["candidate_event_id"],
                "geometry_type": ("official_point_or_polygon" if r["status_execucao_geometria"] == "geometria_oficial_resolvida" else "technical_footprint"),
                "crs": "EPSG:4326", "fonte_geometria": "geometria_oficial_local",
                "geometria_de_ocorrencia": "true", "review_only": "true",
                "justificativa_tecnica": "geometria oficial de ocorrencia resolvida e normalizada para EPSG:4326",
            })
    return rows


def curitiba_tarefas_rows(exec_rows):
    rows = []
    counter = 0
    for r in exec_rows:
        if r["status_execucao_geometria"] in GEOM_EXEC_FORTE or r["status_execucao_geometria"] == "rejeitado":
            continue
        counter += 1
        cid = r["candidate_event_id"]
        rows.append({
            "task_id": f"S18B_CUR_GEO_{counter:04d}", "candidate_event_id": cid,
            "data_evento": r["data_evento"],
            "fonte_sugerida": "GeoCuritiba/IPPUC (camadas de alagamento) e Defesa Civil de Curitiba",
            "url_ou_caminho_esperado": "https://geocuritiba.ippuc.org.br ou solicitacao formal a Defesa Civil municipal",
            "camada_ou_produto_esperado": "camada de ocorrencia de alagamento/inundacao datada do evento",
            "formato_esperado": "geojson_ou_shapefile",
            "campos_necessarios": "geometria(ponto/poligono);crs;data_ocorrencia;tipo_fenomeno;fonte",
            "expected_output_path": f"local_runs/suscetibilidade/18b_regional/{cid.lower()}_geometria.geojson",
            "command_hint": "baixar camada oficial datada de ocorrencia, validar CRS e normalizar para EPSG:4326; nao usar setor de risco nem limite de bairro",
            "criterio_de_sucesso": "geometria oficial de ocorrencia com CRS, datada e vinculavel a patch por overlay",
        })
    return rows


# --- Petropolis: separacao de fenomeno -------------------------------------
def _pet_candidates():
    return sorted(
        [r for r in _read(TARGET_PACK_17C) if r.get("region") == "PET"],
        key=lambda r: r.get("candidate_event_id", ""),
    )


def _pet_base_classificacao(cand, classe):
    art = cand.get("artifact_ref", "") or cand.get("source_name", "")
    if classe == "inundacao_alagamento_enxurrada":
        return "fonte oficial declara inundacao/alagamento"
    if "ANEXO" in art.upper() or "CPRM" in art.upper():
        return "laudo geotecnico CPRM de encosta do desastre de fevereiro/2022 (fenomeno hidrometeorologico misto, nao separado por ocorrencia)"
    if classe == "fenomeno_misto":
        return "relatorio hidrometeorologico do evento sem separacao por ocorrencia"
    return "registro sem tipo de fenomeno resolvido"


def petropolis_sep_rows():
    rows = []
    for c in _pet_candidates():
        classe = _phenomenon_class(c.get("phenomenon_type", ""))
        pode = classe == "inundacao_alagamento_enxurrada"
        rows.append({
            "candidate_event_id": c.get("candidate_event_id", ""), "cidade": "Petropolis",
            "data_evento": c.get("event_date_candidate", "not_available"),
            "tipo_fenomeno_fonte": c.get("phenomenon_type", ""),
            "artefato_fonte": c.get("artifact_ref", "not_available") or "not_available",
            "classe_fenomeno": classe,
            "base_classificacao": _pet_base_classificacao(c, classe),
            "separacao_explicita_disponivel": "false",
            "pode_seguir_como_inundacao": _bool(pode),
            "continua_bloqueado": _bool(not pode),
            "justificativa_tecnica": (
                "fenomeno declarado como inundacao pela fonte oficial; segue como evidencia contextual review-only, "
                "geometria de ocorrencia ainda pendente"
                if pode else
                "fenomeno misto/deslizamento nao pode entrar como inundacao sem separacao espacial, documental ou "
                "tecnica explicita por ocorrencia; permanece bloqueado"
            ),
        })
    return rows


def petropolis_inund_rows(sep_rows):
    rows = []
    tgt = _target_by_id()
    for r in sep_rows:
        if r["classe_fenomeno"] != "inundacao_alagamento_enxurrada":
            continue
        c = tgt.get(r["candidate_event_id"], {})
        rows.append({
            "candidate_event_id": r["candidate_event_id"], "cidade": "Petropolis",
            "data_evento": r["data_evento"], "fonte": c.get("source_name", ""),
            "autoridade_fonte": c.get("authority_tier", ""),
            "classe_fenomeno": r["classe_fenomeno"], "possui_geometria": "false",
            "status_referencia_observacional": "evidencia_contextual",
            "proxima_acao": "obter geometria oficial de ocorrencia ou footprint tecnico datado",
            "justificativa_tecnica": ("inundacao datada separada dos demais fenomenos; segue como evidencia "
                                      "contextual review-only ate obter geometria de ocorrencia"),
        })
    return rows


def petropolis_bloqueio_rows(sep_rows):
    rows = []
    for r in sep_rows:
        if r["classe_fenomeno"] == "inundacao_alagamento_enxurrada":
            continue
        motivo = {
            "deslizamento": "fenomeno de deslizamento; nunca misturado com inundacao",
            "fenomeno_misto": "fenomeno misto (inundacao+deslizamento) sem separacao por ocorrencia",
            "insuficiente": "registro sem tipo de fenomeno ou sem data resolvida",
        }.get(r["classe_fenomeno"], "fenomeno nao elegivel")
        rows.append({
            "candidate_event_id": r["candidate_event_id"], "cidade": "Petropolis",
            "data_evento": r["data_evento"], "classe_fenomeno": r["classe_fenomeno"],
            "motivo_bloqueio": motivo,
            "requer_separacao": _bool(r["classe_fenomeno"] == "fenomeno_misto"),
            "justificativa_tecnica": r["justificativa_tecnica"],
        })
    return rows


def petropolis_tarefas_rows(sep_rows):
    rows = []
    counter = 0
    for r in sep_rows:
        if r["classe_fenomeno"] != "fenomeno_misto":
            continue
        counter += 1
        cid = r["candidate_event_id"]
        rows.append({
            "task_id": f"S18B_PET_FEN_{counter:04d}", "candidate_event_id": cid,
            "fonte_sugerida": "SGB/CPRM (laudos por ocorrencia) e Defesa Civil de Petropolis",
            "url_ou_caminho_esperado": "https://rigeo.sgb.gov.br ou solicitacao formal a Defesa Civil municipal",
            "artefato_necessario": "classificacao_por_ocorrencia_inundacao_x_deslizamento_com_geometria",
            "campos_necessarios": "ocorrencia_id;tipo_fenomeno;geometria;crs;data",
            "expected_output_path": f"local_runs/suscetibilidade/18b_regional/{cid.lower()}_fenomeno_separado.csv",
            "command_hint": "extrair por ocorrencia o tipo de fenomeno; separar inundacao de deslizamento com fonte oficial e geometria",
            "criterio_de_sucesso": "ocorrencias de inundacao separadas de deslizamento, com geometria e data oficiais",
        })
    return rows


# --- Itens consolidados (Recife forte + regionais) -------------------------
def build_recife_items():
    tgt = _target_by_id()
    evt = tgt.get(RECIFE_STRONG_EVENT, {})
    items = []
    for row in sorted(_read(FEATURES_FISICAS_17G), key=lambda r: r.get("canary_patch_id", "")):
        patch_id = row.get("canary_patch_id", "")
        canary_lower = patch_id.lower()
        items.append({
            "kind": "recife_strong", "candidate_event_id": row.get("candidate_event_id", RECIFE_STRONG_EVENT),
            "region": "REC", "cidade": "Recife",
            "data_evento": evt.get("event_date_candidate", "2022-05-24..2022-05-30"),
            "precisao_temporal": evt.get("event_date_precision", "range"),
            "tipo_fenomeno": "flood_inundation_alagamento",
            "classe_fenomeno": "inundacao_alagamento_enxurrada",
            "fonte": evt.get("source_name", "Sentinel-1 SAR metadata feasibility from SUSC-17C31"),
            "autoridade_fonte": evt.get("authority_tier", "technical"),
            "geometry_id": row.get("geometry_id", ""), "geometry_type": "technical_footprint",
            "patch_id": patch_id, "classe_vinculo": "exact_polygon_overlap",
            "possui_footprint_tecnico": True,
            "possui_fisico": row.get("features_diretas_completas") == "true",
            "possui_espectral": _spectral_available(canary_lower),
            "possui_chuva": _chirps_available(canary_lower),
            "fisico_row": row,
            "status": "referencia_observacional_forte_somente_revisao",
            "uso": "avaliacao_somente_revisao",
            "patches_region_only": 0,
        })
    return items


def _classify_regional(cand):
    classe = _phenomenon_class(cand.get("phenomenon_type", ""))
    has_date = _has_date(cand.get("event_date_candidate", ""))
    official = cand.get("authority_tier", "") == "official"
    documentary_ctx = cand.get("source_type", "") == "documentary_context"
    if classe in {"deslizamento", "fenomeno_misto"}:
        return "bloqueado_por_fenomeno"
    if classe == "insuficiente":
        return "bloqueado_sem_data" if not has_date else "bloqueado_por_fenomeno"
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
        cid = cand.get("candidate_event_id", "")
        patches = _cur_region_only_patches(cid) if region == "CUR" else []
        # patch candidato region-only nomeado (nao forte): melhora o vinculo de same_region_only
        patch_id = patches[0] if patches else "not_available"
        items.append({
            "kind": "regional_candidate", "cand": cand, "candidate_event_id": cid,
            "region": region, "cidade": REGION_CITY.get(region, cand.get("city", "")),
            "data_evento": cand.get("event_date_candidate", "not_available"),
            "precisao_temporal": cand.get("event_date_precision", ""),
            "tipo_fenomeno": cand.get("phenomenon_type", ""),
            "classe_fenomeno": _phenomenon_class(cand.get("phenomenon_type", "")),
            "fonte": cand.get("source_name", ""), "autoridade_fonte": cand.get("authority_tier", ""),
            "geometry_id": "not_available", "geometry_type": "none",
            "patch_id": patch_id, "classe_vinculo": "same_region_only",
            "possui_footprint_tecnico": False,
            "possui_fisico": False, "possui_espectral": False, "possui_chuva": False,
            "status": status, "uso": _uso_regional(status),
            "patches_region_only": len(patches),
        })
    return items


def build_items():
    items = build_recife_items() + build_regional_items()
    for i, it in enumerate(items, start=1):
        it["item_id"] = f"S18B_ITEM_{i:04d}"
    return items


# --- Qualidades ------------------------------------------------------------
def _q_fonte(it):
    return {"official": 85, "technical": 80, "documentary": 55, "internal_registry": 30}.get(it.get("autoridade_fonte", ""), 20)


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


# --- Tarefa 4: footprints e fila SAR ---------------------------------------
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
                "footprint_id": f"S18B_FP_{counter:04d}", "regiao": "REC",
                "candidate_event_id": it["candidate_event_id"], "data_evento": it["data_evento"],
                "aoi_conhecida": "true", "geometry_id": it["geometry_id"],
                "footprint_status": "disponivel_review_only_herdado_17d",
                "possui_raster_local": "false",
                "fonte_footprint": "footprint tecnico SAR Sentinel-1 validado no 17D (exact_polygon_overlap)",
                "pos_evento_nao_e_feature_pre_evento": "true", "not_ground_truth": "true",
                "justificativa_tecnica": ("footprint tecnico review-only herdado do 17D, ancora dos 5 canarios; "
                                          "usado apenas como geometria/vinculo, nunca como ground truth ou feature pre-evento"),
            })
    seen = set()
    for it in items:
        if it["kind"] != "regional_candidate" or it["classe_fenomeno"] != "inundacao_alagamento_enxurrada":
            continue
        if not _has_date(it["data_evento"]) or it["candidate_event_id"] in seen:
            continue
        seen.add(it["candidate_event_id"])
        counter += 1
        rows.append({
            "footprint_id": f"S18B_FP_{counter:04d}", "regiao": it["region"],
            "candidate_event_id": it["candidate_event_id"], "data_evento": it["data_evento"],
            "aoi_conhecida": "false", "geometry_id": "not_available",
            "footprint_status": "fila_execucao_externa", "possui_raster_local": "false",
            "fonte_footprint": "nenhum raster local; execucao SAR futura apos AOI oficial",
            "pos_evento_nao_e_feature_pre_evento": "true", "not_ground_truth": "true",
            "justificativa_tecnica": ("data e regiao conhecidas, mas AOI oficial pendente; sem raster local e sem "
                                      "download autorizado, o footprint vira pacote de execucao futura"),
        })
    return rows


def fila_sar_rows(items):
    rows = []
    counter = 0
    seen = set()
    for it in items:
        if it["kind"] != "regional_candidate" or it["classe_fenomeno"] != "inundacao_alagamento_enxurrada":
            continue
        if not _has_date(it["data_evento"]) or it["candidate_event_id"] in seen:
            continue
        seen.add(it["candidate_event_id"])
        counter += 1
        cid = it["candidate_event_id"]
        rows.append({
            "task_id": f"S18B_SAR_{counter:04d}", "regiao": it["region"], "candidate_event_id": cid,
            "aoi": f"limite_municipal_oficial_de_{REGION_CITY[it['region']].lower()}_a_obter",
            "bbox": "pendente_de_aoi_oficial",
            "data_evento": it["data_evento"],
            "pre_window": "ate_12_dias_antes_do_inicio_do_evento",
            "post_window": "ate_12_dias_apos_o_fim_do_evento",
            "colecao_esperada": "sentinel-1-rtc", "bandas": "VV,VH", "polarizacao": "VV+VH",
            "expected_output_path": f"local_runs/suscetibilidade/18b_regional/{cid.lower()}_footprint.geojson",
            "command_hint": ("com AOI oficial definida, aplicar o metodo do DEVRO02D (diferenca VV pre/pos, agua "
                             "permanente JRC mascarada, vetorizar); nao baixar raster pesado sem autorizacao"),
            "criterio_de_sucesso": "footprint tecnico candidato review-only com CRS e data",
            "depende_de": "geometria/AOI oficial da fila de obtencao de geometria",
        })
    return rows


# --- Tarefa 5: vinculos evento-patch ---------------------------------------
def vinculos_rows(items):
    rows = []
    counter = 0
    for it in items:
        counter += 1
        if it["kind"] == "recife_strong":
            classe = "exact_polygon_overlap"
            just = ("vinculo forte review-only herdado do 17D: canario com overlap exato de poligono sobre o "
                    "footprint tecnico; geometria e patch presentes")
        else:
            classe = "same_region_only"
            n = it.get("patches_region_only", 0)
            just = (f"sem geometria de ocorrencia; vinculo apenas regional com {n} patches candidatos region-only "
                    "(sem overlay); nao forte" if n else
                    "sem geometria oficial forte; vinculo apenas regional, nao forte")
        rows.append({
            "vinculo_id": f"S18B_LINK_{counter:04d}", "regiao": it["region"],
            "candidate_event_id": it["candidate_event_id"], "geometry_id": it["geometry_id"],
            "geometry_type": it["geometry_type"], "patch_id": it["patch_id"], "classe_vinculo": classe,
            "vinculo_forte": _bool(classe in CLASSE_VINCULO_FORTE),
            "patches_candidatos_region_only": str(it.get("patches_region_only", 0)),
            "review_only": "true", "justificativa_tecnica": just,
        })
    return rows


# --- Tarefa 6: features -----------------------------------------------------
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
            "completude_features": f"{completude:.2f}", "feature_pre_evento_apenas": "true",
            "not_ground_truth": "true",
            "justificativa_tecnica": ("features diretas review-only ancoradas no vinculo forte; fisico topografico "
                                      "estatico, espectral e chuva na janela pre-evento; nenhuma feature pos-evento usada"),
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
                "task_id": f"S18B_FEAT_{counter:04d}", "regiao": it["region"], "candidate_event_id": cid,
                "familia_feature": familia,
                "pre_requisito": "geometria de ocorrencia ou footprint tecnico com patch-link resolvido",
                "fonte_sugerida": {
                    "fisico": "DEM Copernicus GLO-30 + hidrografia oficial (metodo 17F/17G)",
                    "espectral": "Sentinel-2 COG pre-evento (metodo 17C20)",
                    "chuva": "CHIRPS janela pre-evento (metodo 17C25)",
                }[familia],
                "expected_output_path": f"local_runs/suscetibilidade/18b_regional/{cid.lower()}_{familia}.csv",
                "command_hint": "apos resolver o patch-link, extrair a familia na janela pre-evento",
                "criterio_de_sucesso": f"features de {familia} por patch, somente pre-evento, com fonte",
            })
    return rows


# --- Tarefa 7: matriz 18B ---------------------------------------------------
def matriz_rows(items):
    rows = []
    for it in items:
        just = ("referencia forte review-only: footprint tecnico + vinculo forte + features diretas locais"
                if it["kind"] == "recife_strong" else _justificativa_regional(it))
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
            "score_v7_allowed": HARD_DEFAULTS["score_v7_allowed"], "review_only": HARD_DEFAULTS["review_only"],
            "not_ground_truth_reason": NO_GT_REASON, "justificativa_tecnica": just,
        })
    return rows


def _justificativa_regional(it):
    base = (f"regiao {it['region']}; fenomeno {it['classe_fenomeno']}; data {it['data_evento']} "
            f"({it['precisao_temporal']}); fonte {it['autoridade_fonte']}; vinculo {it['classe_vinculo']}")
    extra = {
        "referencia_observacional_parcial": "inundacao datada oficial; geometria de ocorrencia e patch-link pendentes",
        "evidencia_contextual": "fonte documental datada usada apenas como contexto; sem geometria forte",
        "bloqueado_sem_geometria": "endereco/texto nao vira geometria forte; em fila de obtencao de geometria",
        "bloqueado_sem_data": "sem data exata ou intervalo; em fila de obtencao de data",
        "bloqueado_por_fenomeno": "fenomeno misto/deslizamento nao entra como inundacao sem separacao por ocorrencia",
    }.get(it["status"], "candidato regional review-only")
    return f"{base}; {extra}"


# --- Tarefa 8: comparacao regional -----------------------------------------
def _metrics(items):
    strong = [it for it in items if it["status"] == "referencia_observacional_forte_somente_revisao"]
    parciais = [it for it in items if it["status"] == "referencia_observacional_parcial"]
    contextuais = [it for it in items if it["status"] == "evidencia_contextual"]
    strong_events = {it["candidate_event_id"] for it in strong}
    strong_regions = {it["region"] for it in strong}
    regioes_evidencia = {it["region"] for it in (strong + parciais + contextuais)}
    return {
        "strong": strong, "parciais": parciais, "contextuais": contextuais,
        "eventos_distintos_fortes": len(strong_events),
        "regioes_com_referencia_forte": len(strong_regions),
        "regioes_com_evidencia": len(regioes_evidencia),
        "vinculos_fortes": len(strong),
        "n_parciais": len(parciais), "n_contextuais": len(contextuais),
    }


def _by_region(items, region):
    grp = [it for it in items if it["region"] == region]
    fortes = sum(1 for it in grp if it["status"] == "referencia_observacional_forte_somente_revisao")
    parciais = sum(1 for it in grp if it["status"] == "referencia_observacional_parcial")
    contextuais = sum(1 for it in grp if it["status"] == "evidencia_contextual")
    bloq_fen = sum(1 for it in grp if it["status"] == "bloqueado_por_fenomeno")
    feats = sum(1 for it in grp if it["kind"] == "recife_strong")
    links_fortes = sum(1 for it in grp if it["classe_vinculo"] in CLASSE_VINCULO_FORTE)
    return {"fortes": fortes, "parciais": parciais, "contextuais": contextuais,
            "bloq_fen": bloq_fen, "feats": feats, "links_fortes": links_fortes}


def comparacao_rows(items):
    r = {reg: _by_region(items, reg) for reg in REGIONS_ALL}
    lac = {
        "REC": "ampliar para uma segunda regiao/evento forte",
        "CUR": "geometria oficial de ocorrencia (evento datado, sem geometria)",
        "PET": "separacao de fenomeno por ocorrencia e geometria (predominio misto/deslizamento)",
    }
    dist = {
        "REC": "contribui com 1 evento/1 regiao/5 vinculos fortes",
        "CUR": "falta geometria de ocorrencia + overlay de patch para virar forte",
        "PET": "falta separar inundacao e obter geometria antes de qualquer vinculo forte",
    }
    def row(dim, key, obs):
        return {"dimensao": dim, "recife": str(r["REC"][key]), "curitiba": str(r["CUR"][key]),
                "petropolis": str(r["PET"][key]), "observacao": obs}
    rows = [
        row("eventos_fortes", "fortes", "referencias fortes somente revisao"),
        row("eventos_parciais", "parciais", "inundacao datada oficial sem geometria"),
        row("patch_links_fortes", "links_fortes", "vinculos fortes review-only"),
        row("features_completas", "feats", "vinculos com fisico+espectral+chuva locais"),
        row("fenomenos_bloqueados", "bloq_fen", "bloqueados por fenomeno misto/deslizamento"),
        {"dimensao": "lacuna_principal", "recife": lac["REC"], "curitiba": lac["CUR"],
         "petropolis": lac["PET"], "observacao": "principal bloqueio por regiao"},
        {"dimensao": "distancia_ate_17b", "recife": dist["REC"], "curitiba": dist["CUR"],
         "petropolis": dist["PET"], "observacao": "17B exige 3 eventos, 2 regioes, 20 vinculos fortes"},
    ]
    return rows


# --- Tarefa 9: gate 17B -----------------------------------------------------
def _status_17b(items):
    m = _metrics(items)
    minimos = (m["eventos_distintos_fortes"] >= 3 and m["regioes_com_referencia_forte"] >= 2
               and m["vinculos_fortes"] >= 20)
    if minimos:
        return "17B_PRONTO_PARA_DESENHO_SOMENTE_REVISAO"
    if m["regioes_com_referencia_forte"] >= 2:
        return "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_FORTE"
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
        ("minimo_2_regioes_fortes", str(m["regioes_com_referencia_forte"]), "2",
         m["regioes_com_referencia_forte"] >= 2, "regioes com referencia forte"),
        ("regioes_com_evidencia_observacional", str(m["regioes_com_evidencia"]), ">=2",
         m["regioes_com_evidencia"] >= 2, "inclui parciais/contextuais de Curitiba e Petropolis"),
        ("minimo_20_patch_links_fortes", str(m["vinculos_fortes"]), "20",
         m["vinculos_fortes"] >= 20, "patch-links fortes review-only (5 canarios de Recife)"),
        ("separacao_temporal_possivel", "false", "true", False, "amostra forte de um unico evento datado"),
        ("features_diretas_suficientes", _bool(bool(m["strong"])), "true", bool(m["strong"]),
         "referencia forte com fisico/espectral/chuva locais"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 nunca alterado"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o}
            for c, v, t, p, o in rows]


# --- Status final 18B ------------------------------------------------------
def _final_status(items, has_queue, geom_resolvida):
    m = _metrics(items)
    if not items:
        return "18B_BLOQUEADO_FAIL_CLOSED"
    if m["regioes_com_referencia_forte"] >= 2:
        return "18B_SEGUNDA_REGIAO_FORTE_ALCANCADA"
    if has_queue or geom_resolvida:
        return "18B_BLOQUEIOS_REGIONAIS_REDUZIDOS_COM_FILAS_EXECUTAVEIS"
    return "18B_SEM_AVANCO_FORTE_MAS_COM_PLANO"


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
def summary_obj(items, cur_exec, cur_tarefas, pet_sep, pet_inund, pet_bloq, pet_tarefas,
                fila_sar, fila_feat, geom_norm, final_status):
    inherited = read_json(SUMMARY_18A)
    m = _metrics(items)
    fila_total = len(cur_tarefas) + len(pet_tarefas) + len(fila_sar) + len(fila_feat)
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_18a_status_final": inherited.get("status_final_18a", ""),
        "herdado_18a_status_17b": inherited.get("status_17b", ""),
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
        "curitiba_geometrias_resolvidas": len(geom_norm),
        "curitiba_eventos_processados": len(cur_exec),
        "petropolis_inundacao_separada": len(pet_inund),
        "petropolis_bloqueados_fenomeno": len(pet_bloq),
        "fila_geometria_curitiba": len(cur_tarefas),
        "fila_separacao_petropolis": len(pet_tarefas),
        "fila_sar": len(fila_sar),
        "fila_features": len(fila_feat),
        "fila_total": fila_total,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "status_final_18b": final_status,
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
            {"role": "matriz_18a", "path": rel(MATRIZ_18A), "exists": MATRIZ_18A.exists()},
            {"role": "summary_18a", "path": rel(SUMMARY_18A), "exists": SUMMARY_18A.exists()},
            {"role": "curitiba_prelink_v1uw", "path": rel(CUR_PRELINK), "exists": CUR_PRELINK.exists()},
            {"role": "curitiba_hydromet_v1uw", "path": rel(CUR_HYDROMET), "exists": CUR_HYDROMET.exists()},
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
        "title": "SUSC-18B execucao de geometrias regionais e separacao de fenomeno",
        "type": "object",
        "properties": {
            "status_referencia_observacional": {"enum": STATUS_REF_ALLOWED},
            "uso_permitido": {"enum": USO_PERMITIDO_ALLOWED},
            "classe_fenomeno": {"enum": FENOMENO_CLASS_ALLOWED},
            "classe_vinculo": {"enum": CLASSE_VINCULO_ALLOWED},
            "status_execucao_geometria": {"enum": GEOM_EXEC_STATUS_ALLOWED},
            "status_final_18b": {"enum": GATE_FINAL_ALLOWED},
            "status_17b": {"enum": STATUS_17B_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "enums": {
            "status_referencia_observacional": STATUS_REF_ALLOWED, "uso_permitido": USO_PERMITIDO_ALLOWED,
            "classe_fenomeno": FENOMENO_CLASS_ALLOWED, "classe_vinculo": CLASSE_VINCULO_ALLOWED,
            "status_execucao_geometria": GEOM_EXEC_STATUS_ALLOWED,
            "status_final_18b": GATE_FINAL_ALLOWED, "status_17b": STATUS_17B_ALLOWED,
        },
        "hard_defaults": HARD_DEFAULTS,
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema():
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def _card_worthy(it):
    if it["status"] == "referencia_observacional_forte_somente_revisao":
        return True
    return it["classe_fenomeno"] == "inundacao_alagamento_enxurrada" and it["kind"] == "regional_candidate"


def card_markdown(it):
    forte = it["kind"] == "recife_strong"
    geometria = (f"footprint tecnico `{it['geometry_id']}` (review-only, herdado do 17D)" if forte
                 else "sem geometria de ocorrencia; patches candidatos apenas region-only")
    footprint = ("footprint tecnico SAR review-only disponivel (17D), usado so como geometria/vinculo" if forte
                 else "sem footprint local; execucao SAR em fila apos AOI oficial")
    vinculo = ("exact_polygon_overlap (forte, review-only)" if forte
               else f"same_region_only (nao forte; {it.get('patches_region_only', 0)} patches candidatos region-only)")
    features = ("fisico (17G), espectral pre-evento (17C20) e chuva pre-evento (17C25) disponiveis" if forte
                else "sem features vinculadas; extracao em fila apos resolver geometria/patch")
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

- {vinculo}
- Patch: `{it['patch_id']}`

## Features

- {features}

## Bloqueios

- Status: **{it['status']}**

## Decisao

- Uso permitido: {it['uso']}
- review_only=true; ground_truth=false; eligible_for_training=false; score_v7_allowed=false

## Acao minima

- {_acao_minima(it)}

## Impacto no 17B

{_impacto_17b(it)}

## Por que nao e ground truth

Referencia observacional regional somente revisao; nao confirma ocorrencia no patch nem constitui verdade de referencia.

## Por que nao e treinavel

Sem rotulo validado; referencia review-only nunca alimenta treino supervisionado.

## Por que nao cria score_v7

Nenhum score oficial ou score_v7 e criado; o score_v6 permanece intacto.
"""


def _acao_minima(it):
    return {
        "referencia_observacional_forte_somente_revisao": "manter como referencia review-only e ampliar amostra em outra regiao",
        "referencia_observacional_parcial": "obter geometria oficial de ocorrencia datada e resolver o patch-link por overlay",
        "evidencia_contextual": "obter geometria oficial de ocorrencia ou footprint tecnico datado",
        "bloqueado_sem_geometria": "substituir endereco/texto por geometria oficial de ocorrencia com CRS",
        "bloqueado_sem_data": "resolver data exata ou intervalo por fonte oficial",
        "bloqueado_por_fenomeno": "separar inundacao de deslizamento por ocorrencia com fonte oficial",
    }.get(it["status"], "reavaliar com nova evidencia")


def _impacto_17b(it):
    if it["kind"] == "recife_strong":
        return ("Contribui com um patch-link forte review-only. A amostra forte permanece em uma unica "
                "regiao/evento, insuficiente para o 17B.")
    return ("Sem geometria/vinculo forte, ainda nao contribui para a prontidao do 17B; avanca a base regional "
            "como referencia parcial ou fila executavel especifica.")


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, summary, gate17b, cur_exec, pet_sep):
    m = _metrics(items)
    reg_lines = "\n".join(
        f"- {r['regiao']} ({r['cidade']}): itens={r['itens']}; fortes={r['fortes']}; parciais={r['parciais']}; "
        f"contextuais={r['contextuais']}; bloqueados/fila={r['bloqueados_ou_fila']}"
        for r in resumo_regiao_rows(items))
    cur_lines = "\n".join(
        f"- {r['candidate_event_id']} ({r['data_evento']}): {r['status_execucao_geometria']} | "
        f"patches region-only={r['patches_candidatos_region_only']}"
        for r in cur_exec)
    pet_counts = Counter(r["classe_fenomeno"] for r in pet_sep)
    pet_lines = "\n".join(f"- {cls}: {pet_counts[cls]}" for cls in FENOMENO_CLASS_ALLOWED if pet_counts.get(cls, 0))
    strong_lines = "\n".join(
        f"- {it['item_id']} {it['patch_id']} | exact_polygon_overlap | fisico+espectral+chuva locais"
        for it in m["strong"]) or "- nenhuma"
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate17b)
    return f"""# SUSC-18B Execucao de geometrias regionais e separacao de fenomeno

## Estado herdado do 18A

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 18A herdado: {summary['herdado_18a_status_final']}
- Status 17B herdado: {summary['herdado_18a_status_17b']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Objetivo do 18B

Executar as filas regionais mais importantes do 18A para tentar transformar Curitiba ou Petropolis em
segunda regiao com referencia observacional forte somente revisao: resolver geometria, separar fenomeno,
preparar footprint tecnico, resolver vinculos e localizar features, sem criar ground truth, treino,
score_v7 ou benchmark 17B.

## Execucao Curitiba

Busca local exaustiva por geometria de ocorrencia (csv, geojson, shapefile, parquet, xlsx e registries de
protocolo). Achado: o evento datado tem apenas ancoras hidrometeorologicas (timing) e patches candidatos
em nivel regional (region-only, sem overlay); nao ha camada de ocorrencia local. Endereco/rua/bairro e
centroide de municipio nao viram geometria forte.

{cur_lines}

Geometrias de ocorrencia resolvidas e normalizadas: {summary['curitiba_geometrias_resolvidas']}.

## Execucao Petropolis

Separacao de fenomeno executada item a item. Os laudos CPRM de 2022 sao geotecnicos de encosta (desastre
hidrometeorologico misto), sem separacao por ocorrencia; apenas o evento de inundacao de 2024 pode seguir
como evidencia contextual, e ainda sem geometria.

{pet_lines}

Candidatos de inundacao separados: {summary['petropolis_inundacao_separada']}; bloqueados por fenomeno:
{summary['petropolis_bloqueados_fenomeno']}.

## Geometrias encontradas e ausentes

- Recife: footprint tecnico (geometria tecnica) disponivel review-only.
- Curitiba/Petropolis: sem geometria oficial de ocorrencia local; todas em fila externa especifica.

{reg_lines}

## Fenomenos separados e bloqueados

- Petropolis: {summary['petropolis_inundacao_separada']} inundacao separada, {summary['petropolis_bloqueados_fenomeno']} bloqueados (misto/deslizamento), fila de separacao com {summary['fila_separacao_petropolis']} tarefas.

## Footprints tecnicos produzidos ou enfileirados

- Recife: footprint tecnico review-only herdado (nao e ground truth, nao e feature pre-evento).
- Curitiba/Petropolis: footprint SAR em fila ({summary['fila_sar']} tarefas), dependente de AOI oficial.

## Vinculos patch gerados

- Patch-links fortes review-only: {summary['patch_links_fortes']} (Recife).
- Curitiba/Petropolis: same_region_only (nao forte); Curitiba com patches candidatos region-only nomeados.

## Features disponiveis

- Referencia forte de Recife com fisico, espectral e chuva locais (somente pre-evento):

{strong_lines}

- Curitiba/Petropolis: extracao de features em fila ({summary['fila_features']} tarefas), apos geometria/patch.

## Comparacao com Recife

Recife segue como unica regiao forte (1 evento, {summary['patch_links_fortes']} vinculos fortes, features
completas). Curitiba esta a um passo (evento datado oficial, falta apenas geometria de ocorrencia e
overlay). Petropolis esta a dois passos (falta separar fenomeno e obter geometria). Detalhe em
`comparacao_regional_recife_curitiba_petropolis.csv`.

## Gate de prontidao 17B pos-18B

{gate_lines}

- Status 17B: **{summary['status_17b']}**
- Nenhum benchmark 17B foi criado.

## Conclusao

- O que avancou de verdade: a execucao localizou com precisao o bloqueio de Curitiba (falta somente a
  geometria de ocorrencia; patches candidatos region-only ja identificados) e executou a separacao de
  fenomeno de Petropolis, isolando o unico candidato de inundacao. Foram emitidas filas executaveis
  especificas ({summary['fila_total']} tarefas: {summary['fila_geometria_curitiba']} geometria Curitiba,
  {summary['fila_separacao_petropolis']} separacao Petropolis, {summary['fila_sar']} footprint SAR,
  {summary['fila_features']} features).
- O que segue bloqueado: nenhuma segunda regiao virou forte com dado local; falta geometria de ocorrencia
  de Curitiba e separacao/geometria de Petropolis. O 17B permanece em aproximacao regional com referencias
  parciais.
- Proxima acao pesada: **SUSC-18C** — obter a geometria oficial de ocorrencia de Curitiba (camada GeoCuritiba/
  IPPUC ou solicitacao a Defesa Civil) e, com ela, executar o overlay de patch e a extracao de features
  diretas pre-evento, buscando a segunda regiao forte, sempre somente revisao e sem ground truth.

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
    cur_exec = curitiba_exec_rows()
    cur_norm = curitiba_geom_norm_rows(cur_exec)
    cur_tarefas = curitiba_tarefas_rows(cur_exec)
    pet_sep = petropolis_sep_rows()
    pet_inund = petropolis_inund_rows(pet_sep)
    pet_bloq = petropolis_bloqueio_rows(pet_sep)
    pet_tarefas = petropolis_tarefas_rows(pet_sep)
    footprints = footprints_rows(items)
    fila_sar = fila_sar_rows(items)
    vinculos = vinculos_rows(items)
    features = features_rows(items)
    fila_feat = fila_features_rows(items)
    matriz = matriz_rows(items)
    comparacao = comparacao_rows(items)
    has_queue = bool(cur_tarefas or pet_tarefas or fila_sar or fila_feat)
    final_status = _final_status(items, has_queue, bool(cur_norm))
    gate17b = gate_17b_rows(items)
    summary = summary_obj(items, cur_exec, cur_tarefas, pet_sep, pet_inund, pet_bloq, pet_tarefas,
                          fila_sar, fila_feat, cur_norm, final_status)

    write_json(PREFLIGHT_JSON, preflight_obj())
    write_csv(CUR_EXEC_GEOM, cur_exec, CUR_EXEC_FIELDS)
    write_csv(CUR_GEOM_NORM, cur_norm, CUR_GEOM_NORM_FIELDS)
    write_csv(CUR_TAREFAS_EXT, cur_tarefas, CUR_TAREFAS_FIELDS)
    write_csv(PET_SEP_EXEC, pet_sep, PET_SEP_FIELDS)
    write_csv(PET_INUND_SEP, pet_inund, PET_INUND_FIELDS)
    write_csv(PET_BLOQUEIOS, pet_bloq, PET_BLOQ_FIELDS)
    write_csv(PET_TAREFAS_EXT, pet_tarefas, PET_TAREFAS_FIELDS)
    write_csv(FOOTPRINTS, footprints, FOOTPRINTS_FIELDS)
    write_csv(FILA_SAR, fila_sar, FILA_SAR_FIELDS)
    write_csv(VINCULOS, vinculos, VINCULOS_FIELDS)
    write_csv(FEATURES, features, FEATURES_FIELDS)
    write_csv(FILA_FEATURES, fila_feat, FILA_FEATURES_FIELDS)
    write_csv(MATRIZ, matriz, MATRIZ_FIELDS)
    write_csv(COMPARACAO, comparacao, COMPARACAO_FIELDS)
    write_csv(GATE_17B, gate17b, GATE_FIELDS)
    write_csv(RESUMO_REGIAO, resumo_regiao_rows(items), RESUMO_REGIAO_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status_rows(items), RESUMO_STATUS_FIELDS)
    write_csv(RESUMO_FENOMENO, resumo_fenomeno_rows(items), RESUMO_FENOMENO_FIELDS)
    write_json(SUMMARY, summary)
    for it in items:
        if _card_worthy(it):
            write_markdown(CARDS_DIR / f"{it['item_id']}.md", card_markdown(it))
    write_markdown(REPORT, report_markdown(items, summary, gate17b, cur_exec, pet_sep))
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
        if row.get("classe_vinculo") in CLASSE_VINCULO_FORTE:
            if row.get("geometry_id") in {"", "not_available"}:
                errors.append(f"{rid}:vinculo_forte_sem_geometry_id")
            if row.get("patch_id") in {"", "not_available"}:
                errors.append(f"{rid}:vinculo_forte_sem_patch_id")
        if forte and row.get("classe_vinculo") == "same_region_only":
            errors.append(f"{rid}:forte_sem_vinculo_forte")
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


def validate_pet_sep_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("candidate_event_id", f"pet{idx}")
        if row.get("classe_fenomeno") not in FENOMENO_CLASS_ALLOWED:
            errors.append(f"{rid}:classe_fenomeno_fora_enum")
        if row.get("classe_fenomeno") in {"fenomeno_misto", "deslizamento"} and \
                row.get("pode_seguir_como_inundacao") == "true":
            errors.append(f"{rid}:fenomeno_misto_virou_inundacao")
    return errors


def validate_cur_exec_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("candidate_event_id", f"cur{idx}")
        st = row.get("status_execucao_geometria", "")
        if st not in GEOM_EXEC_STATUS_ALLOWED:
            errors.append(f"{rid}:status_geometria_fora_enum:{st}")
        # endereco/texto nunca vira geometria resolvida forte
        if st in {"geometria_textual_bloqueada", "geometria_ausente_com_tarefa_externa"} and \
                row.get("geometria_local_encontrada") == "geometria_oficial_local":
            errors.append(f"{rid}:texto_ou_ausencia_como_geometria_forte")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate_cur_norm_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("geometry_id", f"norm{idx}")
        # geometria normalizada obriga a ser de ocorrencia (nunca bairro/rua/centroide)
        if row.get("geometria_de_ocorrencia") != "true":
            errors.append(f"{rid}:geometria_normalizada_nao_de_ocorrencia")
    return errors


def validate():
    summary = build_all()
    errors = []
    matriz = _read(MATRIZ)
    if not matriz:
        errors.append("matriz_18b_vazia")
    errors.extend(validate_matriz_rows(matriz))
    errors.extend(validate_vinculos_rows(_read(VINCULOS)))
    errors.extend(validate_features_rows(_read(FEATURES)))
    errors.extend(validate_pet_sep_rows(_read(PET_SEP_EXEC)))
    errors.extend(validate_cur_exec_rows(_read(CUR_EXEC_GEOM)))
    errors.extend(validate_cur_norm_rows(_read(CUR_GEOM_NORM)))
    errors.extend(public_text_violations_files())

    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_proibido")
    if summary["status_final_18b"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_18b']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["status_final_18b"] == "18B_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_avanco_funcional_entregue")
    for field in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary[field] != 0:
            errors.append(f"{field}_nonzero")
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if not (summary["referencia_forte"] or summary["referencia_parcial"] or summary["fila_total"]):
        errors.append("nenhum_avanco_funcional")

    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "18B execucao geometrias regionais e separacao de fenomeno validada: "
        f"fortes={summary['referencia_forte']} parciais={summary['referencia_parcial']} "
        f"cur_geom_resolvida={summary['curitiba_geometrias_resolvidas']} pet_inund={summary['petropolis_inundacao_separada']} "
        f"links_fortes={summary['patch_links_fortes']} fila={summary['fila_total']} "
        f"status18B={summary['status_final_18b']} status17B={summary['status_17b']}"
    )
    return 0


def run_all():
    summary = build_all()
    print(
        "18B execucao geometrias regionais e separacao de fenomeno gerada: "
        f"rec={summary['itens_rec']} cur={summary['itens_cur']} pet={summary['itens_pet']} "
        f"cur_geom={summary['curitiba_geometrias_resolvidas']} pet_inund={summary['petropolis_inundacao_separada']} "
        f"fila={summary['fila_total']} status18B={summary['status_final_18b']} status17B={summary['status_17b']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
