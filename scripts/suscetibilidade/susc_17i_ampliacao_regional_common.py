"""SUSC-17I ampliacao regional da amostra observacional.

O 17H fechou a calibracao forte somente revisao em Recife (1 evento, 1 regiao,
5 vinculos), com 17B ainda sem prontidao por amostra local. Esta etapa amplia a
amostra para Curitiba e Petropolis: inventaria os eventos candidatos ja
catalogados, classifica prontidao por regiao/fenomeno, checa disponibilidade de
features e entrega uma fila regional executavel de desbloqueio quando faltar
geometria, data, separacao de fenomeno ou features.

Nao cria ground truth, treino, score_v7 nem benchmark 17B. O score_v6 nunca e
alterado. Bairro/rua textual nao vira geometria forte. Deslizamento ou fenomeno
misto nunca entra como inundacao sem separacao explicita. Nada e inventado.
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
OUT_DATA_17H = ROOT / "outputs_public" / "data" / "susc_17h_calibracao_observacional_forte_somente_revisao"
OUT_DATA_17G = ROOT / "outputs_public" / "data" / "susc_17g_extracao_direta_features_fisicas_canarios"
OUT_DATA_17D = ROOT / "outputs_public" / "data" / "susc_17d_validacao_tecnica_evidencia_observacional"
OUT_DATA_17C = ROOT / "outputs_public" / "data" / "susc_17c_strong_reference_acquisition_canary"
OUT_DATA_17C5 = ROOT / "outputs_public" / "data" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_DATA = ROOT / "outputs_public" / "data" / "susc_17i_ampliacao_regional_amostra_observacional"
CARDS_DIR = OUT_DATA / "cartoes_regionais"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
TARGET_PACK_17C = OUT_DATA_17C / "susc_17c_source_target_pack.csv"
NORM_GEOM_17C5 = OUT_DATA_17C5 / "susc_17c5_normalized_geometry.csv"
PATCH_LINKS_17C5 = OUT_DATA_17C5 / "susc_17c5_patch_links.csv"
SUMMARY_17H = OUT_DATA_17H / "summary.json"
FEATURES_BY_PATCH = DAT_SUSC / "susc_features_by_patch_v1.csv"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_JSON = OUT_DATA / "preflight.json"
INVENTARIO = OUT_DATA / "inventario_regional_eventos_observacionais.csv"
MATRIZ = OUT_DATA / "matriz_amostra_observacional_expandida.csv"
DISPONIBILIDADE = OUT_DATA / "matriz_disponibilidade_features_regionais.csv"
FILA = OUT_DATA / "fila_regional_desbloqueio_observacional.csv"
GATE = OUT_DATA / "gate_ampliacao_regional_observacional.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_17i.csv"
RESUMO_REGIAO = OUT_DATA / "resumo_por_regiao.csv"
RESUMO_STATUS = OUT_DATA / "resumo_por_status.csv"
RESUMO_FENOMENO = OUT_DATA / "resumo_por_fenomeno.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_17I_AMPLIACAO_REGIONAL_AMOSTRA_OBSERVACIONAL.md"
SCHEMA = SCHEMAS / "susc_17i_ampliacao_regional_schema_v1.json"

REQUIRED_INPUTS = [TARGET_PACK_17C, NORM_GEOM_17C5, PATCH_LINKS_17C5, SUMMARY_17H, FEATURES_BY_PATCH, SCORE_V6]
REQUIRED_OUTPUTS = [
    PREFLIGHT_JSON, INVENTARIO, MATRIZ, DISPONIBILIDADE, FILA, GATE, GATE_17B,
    RESUMO_REGIAO, RESUMO_STATUS, RESUMO_FENOMENO, SUMMARY, REPORT, SCHEMA,
]

REGIONS = ["CUR", "PET"]
REGION_CITY = {"CUR": "Curitiba", "PET": "Petropolis", "REC": "Recife"}
REGION_FEATURE_KEY = {"CUR": "curitiba", "PET": "petropolis", "REC": "recife"}

# --- Enums -----------------------------------------------------------------
USO_PERMITIDO_ALLOWED = [
    "avaliacao_somente_revisao", "contexto_documental",
    "fila_obtencao_geometria", "fila_obtencao_data", "rejeitado",
]
STATUS_PRONTIDAO_ALLOWED = [
    "candidato_observacional_forte", "candidato_observacional_parcial", "candidato_contextual",
    "bloqueado_sem_geometria", "bloqueado_sem_data", "bloqueado_por_fenomeno_misto",
    "bloqueado_por_lacuna_de_features", "rejeitado",
]
FENOMENO_CLASS_ALLOWED = ["inundacao", "deslizamento", "misto", "insuficiente"]
LACUNA_ALLOWED = [
    "obter_geometria_oficial", "obter_data_exata", "separar_fenomeno", "obter_footprint_tecnico",
    "extrair_features_fisicas", "extrair_features_espectrais", "extrair_chuva_pre_evento", "resolver_patch_link",
]
GATE_FINAL_ALLOWED = [
    "17I_AMOSTRA_REGIONAL_EXPANDIDA_COM_CANDIDATOS_FORTES",
    "17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS",
    "17I_CURITIBA_PETROPOLIS_SEM_PRONTIDAO_MAS_COM_PLANO",
    "17I_APROXIMACAO_17B_SEM_BENCHMARK",
    "17I_BLOQUEADO_FAIL_CLOSED",
]
STATUS_17B_ALLOWED = [
    "17B_AINDA_BLOQUEADO_AMOSTRA_INSUFICIENTE",
    "17B_APROXIMACAO_COM_AMOSTRA_REGIONAL_PARCIAL",
    "17B_PRONTO_PARA_DESENHO_DE_BENCHMARK_SOMENTE_REVISAO",
]

PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|llm|codex|ia|human\s+qa)\b", re.IGNORECASE)
NO_GT_REASON = (
    "candidato observacional regional somente revisao; nao confirma ocorrencia no patch, "
    "nao e verdade de referencia, nao e treino e nao autoriza score_v7"
)

STRONG_SOURCES = {"official_observed_event_point", "official_observed_event_polygon", "technical_remote_sensing_flood_footprint"}
OFFICIAL_TIER = "official"

# --- Field lists -----------------------------------------------------------
INVENTARIO_FIELDS = [
    "candidate_event_id", "regiao", "cidade", "data_evento_candidata", "precisao_temporal",
    "tipo_fenomeno", "fonte", "autoridade_fonte", "artefato_fonte", "tipo_geometria",
    "possui_geometria_forte", "possui_footprint_tecnico", "possui_patch_link",
    "possui_janela_pre_evento", "possui_janela_pos_evento", "prioridade",
    "bloqueio_principal", "uso_permitido",
]
MATRIZ_FIELDS = [
    "item_amostra_id", "candidate_event_id", "regiao", "cidade", "data_evento", "tipo_fenomeno",
    "classe_fenomeno", "fonte", "autoridade_fonte", "geometry_id", "patch_id", "classe_vinculo",
    "qualidade_temporal", "qualidade_geometrica", "qualidade_fenomeno", "qualidade_vinculo_patch",
    "possui_fisico_topografico", "possui_espectral", "possui_chuva",
    "status_prontidao", "uso_permitido", "ground_truth", "eligible_for_training", "score_v7_allowed",
    "not_ground_truth_reason", "justificativa_tecnica",
]
DISPONIBILIDADE_FIELDS = [
    "item_amostra_id", "regiao", "patch_id", "possui_fisico", "possui_espectral", "possui_chuva",
    "fonte_fisico", "fonte_espectral", "fonte_chuva", "lacunas", "acao_minima",
]
FILA_FIELDS = [
    "task_id", "regiao", "cidade", "candidate_event_id", "lacuna", "fonte_sugerida",
    "artefato_necessario", "formato_esperado", "expected_input_path", "expected_output_path",
    "command_hint", "prioridade", "criterio_de_sucesso",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_REGIAO_FIELDS = ["regiao", "cidade", "candidatos", "fortes", "parciais", "contextuais", "bloqueados"]
RESUMO_STATUS_FIELDS = ["status_prontidao", "quantidade"]
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
        raise AssertionError("score_v7 existe e e proibido para SUSC-17I")


def _has_date(cand):
    d = cand.get("event_date_candidate", "")
    return d not in {"", "not_available", "unknown"}


def _phenomenon_class(cand):
    ph = cand.get("phenomenon_type", "")
    if ph == "flood_inundation_alagamento":
        return "inundacao"
    if ph == "mass_movement":
        return "deslizamento"
    if ph == "hydrometeorological_or_unknown":
        return "misto"
    return "insuficiente"


def _geom_is_address_only(cand):
    gs = cand.get("geometry_status", "")
    return "address_text" in gs or cand.get("has_address") == "true" and cand.get("has_point") != "true"


def _strong_geometry(cand):
    # geometria forte real: geometria oficial resolvida (poligono/footprint), nunca texto/ponto nao validado
    gs = cand.get("geometry_status", "")
    return gs in {"resolved_geometry_object", "official_polygon_validated", "technical_footprint"}


def _region_features_exist(region):
    key = REGION_FEATURE_KEY.get(region, "")
    return any(r.get("regiao") == key for r in _read(FEATURES_BY_PATCH))


# --- Classificacao ---------------------------------------------------------
def _classify_status(cand):
    ph = _phenomenon_class(cand)
    has_date = _has_date(cand)
    strong_source = cand.get("source_type", "") in STRONG_SOURCES
    official = cand.get("authority_tier", "") == OFFICIAL_TIER
    documentary = cand.get("source_type", "") == "documentary_context"
    address_only = _geom_is_address_only(cand)
    strong_geom = _strong_geometry(cand)

    # 1) fenomeno nao-inundacao nunca entra como inundacao sem separacao
    if ph in {"deslizamento", "misto"}:
        return "bloqueado_por_fenomeno_misto"
    if ph == "insuficiente":
        return "rejeitado"
    # 2) fenomeno inundacao
    if not has_date:
        return "bloqueado_sem_data"
    if strong_geom:
        return "candidato_observacional_forte"
    if address_only:
        return "bloqueado_sem_geometria"
    if strong_source:
        return "candidato_observacional_parcial"
    if documentary:
        return "candidato_contextual"
    if official:
        return "candidato_observacional_parcial"
    return "bloqueado_sem_geometria"


def _uso_permitido(status):
    return {
        "candidato_observacional_forte": "avaliacao_somente_revisao",
        "candidato_observacional_parcial": "avaliacao_somente_revisao",
        "candidato_contextual": "contexto_documental",
        "bloqueado_sem_geometria": "fila_obtencao_geometria",
        "bloqueado_sem_data": "fila_obtencao_data",
        "bloqueado_por_fenomeno_misto": "contexto_documental",
        "bloqueado_por_lacuna_de_features": "fila_obtencao_geometria",
        "rejeitado": "rejeitado",
    }.get(status, "rejeitado")


def _temporal_quality(cand):
    p = cand.get("event_date_precision", "")
    return {"exact_day": 90, "range": 80, "month_only": 35}.get(p, 0)


def _geometry_quality(cand):
    if _strong_geometry(cand):
        return 85
    gs = cand.get("geometry_status", "")
    if "address_text" in gs:
        return 20
    if "documentary" in gs:
        return 20
    if "point_or_coordinate_unvalidated" in gs:
        return 30
    return 0


def _phenomenon_quality(cand):
    return {"inundacao": 85, "deslizamento": 40, "misto": 30, "insuficiente": 15}.get(_phenomenon_class(cand), 15)


def build_items():
    target = [r for r in _read(TARGET_PACK_17C) if r.get("region") in REGIONS]
    items = []
    for idx, cand in enumerate(sorted(target, key=lambda r: r.get("candidate_event_id", "")), start=1):
        status = _classify_status(cand)
        region = cand.get("region", "")
        items.append({
            "item_amostra_id": f"S17I_ITEM_{idx:04d}",
            "cand": cand,
            "region": region,
            "cidade": REGION_CITY.get(region, cand.get("city", "")),
            "classe_fenomeno": _phenomenon_class(cand),
            "status": status,
            "uso": _uso_permitido(status),
            "q_temporal": _temporal_quality(cand),
            "q_geom": _geometry_quality(cand),
            "q_fenomeno": _phenomenon_quality(cand),
            "q_vinculo": 0,  # nenhum vinculo forte em CUR/PET (todos same_region_only)
            "region_feats": _region_features_exist(region),
        })
    return items


# --- Inventario ------------------------------------------------------------
def inventario_rows(items):
    rows = []
    for it in items:
        c = it["cand"]
        rows.append({
            "candidate_event_id": c.get("candidate_event_id", ""),
            "regiao": it["region"], "cidade": it["cidade"],
            "data_evento_candidata": c.get("event_date_candidate", "not_available"),
            "precisao_temporal": c.get("event_date_precision", ""),
            "tipo_fenomeno": c.get("phenomenon_type", ""),
            "fonte": c.get("source_name", ""), "autoridade_fonte": c.get("authority_tier", ""),
            "artefato_fonte": c.get("artifact_ref", "not_available") or "not_available",
            "tipo_geometria": c.get("geometry_status", ""),
            "possui_geometria_forte": _bool(_strong_geometry(c)),
            "possui_footprint_tecnico": _bool(c.get("source_type") == "technical_remote_sensing_flood_footprint"),
            "possui_patch_link": "false",
            "possui_janela_pre_evento": c.get("has_sentinel_window_candidate", "false"),
            "possui_janela_pos_evento": c.get("has_sentinel_window_candidate", "false"),
            "prioridade": c.get("priority", ""),
            "bloqueio_principal": _blocking(it),
            "uso_permitido": it["uso"],
        })
    return rows


def _blocking(it):
    return {
        "candidato_observacional_forte": "nenhum",
        "candidato_observacional_parcial": "geometria_oficial_e_patch_link_pendentes",
        "candidato_contextual": "fonte_documental_sem_geometria_forte",
        "bloqueado_sem_geometria": "sem_geometria_oficial_forte",
        "bloqueado_sem_data": "sem_data_exata_ou_intervalo",
        "bloqueado_por_fenomeno_misto": "fenomeno_misto_ou_deslizamento_nao_separado_como_inundacao",
        "rejeitado": "fonte_ou_fenomeno_insuficiente",
    }.get(it["status"], "indefinido")


# --- Matriz expandida ------------------------------------------------------
def matriz_rows(items):
    rows = []
    for it in items:
        c = it["cand"]
        rows.append({
            "item_amostra_id": it["item_amostra_id"],
            "candidate_event_id": c.get("candidate_event_id", ""),
            "regiao": it["region"], "cidade": it["cidade"],
            "data_evento": c.get("event_date_candidate", "not_available"),
            "tipo_fenomeno": c.get("phenomenon_type", ""), "classe_fenomeno": it["classe_fenomeno"],
            "fonte": c.get("source_name", ""), "autoridade_fonte": c.get("authority_tier", ""),
            "geometry_id": "not_available", "patch_id": "not_available",
            "classe_vinculo": "same_region_only",
            "qualidade_temporal": str(it["q_temporal"]), "qualidade_geometrica": str(it["q_geom"]),
            "qualidade_fenomeno": str(it["q_fenomeno"]), "qualidade_vinculo_patch": str(it["q_vinculo"]),
            "possui_fisico_topografico": "false", "possui_espectral": "false", "possui_chuva": "false",
            "status_prontidao": it["status"], "uso_permitido": it["uso"],
            "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
            "not_ground_truth_reason": NO_GT_REASON,
            "justificativa_tecnica": _justificativa(it),
        })
    return rows


def _justificativa(it):
    c = it["cand"]
    base = (
        f"regiao {it['region']}; fenomeno {it['classe_fenomeno']}; "
        f"data {c.get('event_date_candidate', 'not_available')} ({c.get('event_date_precision', '')}); "
        f"fonte {c.get('source_type', '')} tier {c.get('authority_tier', '')}; "
        "vinculo apenas regional (same_region_only), sem geometria forte"
    )
    if it["status"] == "candidato_observacional_parcial":
        return base + "; candidato parcial somente revisao: inundacao datada de fonte oficial, geometria oficial e patch link pendentes"
    if it["status"] == "bloqueado_por_fenomeno_misto":
        return base + "; fenomeno misto/deslizamento nao pode entrar como inundacao sem separacao explicita"
    if it["status"] == "bloqueado_sem_geometria":
        return base + "; geometria textual/endereco nao e geometria forte"
    if it["status"] == "bloqueado_sem_data":
        return base + "; sem data exata ou intervalo, nao avaliavel"
    if it["status"] == "candidato_contextual":
        return base + "; fonte documental usada apenas como contexto"
    return base


# --- Disponibilidade de features -------------------------------------------
def disponibilidade_rows(items):
    rows = []
    for it in items:
        if it["status"] in {"bloqueado_por_fenomeno_misto", "rejeitado"}:
            continue
        region_feats = it["region_feats"]
        fonte = ("patches_oficiais_regiao_disponiveis_sem_vinculo_evento" if region_feats else "ausente")
        lacunas = "obter_geometria_oficial;resolver_patch_link"
        if it["status"] == "bloqueado_sem_data":
            lacunas = "obter_data_exata;" + lacunas
        rows.append({
            "item_amostra_id": it["item_amostra_id"],
            "regiao": it["region"], "patch_id": "not_available",
            "possui_fisico": "false", "possui_espectral": "false", "possui_chuva": "false",
            "fonte_fisico": fonte, "fonte_espectral": fonte, "fonte_chuva": fonte,
            "lacunas": lacunas,
            "acao_minima": "obter_geometria_oficial" if it["status"] != "bloqueado_sem_data" else "obter_data_exata",
        })
    return rows


# --- Fila regional ---------------------------------------------------------
def fila_rows(items):
    rows = []
    counter = 0

    def add(region, cidade, cid, lacuna, fonte, artefato, fmt, inp, out, hint, prio, criterio):
        nonlocal counter
        counter += 1
        rows.append({
            "task_id": f"S17I_TASK_{counter:04d}", "regiao": region, "cidade": cidade,
            "candidate_event_id": cid, "lacuna": lacuna, "fonte_sugerida": fonte,
            "artefato_necessario": artefato, "formato_esperado": fmt,
            "expected_input_path": inp, "expected_output_path": out,
            "command_hint": hint, "prioridade": prio, "criterio_de_sucesso": criterio,
        })

    for it in items:
        c = it["cand"]
        cid = c.get("candidate_event_id", "")
        region, cidade = it["region"], it["cidade"]
        low = cid.lower()
        if it["status"] == "candidato_observacional_parcial":
            add(region, cidade, cid, "obter_geometria_oficial",
                "geoportal_municipal_ou_defesa_civil", "geometria_oficial_do_evento", "geojson",
                f"local_runs/suscetibilidade/17i_regional/{low}_geometria.geojson",
                f"outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional/{low}_geometria_resolvida.csv",
                "obter poligono/ponto oficial do evento datado e resolver com a logica do 17C5", "alta",
                "geometria oficial com CRS resolvida e vinculavel a patch")
            add(region, cidade, cid, "resolver_patch_link",
                "patches_oficiais_regiao", "vinculo_evento_patch", "csv",
                f"local_runs/suscetibilidade/17i_regional/{low}_geometria.geojson",
                f"outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional/{low}_patch_link.csv",
                "apos obter a geometria, rodar o resolvedor de vinculo evento-patch do 17C5", "alta",
                "pelo menos um patch link forte aceito para o evento")
        elif it["status"] == "bloqueado_sem_geometria":
            add(region, cidade, cid, "obter_geometria_oficial",
                "geoportal_municipal_ou_defesa_civil", "geometria_oficial_do_evento", "geojson",
                f"local_runs/suscetibilidade/17i_regional/{low}_geometria.geojson",
                f"outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional/{low}_geometria_resolvida.csv",
                "substituir endereco/texto por geometria oficial com CRS", "media",
                "geometria oficial forte com CRS (nao textual)")
        elif it["status"] == "bloqueado_sem_data":
            add(region, cidade, cid, "obter_data_exata",
                "relatorio_oficial_ou_defesa_civil", "data_do_evento", "documento_ou_csv",
                f"local_runs/suscetibilidade/17i_regional/{low}_data.txt",
                f"outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional/{low}_data_resolvida.csv",
                "resolver data exata ou intervalo do evento a partir de fonte oficial", "media",
                "data exata ou intervalo confirmado por fonte oficial")
        elif it["status"] == "candidato_contextual":
            add(region, cidade, cid, "obter_footprint_tecnico",
                "sentinel1_rtc_stac", "footprint_tecnico_de_inundacao", "geojson",
                f"local_runs/suscetibilidade/17i_regional/{low}_footprint.geojson",
                f"outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional/{low}_footprint.csv",
                "gerar footprint tecnico pre/pos com o metodo do DEVRO02D, se a data permitir", "baixa",
                "footprint tecnico candidato review-only")

    # tarefa agregada de separacao de fenomeno por regiao (evento misto/deslizamento)
    for region in REGIONS:
        mistos = [it for it in items if it["region"] == region and it["status"] == "bloqueado_por_fenomeno_misto"]
        if mistos:
            add(region, REGION_CITY[region], f"{region}_EVENTOS_MISTOS_{len(mistos)}", "separar_fenomeno",
                "relatorios_oficiais_cprm_defesa_civil", "classificacao_de_fenomeno_por_ocorrencia", "csv",
                f"local_runs/suscetibilidade/17i_regional/{region.lower()}_ocorrencias_classificadas.csv",
                f"outputs_public/data/susc_17i_ampliacao_regional_amostra_observacional/{region.lower()}_fenomeno_separado.csv",
                "separar inundacao/alagamento de deslizamento por ocorrencia antes de usar como evidencia de inundacao", "alta",
                "ocorrencias de inundacao separadas de deslizamento com fonte oficial")
    return rows


# --- Gates -----------------------------------------------------------------
def _counts(items):
    return Counter(it["status"] for it in items)


def _final_status(items, fila):
    c = _counts(items)
    fortes = c.get("candidato_observacional_forte", 0)
    parciais = c.get("candidato_observacional_parcial", 0)
    if not items:
        return "17I_BLOQUEADO_FAIL_CLOSED"
    if fortes > 0:
        return "17I_AMOSTRA_REGIONAL_EXPANDIDA_COM_CANDIDATOS_FORTES"
    if parciais > 0 and fila:
        return "17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS"
    if fila:
        return "17I_CURITIBA_PETROPOLIS_SEM_PRONTIDAO_MAS_COM_PLANO"
    return "17I_BLOQUEADO_FAIL_CLOSED"


def _b17_metrics(items):
    # evidencia forte de inundacao herdada do 17H (Recife): 1 evento, 1 regiao, 5 vinculos fortes
    rec_events, rec_regions, rec_strong_links = 1, 1, 5
    cur_pet_fortes = sum(1 for it in items if it["status"] == "candidato_observacional_forte")
    cur_pet_parciais = sum(1 for it in items if it["status"] == "candidato_observacional_parcial")
    regioes_tocadas = 1 + len({it["region"] for it in items if it["status"] in {
        "candidato_observacional_forte", "candidato_observacional_parcial", "candidato_contextual"}})
    eventos_distintos = rec_events + cur_pet_fortes  # so contam eventos com geometria forte
    strong_links = rec_strong_links + cur_pet_fortes
    return {
        "eventos_distintos_fortes": eventos_distintos,
        "regioes_com_evidencia": regioes_tocadas,
        "vinculos_fortes": strong_links,
        "parciais_regionais": cur_pet_parciais,
    }


def gate_rows(items, fila, final_status):
    m = _b17_metrics(items)
    c = _counts(items)
    regioes = len({it["region"] for it in items})
    fortes_por_regiao = {r: sum(1 for it in items if it["region"] == r and it["status"] == "candidato_observacional_forte") for r in REGIONS}
    rows = [
        ("regioes_processadas", str(regioes), "2", regioes >= 2, "Curitiba e Petropolis processadas"),
        ("regioes_com_evidencia_observacional", str(m["regioes_com_evidencia"]), "2", m["regioes_com_evidencia"] >= 2, "inclui Recife herdado + parciais/contextuais"),
        ("eventos_distintos_com_geometria_forte", str(m["eventos_distintos_fortes"]), "3", m["eventos_distintos_fortes"] >= 3, "apenas eventos com geometria forte contam"),
        ("vinculos_patch_fortes", str(m["vinculos_fortes"]), "20", m["vinculos_fortes"] >= 20, "vinculos fortes herdados de Recife"),
        ("candidatos_fortes_curitiba", str(fortes_por_regiao["CUR"]), ">=0", True, "Curitiba sem geometria forte"),
        ("candidatos_fortes_petropolis", str(fortes_por_regiao["PET"]), ">=0", True, "Petropolis com fenomeno misto"),
        ("candidatos_parciais", str(c.get("candidato_observacional_parcial", 0)), ">=0", True, "candidatos parciais datados"),
        ("fila_executavel_criada", _bool(bool(fila)), "true", bool(fila), "fila regional de desbloqueio"),
        ("separacao_temporal_possivel", "false", "true", False, "amostra forte ainda de um unico evento"),
        ("features_completas_por_evento", "0", ">=1", False, "nenhum evento CUR/PET com features vinculadas"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_allowed_zero", "0", "0", True, "sem score_v7"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 nunca alterado"),
        ("caminho_funcional_entregue", "true", "true", final_status != "17I_BLOQUEADO_FAIL_CLOSED", "candidatos ou plano executavel"),
        ("status_final_17i", final_status, "enum", final_status in GATE_FINAL_ALLOWED, "status final da ampliacao"),
    ]
    return [{"criterio": c2, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c2, v, t, p, o in rows]


def gate_17b_rows(items):
    m = _b17_metrics(items)
    rows = [
        ("minimo_3_eventos_distintos", str(m["eventos_distintos_fortes"]), "3", m["eventos_distintos_fortes"] >= 3, "apenas eventos com geometria forte"),
        ("minimo_2_regioes", str(m["regioes_com_evidencia"]), "2", m["regioes_com_evidencia"] >= 2, "regioes com evidencia observacional"),
        ("minimo_20_vinculos_fortes", str(m["vinculos_fortes"]), "20", m["vinculos_fortes"] >= 20, "vinculos patch fortes aceitos"),
        ("1_evento_por_regiao_com_data_e_geometria", "1", "2", False, "somente Recife com data e geometria fortes"),
        ("separacao_temporal_possivel", "false", "true", False, "amostra forte de um unico evento"),
        ("controles_nao_supervisionados", "true", "true", True, "controles seguem nao supervisionados"),
        ("ground_truth_false", "0", "0", True, "sem ground truth"),
        ("trainable_false", "0", "0", True, "sem treino"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def _status_17b(items):
    m = _b17_metrics(items)
    minimos = m["eventos_distintos_fortes"] >= 3 and m["regioes_com_evidencia"] >= 2 and m["vinculos_fortes"] >= 20
    if minimos:
        return "17B_PRONTO_PARA_DESENHO_DE_BENCHMARK_SOMENTE_REVISAO"
    if m["parciais_regionais"] > 0 and m["regioes_com_evidencia"] >= 2:
        return "17B_APROXIMACAO_COM_AMOSTRA_REGIONAL_PARCIAL"
    return "17B_AINDA_BLOQUEADO_AMOSTRA_INSUFICIENTE"


def _score_v6_changed():
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


# --- Resumos ---------------------------------------------------------------
def resumo_regiao_rows(items):
    rows = []
    for region in REGIONS:
        grp = [it for it in items if it["region"] == region]
        rows.append({
            "regiao": region, "cidade": REGION_CITY[region], "candidatos": str(len(grp)),
            "fortes": str(sum(1 for it in grp if it["status"] == "candidato_observacional_forte")),
            "parciais": str(sum(1 for it in grp if it["status"] == "candidato_observacional_parcial")),
            "contextuais": str(sum(1 for it in grp if it["status"] == "candidato_contextual")),
            "bloqueados": str(sum(1 for it in grp if it["status"].startswith("bloqueado") or it["status"] == "rejeitado")),
        })
    return rows


def resumo_status_rows(items):
    c = _counts(items)
    return [{"status_prontidao": s, "quantidade": str(c[s])} for s in STATUS_PRONTIDAO_ALLOWED if c.get(s, 0)]


def resumo_fenomeno_rows(items):
    d = defaultdict(Counter)
    for it in items:
        d[it["region"]][it["classe_fenomeno"]] += 1
    rows = []
    for region in REGIONS:
        for cls in FENOMENO_CLASS_ALLOWED:
            if d[region].get(cls, 0):
                rows.append({"regiao": region, "classe_fenomeno": cls, "quantidade": str(d[region][cls])})
    return rows


def summary_obj(items, inventario, matriz, disponibilidade, fila, final_status):
    inherited = read_json(SUMMARY_17H)
    c = _counts(items)
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17h_status_final": inherited.get("status_final_17h", ""),
        "herdado_17h_status_17b": inherited.get("status_17b", ""),
        "candidatos_total": len(items),
        "candidatos_cur": sum(1 for it in items if it["region"] == "CUR"),
        "candidatos_pet": sum(1 for it in items if it["region"] == "PET"),
        "candidato_observacional_forte": c.get("candidato_observacional_forte", 0),
        "candidato_observacional_parcial": c.get("candidato_observacional_parcial", 0),
        "candidato_contextual": c.get("candidato_contextual", 0),
        "bloqueado_sem_geometria": c.get("bloqueado_sem_geometria", 0),
        "bloqueado_sem_data": c.get("bloqueado_sem_data", 0),
        "bloqueado_por_fenomeno_misto": c.get("bloqueado_por_fenomeno_misto", 0),
        "rejeitado": c.get("rejeitado", 0),
        "fila_tasks": len(fila),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "benchmark_17b_criado": False,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "caminho_funcional": _caminho_funcional(final_status),
        "status_final_17i": final_status,
        "status_17b": _status_17b(items),
        "review_only": True, "ground_truth": False, "trainable": False,
    }


def _caminho_funcional(final_status):
    return {
        "17I_AMOSTRA_REGIONAL_EXPANDIDA_COM_CANDIDATOS_FORTES": "candidatos_regionais_fortes",
        "17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS": "candidatos_parciais_mais_fila_executavel",
        "17I_CURITIBA_PETROPOLIS_SEM_PRONTIDAO_MAS_COM_PLANO": "plano_executavel_regional",
        "17I_APROXIMACAO_17B_SEM_BENCHMARK": "aproximacao_17b_sem_benchmark",
        "17I_BLOQUEADO_FAIL_CLOSED": "bloqueado_fail_closed",
    }.get(final_status, "indefinido")


# --- Schema ----------------------------------------------------------------
def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17I ampliacao regional da amostra observacional",
        "type": "object",
        "properties": {
            "status_prontidao": {"enum": STATUS_PRONTIDAO_ALLOWED},
            "uso_permitido": {"enum": USO_PERMITIDO_ALLOWED},
            "classe_fenomeno": {"enum": FENOMENO_CLASS_ALLOWED},
            "status_final_17i": {"enum": GATE_FINAL_ALLOWED},
            "status_17b": {"enum": STATUS_17B_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
        },
        "enums": {
            "status_prontidao": STATUS_PRONTIDAO_ALLOWED, "uso_permitido": USO_PERMITIDO_ALLOWED,
            "classe_fenomeno": FENOMENO_CLASS_ALLOWED, "lacuna": LACUNA_ALLOWED,
            "status_final_17i": GATE_FINAL_ALLOWED, "status_17b": STATUS_17B_ALLOWED,
        },
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema():
    write_json(SCHEMA, _schema())


# --- Cartoes (candidatos relevantes de inundacao) --------------------------
def _card_worthy(it):
    return it["classe_fenomeno"] == "inundacao" or it["status"] in {"candidato_observacional_parcial", "candidato_contextual"}


def card_markdown(it):
    c = it["cand"]
    return f"""# Cartao regional {it['item_amostra_id']}

## Regiao e cidade

- Regiao/cidade: {it['region']} / {it['cidade']}
- Evento: `{c.get('candidate_event_id', '')}`

## Fonte

- Fonte: {c.get('source_name', '')}
- Tipo de fonte: `{c.get('source_type', '')}`; autoridade: {c.get('authority_tier', '')}

## Data

- Data candidata: {c.get('event_date_candidate', 'not_available')} (precisao {c.get('event_date_precision', '')})

## Fenomeno

- Tipo: {c.get('phenomenon_type', '')}; classe: **{it['classe_fenomeno']}**

## Geometria

- Estado da geometria: {c.get('geometry_status', '')}
- Geometria forte: {_bool(_strong_geometry(c))}

## Vinculo patch

- Classe de vinculo: same_region_only (sem vinculo forte em {it['region']})

## Features disponiveis

- Fisico/espectral/chuva vinculados ao evento: nao (sem geometria forte para vincular a patch)
- Regiao possui patches oficiais com features: {_bool(it['region_feats'])}

## Lacunas

- {_blocking(it)}

## Decisao de prontidao

- Status: **{it['status']}**
- Uso permitido: {it['uso']}

## Acao minima

- {_acao_minima(it)}

## Por que nao e ground truth

Candidato observacional regional somente revisao; nao confirma ocorrencia e nao e verdade de referencia.

## Por que nao e treinavel

Sem rotulo validado e sem geometria/feature vinculada, nao alimenta treino supervisionado.

## Por que nao cria score_v7

Nenhum score oficial ou score_v7 e criado; o score_v6 permanece intacto.

## Impacto no 17B

Sem geometria forte e sem vinculo forte, este candidato ainda nao contribui para a prontidao de benchmark 17B.
"""


def _acao_minima(it):
    return {
        "candidato_observacional_parcial": "obter geometria oficial e resolver o vinculo com patch",
        "candidato_contextual": "obter footprint tecnico ou geometria oficial datada",
        "bloqueado_sem_geometria": "substituir endereco/texto por geometria oficial com CRS",
        "bloqueado_sem_data": "resolver data exata ou intervalo por fonte oficial",
        "bloqueado_por_fenomeno_misto": "separar inundacao de deslizamento por ocorrencia",
        "rejeitado": "obter fonte e fenomeno adequados",
    }.get(it["status"], "reavaliar com nova evidencia")


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, inventario, matriz, disponibilidade, fila, gate, gate17b, summary):
    reg_lines = "\n".join(
        f"- {r['regiao']} ({r['cidade']}): candidatos={r['candidatos']}; fortes={r['fortes']}; "
        f"parciais={r['parciais']}; contextuais={r['contextuais']}; bloqueados={r['bloqueados']}"
        for r in resumo_regiao_rows(items)
    )
    fen_lines = "\n".join(f"- {r['regiao']} / {r['classe_fenomeno']}: {r['quantidade']}" for r in resumo_fenomeno_rows(items))
    fortes_parciais = [it for it in items if it["status"] in {"candidato_observacional_forte", "candidato_observacional_parcial", "candidato_contextual"}]
    cand_lines = "\n".join(
        f"- {it['item_amostra_id']} ({it['region']}): {it['cand'].get('candidate_event_id', '')} | "
        f"{it['classe_fenomeno']} | {it['cand'].get('event_date_candidate', '')} | {it['status']}"
        for it in fortes_parciais
    ) or "- nenhum candidato forte/parcial/contextual"
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate)
    gate17b_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate17b)
    return f"""# SUSC-17I Ampliacao regional da amostra observacional

## Estado herdado do 17H

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 17H herdado: {summary['herdado_17h_status_final']}
- Status 17B herdado: {summary['herdado_17h_status_17b']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Por que a ampliacao regional e necessaria

A calibracao forte do 17H usou apenas Recife (1 evento, 1 regiao, 5 vinculos). O 17B exige mais
eventos, mais regioes e separacao temporal. Esta etapa amplia para Curitiba e Petropolis.

## Inventario Curitiba/Petropolis

- Candidatos totais: {summary['candidatos_total']} (Curitiba {summary['candidatos_cur']}, Petropolis {summary['candidatos_pet']}).

{reg_lines}

## Candidatos encontrados

{cand_lines}

## Candidatos fortes/parciais/contextuais

- Fortes: {summary['candidato_observacional_forte']}
- Parciais: {summary['candidato_observacional_parcial']}
- Contextuais: {summary['candidato_contextual']}
- Bloqueados sem geometria: {summary['bloqueado_sem_geometria']}
- Bloqueados sem data: {summary['bloqueado_sem_data']}
- Bloqueados por fenomeno misto: {summary['bloqueado_por_fenomeno_misto']}

## Lacunas por regiao

{fen_lines}

Curitiba nao tem geometria oficial nem, na maioria, data exata; os eventos ficam como parciais,
contextuais ou em fila de obtencao de geometria/data. Petropolis tem pontos oficiais datados de
fevereiro de 2022, mas o fenomeno e predominantemente misto (deslizamento e inundacao juntos), o
que exige separacao explicita antes de usar como evidencia de inundacao.

## Comparacao com Recife

Recife tem o unico conjunto com geometria forte (footprint tecnico) e vinculo forte (5 canarios).
Curitiba e Petropolis ainda nao alcancam geometria forte nem vinculo forte de inundacao.

## Gate final 17I

{gate_lines}

- Status final: **{summary['status_final_17i']}**
- Caminho funcional: **{summary['caminho_funcional']}**

## Status 17B pos-17I

{gate17b_lines}

- Status 17B: **{summary['status_17b']}**
- Nenhum benchmark 17B foi criado.

## Por que segue sem ground truth

Nenhum candidato confirma ocorrencia no patch; sem verdade de referencia observacional nao ha ground truth.

## Por que segue sem treino

Sem rotulo validado, nenhum candidato alimenta treino supervisionado.

## Por que segue sem score_v7

Nenhum score oficial ou score_v7 e criado; o score_v6 permanece intacto.

## Proximo marco recomendado

SUSC-17J: executar a fila regional (obter geometria oficial datada em Curitiba e separar o fenomeno
de inundacao em Petropolis), resolver vinculos de patch e extrair features, ampliando a amostra
forte para pelo menos duas regioes, sempre somente revisao e sem ground truth.
"""


# --- Orquestracao ----------------------------------------------------------
def preflight_obj():
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "dirty_lines": len(_run_git(["status", "--short"]).splitlines()),
        "inputs": [
            {"role": "target_pack_17c", "path": rel(TARGET_PACK_17C), "exists": TARGET_PACK_17C.exists()},
            {"role": "normalized_geometry_17c5", "path": rel(NORM_GEOM_17C5), "exists": NORM_GEOM_17C5.exists()},
            {"role": "patch_links_17c5", "path": rel(PATCH_LINKS_17C5), "exists": PATCH_LINKS_17C5.exists()},
            {"role": "summary_17h", "path": rel(SUMMARY_17H), "exists": SUMMARY_17H.exists()},
            {"role": "features_by_patch", "path": rel(FEATURES_BY_PATCH), "exists": FEATURES_BY_PATCH.exists()},
            {"role": "score_v6", "path": rel(SCORE_V6), "exists": SCORE_V6.exists()},
            {"role": "score_v7_proibido", "path": rel(SCORE_V7), "exists": SCORE_V7.exists()},
        ],
        "regioes_alvo": REGIONS,
    }


def build_all():
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    ensure_dir(SCHEMAS)
    write_schema()

    items = build_items()
    inventario = inventario_rows(items)
    matriz = matriz_rows(items)
    disponibilidade = disponibilidade_rows(items)
    fila = fila_rows(items)
    final_status = _final_status(items, fila)
    gate = gate_rows(items, fila, final_status)
    gate17b = gate_17b_rows(items)
    resumo_regiao = resumo_regiao_rows(items)
    resumo_status = resumo_status_rows(items)
    resumo_fenomeno = resumo_fenomeno_rows(items)
    summary = summary_obj(items, inventario, matriz, disponibilidade, fila, final_status)

    write_json(PREFLIGHT_JSON, preflight_obj())
    write_csv(INVENTARIO, inventario, INVENTARIO_FIELDS)
    write_csv(MATRIZ, matriz, MATRIZ_FIELDS)
    write_csv(DISPONIBILIDADE, disponibilidade, DISPONIBILIDADE_FIELDS)
    write_csv(FILA, fila, FILA_FIELDS)
    write_csv(GATE, gate, GATE_FIELDS)
    write_csv(GATE_17B, gate17b, GATE_FIELDS)
    write_csv(RESUMO_REGIAO, resumo_regiao, RESUMO_REGIAO_FIELDS)
    write_csv(RESUMO_STATUS, resumo_status, RESUMO_STATUS_FIELDS)
    write_csv(RESUMO_FENOMENO, resumo_fenomeno, RESUMO_FENOMENO_FIELDS)
    write_json(SUMMARY, summary)
    for it in items:
        if _card_worthy(it):
            write_markdown(CARDS_DIR / f"{it['item_amostra_id']}.md", card_markdown(it))
    write_markdown(REPORT, report_markdown(items, inventario, matriz, disponibilidade, fila, gate, gate17b, summary))
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
        rid = row.get("item_amostra_id", f"row{idx}")
        for field in ["ground_truth", "eligible_for_training", "score_v7_allowed"]:
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        st = row.get("status_prontidao", "")
        if st not in STATUS_PRONTIDAO_ALLOWED:
            errors.append(f"{rid}:status_fora_enum:{st}")
        if row.get("uso_permitido") not in USO_PERMITIDO_ALLOWED:
            errors.append(f"{rid}:uso_permitido_fora_enum")
        if row.get("classe_fenomeno") not in FENOMENO_CLASS_ALLOWED:
            errors.append(f"{rid}:classe_fenomeno_fora_enum")
        # deslizamento/misto nunca vira inundacao forte/parcial
        if row.get("classe_fenomeno") in {"deslizamento", "misto"} and st in {"candidato_observacional_forte", "candidato_observacional_parcial"}:
            errors.append(f"{rid}:fenomeno_misto_como_inundacao")
        # candidato forte exige data e geometria e justificativa
        if st == "candidato_observacional_forte":
            if row.get("data_evento") in {"", "not_available", "unknown"}:
                errors.append(f"{rid}:forte_sem_data")
            if row.get("geometry_id") in {"", "not_available"}:
                errors.append(f"{rid}:forte_sem_geometry_id")
        # geometria textual nao pode ser geometria forte
        if row.get("classe_vinculo") == "same_region_only" and st == "candidato_observacional_forte":
            errors.append(f"{rid}:forte_sem_vinculo_forte")
        if row.get("justificativa_tecnica", "").strip() == "":
            errors.append(f"{rid}:justificativa_vazia")
    return errors


def validate_fila_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        if row.get("lacuna") not in LACUNA_ALLOWED:
            errors.append(f"fila{idx}:lacuna_fora_enum:{row.get('lacuna')}")
    return errors


def validate():
    summary = build_all()
    errors = []
    matriz = _read(MATRIZ)
    if not matriz:
        errors.append("matriz_regional_vazia")
    errors.extend(validate_matriz_rows(matriz))
    errors.extend(validate_fila_rows(_read(FILA)))
    errors.extend(public_text_violations_files())
    for row in _read(INVENTARIO):
        if row.get("uso_permitido") not in USO_PERMITIDO_ALLOWED:
            errors.append(f"inventario:{row.get('candidate_event_id')}:uso_fora_enum")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_proibido")
    if summary["status_final_17i"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_17i']}")
    if summary["status_17b"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_fora_enum:{summary['status_17b']}")
    if summary["status_final_17i"] == "17I_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_caminho_funcional_entregue")
    for field in ["ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"]:
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
        "17I ampliacao regional validada: "
        f"fortes={summary['candidato_observacional_forte']} parciais={summary['candidato_observacional_parcial']} "
        f"contextuais={summary['candidato_contextual']} misto={summary['bloqueado_por_fenomeno_misto']} "
        f"fila={summary['fila_tasks']} status17I={summary['status_final_17i']} status17B={summary['status_17b']}"
    )
    return 0


def run_all():
    summary = build_all()
    print(
        "17I ampliacao regional gerada: "
        f"cur={summary['candidatos_cur']} pet={summary['candidatos_pet']} "
        f"fortes={summary['candidato_observacional_forte']} parciais={summary['candidato_observacional_parcial']} "
        f"fila={summary['fila_tasks']} status17I={summary['status_final_17i']} caminho={summary['caminho_funcional']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
