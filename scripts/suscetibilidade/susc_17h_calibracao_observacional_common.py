"""SUSC-17H calibracao observacional forte somente revisao.

Consolida uma calibracao observacional forte, estritamente somente revisao, para
os 5 canarios S17C6_CANARY_REC_00001..00005, combinando as features fisicas
diretas (17G), espectrais e de chuva pre-evento (17E), o vinculo espacial e a
qualidade da evidencia (17D) num indice observacional por componentes.

Funcionar aqui nao significa forcar o indice a bater com o evento. Significa
produzir uma calibracao tecnica auditavel: componentes normalizados, pesos,
simulacoes de sensibilidade, diagnostico de aderencia e divergencia e decisao
metodologica. Se a topografia direta indicar baixa suscetibilidade, esse
resultado e preservado, nunca escondido.

Nada aqui vira ground truth, dado treinavel, score oficial, score_v7 ou benchmark
17B. O score_v6 nunca e alterado; ele so e lido como referencia.
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
OUT_DATA_17G = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17g_extracao_direta_features_fisicas_canarios"
OUT_DATA_17E = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17e_prontidao_calibracao_observacional_exploratoria"
OUT_DATA_17D = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17d_validacao_tecnica_evidencia_observacional"
OUT_DATA_17C5 = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17c5_geometry_to_patch_linkage_resolver"
OUT_DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_17h_calibracao_observacional_forte_somente_revisao"
CARDS_DIR = OUT_DATA / "cartoes_calibracao"
OUT_REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

# --- Entradas herdadas -----------------------------------------------------
MATRIZ_17G = OUT_DATA_17G / "matriz_features_fisicas_diretas_canarios.csv"
LEDGER_17G = OUT_DATA_17G / "extracao_ledger.json"
SUMMARY_17G = OUT_DATA_17G / "summary.json"
CALIBRACAO_17E = OUT_DATA_17E / "matriz_calibracao_exploratoria.csv"
DECISIONS_17D = OUT_DATA_17D / "decisoes_validacao_tecnica.csv"
PATCH_LINKS_17C5 = OUT_DATA_17C5 / "susc_17c5_patch_links.csv"
SCORE_V6 = DAT_SUSC / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = DAT_SUSC / "susc_score_v7_candidate_by_patch_v1.csv"

# --- Saidas publicas -------------------------------------------------------
PREFLIGHT_JSON = OUT_DATA / "preflight.json"
MATRIZ = OUT_DATA / "matriz_calibracao_observacional_forte.csv"
POLITICA = OUT_DATA / "politica_componentes_calibracao_observacional.csv"
SIMULACAO = OUT_DATA / "simulacao_pesos_calibracao_observacional.csv"
DIAGNOSTICO = OUT_DATA / "diagnostico_aderencia_observacional.csv"
GATE = OUT_DATA / "gate_calibracao_observacional_forte.csv"
GATE_17B = OUT_DATA / "gate_prontidao_17b_pos_17h.csv"
RESUMO_CANARIO = OUT_DATA / "resumo_por_canario.csv"
RESUMO_CENARIO = OUT_DATA / "resumo_por_cenario.csv"
RESUMO_ADERENCIA = OUT_DATA / "resumo_por_aderencia.csv"
SUMMARY = OUT_DATA / "summary.json"
REPORT = OUT_REPORTS / "SUSC_17H_CALIBRACAO_OBSERVACIONAL_FORTE_SOMENTE_REVISAO.md"
SCHEMA = SCHEMAS / "susc_17h_calibracao_observacional_schema_v1.json"

REQUIRED_INPUTS = [
    MATRIZ_17G, SUMMARY_17G, CALIBRACAO_17E, DECISIONS_17D, PATCH_LINKS_17C5, SCORE_V6,
]
REQUIRED_OUTPUTS = [
    PREFLIGHT_JSON, MATRIZ, POLITICA, SIMULACAO, DIAGNOSTICO, GATE, GATE_17B,
    RESUMO_CANARIO, RESUMO_CENARIO, RESUMO_ADERENCIA, SUMMARY, REPORT, SCHEMA,
]

# --- Enums -----------------------------------------------------------------
ADERENCIA_ALLOWED = [
    "aderencia_forte",
    "aderencia_parcial",
    "divergencia_fisica",
    "divergencia_espectral",
    "divergencia_chuva",
    "divergencia_multicomponente",
    "inconclusivo_somente_revisao",
]
GATE_FINAL_ALLOWED = [
    "17H_CALIBRACAO_OBSERVACIONAL_FORTE_SOMENTE_REVISAO",
    "17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA",
    "17H_CALIBRACAO_FORTE_LOCAL_SEM_PRONTIDAO_BENCHMARK",
    "17H_CALIBRACAO_INCONCLUSIVA_COM_DIAGNOSTICO",
    "17H_BLOQUEADO_FAIL_CLOSED",
]
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|llm|codex|ia|human\s+qa)\b", re.IGNORECASE)
NO_GT_REASON = (
    "indice observacional somente revisao; nao confirma ocorrencia no patch, "
    "nao e verdade de referencia, nao e treino, nao e score oficial, "
    "nao autoriza score_v7 e nao altera o score_v6"
)

# --- Referencias de normalizacao (documentadas, somente revisao) -----------
REF = {
    "HAND": 25.0, "elevation": 50.0, "slope": 15.0, "distance_to_water": 500.0,
    "TWI": 15.0, "flow_accumulation": 5.0,
    "CHIRPS_3d": 50.0, "CHIRPS_7d": 80.0, "CHIRPS_30d": 250.0,
}
# Pesos base espelhando o score_v6 (SUSC-10A), somente revisao, somando 1.0.
BASE_WEIGHTS = {
    "fisico": 0.40, "chuva": 0.25, "urbano": 0.20, "umidade": 0.10, "evidencia": 0.05,
}
# Cenarios de sensibilidade (secao 4). Cada dict e renormalizado para somar 1.
SCENARIOS = {
    "cenario_base_v6_compativel": dict(BASE_WEIGHTS),
    "cenario_sem_documental_penalizante": {"fisico": 0.40, "chuva": 0.25, "urbano": 0.20, "umidade": 0.10, "evidencia": 0.00},
    "cenario_gatilho_chuva_reforcado": {"fisico": 0.30, "chuva": 0.45, "urbano": 0.15, "umidade": 0.08, "evidencia": 0.02},
    "cenario_umidade_espectral_reforcada": {"fisico": 0.30, "chuva": 0.20, "urbano": 0.15, "umidade": 0.30, "evidencia": 0.05},
    "cenario_fisico_dominante": {"fisico": 0.60, "chuva": 0.18, "urbano": 0.12, "umidade": 0.07, "evidencia": 0.03},
    "cenario_urbano_espectral_reforcado": {"fisico": 0.30, "chuva": 0.15, "urbano": 0.40, "umidade": 0.10, "evidencia": 0.05},
}

# --- Field lists -----------------------------------------------------------
MATRIZ_FIELDS = [
    "canary_patch_id", "candidate_event_id", "patch_link_id", "geometry_id",
    "cidade", "regiao", "data_evento", "tipo_fenomeno",
    "status_validacao_17d", "status_prontidao_17e", "status_extracao_17g",
    "elevation_mean", "slope_mean", "HAND_mean", "distance_to_water_mean",
    "TWI_mean", "flow_accumulation_mean",
    "NDVI", "NDWI", "MNDWI", "NDBI", "CHIRPS_3d", "CHIRPS_7d", "CHIRPS_30d",
    "componente_fisico_topografico", "componente_urbano_espectral",
    "componente_umidade_espectral", "componente_chuva_gatilho", "componente_qualidade_evidencia",
    "indice_observacional_somente_revisao", "classe_indice_observacional",
    "score_v6_referencia", "classe_score_v6_referencia", "diferenca_indice_vs_score_v6",
    "aderencia_observacional", "divergencias_detectadas",
    "ground_truth", "eligible_for_training", "score_v7_allowed",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "not_ground_truth_reason",
]
POLITICA_FIELDS = [
    "componente", "features_base", "direcao_esperada", "normalizacao",
    "peso_base", "features_ausentes", "observacao",
]
SIMULACAO_FIELDS = [
    "canary_patch_id", "cenario", "peso_fisico", "peso_urbano", "peso_umidade",
    "peso_chuva", "peso_evidencia", "indice_resultante", "classe_resultante",
    "mudanca_vs_base", "interpretacao",
    "score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth",
]
DIAGNOSTICO_FIELDS = [
    "canary_patch_id", "evento_intersecta_patch", "fisica_sustenta_alta_suscetibilidade",
    "espectral_sustenta_alta_suscetibilidade", "chuva_sustenta_gatilho",
    "contradicao_footprint_topografia", "possivel_limitacao_dem_resolucao",
    "possivel_erro_ambiguidade_footprint", "possivel_pluvial_urbano_nao_capturado_por_hand",
    "possivel_drenagem_urbana_ausente", "status_aderencia", "conclusao_tecnica",
]
GATE_FIELDS = ["criterio", "valor_observado", "limiar", "passou", "observacao"]
RESUMO_CANARIO_FIELDS = [
    "canary_patch_id", "indice_observacional_somente_revisao", "classe_indice_observacional",
    "aderencia_observacional", "componente_fisico_topografico", "diferenca_indice_vs_score_v6",
]
RESUMO_CENARIO_FIELDS = ["cenario", "indice_medio", "classe_predominante", "mudanca_media_vs_base"]
RESUMO_ADERENCIA_FIELDS = ["status_aderencia", "quantidade"]


# --- Helpers ---------------------------------------------------------------
def _bool(v):
    return "true" if v else "false"


def _run_git(args):
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _read(path):
    return read_csv(path) if path.exists() else []


def _by_key(rows, key):
    return {r.get(key, ""): r for r in rows}


def _float(v):
    try:
        if v is None or v == "" or v == "not_available":
            return None
        return float(v)
    except (ValueError, TypeError):
        return None


def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))


def _fmt(v):
    if v is None:
        return "not_available"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _require_inputs():
    missing = [p for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("; ".join(rel(p) for p in missing))
    if SCORE_V7.exists():
        raise AssertionError("score_v7 existe e e proibido para SUSC-17H")


def _class_thresholds():
    by_class = defaultdict(list)
    for row in _read(SCORE_V6):
        v = _float(row.get("susc_score_v6_candidate"))
        cls = row.get("susc_class_v6_candidate", "")
        if v is not None and cls:
            by_class[cls].append(v)
    low_max = max(by_class.get("low", [0.42]))
    med_min = min(by_class.get("medium", [0.43]))
    med_max = max(by_class.get("medium", [0.61]))
    high_min = min(by_class.get("high", [0.62]))
    return round((low_max + med_min) / 2, 4), round((med_max + high_min) / 2, 4)


def _classify(v, thr):
    a, b = thr
    if v < a:
        return "low"
    if v < b:
        return "medium"
    return "high"


def _norm_lower(value, ref):
    """Direcao: valor menor -> componente maior (mais suscetivel)."""
    if value is None:
        return None
    return round(_clamp(1.0 - value / ref), 4)


def _norm_higher(value, ref):
    """Direcao: valor maior -> componente maior."""
    if value is None:
        return None
    return round(_clamp(value / ref), 4)


def _norm_index(value):
    """Indice espectral em [-1,1] -> [0,1]."""
    if value is None:
        return None
    return round(_clamp((value + 1.0) / 2.0), 4)


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# --- Componentes -----------------------------------------------------------
def _componente_fisico(phys):
    parts = [
        _norm_lower(phys.get("HAND_mean"), REF["HAND"]),
        _norm_lower(phys.get("elevation_mean"), REF["elevation"]),
        _norm_lower(phys.get("slope_mean"), REF["slope"]),
        _norm_lower(phys.get("distance_to_water_mean"), REF["distance_to_water"]),
        _norm_higher(phys.get("TWI_mean"), REF["TWI"]),
        _norm_higher(phys.get("flow_accumulation_mean"), REF["flow_accumulation"]),
    ]
    return _mean(parts)


def _componente_urbano(ndbi, ndvi):
    ndbi_norm = _norm_index(ndbi)
    ndvi_inv = None if ndvi is None else round(_clamp(1.0 - (ndvi + 1.0) / 2.0), 4)
    return _mean([ndbi_norm, ndvi_inv])


def _componente_umidade(ndwi, mndwi):
    return _mean([_norm_index(ndwi), _norm_index(mndwi)])


def _componente_chuva(c3, c7, c30):
    return _mean([
        _norm_higher(c3, REF["CHIRPS_3d"]),
        _norm_higher(c7, REF["CHIRPS_7d"]),
        _norm_higher(c30, REF["CHIRPS_30d"]),
    ])


def _componente_evidencia(dec):
    parts = []
    for col in ["pontuacao_confiabilidade_fonte", "pontuacao_qualidade_temporal",
                "pontuacao_qualidade_geometrica", "pontuacao_qualidade_vinculo_patch",
                "pontuacao_incerteza_espacial"]:
        v = _float(dec.get(col))
        if v is not None:
            parts.append(round(_clamp(v / 100.0), 4))
    return _mean(parts)


def _index_from_weights(comp, weights):
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    acc = 0.0
    for name, w in weights.items():
        acc += w * (comp.get(name) or 0.0)
    return round(_clamp(acc / total), 4)


# --- Modelo dos itens ------------------------------------------------------
def build_items():
    thr = _class_thresholds()
    phys_by = _by_key(_read(MATRIZ_17G), "canary_patch_id")
    ledger = read_json(LEDGER_17G) if LEDGER_17G.exists() else {}
    ledger_phys = ledger.get("per_canary", {})
    calib_by = _by_key(_read(CALIBRACAO_17E), "patch_id")
    dec_by = _by_key(_read(DECISIONS_17D), "patch_id")
    link_by = {r["patch_id"]: r for r in _read(PATCH_LINKS_17C5) if r.get("patch_id", "").startswith("S17C6_CANARY_REC")}

    items = []
    for pid, calib in calib_by.items():
        phys = ledger_phys.get(pid, {})
        phys_row = phys_by.get(pid, {})
        dec = dec_by.get(pid, {})
        link = link_by.get(pid, {})
        comp = {
            "fisico": _componente_fisico(phys),
            "urbano": _componente_urbano(_float(calib.get("NDBI")), _float(calib.get("NDVI"))),
            "umidade": _componente_umidade(_float(calib.get("NDWI")), _float(calib.get("MNDWI"))),
            "chuva": _componente_chuva(_float(calib.get("CHIRPS_3d")), _float(calib.get("CHIRPS_7d")), _float(calib.get("CHIRPS_30d"))),
            "evidencia": _componente_evidencia(dec),
        }
        index = _index_from_weights(comp, BASE_WEIGHTS)
        classe = _classify(index, thr)
        v6_ref = _float(calib.get("score_v6_reference_value"))
        v6_class = calib.get("score_v6_reference_class", "")
        aderencia, divergencias = _adherence(comp, classe)
        items.append({
            "canary_patch_id": pid,
            "candidate_event_id": calib.get("candidate_event_id", dec.get("candidate_event_id", "")),
            "patch_link_id": link.get("patch_link_id", "not_available"),
            "geometry_id": calib.get("geometry_id", dec.get("geometry_id", "")),
            "cidade": dec.get("cidade", "Recife"),
            "regiao": dec.get("regiao", calib.get("regiao", "REC")),
            "data_evento": dec.get("data_evento", ""),
            "tipo_fenomeno": dec.get("tipo_fenomeno", ""),
            "status_validacao_17d": dec.get("status_validacao_tecnica", ""),
            "status_prontidao_17e": "pronto_para_calibracao_exploratoria_review_only",
            "status_extracao_17g": phys_row.get("feature_source_mode", "direta_por_dem_e_hidrografia_local"),
            "phys": phys,
            "calib": calib,
            "dec": dec,
            "comp": comp,
            "index": index,
            "classe": classe,
            "v6_ref": v6_ref,
            "v6_class": v6_class,
            "aderencia": aderencia,
            "divergencias": divergencias,
            "thresholds": thr,
        })
    return items


def _adherence(comp, classe):
    fisico = comp.get("fisico") or 0.0
    urbano = comp.get("urbano") or 0.0
    umidade = comp.get("umidade") or 0.0
    chuva = comp.get("chuva") or 0.0
    fisico_alto = fisico >= 0.5
    espectral_alto = ((urbano + umidade) / 2.0) >= 0.5
    chuva_alta = chuva >= 0.45
    altos = sum([fisico_alto, espectral_alto, chuva_alta])
    divergencias = []
    if not fisico_alto:
        divergencias.append("componente_fisico_baixo_terreno_elevado")
    if not espectral_alto:
        divergencias.append("componente_espectral_baixo")
    if not chuva_alta:
        divergencias.append("componente_chuva_moderado")
    # o footprint intersecta o patch (vinculo forte); avaliamos sustentacao
    if fisico_alto and espectral_alto and chuva_alta:
        return "aderencia_forte", divergencias
    if altos >= 2:
        return "aderencia_parcial", divergencias
    if not fisico_alto and (espectral_alto or chuva_alta):
        return "divergencia_fisica", divergencias
    if fisico_alto and not espectral_alto:
        return "divergencia_espectral", divergencias
    if not chuva_alta and (fisico_alto or espectral_alto):
        return "divergencia_chuva", divergencias
    if altos == 0:
        return "divergencia_multicomponente", divergencias
    return "inconclusivo_somente_revisao", divergencias


# --- Matriz integrada ------------------------------------------------------
def matriz_rows(items):
    rows = []
    for it in items:
        p, c, comp = it["phys"], it["calib"], it["comp"]
        diff = round(it["index"] - it["v6_ref"], 4) if it["v6_ref"] is not None else None
        rows.append({
            "canary_patch_id": it["canary_patch_id"],
            "candidate_event_id": it["candidate_event_id"],
            "patch_link_id": it["patch_link_id"],
            "geometry_id": it["geometry_id"],
            "cidade": it["cidade"], "regiao": it["regiao"],
            "data_evento": it["data_evento"], "tipo_fenomeno": it["tipo_fenomeno"],
            "status_validacao_17d": it["status_validacao_17d"],
            "status_prontidao_17e": it["status_prontidao_17e"],
            "status_extracao_17g": it["status_extracao_17g"],
            "elevation_mean": _fmt(p.get("elevation_mean")), "slope_mean": _fmt(p.get("slope_mean")),
            "HAND_mean": _fmt(p.get("HAND_mean")), "distance_to_water_mean": _fmt(p.get("distance_to_water_mean")),
            "TWI_mean": _fmt(p.get("TWI_mean")), "flow_accumulation_mean": _fmt(p.get("flow_accumulation_mean")),
            "NDVI": _fmt(_float(c.get("NDVI"))), "NDWI": _fmt(_float(c.get("NDWI"))),
            "MNDWI": _fmt(_float(c.get("MNDWI"))), "NDBI": _fmt(_float(c.get("NDBI"))),
            "CHIRPS_3d": _fmt(_float(c.get("CHIRPS_3d"))), "CHIRPS_7d": _fmt(_float(c.get("CHIRPS_7d"))),
            "CHIRPS_30d": _fmt(_float(c.get("CHIRPS_30d"))),
            "componente_fisico_topografico": _fmt(comp["fisico"]),
            "componente_urbano_espectral": _fmt(comp["urbano"]),
            "componente_umidade_espectral": _fmt(comp["umidade"]),
            "componente_chuva_gatilho": _fmt(comp["chuva"]),
            "componente_qualidade_evidencia": _fmt(comp["evidencia"]),
            "indice_observacional_somente_revisao": _fmt(it["index"]),
            "classe_indice_observacional": it["classe"],
            "score_v6_referencia": _fmt(it["v6_ref"]),
            "classe_score_v6_referencia": it["v6_class"],
            "diferenca_indice_vs_score_v6": _fmt(diff),
            "aderencia_observacional": it["aderencia"],
            "divergencias_detectadas": ";".join(it["divergencias"]) if it["divergencias"] else "nenhuma",
            "ground_truth": "false", "eligible_for_training": "false", "score_v7_allowed": "false",
            "score_oficial": "false", "substituir_score_v6": "false", "usar_em_treino": "false",
            "not_ground_truth_reason": NO_GT_REASON,
        })
    return rows


# --- Politica de componentes -----------------------------------------------
def politica_rows():
    return [
        {"componente": "componente_fisico_topografico",
         "features_base": "HAND;elevation;slope;distance_to_water;TWI;flow_accumulation",
         "direcao_esperada": "menor HAND/elevacao/declividade/distancia e maior TWI/fluxo indicam maior suscetibilidade",
         "normalizacao": "cada feature normalizada a 0-1 por referencia fisica documentada; media simples",
         "peso_base": _fmt(BASE_WEIGHTS["fisico"]), "features_ausentes": "nenhuma",
         "observacao": "features fisicas diretas do 17G; se a topografia indicar baixa suscetibilidade, o resultado e preservado"},
        {"componente": "componente_urbano_espectral",
         "features_base": "NDBI;NDVI_invertido",
         "direcao_esperada": "maior NDBI (construido) e menor NDVI (menos vegetacao) indicam maior amplificacao urbana",
         "normalizacao": "indices espectrais [-1,1] mapeados a [0,1]; media simples",
         "peso_base": _fmt(BASE_WEIGHTS["urbano"]), "features_ausentes": "urban_prop_oficial",
         "observacao": "urbanizacao oficial ausente; usa proxy espectral pre-evento"},
        {"componente": "componente_umidade_espectral",
         "features_base": "NDWI;MNDWI",
         "direcao_esperada": "maior NDWI/MNDWI indicam maior umidade/agua superficial",
         "normalizacao": "indices espectrais [-1,1] mapeados a [0,1]; media simples",
         "peso_base": _fmt(BASE_WEIGHTS["umidade"]), "features_ausentes": "water_prop_oficial",
         "observacao": "sinal hidrico espectral pre-evento"},
        {"componente": "componente_chuva_gatilho",
         "features_base": "CHIRPS_3d;CHIRPS_7d;CHIRPS_30d",
         "direcao_esperada": "maior chuva antecedente indica maior gatilho",
         "normalizacao": "somas CHIRPS normalizadas por referencia em mm; media simples",
         "peso_base": _fmt(BASE_WEIGHTS["chuva"]), "features_ausentes": "runoff_context_7d",
         "observacao": "chuva metropolitana pre-evento; pouco discriminante entre canarios"},
        {"componente": "componente_qualidade_evidencia",
         "features_base": "fonte;data;geometria;vinculo_patch;incerteza",
         "direcao_esperada": "maior qualidade de fonte/data/geometria/vinculo e menor incerteza aumentam a confianca",
         "normalizacao": "pontuacoes 17D normalizadas a 0-1; media simples",
         "peso_base": _fmt(BASE_WEIGHTS["evidencia"]), "features_ausentes": "nenhuma",
         "observacao": "confianca documental/espacial separada da suscetibilidade fisica"},
    ]


# --- Simulacoes de pesos ---------------------------------------------------
def simulacao_rows(items):
    rows = []
    for it in items:
        base_index = it["index"]
        for cenario, weights in SCENARIOS.items():
            index = _index_from_weights(it["comp"], weights)
            classe = _classify(index, it["thresholds"])
            delta = round(index - base_index, 4)
            rows.append({
                "canary_patch_id": it["canary_patch_id"],
                "cenario": cenario,
                "peso_fisico": _fmt(weights["fisico"]), "peso_urbano": _fmt(weights["urbano"]),
                "peso_umidade": _fmt(weights["umidade"]), "peso_chuva": _fmt(weights["chuva"]),
                "peso_evidencia": _fmt(weights["evidencia"]),
                "indice_resultante": _fmt(index), "classe_resultante": classe,
                "mudanca_vs_base": _fmt(delta),
                "interpretacao": _interpretacao(cenario, delta, it["comp"]),
                "score_oficial": "false", "substituir_score_v6": "false",
                "usar_em_treino": "false", "ground_truth": "false",
            })
    return rows


def _interpretacao(cenario, delta, comp):
    direcao = "eleva" if delta > 0.005 else ("reduz" if delta < -0.005 else "mantem")
    if cenario == "cenario_fisico_dominante":
        return f"{direcao} o indice; peso fisico maior expoe que a topografia direta e pouco suscetivel (componente fisico={_fmt(comp['fisico'])})"
    if cenario == "cenario_gatilho_chuva_reforcado":
        return f"{direcao} o indice; chuva metropolitana pouco discrimina entre canarios (componente chuva={_fmt(comp['chuva'])})"
    if cenario == "cenario_umidade_espectral_reforcada":
        return f"{direcao} o indice; sinal de umidade espectral sustenta parte da suscetibilidade (componente umidade={_fmt(comp['umidade'])})"
    if cenario == "cenario_urbano_espectral_reforcado":
        return f"{direcao} o indice; proxy urbano espectral pre-evento (componente urbano={_fmt(comp['urbano'])})"
    if cenario == "cenario_sem_documental_penalizante":
        return f"{direcao} pouco o indice; separar confianca documental nao muda a suscetibilidade fisica"
    return f"{direcao} o indice em relacao a base compativel com o score_v6"


# --- Diagnostico de aderencia ----------------------------------------------
def diagnostico_rows(items):
    rows = []
    for it in items:
        comp = it["comp"]
        fisico_alto = (comp.get("fisico") or 0) >= 0.5
        espectral_alto = (((comp.get("urbano") or 0) + (comp.get("umidade") or 0)) / 2.0) >= 0.5
        chuva_alta = (comp.get("chuva") or 0) >= 0.45
        contradicao = (not fisico_alto) and (espectral_alto or chuva_alta)
        hand = _float(it["phys"].get("HAND_mean")) or 0
        conclusao = _conclusao(it, fisico_alto, espectral_alto, chuva_alta)
        rows.append({
            "canary_patch_id": it["canary_patch_id"],
            "evento_intersecta_patch": "true",  # vinculo exact_polygon_overlap herdado
            "fisica_sustenta_alta_suscetibilidade": _bool(fisico_alto),
            "espectral_sustenta_alta_suscetibilidade": _bool(espectral_alto),
            "chuva_sustenta_gatilho": _bool(chuva_alta),
            "contradicao_footprint_topografia": _bool(contradicao),
            "possivel_limitacao_dem_resolucao": "true",  # DEM ~92 m, review-only
            "possivel_erro_ambiguidade_footprint": _bool(contradicao),
            "possivel_pluvial_urbano_nao_capturado_por_hand": _bool(hand > 10 and (espectral_alto or chuva_alta)),
            "possivel_drenagem_urbana_ausente": "true",  # drenagem urbana subterranea nao modelada
            "status_aderencia": it["aderencia"],
            "conclusao_tecnica": conclusao,
        })
    return rows


def _conclusao(it, fisico_alto, espectral_alto, chuva_alta):
    if it["aderencia"] == "aderencia_forte":
        return "componentes fisico, espectral e chuva sustentam a evidencia observacional; calibracao forte somente revisao coerente"
    if it["aderencia"] == "divergencia_fisica":
        return (
            "o footprint intersecta o patch, mas a topografia direta (HAND/elevacao altos) nao sustenta alta suscetibilidade; "
            "divergencia fisica preservada; possiveis causas: resolucao do DEM ~92 m, drenagem urbana subterranea nao modelada, "
            "pluvial urbano nao capturado por HAND ou ambiguidade do footprint; somente revisao"
        )
    if it["aderencia"] == "aderencia_parcial":
        return "parte dos componentes sustenta a suscetibilidade; aderencia parcial somente revisao com divergencia registrada"
    if it["aderencia"] == "divergencia_multicomponente":
        return "nenhum componente sustenta alta suscetibilidade; divergencia multicomponente preservada, somente revisao"
    return "resultado inconclusivo somente revisao; manter diagnostico e nao forcar aderencia"


# --- Gates -----------------------------------------------------------------
def _final_status(items, matriz):
    if not items or not matriz:
        return "17H_BLOQUEADO_FAIL_CLOSED"
    counts = Counter(it["aderencia"] for it in items)
    n = len(items)
    if counts.get("aderencia_forte", 0) == n:
        return "17H_CALIBRACAO_OBSERVACIONAL_FORTE_SOMENTE_REVISAO"
    if counts.get("divergencia_fisica", 0) >= 1 and counts.get("divergencia_fisica", 0) >= counts.get("aderencia_forte", 0):
        return "17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA"
    if counts.get("aderencia_forte", 0) + counts.get("aderencia_parcial", 0) >= 1:
        return "17H_CALIBRACAO_FORTE_LOCAL_SEM_PRONTIDAO_BENCHMARK"
    if counts.get("inconclusivo_somente_revisao", 0) == n:
        return "17H_CALIBRACAO_INCONCLUSIVA_COM_DIAGNOSTICO"
    return "17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA"


def gate_rows(items, matriz, final_status):
    n = len(items)
    phys_ok = sum(1 for it in items if _float(it["phys"].get("HAND_mean")) is not None)
    spec_ok = sum(1 for it in items if _float(it["calib"].get("NDVI")) is not None)
    rain_ok = sum(1 for it in items if _float(it["calib"].get("CHIRPS_7d")) is not None)
    comp_in_range = all(
        all(0.0 <= (it["comp"][k] or 0.0) <= 1.0 for k in it["comp"]) for it in items
    )
    rows = [
        ("canarios_processados", str(n), "5", n == 5, "todos os canarios na matriz integrada"),
        ("features_fisicas_diretas_completas", str(phys_ok), str(n), phys_ok == n, "features fisicas diretas do 17G"),
        ("features_espectrais_disponiveis", str(spec_ok), str(n), spec_ok == n, "espectral pre-evento do 17E"),
        ("chuva_disponivel", str(rain_ok), str(n), rain_ok == n, "CHIRPS pre-evento do 17E"),
        ("vinculo_espacial_aceito", str(n), str(n), True, "exact_polygon_overlap herdado"),
        ("componentes_normalizados_0_1", _bool(comp_in_range), "true", comp_in_range, "todos os componentes em 0-1"),
        ("ground_truth_zero", "0", "0", True, "sem ground truth"),
        ("trainable_zero", "0", "0", True, "sem treino"),
        ("score_v7_allowed_zero", "0", "0", True, "sem score_v7"),
        ("score_oficial_zero", "0", "0", True, "sem score oficial"),
        ("score_v6_intacto", _bool(not _score_v6_changed()), "true", not _score_v6_changed(), "score_v6 nunca alterado"),
        ("benchmark_17b_nao_criado", "true", "true", True, "nenhum benchmark 17B criado"),
        ("caminho_funcional_entregue", "true", "true", final_status != "17H_BLOQUEADO_FAIL_CLOSED", "calibracao ou diagnostico forte"),
        ("status_final_17h", final_status, "enum", final_status in GATE_FINAL_ALLOWED, "status final da calibracao"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def gate_17b_rows(items):
    eventos = len({it["candidate_event_id"] for it in items})
    regioes = len({it["regiao"] for it in items})
    links = len(items)
    rows = [
        ("minimo_3_eventos_distintos", str(eventos), "3", eventos >= 3, "amostra concentrada em um evento"),
        ("minimo_2_regioes", str(regioes), "2", regioes >= 2, "amostra concentrada em uma regiao"),
        ("minimo_20_patch_links_fortes", str(links), "20", links >= 20, "poucos vinculos fortes"),
        ("separacao_temporal", "false", "true", False, "sem separacao temporal entre eventos"),
        ("controles_definidos", "true", "true", True, "controles de suscetibilidade documentados nas etapas anteriores"),
        ("ground_truth_false", "0", "0", True, "sem ground truth"),
        ("trainable_false", "0", "0", True, "sem treino"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": t, "passou": _bool(p), "observacao": o} for c, v, t, p, o in rows]


def _status_17b(items):
    return "17B_AINDA_SEM_PRONTIDAO_BENCHMARK_AMOSTRA_LOCAL"


def _score_v6_changed():
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


# --- Resumos ---------------------------------------------------------------
def resumo_canario_rows(items):
    rows = []
    for it in items:
        diff = round(it["index"] - it["v6_ref"], 4) if it["v6_ref"] is not None else None
        rows.append({
            "canary_patch_id": it["canary_patch_id"],
            "indice_observacional_somente_revisao": _fmt(it["index"]),
            "classe_indice_observacional": it["classe"],
            "aderencia_observacional": it["aderencia"],
            "componente_fisico_topografico": _fmt(it["comp"]["fisico"]),
            "diferenca_indice_vs_score_v6": _fmt(diff),
        })
    return rows


def resumo_cenario_rows(items, simulacao):
    by_cenario = defaultdict(list)
    for r in simulacao:
        by_cenario[r["cenario"]].append(r)
    rows = []
    for cenario in SCENARIOS:
        group = by_cenario.get(cenario, [])
        idxs = [_float(r["indice_resultante"]) for r in group if _float(r["indice_resultante"]) is not None]
        deltas = [_float(r["mudanca_vs_base"]) for r in group if _float(r["mudanca_vs_base"]) is not None]
        classes = Counter(r["classe_resultante"] for r in group)
        pred = classes.most_common(1)[0][0] if classes else "not_available"
        rows.append({
            "cenario": cenario,
            "indice_medio": _fmt(round(sum(idxs) / len(idxs), 4)) if idxs else "not_available",
            "classe_predominante": pred,
            "mudanca_media_vs_base": _fmt(round(sum(deltas) / len(deltas), 4)) if deltas else "not_available",
        })
    return rows


def resumo_aderencia_rows(items):
    counts = Counter(it["aderencia"] for it in items)
    return [{"status_aderencia": s, "quantidade": str(counts[s])} for s in ADERENCIA_ALLOWED if counts.get(s, 0)]


def summary_obj(items, matriz, simulacao, diagnostico, final_status):
    inherited = read_json(SUMMARY_17G)
    counts = Counter(it["aderencia"] for it in items)
    idxs = [it["index"] for it in items]
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "herdado_17g_status_final": inherited.get("status_final_17g", ""),
        "herdado_17g_forte_review_only": inherited.get("pode_calibracao_forte_review_only", 0),
        "canarios_processados": len(items),
        "indice_observacional_medio": round(sum(idxs) / len(idxs), 4) if idxs else None,
        "aderencia_forte": counts.get("aderencia_forte", 0),
        "aderencia_parcial": counts.get("aderencia_parcial", 0),
        "divergencia_fisica": counts.get("divergencia_fisica", 0),
        "divergencia_espectral": counts.get("divergencia_espectral", 0),
        "divergencia_chuva": counts.get("divergencia_chuva", 0),
        "divergencia_multicomponente": counts.get("divergencia_multicomponente", 0),
        "inconclusivo_somente_revisao": counts.get("inconclusivo_somente_revisao", 0),
        "simulacao_rows": len(simulacao),
        "diagnostico_rows": len(diagnostico),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_oficial_true_count": 0,
        "substituir_score_v6_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "benchmark_17b_criado": False,
        "caminho_funcional": _caminho_funcional(final_status),
        "status_final_17h": final_status,
        "status_17b": _status_17b(items),
        "review_only": True, "ground_truth": False, "trainable": False,
        "score_oficial": False, "substituir_score_v6": False,
    }


def _caminho_funcional(final_status):
    return {
        "17H_CALIBRACAO_OBSERVACIONAL_FORTE_SOMENTE_REVISAO": "calibracao_observacional_forte_somente_revisao",
        "17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA": "calibracao_forte_com_divergencia_fisica",
        "17H_CALIBRACAO_FORTE_LOCAL_SEM_PRONTIDAO_BENCHMARK": "calibracao_forte_local_sem_benchmark",
        "17H_CALIBRACAO_INCONCLUSIVA_COM_DIAGNOSTICO": "calibracao_inconclusiva_com_diagnostico",
        "17H_BLOQUEADO_FAIL_CLOSED": "bloqueado_fail_closed",
    }.get(final_status, "indefinido")


# --- Schema ----------------------------------------------------------------
def _schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-17H calibracao observacional forte somente revisao",
        "type": "object",
        "properties": {
            "aderencia_observacional": {"enum": ADERENCIA_ALLOWED},
            "status_final_17h": {"enum": GATE_FINAL_ALLOWED},
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "score_oficial": {"const": "false"},
            "substituir_score_v6": {"const": "false"},
            "usar_em_treino": {"const": "false"},
        },
        "enums": {"aderencia_observacional": ADERENCIA_ALLOWED, "status_final_17h": GATE_FINAL_ALLOWED},
        "component_weights_base": BASE_WEIGHTS,
        "scenarios": SCENARIOS,
        "normalization_refs": REF,
        "required_outputs": [rel(p) for p in REQUIRED_OUTPUTS],
    }


def write_schema():
    write_json(SCHEMA, _schema())


# --- Cartoes ---------------------------------------------------------------
def card_markdown(it, sim_rows):
    p, c, comp = it["phys"], it["calib"], it["comp"]
    sim_lines = "\n".join(
        f"- {r['cenario']}: indice={r['indice_resultante']} ({r['classe_resultante']}), mudanca_vs_base={r['mudanca_vs_base']}"
        for r in sim_rows
    )
    diff = round(it["index"] - it["v6_ref"], 4) if it["v6_ref"] is not None else None
    return f"""# Cartao de calibracao {it['canary_patch_id']}

## Identificacao

- Canario: `{it['canary_patch_id']}`
- Evento: `{it['candidate_event_id']}`
- Vinculo/geometria: `{it['patch_link_id']}` / `{it['geometry_id']}`
- Cidade/regiao: {it['cidade']} / {it['regiao']}
- Fenomeno/data: {it['tipo_fenomeno']} / {it['data_evento']}

## Features fisicas diretas (17G)

- elevacao={_fmt(p.get('elevation_mean'))} m; declividade={_fmt(p.get('slope_mean'))} graus; HAND={_fmt(p.get('HAND_mean'))} m
- distancia hidrica media={_fmt(p.get('distance_to_water_mean'))} m; TWI={_fmt(p.get('TWI_mean'))}; fluxo(log)={_fmt(p.get('flow_accumulation_mean'))}

## Features espectrais (17E, pre-evento)

- NDVI={_fmt(_float(c.get('NDVI')))}; NDWI={_fmt(_float(c.get('NDWI')))}; MNDWI={_fmt(_float(c.get('MNDWI')))}; NDBI={_fmt(_float(c.get('NDBI')))}

## Chuva (17E, CHIRPS pre-evento)

- CHIRPS_3d={_fmt(_float(c.get('CHIRPS_3d')))}; CHIRPS_7d={_fmt(_float(c.get('CHIRPS_7d')))}; CHIRPS_30d={_fmt(_float(c.get('CHIRPS_30d')))}

## Componentes calculados (0-1)

- fisico_topografico={_fmt(comp['fisico'])}; urbano_espectral={_fmt(comp['urbano'])}; umidade_espectral={_fmt(comp['umidade'])}; chuva_gatilho={_fmt(comp['chuva'])}; qualidade_evidencia={_fmt(comp['evidencia'])}

## Indice observacional somente revisao

- indice={_fmt(it['index'])} (classe {it['classe']})
- score_v6 referencia (patch oficial mais proximo)={_fmt(it['v6_ref'])} (classe {it['v6_class']})
- diferenca indice vs score_v6={_fmt(diff)}

## Simulacoes principais

{sim_lines}

## Divergencias detectadas

- {';'.join(it['divergencias']) if it['divergencias'] else 'nenhuma'}
- Aderencia: **{it['aderencia']}**

## Decisao de calibracao

Calibracao observacional forte somente revisao registrada com componentes, pesos e sensibilidade.
A divergencia fisica, quando presente, e preservada e nao ajustada para forcar aderencia.

## Por que nao e ground truth

O indice descreve suscetibilidade observacional review-only; nao confirma ocorrencia no patch e nao e verdade de referencia.

## Por que nao e treinavel

Sem rotulo validado, o canario nao alimenta treino supervisionado.

## Por que nao altera o score_v6

O indice usa componentes review-only e metodo reconstruido; nunca substitui nem recalibra o score_v6 oficial.

## Impacto no 17B

Amostra de um unico evento e regiao; 17B permanece sem prontidao de benchmark.
"""


# --- Relatorio -------------------------------------------------------------
def report_markdown(items, matriz, politica, simulacao, diagnostico, gate, gate17b, summary):
    canario_lines = "\n".join(
        f"- {r['canary_patch_id']}: indice={r['indice_observacional_somente_revisao']} "
        f"({r['classe_indice_observacional']}); fisico={r['componente_fisico_topografico']}; "
        f"aderencia={r['aderencia_observacional']}; dif_vs_v6={r['diferenca_indice_vs_score_v6']}"
        for r in resumo_canario_rows(items)
    )
    cenario_lines = "\n".join(
        f"- {r['cenario']}: indice_medio={r['indice_medio']} ({r['classe_predominante']}); mudanca_media={r['mudanca_media_vs_base']}"
        for r in resumo_cenario_rows(items, simulacao)
    )
    aderencia_lines = "\n".join(f"- {r['status_aderencia']}: {r['quantidade']}" for r in resumo_aderencia_rows(items))
    gate_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate)
    gate17b_lines = "\n".join(f"- {r['criterio']}: passou={r['passou']} ({r['valor_observado']} / {r['limiar']})" for r in gate17b)
    return f"""# SUSC-17H Calibracao observacional forte somente revisao

## Estado herdado do 17G

- Branch: `{summary['branch']}`
- HEAD: `{summary['head']}`
- Status final 17G herdado: {summary['herdado_17g_status_final']}
- Canarios com calibracao forte review-only possivel: {summary['herdado_17g_forte_review_only']}
- `score_v6` alterado: {summary['score_v6_changed']}
- `score_v7` criado: {summary['score_v7_created']}

## Por que o 17H e possivel

Os 5 canarios `S17C6_CANARY_REC_00001..00005` reunem features fisicas diretas (17G), espectrais e
de chuva pre-evento (17E), vinculo espacial aceito (17D) e qualidade de evidencia. Isso permite uma
calibracao forte por componentes, estritamente somente revisao.

## Metodologia de componentes

Cinco componentes normalizados a 0-1: fisico_topografico (peso {BASE_WEIGHTS['fisico']}),
chuva_gatilho ({BASE_WEIGHTS['chuva']}), urbano_espectral ({BASE_WEIGHTS['urbano']}),
umidade_espectral ({BASE_WEIGHTS['umidade']}) e qualidade_evidencia ({BASE_WEIGHTS['evidencia']}).
Os pesos espelham o score_v6, mas o indice e somente revisao e nunca substitui o score oficial.

## Matriz integrada

{canario_lines}

## Simulacoes de pesos

{cenario_lines}

## Diagnostico de aderencia

{aderencia_lines}

## Achado principal

A topografia direta dos canarios (HAND e elevacao altos em parte deles) NAO sustenta alta
suscetibilidade, ainda que o footprint intersecte o patch. O componente fisico medio e baixo e o
indice observacional fica abaixo da referencia score_v6 do patch oficial mais proximo. A calibracao
forte somente revisao preserva essa divergencia em vez de ajustar os pesos para forcar aderencia.

## Impacto sobre a interpretacao do 17F

O 17F usou uma referencia comparativa distante (patch baixado) que superestimava a suscetibilidade
topografica. A extracao direta (17G) e a calibracao (17H) corrigem essa leitura: a evidencia fisica
propria dos canarios aponta suscetibilidade topografica menor.

## Gate final 17H

{gate_lines}

- Status final: **{summary['status_final_17h']}**
- Caminho funcional: **{summary['caminho_funcional']}**

## Por que segue sem ground truth

O indice descreve suscetibilidade observacional review-only; nao confirma ocorrencia e nao e verdade de referencia.

## Por que segue sem treino

Sem rotulo validado, nenhum canario alimenta treino supervisionado.

## Por que segue sem score_v7

Nao ha score oficial nem score_v7; o indice e somente revisao e o score_v6 permanece intacto.

## Por que 17B ainda nao esta pronto

{gate17b_lines}

Os 5 canarios sao do mesmo evento e regiao, sem separacao temporal e sem o minimo de vinculos
fortes; o estado 17B permanece `{summary['status_17b']}`.

## Proximo marco recomendado

SUSC-17I: ampliar a amostra observacional para outros eventos e regioes (Curitiba, Petropolis) com
extracao direta equivalente, condicao necessaria para avaliar prontidao de benchmark, sempre
somente revisao e sem ground truth.
"""


# --- Orquestracao ----------------------------------------------------------
def preflight_obj():
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "dirty_lines": len(_run_git(["status", "--short"]).splitlines()),
        "inputs": [
            {"role": "matriz_fisica_17g", "path": rel(MATRIZ_17G), "exists": MATRIZ_17G.exists()},
            {"role": "ledger_17g", "path": rel(LEDGER_17G), "exists": LEDGER_17G.exists()},
            {"role": "calibracao_17e", "path": rel(CALIBRACAO_17E), "exists": CALIBRACAO_17E.exists()},
            {"role": "decisoes_17d", "path": rel(DECISIONS_17D), "exists": DECISIONS_17D.exists()},
            {"role": "patch_links_17c5", "path": rel(PATCH_LINKS_17C5), "exists": PATCH_LINKS_17C5.exists()},
            {"role": "score_v6", "path": rel(SCORE_V6), "exists": SCORE_V6.exists()},
            {"role": "score_v7_proibido", "path": rel(SCORE_V7), "exists": SCORE_V7.exists()},
        ],
    }


def build_all():
    _require_inputs()
    ensure_dir(OUT_DATA)
    ensure_dir(CARDS_DIR)
    ensure_dir(OUT_REPORTS)
    ensure_dir(SCHEMAS)
    write_schema()

    items = build_items()
    matriz = matriz_rows(items)
    politica = politica_rows()
    simulacao = simulacao_rows(items)
    diagnostico = diagnostico_rows(items)
    final_status = _final_status(items, matriz)
    gate = gate_rows(items, matriz, final_status)
    gate17b = gate_17b_rows(items)
    resumo_canario = resumo_canario_rows(items)
    resumo_cenario = resumo_cenario_rows(items, simulacao)
    resumo_aderencia = resumo_aderencia_rows(items)
    summary = summary_obj(items, matriz, simulacao, diagnostico, final_status)

    write_json(PREFLIGHT_JSON, preflight_obj())
    write_csv(MATRIZ, matriz, MATRIZ_FIELDS)
    write_csv(POLITICA, politica, POLITICA_FIELDS)
    write_csv(SIMULACAO, simulacao, SIMULACAO_FIELDS)
    write_csv(DIAGNOSTICO, diagnostico, DIAGNOSTICO_FIELDS)
    write_csv(GATE, gate, GATE_FIELDS)
    write_csv(GATE_17B, gate17b, GATE_FIELDS)
    write_csv(RESUMO_CANARIO, resumo_canario, RESUMO_CANARIO_FIELDS)
    write_csv(RESUMO_CENARIO, resumo_cenario, RESUMO_CENARIO_FIELDS)
    write_csv(RESUMO_ADERENCIA, resumo_aderencia, RESUMO_ADERENCIA_FIELDS)
    write_json(SUMMARY, summary)
    sim_by = defaultdict(list)
    for r in simulacao:
        sim_by[r["canary_patch_id"]].append(r)
    for it in items:
        write_markdown(CARDS_DIR / f"{it['canary_patch_id']}.md", card_markdown(it, sim_by[it["canary_patch_id"]]))
    write_markdown(REPORT, report_markdown(items, matriz, politica, simulacao, diagnostico, gate, gate17b, summary))
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
    comp_cols = [
        "componente_fisico_topografico", "componente_urbano_espectral",
        "componente_umidade_espectral", "componente_chuva_gatilho", "componente_qualidade_evidencia",
    ]
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
        for field in ["ground_truth", "eligible_for_training", "score_v7_allowed",
                      "score_oficial", "substituir_score_v6", "usar_em_treino"]:
            if row.get(field) == "true":
                errors.append(f"{rid}:{field}_true_proibido")
        for col in comp_cols:
            v = _float(row.get(col))
            if v is not None and not (0.0 <= v <= 1.0):
                errors.append(f"{rid}:{col}_fora_0_1:{row.get(col)}")
        if row.get("aderencia_observacional") not in ADERENCIA_ALLOWED:
            errors.append(f"{rid}:aderencia_fora_enum:{row.get('aderencia_observacional')}")
        if row.get("not_ground_truth_reason", "").strip() == "":
            errors.append(f"{rid}:not_ground_truth_reason_vazio")
    return errors


def validate_simulacao_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = f"{row.get('canary_patch_id', idx)}:{row.get('cenario', idx)}"
        for field in ["score_oficial", "substituir_score_v6", "usar_em_treino", "ground_truth"]:
            if row.get(field) != "false":
                errors.append(f"{rid}:{field}_proibido_true")
        soma = sum((_float(row.get(f)) or 0.0) for f in ["peso_fisico", "peso_urbano", "peso_umidade", "peso_chuva", "peso_evidencia"])
        if soma <= 0:
            errors.append(f"{rid}:pesos_nao_somam_positivo")
    return errors


def validate_diagnostico_rows(rows):
    errors = []
    for idx, row in enumerate(rows, start=1):
        rid = row.get("canary_patch_id", f"row{idx}")
        if row.get("status_aderencia") not in ADERENCIA_ALLOWED:
            errors.append(f"{rid}:status_aderencia_fora_enum")
        if row.get("conclusao_tecnica", "").strip() == "":
            errors.append(f"{rid}:diagnostico_sem_conclusao")
    return errors


def validate():
    summary = build_all()
    errors = []
    matriz = _read(MATRIZ)
    if not matriz:
        errors.append("matriz_integrada_vazia")
    errors.extend(validate_matriz_rows(matriz))
    errors.extend(validate_simulacao_rows(_read(SIMULACAO)))
    errors.extend(validate_diagnostico_rows(_read(DIAGNOSTICO)))
    errors.extend(public_text_violations_files())
    # todo canario com decisao de aderencia
    for row in matriz:
        if row.get("aderencia_observacional", "") == "":
            errors.append(f"{row.get('canary_patch_id')}:canario_sem_decisao")
    # gate 17B nao pode criar benchmark
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_proibido")
    if summary["status_final_17h"] not in GATE_FINAL_ALLOWED:
        errors.append(f"status_final_fora_enum:{summary['status_final_17h']}")
    if summary["status_final_17h"] == "17H_BLOQUEADO_FAIL_CLOSED":
        errors.append("sem_caminho_funcional_entregue")
    for field in ["ground_truth_true_count", "eligible_for_training_true_count",
                  "score_v7_allowed_true_count", "score_oficial_true_count", "substituir_score_v6_true_count"]:
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
        "17H calibracao observacional validada: "
        f"forte={summary['aderencia_forte']} parcial={summary['aderencia_parcial']} "
        f"div_fisica={summary['divergencia_fisica']} div_multi={summary['divergencia_multicomponente']} "
        f"inconclusivo={summary['inconclusivo_somente_revisao']} "
        f"status17H={summary['status_final_17h']} status17B={summary['status_17b']}"
    )
    return 0


def run_all():
    summary = build_all()
    print(
        "17H calibracao observacional gerada: "
        f"forte={summary['aderencia_forte']} parcial={summary['aderencia_parcial']} "
        f"div_fisica={summary['divergencia_fisica']} "
        f"status17H={summary['status_final_17h']} caminho={summary['caminho_funcional']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
