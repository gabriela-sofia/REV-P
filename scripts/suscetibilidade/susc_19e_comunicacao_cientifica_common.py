"""SUSC-19E pacote de comunicacao cientifica review-only.

Transforma os resultados de 18H, 19A, 19B, 19C e 19D em um pacote de comunicacao
para trabalho de conclusao, artigo e apresentacao, sem inflar claims. Nao cria
dado novo, score novo, benchmark, treino nem ground truth. Le os numeros dos
marcos anteriores (nao regenera) e organiza narrativa, tabelas, plano de figuras,
politica de claims, glossario, perguntas de banca, matriz de riscos e gates.

Enquadramento cientifico do REV-P: framework multimodal auditavel de
suscetibilidade urbana a enchentes, com avaliacao observacional review-only. Nao e
predicao operacional, nao e ground truth patch-level, nao e modelo treinado, nao e
benchmark supervisionado, nao e score_v7, nao e validacao estatistica forte e nao
e sistema de alerta.

Separacao obrigatoria em toda a narrativa: event_record, source_footprint,
derived_patch_link, feature_evidence, score_evaluation.

Guardrails: ground_truth=false, eligible_for_training=false, score_v7_allowed=false,
sem benchmark 17B, score_v6 intacto. Background e unlabeled_background (nunca
negativo). Petropolis misto/contextual nunca aparece como observado. SAR e
pos-evento e nao e geometria oficial. hit-rate e exploratorio review-only, nunca
benchmark. n=7 nao sustenta conclusao estatistica forte.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

from susc_io import (  # noqa: E402
    ensure_dir,
    read_csv,
    read_json,
    rel,
    write_csv,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "data" / "linhagem_anterior" / "susc_19e_pacote_comunicacao_cientifica_review_only"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"
FIGURES_DIR = ROOT / "outputs_public" / "figures"

REPORT = REPORTS / "SUSC_19E_PACOTE_COMUNICACAO_CIENTIFICA_REVIEW_ONLY.md"
SCHEMA = SCHEMAS_DIR / "susc_19e_comunicacao_cientifica_schema_v1.json"

SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"

DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior"
SUM_18H = DATA / "susc_18h_consolidacao_mestre_cadeia_observacional" / "summary.json"
REG_18H = DATA / "susc_18h_consolidacao_mestre_cadeia_observacional" / "matriz_regional_recife_curitiba_petropolis.csv"
SUM_19A = DATA / "susc_19a_matriz_multimodal_escalavel_por_patch" / "summary.json"
SUM_19B = DATA / "susc_19b_auditoria_lacunas_territoriais" / "summary.json"
SUM_19C = DATA / "susc_19c_avaliacao_observacional_review_only" / "summary.json"
SUM_19D = DATA / "susc_19d_diagnostico_divergencias_score_v6_review_only" / "summary.json"

# Saidas (markdown)
SINTESE = OUT / "sintese_executiva_rev_p_19e.md"
ART_METODO = OUT / "blocos_artigo_metodologia_19e.md"
ART_RESULT = OUT / "blocos_artigo_resultados_19e.md"
SLIDES = OUT / "roteiro_slides_rev_p_19e.md"
BANCA = OUT / "perguntas_respostas_banca_19e.md"

# Saidas (csv)
CLAIMS = OUT / "politica_claims_rev_p_19e.csv"
TAB_REGIAO = OUT / "tabela_estado_por_regiao_19e.csv"
TAB_COBERTURA = OUT / "tabela_cobertura_multimodal_19e.csv"
TAB_RESULTADOS = OUT / "tabela_resultados_observacionais_19e.csv"
TAB_BLOQUEIOS = OUT / "tabela_bloqueios_score_v7_19e.csv"
TAB_PROXIMOS = OUT / "tabela_proximos_passos_19e.csv"
PLANO_FIGURAS = OUT / "plano_figuras_artigo_slides_19e.csv"
GLOSSARIO = OUT / "glossario_metodologico_rev_p_19e.csv"
RISCOS = OUT / "matriz_riscos_comunicacao_19e.csv"
GATE_COM = OUT / "gate_comunicacao_cientifica_19e.csv"
GATE_ART = OUT / "gate_artigo_slides_pos_19e.csv"
RESUMO_SECAO = OUT / "resumo_por_secao.csv"
RESUMO_CLAIM = OUT / "resumo_por_claim.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"

REQUIRED_INPUTS = [SUM_18H, REG_18H, SUM_19A, SUM_19C, SUM_19D]
REQUIRED_OUTPUTS = [
    SINTESE, ART_METODO, ART_RESULT, SLIDES, BANCA,
    CLAIMS, TAB_REGIAO, TAB_COBERTURA, TAB_RESULTADOS, TAB_BLOQUEIOS, TAB_PROXIMOS,
    PLANO_FIGURAS, GLOSSARIO, RISCOS, GATE_COM, GATE_ART, RESUMO_SECAO, RESUMO_CLAIM,
    SUMMARY, PREFLIGHT, SCHEMA, REPORT,
]

# ---------------------------------------------------------------------------
# Guardrails / constantes
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)
STRONG_STATS_RE = re.compile(r"p[-_ ]?valor|p[-_ ]?value|significancia estatistica|estatisticamente significativo", re.IGNORECASE)
# Marcadores que enquadram uma frase como errada/proibida/negada (permitem citar a frase perigosa).
NEG_MARKER_RE = re.compile(r"(proib|errad|incorret|ressalva|bloquead|nunca|risco|fals|\bnao\b|n[aã]o[ _]?[eé]|sem[ _]|deve[ _]dizer|corre[cç])", re.IGNORECASE)

# Padroes perigosos: so podem aparecer se a mesma linha estiver enquadrada como erro/negacao.
DANGER_PATTERNS = {
    "background_negativo": re.compile(r"background[^.\n]{0,25}negativ", re.IGNORECASE),
    "sar_geometria_oficial": re.compile(r"sar[^.\n]{0,30}geometria oficial", re.IGNORECASE),
    "previsao_operacional": re.compile(r"(prev[eê]|predi[çc]|previs[aã]o)[^.\n]{0,30}enchente", re.IGNORECASE),
    "hitrate_benchmark": re.compile(r"hit[- ]?rate[^.\n]{0,25}benchmark", re.IGNORECASE),
    "ground_truth_patch": re.compile(r"ground truth[^.\n]{0,20}patch", re.IGNORECASE),
}

STATUS_19E_COM_ALLOWED = {
    "19E_COMUNICACAO_REVIEW_ONLY_PRONTA",
    "19E_COMUNICACAO_PRONTA_COM_RESSALVAS",
    "19E_BLOQUEADO_POR_LACUNA_NARRATIVA",
    "19E_BLOQUEADO_FAIL_CLOSED",
}
STATUS_ART_ALLOWED = {
    "ARTIGO_SLIDES_PRONTOS_PARA_REDACAO",
    "ARTIGO_SLIDES_PRONTOS_COM_RESSALVAS",
    "ARTIGO_SLIDES_BLOQUEADOS_POR_FIGURAS",
    "ARTIGO_SLIDES_BLOQUEADOS_POR_LACUNAS",
}
CLAIM_STATUS_ALLOWED = {
    "permitido", "permitido_com_ressalva", "proibido", "bloqueado_por_evidencia_insuficiente",
}

SEPARACAO = ["event_record", "source_footprint", "derived_patch_link", "feature_evidence", "score_evaluation"]


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def git_ignored(paths: list[Path]) -> set[str]:
    if not paths:
        return set()
    rels = [rel(p) for p in paths]
    r = subprocess.run(
        ["git", "check-ignore", "-z", "--stdin"], cwd=ROOT,
        input=("\0".join(rels) + "\0").encode("utf-8"), capture_output=True, check=False,
    )
    out = r.stdout.decode("utf-8", errors="ignore")
    return {piece.replace("\\", "/") for piece in out.split("\0") if piece.strip()}


def require_inputs() -> None:
    missing = [rel(p) for p in REQUIRED_INPUTS if not p.exists()]
    if missing:
        raise FileNotFoundError("entradas obrigatorias ausentes: " + "; ".join(missing))


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.1f}%"
    except (TypeError, ValueError):
        return "NA"


# ---------------------------------------------------------------------------
# Contexto herdado (le summaries, nao regenera)
# ---------------------------------------------------------------------------
def load_context() -> dict:
    s19a = read_json(SUM_19A)
    s19c = read_json(SUM_19C)
    s19d = read_json(SUM_19D)
    s18h = read_json(SUM_18H)
    reg = {r["regiao"]: r for r in read_csv(REG_18H)}
    return {
        "s18h": s18h, "s19a": s19a, "s19c": s19c, "s19d": s19d, "reg_18h": reg,
        "total_patches": s19a.get("total_patches", 300),
        "cov_fisico": s19a.get("coverage_fisico_medio", 1.0),
        "cov_territorial": s19a.get("coverage_territorial_medio", 0.3333),
        "cov_espectral": s19a.get("coverage_espectral_medio", 1.0),
        "cov_chuva": s19a.get("coverage_chuva_medio", 1.0),
        "cov_documental": s19a.get("coverage_documental_medio", 0.02),
        "cov_observacional": s19a.get("coverage_observacional_medio", 0.0267),
        "observed": s19c.get("observed_patches", 7),
        "obs_recife": s19c.get("observed_recife", 5),
        "obs_curitiba": s19c.get("observed_curitiba", 2),
        "background": s19c.get("background_patches", 293),
        "score_obs": s19c.get("score_medio_observado", 0.5958),
        "score_bg": s19c.get("score_medio_background", 0.5208),
        "top30_global": s19c.get("observed_in_top_30_global", 0),
        "top30_regional": s19c.get("observed_in_top_30_regional", 3),
        "familias_aderentes": s19d.get("familias_aderentes", []),
        "familias_divergentes": s19d.get("familias_divergentes", []),
        "divergencia_forte": s19d.get("divergencia_forte", "curitiba_01101"),
    }


# ---------------------------------------------------------------------------
# 3. Politica de claims
# ---------------------------------------------------------------------------
def claims_rows() -> list[dict]:
    permitidos = [
        ("CLM_P01", "O REV-P estima suscetibilidade urbana a enchentes por patch (score_v6 candidato).",
         "score_v6 documentado no contrato 17c35; suscetibilidade nao e previsao de evento",
         "susc_score_v6_candidate_by_patch_v1.csv", "descricao_de_suscetibilidade_review_only"),
        ("CLM_P02", "O framework integra features fisicas, hidrologicas, espectrais, territoriais, de chuva e evidencia documental.",
         "matriz multimodal 19A com 6 familias; feature_evidence separada de score_evaluation",
         "susc_19a_matriz_multimodal_escalavel_por_patch", "descricao_do_pipeline_multimodal"),
        ("CLM_P03", "A avaliacao observacional e review-only, sem treino e sem benchmark.",
         "avaliacao 19C exploratoria; background e unlabeled_background",
         "susc_19c_avaliacao_observacional_review_only", "descricao_da_avaliacao_review_only"),
        ("CLM_P04", "O diagnostico 19D identifica divergencias do score_v6 sem alterar o score.",
         "score_v6 intacto; diagnostico por familia e por patch",
         "susc_19d_diagnostico_divergencias_score_v6_review_only", "descricao_do_diagnostico_review_only"),
        ("CLM_P05", "O score_v7 permanece bloqueado ate ampliar amostra, preencher territorial e ter benchmark.",
         "gates de prontidao 19C/19D; score_v7 nao autorizado",
         "gate_prontidao_score_v7_pos_19d.csv", "descricao_de_bloqueio_metodologico"),
    ]
    proibidos = [
        ("CLM_X01", "O REV-P preve enchentes operacionalmente.", "proibido",
         "nao ha modelo preditivo operacional nem sistema de alerta; suscetibilidade nao e previsao",
         "enquadramento_cientifico_rev_p", "nunca_usar_este_claim"),
        ("CLM_X02", "O REV-P possui ground truth patch-level.", "proibido",
         "nao ha verdade de campo por patch; evidencia e review-only sem geometria de ocorrencia confirmada",
         "susc_gt01_ground_reference_policy", "nunca_usar_este_claim"),
        ("CLM_X03", "O REV-P treina um modelo supervisionado.", "proibido",
         "eligible_for_training=false em toda a cadeia; nao ha treino supervisionado",
         "guardrails_cadeia_susc", "nunca_usar_este_claim"),
        ("CLM_X04", "O REV-P validou estatisticamente com amostra suficiente.", "proibido",
         "apenas 7 patches observacionais; sem conclusao estatistica forte",
         "susc_19c_avaliacao_observacional_review_only", "nunca_usar_este_claim"),
        ("CLM_X05", "O REV-P tem um benchmark 17B.", "proibido",
         "benchmark_17b_criado=false; 17B nao criado",
         "gate_prontidao_17b_pos_19d.csv", "nunca_usar_este_claim"),
        ("CLM_X06", "A referencia SAR de Curitiba e geometria oficial de ocorrencia.", "proibido",
         "SAR e footprint pos-evento; nao e geometria oficial; Curitiba aguarda resposta oficial 18D",
         "susc_18h_consolidacao_mestre_cadeia_observacional", "nunca_usar_este_claim"),
        ("CLM_X07", "O universo sem evidencia documentada e um conjunto negativo.", "proibido",
         "background e unlabeled_background; ausencia de evidencia nao e evidencia de ausencia; nunca e negativo",
         "susc_19c_avaliacao_observacional_review_only", "nunca_usar_este_claim"),
    ]
    ressalva = [
        ("CLM_R01", "Areas com evidencia observacional review-only tendem a ter score_v6 maior que o universo nao rotulado.",
         "permitido_com_ressalva",
         "valido como leitura exploratoria review-only (score obs > background) mas 0/7 no top-30 global e n pequeno",
         "susc_19c_avaliacao_observacional_review_only", "usar_sempre_com_ressalva_de_amostra_e_review_only"),
        ("CLM_R02", "O sinal urbano e topografico e coerente com maior suscetibilidade nos observados.",
         "permitido_com_ressalva",
         "coerencia por familia (19D) valida como leitura exploratoria; divergencia hidrologica/chuva no mesmo conjunto",
         "susc_19d_diagnostico_divergencias_score_v6_review_only", "usar_com_ressalva_de_divergencia_e_amostra"),
    ]
    bloqueado = [
        ("CLM_B01", "Curitiba e a segunda regiao de referencia forte confirmada.",
         "bloqueado_por_evidencia_insuficiente",
         "Curitiba e apenas tecnica SAR sem geometria oficial; nao ha referencia forte confirmada",
         "susc_18h_consolidacao_mestre_cadeia_observacional", "so_apos_geometria_oficial_18D"),
        ("CLM_B02", "Petropolis e uma regiao observacional avaliada.",
         "bloqueado_por_evidencia_insuficiente",
         "Petropolis e fenomeno misto sem separacao; permanece bloqueado e fora de observado",
         "susc_19c_avaliacao_observacional_review_only", "so_apos_separacao_de_fenomeno"),
    ]
    rows = []
    for cid, frase, justif, fonte, uso in permitidos:
        rows.append({"claim_id": cid, "frase": frase, "status": "permitido",
                     "justificativa": justif, "fonte_interna": fonte, "uso_permitido": uso})
    for cid, frase, status, justif, fonte, uso in ressalva:
        rows.append({"claim_id": cid, "frase": frase, "status": status,
                     "justificativa": justif, "fonte_interna": fonte, "uso_permitido": uso})
    for cid, frase, status, justif, fonte, uso in proibidos:
        rows.append({"claim_id": cid, "frase": frase, "status": status,
                     "justificativa": justif, "fonte_interna": fonte, "uso_permitido": uso})
    for cid, frase, status, justif, fonte, uso in bloqueado:
        rows.append({"claim_id": cid, "frase": frase, "status": status,
                     "justificativa": justif, "fonte_interna": fonte, "uso_permitido": uso})
    return rows


# ---------------------------------------------------------------------------
# 7. Tabelas finais
# ---------------------------------------------------------------------------
def tabela_regiao_rows(ctx: dict) -> list[dict]:
    reg = ctx["reg_18h"]
    papel = {
        "recife": "referencia_forte_review_only",
        "curitiba": "segunda_regiao_tecnica_sar",
        "petropolis": "bloqueada_fenomeno_misto",
    }
    out = []
    for r in ("recife", "curitiba", "petropolis"):
        m = reg.get(r, {})
        out.append({
            "regiao": r, "papel": papel[r],
            "eventos_fortes": m.get("eventos_fortes", "0"),
            "eventos_tecnicos": m.get("eventos_tecnicos", "0"),
            "eventos_bloqueados": m.get("eventos_bloqueados", "0"),
            "patch_links_fortes": m.get("patch_links_fortes", "0"),
            "status_regional": m.get("status_regional", "NA"),
            "bloqueio_principal": m.get("bloqueio_principal", "NA"),
            "proxima_acao": m.get("proxima_acao", "NA"),
        })
    return out


def tabela_cobertura_rows(ctx: dict) -> list[dict]:
    familias = [
        ("fisica_topografica", ctx["cov_fisico"], "completa", "nenhuma"),
        ("espectral_umidade", ctx["cov_espectral"], "completa", "nenhuma"),
        ("chuva_hidrometeorologica", ctx["cov_chuva"], "completa", "nenhuma"),
        ("territorial", ctx["cov_territorial"], "parcial",
         "MapBiomas_class;exposed_soil_prop;water_prop;impervious_proxy (missingness herdado do 19B)"),
        ("documental", ctx["cov_documental"], "rara", "documental so em Recife e contexto misto em Petropolis"),
        ("observacional", ctx["cov_observacional"], "rara", "evidencia so em Recife e SAR de Curitiba"),
    ]
    return [{
        "familia": f, "cobertura_media": f"{c:.4f}", "cobertura_pct": _pct(c),
        "status_cobertura": st, "principal_lacuna": lac,
        "observacao": "cobertura nao e score de suscetibilidade",
    } for f, c, st, lac in familias]


def tabela_resultados_rows(ctx: dict) -> list[dict]:
    ader = ";".join(ctx["familias_aderentes"]) or "nenhuma"
    diver = ";".join(ctx["familias_divergentes"]) or "nenhuma"
    data = [
        ("patches_observacionais", str(ctx["observed"]), "global", "Recife 5 + Curitiba 2; review-only"),
        ("patches_recife", str(ctx["obs_recife"]), "recife", "canarios observacionais review-only"),
        ("patches_curitiba", str(ctx["obs_curitiba"]), "curitiba", "overlays tecnicos SAR review-only"),
        ("background_nao_rotulado", str(ctx["background"]), "global", "unlabeled_background; nunca negativo"),
        ("score_medio_observado", f"{ctx['score_obs']:.3f}", "global", "leitura exploratoria review-only"),
        ("score_medio_background", f"{ctx['score_bg']:.3f}", "global", "unlabeled_background; nunca negativo"),
        ("observados_top30_global", f"{ctx['top30_global']}/{ctx['observed']}", "global", "nenhum no extremo do ranking"),
        ("observados_top30_regional", f"{ctx['top30_regional']}/{ctx['observed']}", "regional", "hit-rate exploratorio review-only, nao e benchmark"),
        ("familias_que_sustentam_aderencia", ader, "global", "coerencia review-only"),
        ("familias_que_derrubam_observados", diver, "global", "divergencia review-only"),
        ("divergencia_forte", ctx["divergencia_forte"], "curitiba", "score_v6 baixo apesar de evidencia review-only"),
    ]
    return [{"metrica": m, "valor": v, "escopo": e, "interpretacao_review_only": i} for m, v, e, i in data]


def tabela_bloqueios_rows() -> list[dict]:
    data = [
        ("amostra_minima", "apenas 7 patches observacionais; 0 no top-30 global", "19C", "SCORE_V7_BLOQUEADO_POR_AMOSTRA"),
        ("missingness_territorial", "faltam MapBiomas/solo exposto/agua/impervious nos 300 patches", "19B", "SCORE_V7_BLOQUEADO_POR_MISSINGNESS_TERRITORIAL"),
        ("ausencia_benchmark", "sem benchmark 17B supervisionado ou observacional consolidado", "19D", "SCORE_V7_BLOQUEADO_POR_AUSENCIA_BENCHMARK"),
        ("geometria_oficial_curitiba", "Curitiba so tem SAR pos-evento; sem geometria oficial de ocorrencia", "18D/18H", "SCORE_V7_NAO_AUTORIZADO"),
        ("fenomeno_misto_petropolis", "Petropolis misto deslizamento/inundacao sem separacao", "18H/19C", "SCORE_V7_NAO_AUTORIZADO"),
    ]
    return [{"bloqueio": b, "descricao": d, "origem": o, "status_score_v7": s} for b, d, o, s in data]


def tabela_proximos_rows() -> list[dict]:
    data = [
        ("ampliar_amostra_observacional", "mais eventos com geometria oficial nas 3 regioes", "geometria oficial (fila 17B/GT)", "alta", "nao cria score_v7 agora"),
        ("preencher_territorial", "executar pacote MapBiomas/GEE preparado no 19B", "acesso GEE/MapBiomas", "alta", "nao calibra sobre missingness"),
        ("separar_fenomeno_petropolis", "separar deslizamento de inundacao e buscar evento 2024", "geometria de ocorrencia", "media", "nao promove Petropolis misto"),
        ("consolidar_geometria_curitiba", "ingerir resposta oficial 18D e consolidar referencia tecnica SAR", "resposta oficial 18D", "media", "SAR nao vira geometria oficial"),
        ("comunicacao_cientifica", "redigir artigo/slides review-only com ressalvas", "este pacote 19E", "alta", "sem overclaim, sem previsao"),
    ]
    return [{"passo": p, "descricao": d, "pre_requisito": pr, "prioridade": pri, "nao_faz_agora": n}
            for p, d, pr, pri, n in data]


# ---------------------------------------------------------------------------
# 8. Plano de figuras
# ---------------------------------------------------------------------------
def _fig_exists(name: str) -> str:
    p = FIGURES_DIR / name
    return name if p.exists() else "nenhum_reuso_direto"


def plano_figuras_rows() -> list[dict]:
    data = [
        ("FIG_01", "Pipeline geral do REV-P", "mostrar Sentinel-first -> features multimodais -> score_v6 -> avaliacao review-only",
         "narrativa 19A/19D", "fig_methodological_contribution_matrix.png", "true", "alta", "true", "true"),
        ("FIG_02", "Matriz multimodal por patch", "6 familias de features por patch nos 300 patches",
         "susc_19a matriz_multimodal_escalavel_por_patch.csv", _fig_exists("fig_local_context_coverage.png"), "true", "alta", "true", "true"),
        ("FIG_03", "Separacao event_record/footprint/patch_link/score_evaluation", "explicitar as 5 camadas separadas",
         "narrativa 19E", "nenhum_reuso_direto", "true", "alta", "true", "true"),
        ("FIG_04", "Mapa conceitual por regiao", "Recife forte / Curitiba tecnica SAR / Petropolis bloqueado",
         "susc_18h matriz_regional", _fig_exists("fig_regional_roles_summary.png"), "false", "media", "true", "true"),
        ("FIG_05", "Barras de cobertura multimodal", "cobertura por familia (fisico/espectral/chuva completos; territorial 0.33)",
         "susc_19a summary.json", _fig_exists("fig_evidence_layer_availability.png"), "true", "alta", "true", "true"),
        ("FIG_06", "Ranking score_v6 observacional", "posicao dos 7 observados no ranking (0/7 top-30 global)",
         "susc_19c ranking_score_v6_observacional_19c.csv", "nenhum_reuso_direto", "true", "alta", "true", "true"),
        ("FIG_07", "Contraste de features observacional", "aderencia topo/urbano/espectral e divergencia hidrologia/chuva",
         "susc_19c contraste_features_observacional_19c.csv; susc_19d decomposicao_familias", "nenhum_reuso_direto", "true", "alta", "true", "true"),
        ("FIG_08", "Bloqueios do score_v7", "amostra, missingness territorial, ausencia de benchmark",
         "susc_19d gate_prontidao_score_v7_pos_19d.csv", "nenhum_reuso_direto", "false", "media", "true", "true"),
    ]
    return [{
        "figura_id": fid, "titulo": t, "objetivo": o, "fonte_dados": fd,
        "arquivo_existente_sugerido": ex, "precisa_criar": pc, "prioridade": pr,
        "uso_artigo": ua, "uso_slide": us,
    } for fid, t, o, fd, ex, pc, pr, ua, us in data]


# ---------------------------------------------------------------------------
# 9. Glossario
# ---------------------------------------------------------------------------
def glossario_rows() -> list[dict]:
    data = [
        ("suscetibilidade", "propensao estrutural de um local a um fenomeno",
         "score_v6 candidato por patch a partir de features do terreno e contexto",
         "confundir com probabilidade de evento ou previsao",
         "estima suscetibilidade urbana a enchentes (nao preve enchentes)"),
        ("alagamento", "acumulo temporario de agua por drenagem insuficiente",
         "fenomeno urbano alvo em Recife/Curitiba, distinto de inundacao fluvial",
         "tratar alagamento e inundacao como sinonimos",
         "alagamento urbano review-only"),
        ("inundacao", "transbordamento de corpo d'agua sobre a planicie",
         "fenomeno distinto do alagamento; relevante na separacao de Petropolis",
         "misturar inundacao com deslizamento",
         "inundacao separada de deslizamento"),
        ("patch", "recorte espacial fixo usado como unidade de analise",
         "unidade de features e de score_v6; 300 patches (100 por regiao)",
         "tratar patch como ocorrencia confirmada",
         "unidade de analise por patch"),
        ("score_v6", "indice candidato de suscetibilidade por patch",
         "robust_minmax de combinacao linear ponderada de 5 componentes (contrato 17c35)",
         "chamar de previsao ou de probabilidade validada",
         "score_v6 candidato review-only"),
        ("score_v7", "hipotetico score futuro recalibrado",
         "nao criado; bloqueado por amostra, missingness territorial e ausencia de benchmark",
         "apresentar score_v7 como existente",
         "score_v7 permanece bloqueado"),
        ("review-only", "uso restrito a revisao, sem operacao nem treino",
         "modo de toda a avaliacao observacional; nao habilita treino/GT/score_v7",
         "ler review-only como validacao operacional",
         "avaliacao observacional review-only"),
        ("ground truth", "verdade de campo confirmada por observacao direta",
         "nao existe patch-level no REV-P; evidencia e review-only sem geometria confirmada",
         "chamar canario ou SAR de ground truth",
         "nao ha ground truth patch-level"),
        ("referencia observacional", "evidencia de ocorrencia usada so para revisao",
         "canarios de Recife e overlays SAR de Curitiba; nunca label nem target",
         "usar referencia como rotulo de treino",
         "referencia observacional review-only"),
        ("footprint", "area atingida derivada de sensor pos-evento",
         "footprint SAR e pos-evento; nao e feature pre-evento nem geometria oficial",
         "usar footprint como feature preditiva pre-evento",
         "footprint pos-evento review-only"),
        ("patch_link", "vinculo derivado entre evidencia e patch",
         "derived_patch_link; separado de event_record e de score_evaluation",
         "tratar patch_link como ocorrencia oficial por patch",
         "derived_patch_link review-only"),
        ("background/unlabeled", "patches sem evidencia documentada",
         "unlabeled_background; ausencia de evidencia nao e evidencia de ausencia",
         "chamar background de conjunto negativo",
         "unlabeled_background (nunca negativo)"),
        ("hard negative auditado", "negativo confirmado por auditoria explicita",
         "estado raro e nao usado; nao ha negativos no REV-P",
         "inferir negativo a partir de ausencia de evidencia",
         "sem hard negative; ausencia nao e negativo"),
        ("feature pre-evento", "atributo do terreno anterior ao evento",
         "base do score_v6; separada de footprint pos-evento",
         "misturar feature pre-evento com footprint pos-evento",
         "feature pre-evento separada do footprint"),
        ("footprint pos-evento", "extensao observada apos o evento",
         "usado so como referencia observacional review-only",
         "usar footprint pos-evento como preditor",
         "footprint pos-evento como referencia review-only"),
        ("vazamento temporal", "uso de informacao do futuro no calculo",
         "controlado: features pre-evento nao incorporam footprint pos-evento",
         "ignorar a separacao temporal entre feature e footprint",
         "controle de vazamento temporal explicito"),
        ("MapBiomas", "base de uso e cobertura da terra",
         "lacuna territorial: classe/solo/agua/impervious nao materializados nos 300",
         "afirmar cobertura territorial completa",
         "lacuna territorial MapBiomas explicita"),
        ("HAND", "altura acima da drenagem mais proxima",
         "feature hidrologica; parte da familia fisica_topografica/hidrologica",
         "interpretar HAND como probabilidade de enchente",
         "HAND como feature topografica-hidrologica"),
        ("TWI", "indice topografico de umidade",
         "feature hidrologica; divergente nos observados (19D)",
         "usar TWI natural como drenagem urbana",
         "TWI como indice de umidade topografica"),
        ("MNDWI", "indice de agua modificado",
         "feature espectral de umidade; coerente nos observados",
         "ler MNDWI como deteccao de enchente",
         "MNDWI como indice espectral de umidade"),
        ("NDBI", "indice de area construida",
         "feature espectral urbana; coerente nos observados",
         "ler NDBI como risco confirmado",
         "NDBI como indice de area construida"),
        ("SAR", "radar de abertura sintetica",
         "footprint pos-evento em Curitiba; nao e geometria oficial",
         "chamar SAR de geometria oficial de ocorrencia",
         "SAR pos-evento (nao e geometria oficial)"),
    ]
    return [{
        "termo": t, "definicao_curta": dc, "definicao_no_rev_p": dr,
        "erro_comum": ec, "frase_segura_para_usar": fs,
    } for t, dc, dr, ec, fs in data]


# ---------------------------------------------------------------------------
# 11. Matriz de riscos
# ---------------------------------------------------------------------------
def riscos_rows() -> list[dict]:
    data = [
        ("overclaim_de_previsao", "O REV-P preve enchentes em tempo real.",
         "O REV-P estima suscetibilidade urbana review-only; nao preve enchentes.",
         "alta", "previsao_operacional_exige_negacao_na_linha"),
        ("ground_truth_falso", "Cada patch tem ground truth de enchente.",
         "Nao ha ground truth patch-level; a evidencia e review-only.",
         "alta", "ground_truth_patch_exige_negacao_na_linha"),
        ("negativo_falso", "O background e o conjunto de patches negativos.",
         "O background e unlabeled_background e nunca e negativo.",
         "alta", "background_negativo_exige_negacao_na_linha"),
        ("sar_como_geometria_oficial", "O SAR de Curitiba e a geometria oficial de ocorrencia.",
         "O SAR e footprint pos-evento e nao e geometria oficial.",
         "alta", "sar_geometria_oficial_exige_negacao_na_linha"),
        ("score_v7_implicito", "A nova calibracao (score_v7) ja esta pronta.",
         "O score_v7 nao foi criado e permanece bloqueado.",
         "media", "score_v7_created_deve_ser_false"),
        ("estatistica_forte_n7", "O resultado e estatisticamente significativo.",
         "Com 7 patches nao ha conclusao estatistica forte; leitura exploratoria review-only.",
         "alta", "strong_stats_re_bloqueia_texto_publico"),
        ("petropolis_promovido", "Petropolis e uma regiao observacional avaliada.",
         "Petropolis e fenomeno misto sem separacao e permanece bloqueado, fora de observado.",
         "media", "petropolis_nao_pode_aparecer_como_observado"),
        ("missingness_omitido", "A cobertura territorial do REV-P e completa.",
         "Falta cobertura territorial (MapBiomas/solo/agua/impervious); missingness explicito.",
         "media", "missingness_territorial_deve_aparecer"),
    ]
    return [{
        "risco": r, "exemplo_de_frase_errada": e, "correcao": c,
        "severidade": s, "validator_rule": v,
    } for r, e, c, s, v in data]


# ---------------------------------------------------------------------------
# 12. Gates
# ---------------------------------------------------------------------------
def gate_comunicacao_rows(summary: dict) -> list[dict]:
    status = summary["status_19e_comunicacao"]
    data = [
        ("narrativa_consolidada", "true", "true", "sintese, metodologia e resultados gerados"),
        ("claims_policy_definida", "true", "true", "permitidos e proibidos explicitos"),
        ("glossario_completo", "true", "true", "22 termos metodologicos"),
        ("score_v6_intacto", str(not summary["score_v6_changed"]).lower(), "true", "score_v6 nao alterado"),
        ("ressalvas_amostra_e_missingness", "true", "true", "amostra minima e missingness territorial explicitos"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": l, "status_19e": status, "observacao": o}
            for c, v, l, o in data]


def gate_artigo_rows(summary: dict) -> list[dict]:
    status = summary["status_artigo_slides"]
    data = [
        ("blocos_artigo_prontos", "true", "true", "metodologia e resultados redigiveis"),
        ("roteiro_slides_pronto", "true", "true", "15 slides com limitacoes e proximos passos"),
        ("figuras_criticas_pendentes", "true", "false", "algumas figuras precisam ser criadas (plano definido)"),
        ("tabelas_finais_prontas", "true", "true", "5 tabelas finais geradas"),
    ]
    return [{"criterio": c, "valor_observado": v, "limiar": l, "status_artigo_slides": status, "observacao": o}
            for c, v, l, o in data]


# ---------------------------------------------------------------------------
# 6. Roteiro de slides
# ---------------------------------------------------------------------------
def slides_data(ctx: dict) -> list[tuple]:
    return [
        (1, "Problema", "Enchentes urbanas exigem leitura de suscetibilidade, nao previsao",
         "impacto urbano de enchentes;Recife/Curitiba/Petropolis;lacuna de dados observacionais",
         "FIG_01", "tabela_estado_por_regiao_19e", "Abrir com o problema urbano e o recorte review-only."),
        (2, "O que o REV-P faz", "Framework multimodal auditavel de suscetibilidade urbana review-only",
         "features multimodais por patch;score_v6 candidato;avaliacao observacional review-only",
         "FIG_01", "tabela_cobertura_multimodal_19e", "Enfatizar auditavel e review-only."),
        (3, "O que o REV-P nao promete", "Nao e previsao, nao e ground truth, nao e modelo treinado",
         "sem previsao operacional;sem ground truth patch-level;sem benchmark;sem score_v7;sem alerta",
         "FIG_08", "politica_claims_rev_p_19e", "Deixar claros os limites antes dos resultados."),
        (4, "Pipeline multimodal", "Sentinel-first e 6 familias de features por patch",
         "fisico;hidrologico;espectral;territorial;chuva;documental",
         "FIG_02", "tabela_cobertura_multimodal_19e", "Mostrar o pipeline e a separacao de camadas."),
        (5, "Features por patch", "300 patches com fisico/espectral/chuva completos e territorial parcial",
         f"300 patches;cobertura territorial {_pct(ctx['cov_territorial'])};missingness MapBiomas/solo/agua/impervious",
         "FIG_05", "tabela_cobertura_multimodal_19e", "Explicitar a lacuna territorial."),
        (6, "Evidencia observacional review-only", "7 patches com evidencia review-only, nunca label",
         "event_record;source_footprint;derived_patch_link;feature_evidence;score_evaluation",
         "FIG_03", "tabela_resultados_observacionais_19e", "Explicar a separacao das 5 camadas."),
        (7, "Recife", "Referencia forte review-only de uma regiao e um evento",
         "5 canarios;coerencia urbana/topografica;amostra pequena",
         "FIG_04", "tabela_estado_por_regiao_19e", "Recife e o caso mais solido, ainda review-only."),
        (8, "Curitiba SAR", "Segunda regiao tecnica SAR sem geometria oficial",
         "2 overlays SAR;SAR e pos-evento;nao e geometria oficial",
         "FIG_04", "tabela_estado_por_regiao_19e", "SAR nao e geometria oficial."),
        (9, "Petropolis bloqueado", "Fenomeno misto sem separacao permanece bloqueado",
         "deslizamento e inundacao misturados;fora de observado;29 patches bloqueados",
         "FIG_04", "tabela_estado_por_regiao_19e", "Petropolis nao e promovido."),
        (10, "Matriz multimodal 300 patches", "Consolidacao escalavel por patch",
         "1 linha por patch;6 familias;cobertura nao e score",
         "FIG_02", "tabela_cobertura_multimodal_19e", "Coverage nao e suscetibilidade."),
        (11, "Avaliacao 19C", "Score observado maior que background, mas 0/7 no top-30 global",
         f"score obs {ctx['score_obs']:.3f} vs background {ctx['score_bg']:.3f};0/7 top-30 global;3/7 top-30 regional",
         "FIG_06", "tabela_resultados_observacionais_19e", "hit-rate e exploratorio, nao benchmark."),
        (12, "Diagnostico 19D", "Aderencia urbana/topografica e divergencia hidrologica/chuva",
         "familias aderentes;familias divergentes;curitiba_01101 divergencia forte",
         "FIG_07", "tabela_resultados_observacionais_19e", "Explicar por que os observados nao vao ao extremo."),
        (13, "Limitacoes", "Amostra minima, missingness territorial, sem benchmark, sem geometria oficial",
         "n=7 sem conclusao forte;territorial incompleto;score_v7 bloqueado;background nunca negativo",
         "FIG_08", "tabela_bloqueios_score_v7_19e", "Ser explicito nas limitacoes."),
        (14, "Proximos passos", "Ampliar amostra, preencher territorial, separar fenomeno",
         "mais eventos com geometria oficial;pacote MapBiomas/GEE;separar Petropolis",
         "FIG_08", "tabela_proximos_passos_19e", "Roadmap sem prometer score_v7."),
        (15, "Conclusao", "Contribuicao: framework multimodal auditavel review-only, com diagnostico honesto",
         "suscetibilidade review-only;divergencias documentadas;score_v6 intacto",
         "FIG_01", "tabela_resultados_observacionais_19e", "Fechar com a contribuicao e a honestidade metodologica."),
    ]


def slides_md(ctx: dict) -> str:
    linhas = []
    linhas.append("# Roteiro de slides - REV-P (comunicacao review-only)\n")
    linhas.append("Formato por slide: titulo, mensagem principal, bullets, figura sugerida, "
                  "tabela sugerida e fala do apresentador.\n")
    for num, titulo, msg, bullets, fig, tab, fala in slides_data(ctx):
        linhas.append(f"## Slide {num} - {titulo}\n")
        linhas.append(f"- **Mensagem principal**: {msg}")
        linhas.append("- **Bullets**:")
        for b in bullets.split(";"):
            linhas.append(f"  - {b}")
        linhas.append(f"- **Figura sugerida**: {fig}")
        linhas.append(f"- **Tabela sugerida**: {tab}")
        linhas.append(f"- **Fala do apresentador**: {fala}\n")
    linhas.append("> Todos os slides mantem o enquadramento review-only: sem previsao operacional, "
                  "sem ground truth patch-level, sem modelo treinado, sem benchmark e sem score_v7. "
                  "O background nunca e negativo e o SAR nao e geometria oficial.\n")
    return "\n".join(linhas)


def slides_csv_rows(ctx: dict) -> list[dict]:
    return [{
        "slide_num": str(num), "titulo": titulo, "mensagem_principal": msg,
        "bullets": bullets, "figura_sugerida": fig, "tabela_sugerida": tab,
        "fala_do_apresentador": fala,
    } for num, titulo, msg, bullets, fig, tab, fala in slides_data(ctx)]


# ---------------------------------------------------------------------------
# 2/4/5/10. Narrativas markdown
# ---------------------------------------------------------------------------
def sintese_md(ctx: dict) -> str:
    return f"""# Sintese executiva - REV-P (review-only)

## O que o REV-P e
Framework multimodal auditavel de suscetibilidade urbana a enchentes, com avaliacao
observacional review-only. Integra features fisicas, hidrologicas, espectrais,
territoriais, de chuva e evidencia documental por patch e produz o score_v6
candidato de suscetibilidade.

## O que o REV-P nao e
Nao e predicao operacional de enchente, nao e ground truth patch-level, nao e modelo
treinado, nao e benchmark supervisionado, nao e score_v7 e nao e sistema de alerta.
A avaliacao e review-only e nao substitui validacao operacional.

## Estado atual
{ctx['total_patches']} patches consolidados (100 por regiao). Cobertura fisica,
espectral e de chuva completas; cobertura territorial parcial
({_pct(ctx['cov_territorial'])}) por missingness de MapBiomas/solo exposto/agua/
impermeabilizacao. {ctx['observed']} patches com evidencia observacional review-only
(Recife {ctx['obs_recife']}, Curitiba {ctx['obs_curitiba']}); background nao rotulado
de {ctx['background']} (nunca negativo).

## Principal contribuicao
Um pipeline multimodal auditavel por patch com separacao explicita entre
event_record, source_footprint, derived_patch_link, feature_evidence e
score_evaluation, e um diagnostico honesto das divergencias do score_v6 sem inflar
claims.

## Achados de Recife
Cinco canarios observacionais review-only com score_v6 medio acima do background
regional e coerencia urbana e topografica. Referencia mais solida da cadeia, ainda
assim review-only e de uma unica regiao e um unico evento.

## Achados de Curitiba
Dois overlays tecnicos SAR review-only. O SAR e footprint pos-evento e nao e
geometria oficial. Um patch com score alto e outro (curitiba_01101) com score baixo,
a divergencia observacional mais forte.

## Bloqueio de Petropolis
Fenomeno misto (deslizamento e inundacao) sem separacao. Permanece bloqueado e fora
do conjunto observado; nao e promovido.

## Por que 17B nao existe
Nao ha benchmark 17B: faltam geometria oficial de ocorrencia e eventos suficientes;
Curitiba e apenas tecnica SAR. benchmark_17b_criado=false.

## Por que score_v7 nao existe
O score_v7 permanece bloqueado por amostra minima ({ctx['observed']} observados),
por missingness territorial e por ausencia de benchmark. O score_v6 permanece intacto.
"""


def artigo_metodologia_md(ctx: dict) -> str:
    return f"""# Blocos para artigo - Metodologia (review-only)

## Unidade de analise por patch
A unidade de analise e o patch, um recorte espacial fixo. Sao {ctx['total_patches']}
patches (100 por regiao em Recife, Curitiba e Petropolis). O patch e unidade de
feature_evidence e de score_evaluation; nunca e ocorrencia confirmada.

## Matriz multimodal
Cada patch recebe seis familias de features: fisica_topografica, hidrologica,
urbana/territorial, espectral_umidade, chuva_hidrometeorologica e documental. A
cobertura fisica, espectral e de chuva e completa; a territorial e parcial por
missingness de MapBiomas/solo exposto/agua/impermeabilizacao. Cobertura nao e score
de suscetibilidade.

## score_v6
O score_v6 candidato e robust_minmax de uma combinacao linear ponderada de cinco
componentes (topografia_hidrologia 0.4, chuva 0.25, urbano_espectral 0.2, mitigacao
por vegetacao -0.1, evidencia documental 0.05), conforme o contrato 17c35. E um
indice de suscetibilidade, nao uma previsao de evento.

## Evidencia observacional
A evidencia observacional e review-only: canarios documentais em Recife e overlays
tecnicos SAR em Curitiba. Separa-se explicitamente event_record, source_footprint,
derived_patch_link, feature_evidence e score_evaluation. O footprint SAR e
pos-evento e nao e feature pre-evento nem geometria oficial.

## Avaliacao review-only
Os patches observacionais sao comparados ao universo nao rotulado
(unlabeled_background) no score_v6 e nas features. O background nunca e negativo:
ausencia de evidencia documentada nao e evidencia de ausencia. As metricas de
hit-rate e enrichment sao exploratorias review-only e nunca sao benchmark.

## Controle de vazamento temporal
As features pre-evento nao incorporam o footprint pos-evento. A separacao entre
feature_evidence (pre-evento) e source_footprint (pos-evento) e mantida para evitar
vazamento temporal.

## Limitacoes
Amostra observacional minima ({ctx['observed']} patches); missingness territorial;
ausencia de benchmark; ausencia de geometria oficial em Curitiba; fenomeno misto em
Petropolis. Com {ctx['observed']} patches nao ha conclusao estatistica forte.

## Proximos passos
Ampliar a amostra observacional com geometria oficial, preencher a cobertura
territorial (pacote MapBiomas/GEE), separar o fenomeno de Petropolis e consolidar a
geometria de Curitiba. Nenhum passo cria score_v7, benchmark, treino ou ground truth
agora.
"""


def artigo_resultados_md(ctx: dict) -> str:
    ader = ", ".join(ctx["familias_aderentes"]) or "nenhuma"
    diver = ", ".join(ctx["familias_divergentes"]) or "nenhuma"
    return f"""# Blocos para artigo - Resultados (review-only)

## Consolidacao multimodal
Foram consolidados {ctx['total_patches']} patches (100 por regiao). As familias
fisica, espectral e de chuva tem cobertura completa; a familia territorial tem
cobertura parcial ({_pct(ctx['cov_territorial'])}) por lacuna de MapBiomas, solo
exposto, agua e impermeabilizacao.

## Amostra observacional
{ctx['observed']} patches com evidencia observacional review-only (Recife
{ctx['obs_recife']}, Curitiba {ctx['obs_curitiba']}); background nao rotulado de
{ctx['background']}. Petropolis fica fora, como fenomeno misto bloqueado.

## score_v6 observacional
O score_v6 medio dos observados e {ctx['score_obs']:.3f} contra {ctx['score_bg']:.3f}
do background. Ainda assim, {ctx['top30_global']}/{ctx['observed']} observados estao
no top-30 global e {ctx['top30_regional']}/{ctx['observed']} no top-30 regional. O
hit-rate e exploratorio review-only e nao e benchmark.

## Aderencia e divergencia
Familias que sustentam a aderencia: {ader}. Familias que derrubam os observados:
{diver}. Ha coerencia urbana e topografica e divergencia hidrologica e de chuva
antecedente.

## Divergencia relevante
O patch {ctx['divergencia_forte']} tem score_v6 baixo apesar da evidencia
observacional review-only, sendo a divergencia mais forte; permanece baixo nos
cenarios de sensibilidade nao oficiais.

## Bloqueio do score_v7
O score_v7 permanece bloqueado por amostra minima, missingness territorial e ausencia
de benchmark. O score_v6 permanece intacto e nenhum score_v7 e criado.
"""


def banca_md(ctx: dict) -> str:
    return f"""# Perguntas e respostas de banca - REV-P (review-only)

## Por que nao e ground truth?
Porque nao ha verdade de campo confirmada por patch. A evidencia e review-only, sem
geometria de ocorrencia confirmada; canarios e footprints SAR nao sao ground truth
patch-level.

## Por que nao e previsao?
Porque o REV-P estima suscetibilidade estrutural, nao a ocorrencia de um evento no
tempo. Nao ha modelo preditivo operacional nem sistema de alerta.

## Por que nao treinou modelo?
Porque nao ha rotulos nem targets: eligible_for_training=false em toda a cadeia. A
avaliacao e review-only e nao supervisionada.

## Por que usar SAR?
O SAR fornece footprint pos-evento em Curitiba como referencia observacional
review-only, util para revisao. Ele nao entra como feature pre-evento.

## Por que Curitiba SAR nao e oficial?
Porque o footprint SAR e derivado de sensor pos-evento e nao e geometria oficial de
ocorrencia; Curitiba aguarda resposta oficial (18D).

## Por que Petropolis ficou bloqueado?
Porque o fenomeno e misto (deslizamento e inundacao) sem separacao e sem geometria
forte. Permanece bloqueado e fora do conjunto observado.

## Por que score_v7 nao foi criado?
Porque ha bloqueio por amostra minima ({ctx['observed']} observados), por missingness
territorial e por ausencia de benchmark. Criar score_v7 agora seria overclaim.

## O que significa review-only?
Significa uso restrito a revisao: a evidencia e as metricas servem para inspecao e
diagnostico, nao para treino, ground truth, score_v7 ou operacao.

## O que o REV-P entrega de util?
Um pipeline multimodal auditavel por patch, com separacao explicita das camadas e um
diagnostico honesto das divergencias do score_v6, reproduzivel e sem overclaim.

## Qual e a principal limitacao?
A amostra observacional minima ({ctx['observed']} patches) combinada com missingness
territorial e ausencia de benchmark; com {ctx['observed']} patches nao ha conclusao
estatistica forte.

## Qual seria o proximo passo cientifico?
Ampliar a amostra observacional com geometria oficial, preencher a cobertura
territorial e separar o fenomeno de Petropolis, mantendo tudo review-only ate haver
base para benchmark.
"""


# ---------------------------------------------------------------------------
# Resumos
# ---------------------------------------------------------------------------
def resumo_secao_rows() -> list[dict]:
    data = [
        ("sintese_executiva", rel(SINTESE), "narrativa"),
        ("blocos_artigo_metodologia", rel(ART_METODO), "artigo"),
        ("blocos_artigo_resultados", rel(ART_RESULT), "artigo"),
        ("roteiro_slides", rel(SLIDES), "slides"),
        ("perguntas_banca", rel(BANCA), "defesa"),
        ("politica_claims", rel(CLAIMS), "governanca"),
        ("glossario", rel(GLOSSARIO), "governanca"),
        ("matriz_riscos", rel(RISCOS), "governanca"),
        ("plano_figuras", rel(PLANO_FIGURAS), "figuras"),
    ]
    return [{"secao": s, "arquivo": a, "tipo": t, "review_only": "true"} for s, a, t in data]


def resumo_claim_rows(claims: list[dict]) -> list[dict]:
    counts = {}
    for r in claims:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return [{"status": k, "quantidade": str(v)} for k, v in sorted(counts.items())]


# ---------------------------------------------------------------------------
# Schema / preflight / summary
# ---------------------------------------------------------------------------
def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-19E pacote de comunicacao cientifica review-only",
        "type": "object",
        "description": "Comunicacao review-only; nao cria dado/score/benchmark/treino/ground truth; score_v6 intacto; sem overclaim.",
        "enquadramento_rev_p": "framework multimodal auditavel de suscetibilidade urbana a enchentes, com avaliacao observacional review-only",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "benchmark_17b_criado": {"const": "false"},
            "score_v6_changed": {"const": "false"},
            "review_only": {"const": "true"},
        },
        "separacao_obrigatoria": SEPARACAO,
        "status_19e_comunicacao_allowed": sorted(STATUS_19E_COM_ALLOWED),
        "status_artigo_slides_allowed": sorted(STATUS_ART_ALLOWED),
        "claim_status_allowed": sorted(CLAIM_STATUS_ALLOWED),
        "background_label": "unlabeled_background",
    }


def preflight_obj(ctx: dict) -> dict:
    status_full = _run_git(["status", "--short"]).splitlines()
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "status_short_count": len(status_full),
        "entradas_lidas": {rel(p): p.exists() for p in REQUIRED_INPUTS},
        "score_v6_baseline_path": rel(SCORE_V6),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "figuras_existentes": sorted(p.name for p in FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else [],
        "lacunas_herdadas": {
            "cobertura_territorial": ctx["cov_territorial"],
            "observados": ctx["observed"],
            "top30_global": ctx["top30_global"],
            "divergencia_forte": ctx["divergencia_forte"],
        },
        "outputs_ignorados_pelo_git": sorted(git_ignored([OUT])),
        "summary_versionavel": rel(SUMMARY) not in git_ignored([SUMMARY]),
    }


def summary_obj(ctx: dict, claims: list[dict]) -> dict:
    status_counts = {}
    for r in claims:
        status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "total_patches": ctx["total_patches"],
        "observed_patches": ctx["observed"],
        "background_patches": ctx["background"],
        "score_medio_observado": ctx["score_obs"],
        "score_medio_background": ctx["score_bg"],
        "observed_in_top_30_global": ctx["top30_global"],
        "observed_in_top_30_regional": ctx["top30_regional"],
        "claims_total": len(claims),
        "claims_por_status": status_counts,
        "claims_permitidos": status_counts.get("permitido", 0) + status_counts.get("permitido_com_ressalva", 0),
        "claims_proibidos": status_counts.get("proibido", 0),
        "n_slides": 15,
        "n_glossario": len(glossario_rows()),
        "n_figuras_plano": len(plano_figuras_rows()),
        "n_riscos": len(riscos_rows()),
        "status_19e_comunicacao": "19E_COMUNICACAO_PRONTA_COM_RESSALVAS",
        "status_artigo_slides": "ARTIGO_SLIDES_PRONTOS_COM_RESSALVAS",
        "status_score_v7": "SCORE_V7_NAO_AUTORIZADO",
        "status_17b": "17B_NAO_CRIADO",
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "proximo_marco": "SUSC-19F ou execucao das filas (geometria oficial, MapBiomas/GEE)",
    }


# ---------------------------------------------------------------------------
# 17. Relatorio publico
# ---------------------------------------------------------------------------
def report_text(ctx: dict, summary: dict, claims: list[dict], tab_reg: list[dict],
                tab_cob: list[dict], figuras: list[dict]) -> str:
    permit = [c for c in claims if c["status"] in ("permitido", "permitido_com_ressalva")]
    proib = [c for c in claims if c["status"] == "proibido"]
    linhas_permit = "\n".join(f"- {c['frase']} (`{c['status']}`)" for c in permit)
    linhas_proib = "\n".join(f"- {c['frase']} (`proibido`)" for c in proib)
    linhas_reg = "\n".join(
        f"| {r['regiao']} | {r['papel']} | {r['patch_links_fortes']} | {r['bloqueio_principal']} |"
        for r in tab_reg
    )
    linhas_cob = "\n".join(
        f"| {r['familia']} | {r['cobertura_pct']} | {r['status_cobertura']} | {r['principal_lacuna']} |"
        for r in tab_cob
    )
    linhas_fig = "\n".join(
        f"| {f['figura_id']} | {f['titulo']} | {f['precisa_criar']} | {f['prioridade']} |"
        for f in figuras
    )
    return f"""# SUSC-19E - Pacote de comunicacao cientifica review-only

## Objetivo

Transformar os resultados de 18H, 19A, 19B, 19C e 19D em um pacote de comunicacao
claro para trabalho de conclusao, artigo e apresentacao, sem inflar claims. Nao cria
dado novo, score novo, benchmark, treino nem ground truth.

## Enquadramento cientifico

O REV-P e um framework multimodal auditavel de suscetibilidade urbana a enchentes,
com avaliacao observacional review-only. Nao e predicao operacional, nao e ground
truth patch-level, nao e modelo treinado, nao e benchmark supervisionado, nao e
score_v7 e nao e sistema de alerta.

## Sintese do estado atual

{ctx['total_patches']} patches consolidados; cobertura fisica/espectral/chuva
completas e territorial parcial ({_pct(ctx['cov_territorial'])}). {ctx['observed']}
patches observacionais review-only (Recife {ctx['obs_recife']}, Curitiba
{ctx['obs_curitiba']}); background nao rotulado de {ctx['background']} (nunca
negativo). score_v6 medio observado {ctx['score_obs']:.3f} contra {ctx['score_bg']:.3f};
{ctx['top30_global']}/{ctx['observed']} no top-30 global e
{ctx['top30_regional']}/{ctx['observed']} no top-30 regional.

## Estado por regiao

| Regiao | Papel | Patch-links fortes | Bloqueio principal |
| --- | --- | --- | --- |
{linhas_reg}

## Cobertura multimodal

| Familia | Cobertura | Status | Principal lacuna |
| --- | --- | --- | --- |
{linhas_cob}

## Claims permitidos

{linhas_permit}

## Claims proibidos

{linhas_proib}

## Plano de figuras

| Figura | Titulo | Precisa criar | Prioridade |
| --- | --- | --- | --- |
{linhas_fig}

## Roteiro de slides

15 slides em `roteiro_slides_rev_p_19e.md`, do problema a conclusao, sempre com o
enquadramento review-only, incluindo limitacoes (slide 13) e proximos passos
(slide 14).

## Riscos de comunicacao

Oito riscos em `matriz_riscos_comunicacao_19e.csv` (overclaim de previsao, ground
truth falso, background chamado de negativo, SAR como geometria oficial, score_v7
implicito, estatistica forte com n pequeno, Petropolis promovido, missingness
omitido), cada um com correcao e regra de validacao.

## Por que nao ha ground truth, treino, benchmark nem score_v7

Nao ha verdade de campo por patch; nao ha treino supervisionado; nao ha benchmark
17B; o score_v7 permanece bloqueado por amostra, missingness territorial e ausencia
de benchmark. O score_v6 permanece intacto.

## Proximos passos

Ampliar amostra observacional com geometria oficial, preencher a cobertura
territorial (pacote MapBiomas/GEE), separar o fenomeno de Petropolis e consolidar a
geometria de Curitiba. Estado da comunicacao: `{summary['status_19e_comunicacao']}`;
estado de artigo/slides: `{summary['status_artigo_slides']}`.
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    require_inputs()
    ensure_dir(OUT)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS_DIR)

    ctx = load_context()
    claims = claims_rows()
    tab_reg = tabela_regiao_rows(ctx)
    tab_cob = tabela_cobertura_rows(ctx)
    tab_res = tabela_resultados_rows(ctx)
    tab_blo = tabela_bloqueios_rows()
    tab_prox = tabela_proximos_rows()
    figuras = plano_figuras_rows()
    glossario = glossario_rows()
    riscos = riscos_rows()
    summary = summary_obj(ctx, claims)

    # markdown
    write_markdown(SINTESE, sintese_md(ctx))
    write_markdown(ART_METODO, artigo_metodologia_md(ctx))
    write_markdown(ART_RESULT, artigo_resultados_md(ctx))
    write_markdown(SLIDES, slides_md(ctx))
    write_markdown(BANCA, banca_md(ctx))

    # csv
    write_csv(CLAIMS, claims)
    write_csv(TAB_REGIAO, tab_reg)
    write_csv(TAB_COBERTURA, tab_cob)
    write_csv(TAB_RESULTADOS, tab_res)
    write_csv(TAB_BLOQUEIOS, tab_blo)
    write_csv(TAB_PROXIMOS, tab_prox)
    write_csv(PLANO_FIGURAS, figuras)
    write_csv(GLOSSARIO, glossario)
    write_csv(RISCOS, riscos)
    write_csv(GATE_COM, gate_comunicacao_rows(summary))
    write_csv(GATE_ART, gate_artigo_rows(summary))
    write_csv(RESUMO_SECAO, resumo_secao_rows())
    write_csv(RESUMO_CLAIM, resumo_claim_rows(claims))

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj(ctx))
    write_json(SUMMARY, summary)
    write_markdown(REPORT, report_text(ctx, summary, claims, tab_reg, tab_cob, figuras))
    return summary


# ---------------------------------------------------------------------------
# Validacao
# ---------------------------------------------------------------------------
def _public_paths() -> list[Path]:
    paths = []
    if OUT.exists():
        paths.extend(p for p in OUT.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".json", ".md"})
    if REPORT.exists():
        paths.append(REPORT)
    return sorted(dict.fromkeys(paths), key=lambda p: rel(p))


def _prose_paths() -> list[Path]:
    """Somente prosa/tabelas (md/csv). JSON de maquina nao entra nas checagens
    semanticas de linha (chaves nao relacionadas cairiam na mesma linha)."""
    return [p for p in _public_paths() if p.suffix.lower() in {".csv", ".md"}]


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def validate_public_text() -> list[str]:
    errors = []
    for path in _public_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:vocabulario_publico_proibido:{','.join(hits)}")
    return errors


def _danger_unframed(text: str) -> list[str]:
    """Linhas em que um padrao perigoso aparece SEM marcador de negacao/erro na mesma linha."""
    problems = []
    for line in text.splitlines():
        for name, pat in DANGER_PATTERNS.items():
            if pat.search(line) and not NEG_MARKER_RE.search(line):
                problems.append(name)
    return problems


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]

    # guardrails do summary
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"] or summary["marco_17b_criado"]:
        errors.append("benchmark_ou_marco_17b_criado_forbidden")
    for f in ("ground_truth_true_count", "eligible_for_training_true_count", "score_v7_allowed_true_count"):
        if summary[f]:
            errors.append(f"{f}_forbidden")
    if summary["status_19e_comunicacao"] not in STATUS_19E_COM_ALLOWED:
        errors.append(f"status_19e_comunicacao_fora_enum:{summary['status_19e_comunicacao']}")
    if summary["status_artigo_slides"] not in STATUS_ART_ALLOWED:
        errors.append(f"status_artigo_slides_fora_enum:{summary['status_artigo_slides']}")

    # claim policy: enums, permitidos e proibidos presentes, proibido nunca como permitido
    claims = read_csv(CLAIMS)
    status_set = {r["status"] for r in claims}
    for r in claims:
        if r["status"] not in CLAIM_STATUS_ALLOWED:
            errors.append(f"claim_status_fora_enum:{r['claim_id']}:{r['status']}")
    if "permitido" not in status_set:
        errors.append("claim_policy_sem_permitido")
    if "proibido" not in status_set:
        errors.append("claim_policy_sem_proibido")
    # nenhum claim permitido pode conter padrao perigoso afirmativo
    for r in claims:
        if r["status"] in ("permitido", "permitido_com_ressalva"):
            for _name, pat in DANGER_PATTERNS.items():
                if pat.search(r["frase"]):
                    errors.append(f"claim_permitido_com_frase_perigosa:{r['claim_id']}")
    # claims proibidos canonicos precisam existir com status proibido
    proib_frases = {r["frase"].lower() for r in claims if r["status"] == "proibido"}
    for termo in ("preve enchentes", "ground truth patch-level", "benchmark 17b", "geometria oficial", "negativo"):
        if not any(termo in f for f in proib_frases):
            errors.append(f"claim_proibido_canonico_ausente:{termo}")

    # perguntas de banca: precisam cobrir ground truth, previsao, score_v7
    banca_txt = BANCA.read_text(encoding="utf-8", errors="ignore").lower()
    for termo in ("ground truth", "previs", "score_v7", "review-only"):
        if termo not in banca_txt:
            errors.append(f"banca_sem_topico:{termo}")

    # slides: precisam conter limitacoes e proximos passos
    slides_txt = SLIDES.read_text(encoding="utf-8", errors="ignore").lower()
    if "limitac" not in slides_txt:
        errors.append("slides_sem_limitacoes")
    if "proximos passos" not in slides_txt and "proximo" not in slides_txt:
        errors.append("slides_sem_proximos_passos")

    # artigo metodologia precisa mencionar review-only e a separacao das camadas
    metodo_txt = ART_METODO.read_text(encoding="utf-8", errors="ignore").lower()
    if "review-only" not in metodo_txt:
        errors.append("artigo_metodologia_sem_review_only")
    for camada in SEPARACAO:
        if camada not in metodo_txt:
            errors.append(f"artigo_metodologia_sem_camada:{camada}")

    # Petropolis nunca como observado; sempre bloqueado/fora
    for path in _prose_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore").lower()
        for line in txt.splitlines():
            if "petropolis" in line and ("observ" in line or "avaliad" in line):
                if not re.search(r"(bloque|fora|misto|exclu|nao entra|n[aã]o (e|eh) )", line):
                    errors.append(f"{rel(path)}:petropolis_como_observado")
                    break

    # missingness territorial deve aparecer em texto publico
    joined = " ".join(p.read_text(encoding="utf-8", errors="ignore") for p in _public_paths())
    if "MapBiomas" not in joined or "territorial" not in joined.lower():
        errors.append("missingness_territorial_omitido")

    # padroes perigosos sem enquadramento de negacao (somente prosa/tabelas)
    for path in _prose_paths():
        txt = path.read_text(encoding="utf-8", errors="ignore")
        for line in txt.splitlines():
            if STRONG_STATS_RE.search(line) and not NEG_MARKER_RE.search(line):
                errors.append(f"{rel(path)}:conclusao_estatistica_forte_com_n_pequeno")
                break
        for name in _danger_unframed(txt):
            errors.append(f"{rel(path)}:padrao_perigoso_sem_negacao:{name}")

    # gates enums
    for r in read_csv(GATE_COM):
        if r.get("status_19e") not in STATUS_19E_COM_ALLOWED:
            errors.append(f"gate_com:status_fora_enum:{r.get('status_19e')}")
    for r in read_csv(GATE_ART):
        if r.get("status_artigo_slides") not in STATUS_ART_ALLOWED:
            errors.append(f"gate_art:status_fora_enum:{r.get('status_artigo_slides')}")

    # output 19E nao pode ser ignorado pelo git
    if rel(SUMMARY) in git_ignored([SUMMARY, CLAIMS]):
        errors.append("output_19e_ignorado_pelo_git")

    errors.extend(validate_public_text())
    return errors


def validate() -> int:
    summary = build_all()
    errors = validate_outputs(summary)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        "19E comunicacao cientifica validada: "
        f"claims={summary['claims_total']} (permit={summary['claims_permitidos']} proib={summary['claims_proibidos']}) "
        f"slides={summary['n_slides']} glossario={summary['n_glossario']} figuras={summary['n_figuras_plano']} "
        f"status19E={summary['status_19e_comunicacao']} artigo={summary['status_artigo_slides']} "
        f"scorev7={summary['status_score_v7']} 17B={summary['status_17b']} "
        f"score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "19E comunicacao cientifica gerada: "
        f"claims={summary['claims_total']} slides={summary['n_slides']} "
        f"status19E={summary['status_19e_comunicacao']} artigo={summary['status_artigo_slides']} "
        f"scorev7={summary['status_score_v7']}"
    )
    return 0
