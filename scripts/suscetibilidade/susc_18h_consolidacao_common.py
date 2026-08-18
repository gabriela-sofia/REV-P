"""SUSC-18H consolidacao mestre da cadeia observacional (17C ate 18G).

Etapa de consolidacao metodologica: nao cria nova geometria, novo score, novo
benchmark nem novo rotulo. Le os marcos ja produzidos (17C ate 18G), organiza o
estado real em matrizes auditaveis e prepara a estrategia das proximas frentes.

Todos os textos publicos ficam em portugues tecnico do Brasil. As matrizes sao
somente revisao (review-only): ground_truth=false, eligible_for_training=false,
score_v7_allowed=false e nenhum benchmark 17B e criado.
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
    read_json,
    rel,
    sha256_file,
    write_csv,
    write_json,
    write_markdown,
)

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
OUT = ROOT / "outputs_public" / "data" / "susc_18h_consolidacao_mestre_cadeia_observacional"
REPORTS = ROOT / "outputs_public" / "reports"
DATA_ROOT = ROOT / "outputs_public" / "data"
SCRIPTS_DIR = ROOT / "scripts" / "suscetibilidade"
TESTS_DIR = ROOT / "tests" / "suscetibilidade"
SCHEMAS_DIR = ROOT / "schemas" / "suscetibilidade"

REPORT = REPORTS / "SUSC_18H_CONSOLIDACAO_MESTRE_CADEIA_OBSERVACIONAL.md"
SCHEMA = SCHEMAS_DIR / "susc_18h_consolidacao_mestre_schema_v1.json"

SCORE_V6 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
SCORE_V7 = ROOT / "datasets" / "suscetibilidade" / "susc_score_v7_candidate_by_patch_v1.csv"

# Arquivos de saida
INVENTARIO = OUT / "inventario_marcos_susc_17c_18g.csv"
LINHAGEM = OUT / "linhagem_mestre_evidencia_observacional.csv"
QUALIDADE = OUT / "matriz_qualidade_evidencia.csv"
REGIONAL = OUT / "matriz_regional_recife_curitiba_petropolis.csv"
DIAG_17B = OUT / "diagnostico_prontidao_17b_mestre.csv"
PENDENCIAS = OUT / "matriz_pendencias_priorizadas.csv"
PLANO_MARCOS = OUT / "plano_proximos_marcos_pos_18g.csv"
MANIFESTO = OUT / "manifesto_outputs_publicos_17c_18g.csv"
RESUMO_REGIAO = OUT / "resumo_por_regiao.csv"
RESUMO_STATUS = OUT / "resumo_por_status.csv"
SUMMARY = OUT / "summary.json"
PREFLIGHT = OUT / "preflight.json"
RESUMO_ARTIGO = OUT / "resumo_tecnico_para_artigo.md"
RESUMO_SLIDES = OUT / "resumo_visual_para_slides.md"
PLANO_COMMIT = OUT / "plano_commit_seletivo_pos_18g.md"

REQUIRED_OUTPUTS = [
    INVENTARIO, LINHAGEM, QUALIDADE, REGIONAL, DIAG_17B, PENDENCIAS, PLANO_MARCOS,
    MANIFESTO, RESUMO_REGIAO, RESUMO_STATUS, SUMMARY, PREFLIGHT, RESUMO_ARTIGO,
    RESUMO_SLIDES, PLANO_COMMIT, SCHEMA, REPORT,
]

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
PUBLIC_FORBIDDEN_RE = re.compile(r"\b(?:agentic|agente|codex|llm|ia)\b", re.IGNORECASE)

HARD_DEFAULTS = {
    "review_only": "true",
    "ground_truth": "false",
    "eligible_for_training": "false",
    "score_v7_allowed": "false",
}

STATUS_17B_ALLOWED = {
    "17B_BLOQUEADO_OFICIALMENTE",
    "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA",
    "17B_APROXIMACAO_COM_AMOSTRA_PARCIAL",
    "17B_PRONTO_APENAS_PARA_DESENHO_REVIEW_ONLY",
    "17B_NAO_CRIADO",
}
STATUS_17B_MESTRE = "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA"

EVIDENCE_TIERS = {
    "A_oficial_poligono_observado",
    "B_oficial_ponto_observado",
    "C_footprint_tecnico_sar",
    "D_endereco_oficial_resolvido",
    "E_contexto_rua_bairro",
    "F_area_risco_ou_suscetibilidade_nao_evento",
    "G_rejeitado_ou_insuficiente",
}

# ---------------------------------------------------------------------------
# Registro dos marcos 17C ate 18G
# ---------------------------------------------------------------------------
# token: prefixo usado para localizar scripts/schemas/testes (com _ final para
# nao confundir 17c com 17c5, nem 18e com 18e2).
MARCOS = [
    {
        "marco": "SUSC-17C",
        "titulo_publico": "Aquisicao de referencia forte e canarios observacionais",
        "token": "susc_17c_",
        "pasta": "susc_17c_strong_reference_acquisition_canary",
        "relatorio": "SUSC_17C_STRONG_REFERENCE_ACQUISITION_CANARY_REPORT.md",
        "status": "17C_REFERENCIA_FORTE_ADQUIRIDA_SEM_PATCH_LINK_SUFICIENTE",
        "resultados": "10 candidatos fortes, 11 geometrias, apenas 1 vinculo patch, viabilidade SAR registrada",
    },
    {
        "marco": "SUSC-17C5",
        "titulo_publico": "Resolucao de vinculo geometria para patch",
        "token": "susc_17c5_",
        "pasta": "susc_17c5_geometry_to_patch_linkage_resolver",
        "relatorio": "SUSC_17C5_GEOMETRY_TO_PATCH_LINKAGE_RESOLVER_REPORT.md",
        "status": "17C5_PATCH_LINKS_PARCIAIS_SEM_VINCULO_OFICIAL_FORTE",
        "resultados": "67 vinculos patch, 5 candidatos fortes, 0 vinculo oficial forte",
    },
    {
        "marco": "SUSC-17D",
        "titulo_publico": "Validacao tecnica da evidencia observacional",
        "token": "susc_17d_",
        "pasta": "susc_17d_validacao_tecnica_evidencia_observacional",
        "relatorio": "SUSC_17D_VALIDACAO_TECNICA_EVIDENCIA_OBSERVACIONAL.md",
        "status": "17D_VALIDACAO_TECNICA_SEM_PRONTIDAO_BENCHMARK",
        "resultados": "cartoes de evidencia e matriz de consistencia; footprint tecnico dos canarios; sem prontidao de benchmark",
    },
    {
        "marco": "SUSC-17E",
        "titulo_publico": "Prontidao e calibracao observacional exploratoria",
        "token": "susc_17e_",
        "pasta": "susc_17e_prontidao_calibracao_observacional_exploratoria",
        "relatorio": "SUSC_17E_PRONTIDAO_CALIBRACAO_OBSERVACIONAL_EXPLORATORIA.md",
        "status": "17E_CALIBRACAO_EXPLORATORIA_REVIEW_ONLY",
        "resultados": "calibracao exploratoria somente revisao; features espectral e chuva presentes; componente fisico ausente bloqueia forte",
    },
    {
        "marco": "SUSC-17F",
        "titulo_publico": "Extracao fisica topografica dos canarios",
        "token": "susc_17f_",
        "pasta": "susc_17f_extracao_fisica_topografica_canarios_observacionais",
        "relatorio": "SUSC_17F_EXTRACAO_FISICA_TOPOGRAFICA_CANARIOS_OBSERVACIONAIS.md",
        "status": "17F_EXPLORATORIA_APRIMORADA_COM_COMPONENTE_FISICO",
        "resultados": "componente fisico por referencia comparativa; 5 canarios; fila de 10 tarefas; extracoes diretas ainda em 0",
    },
    {
        "marco": "SUSC-17G",
        "titulo_publico": "Extracao direta de features fisicas dos canarios",
        "token": "susc_17g_",
        "pasta": "susc_17g_extracao_direta_features_fisicas_canarios",
        "relatorio": "SUSC_17G_EXTRACAO_DIRETA_FEATURES_FISICAS_CANARIOS.md",
        "status": "17G_CALIBRACAO_FORTE_REVIEW_ONLY_POSSIVEL",
        "resultados": "extracao direta DEM GLO-30 e hidrografia; 5 canarios completos; terreno elevado corrige o 17F",
    },
    {
        "marco": "SUSC-17H",
        "titulo_publico": "Calibracao observacional forte somente revisao",
        "token": "susc_17h_",
        "pasta": "susc_17h_calibracao_observacional_forte_somente_revisao",
        "relatorio": "SUSC_17H_CALIBRACAO_OBSERVACIONAL_FORTE_SOMENTE_REVISAO.md",
        "status": "17H_CALIBRACAO_FORTE_COM_DIVERGENCIA_FISICA",
        "resultados": "indice de 5 componentes espelha o score_v6; 0.40-0.47 abaixo da referencia 0.60; divergencia fisica preservada",
    },
    {
        "marco": "SUSC-17I",
        "titulo_publico": "Ampliacao regional da amostra observacional",
        "token": "susc_17i_",
        "pasta": "susc_17i_ampliacao_regional_amostra_observacional",
        "relatorio": "SUSC_17I_AMPLIACAO_REGIONAL_AMOSTRA_OBSERVACIONAL.md",
        "status": "17I_AMOSTRA_REGIONAL_PARCIAL_COM_FILAS_EXECUTAVEIS",
        "resultados": "Curitiba e Petropolis com 41 candidatos; 0 fortes, 2 parciais em Curitiba, 36 de fenomeno misto em Petropolis",
    },
    {
        "marco": "SUSC-18A",
        "titulo_publico": "Execucao da referencia observacional regional",
        "token": "susc_18a_",
        "pasta": "susc_18a_execucao_referencia_observacional_regional",
        "relatorio": "SUSC_18A_EXECUCAO_REFERENCIA_OBSERVACIONAL_REGIONAL.md",
        "status": "18A_REFERENCIA_REGIONAL_FORTE_MAIS_FILAS_EXECUTAVEIS",
        "resultados": "Recife 5 fortes somente revisao; Curitiba 2 parciais e fila de geometria; Petropolis 29 misto e 1 contextual",
    },
    {
        "marco": "SUSC-18B",
        "titulo_publico": "Execucao de geometrias regionais e separacao de fenomeno",
        "token": "susc_18b_",
        "pasta": "susc_18b_execucao_geometrias_regionais_separacao_fenomeno",
        "relatorio": "SUSC_18B_EXECUCAO_GEOMETRIAS_REGIONAIS_SEPARACAO_FENOMENO.md",
        "status": "18B_BLOQUEIOS_REGIONAIS_REDUZIDOS_COM_FILAS_EXECUTAVEIS",
        "resultados": "Curitiba 54 prelinks region-only e ancora hidrometeorologica sem geometria; Petropolis 1 inundacao 2024 separada",
    },
    {
        "marco": "SUSC-18C",
        "titulo_publico": "Aquisicao de geometria oficial de Curitiba",
        "token": "susc_18c_",
        "pasta": "susc_18c_aquisicao_geometria_oficial_curitiba",
        "relatorio": "SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md",
        "status": "18C_CURITIBA_GEOMETRIA_AUSENTE_COM_SOLICITACAO_FORMAL",
        "resultados": "geometria oficial de ocorrencia ausente; solicitacao formal preparada; 43 poligonos de patch CUR reais",
    },
    {
        "marco": "SUSC-18D",
        "titulo_publico": "Protocolo externo de Curitiba com solicitacao formal",
        "token": "susc_18d_",
        "pasta": "susc_18d_protocolo_externo_curitiba",
        "relatorio": "SUSC_18D_PROTOCOLO_EXTERNO_CURITIBA.md",
        "status": "18D_AGUARDANDO_RESPOSTA_OFICIAL",
        "resultados": "pacote formal enviado ao orgao oficial; aguardando resposta com geometria de ocorrencia",
    },
    {
        "marco": "SUSC-18E",
        "titulo_publico": "Footprint tecnico SAR de Curitiba",
        "token": "susc_18e_",
        "pasta": "susc_18e_footprint_tecnico_sar_curitiba",
        "relatorio": "SUSC_18E_FOOTPRINT_TECNICO_SAR_CURITIBA.md",
        "status": "18E_PACOTE_GEE_SENTINEL1_PRONTO",
        "resultados": "pacote Sentinel-1 preparado com AOI tecnica, janelas pre e pos evento e script de execucao",
    },
    {
        "marco": "SUSC-18E2",
        "titulo_publico": "Execucao controlada Sentinel-1 de Curitiba",
        "token": "susc_18e2_",
        "pasta": "susc_18e2_execucao_controlada_sentinel1_curitiba",
        "relatorio": "SUSC_18E2_EXECUCAO_CONTROLADA_SENTINEL1_CURITIBA.md",
        "status": "18E2_TAREFAS_GEE_INICIADAS_AGUARDANDO_CONCLUSAO",
        "resultados": "execucao controlada iniciada; Sentinel-1 pre=3 e pos=3; exports de flood_mask, flood_vector e patch_stats iniciados",
    },
    {
        "marco": "SUSC-18F",
        "titulo_publico": "Ingestao e validacao do footprint SAR de Curitiba",
        "token": "susc_18f_",
        "pasta": "susc_18f_ingestao_validacao_footprint_sar_curitiba",
        "relatorio": "SUSC_18F_INGESTAO_VALIDACAO_FOOTPRINT_SAR_CURITIBA.md",
        "status": "18F_REFERENCIA_TECNICA_SAR_CURITIBA_PARCIAL_POR_PATCH_STATS",
        "resultados": "patch_stats real com 43 linhas; referencia tecnica SAR parcial porque o vetor completo excedeu o limite de feicoes",
    },
    {
        "marco": "SUSC-18G",
        "titulo_publico": "Recuperacao e compactacao vetorial SAR de Curitiba",
        "token": "susc_18g_",
        "pasta": "susc_18g_recuperacao_compactacao_vetorial_sar_curitiba",
        "relatorio": "SUSC_18G_RECUPERACAO_COMPACTACAO_VETORIAL_SAR_CURITIBA.md",
        "status": "18G_REFERENCIA_TECNICA_SAR_FORTE_COM_OVERLAY",
        "resultados": "vetor compacto tecnico por patch; 2 feicoes compactas; 2 overlays tecnicos somente revisao em CUR_01050 e CUR_01101",
    },
]

# ---------------------------------------------------------------------------
# Linhagem mestre da evidencia (autoral, derivada da leitura dos marcos)
# ---------------------------------------------------------------------------
LINHAGEM_FIELDS = [
    "lineage_id", "entidade", "origem_marco", "region", "city", "candidate_event_id",
    "event_date", "phenomenon_type", "geometry_id", "geometry_type", "patch_id",
    "patch_link_id", "feature_family", "score_reference", "evidence_tier", "review_only",
    "ground_truth", "eligible_for_training", "score_v7_allowed", "uncertainty_m",
    "status", "justificativa_tecnica",
]

ENTIDADES = {"event_record", "source_footprint", "derived_patch_link", "feature_evidence", "score_evaluation", "gate_status"}


def _lin(**kw) -> dict:
    row = dict(HARD_DEFAULTS)
    row.setdefault("uncertainty_m", "not_available")
    row.update(kw)
    return row


LINHAGEM_ROWS = [
    # ------------------------- Recife -------------------------
    _lin(
        lineage_id="LIN_REC_0001", entidade="event_record", origem_marco="SUSC-17C",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="not_available", geometry_type="none", patch_id="not_available",
        patch_link_id="not_available", feature_family="not_available", score_reference="not_available",
        evidence_tier="C_footprint_tecnico_sar",
        status="referencia_forte_review_only",
        justificativa_tecnica="evento de referencia forte para revisao; ocorrencias de Recife nao georreferenciadas por ocorrencia; nao e ground truth",
    ),
    _lin(
        lineage_id="LIN_REC_0002", entidade="source_footprint", origem_marco="SUSC-17D",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="REC_FOOTPRINT_TECNICO", geometry_type="footprint_tecnico_observacional",
        patch_id="not_available", patch_link_id="not_available", feature_family="not_available",
        score_reference="not_available", evidence_tier="C_footprint_tecnico_sar",
        status="footprint_tecnico_review_only",
        justificativa_tecnica="footprint tecnico observacional para revisao; nao substitui geometria oficial de ocorrencia; nao e ground truth",
    ),
    _lin(
        lineage_id="LIN_REC_0003", entidade="derived_patch_link", origem_marco="SUSC-17C5",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="REC_FOOTPRINT_TECNICO", geometry_type="footprint_tecnico_observacional",
        patch_id="recife_canarios", patch_link_id="REC_PATCH_LINKS_5",
        feature_family="not_available", score_reference="not_available",
        evidence_tier="C_footprint_tecnico_sar",
        status="cinco_canarios_fortes_review_only",
        justificativa_tecnica="cinco vinculos patch de canario para revisao; vinculo forte somente revisao, nao rotulo e nao treino",
    ),
    _lin(
        lineage_id="LIN_REC_0004", entidade="feature_evidence", origem_marco="SUSC-17G",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="not_available", geometry_type="none", patch_id="recife_canarios",
        patch_link_id="REC_PATCH_LINKS_5", feature_family="fisica_topografica",
        score_reference="not_available", evidence_tier="C_footprint_tecnico_sar",
        status="features_fisicas_diretas_completas",
        justificativa_tecnica="features fisicas diretas (DEM GLO-30, hidrografia) para revisao; terreno elevado gera divergencia fisica preservada",
    ),
    _lin(
        lineage_id="LIN_REC_0005", entidade="feature_evidence", origem_marco="SUSC-17E",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="not_available", geometry_type="none", patch_id="recife_canarios",
        patch_link_id="REC_PATCH_LINKS_5", feature_family="espectral_e_chuva",
        score_reference="not_available", evidence_tier="C_footprint_tecnico_sar",
        status="features_espectrais_e_chuva_presentes",
        justificativa_tecnica="features espectral e chuva presentes para revisao; nao constituem rotulo e nao habilitam treino",
    ),
    _lin(
        lineage_id="LIN_REC_0006", entidade="score_evaluation", origem_marco="SUSC-17H",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="not_available", geometry_type="none", patch_id="recife_canarios",
        patch_link_id="REC_PATCH_LINKS_5", feature_family="indice_5_componentes",
        score_reference="score_v6_candidate", evidence_tier="C_footprint_tecnico_sar",
        status="calibracao_forte_review_only_com_divergencia_fisica",
        justificativa_tecnica="indice de calibracao espelha o score_v6 apenas para revisao; score_v6 intacto e score_v7 nao criado",
    ),
    _lin(
        lineage_id="LIN_REC_0007", entidade="gate_status", origem_marco="SUSC-18A",
        region="nordeste", city="recife", candidate_event_id="REC_CANARY_SET",
        event_date="not_available", phenomenon_type="inundacao_alagamento",
        geometry_id="not_available", geometry_type="none", patch_id="recife_canarios",
        patch_link_id="REC_PATCH_LINKS_5", feature_family="not_available",
        score_reference="not_available", evidence_tier="C_footprint_tecnico_sar",
        status="aproximacao_regional_uma_regiao_um_evento",
        justificativa_tecnica="gate de aproximacao com uma regiao e um evento forte; nao habilita benchmark 17B",
    ),
    # ------------------------- Curitiba oficial -------------------------
    _lin(
        lineage_id="LIN_CUR_0001", entidade="event_record", origem_marco="SUSC-17I",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="not_available",
        patch_link_id="not_available", feature_family="not_available", score_reference="not_available",
        evidence_tier="G_rejeitado_ou_insuficiente",
        status="evento_reconhecido_sem_geometria_oficial",
        justificativa_tecnica="evento oficial reconhecido (CUR_2022_01_15) porem sem geometria oficial de ocorrencia liberada; insuficiente ate resposta do 18D",
    ),
    _lin(
        lineage_id="LIN_CUR_0002", entidade="derived_patch_link", origem_marco="SUSC-18B",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="curitiba_region",
        patch_link_id="CUR_PRELINK_54", feature_family="not_available", score_reference="not_available",
        evidence_tier="F_area_risco_ou_suscetibilidade_nao_evento",
        status="prelink_region_only_sem_geometria",
        justificativa_tecnica="54 prelinks region-only e ancora hidrometeorologica; vinculo apenas regional, nao patch de ocorrencia; nao e evidencia forte",
    ),
    _lin(
        lineage_id="LIN_CUR_0003", entidade="gate_status", origem_marco="SUSC-18D",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="not_available",
        patch_link_id="not_available", feature_family="not_available", score_reference="not_available",
        evidence_tier="G_rejeitado_ou_insuficiente",
        status="bloqueado_aguardando_resposta_oficial_18D",
        justificativa_tecnica="rota oficial bloqueada por ausencia de geometria; pacote formal 18D aguardando resposta; nao promover sem geometria",
    ),
    # ------------------------- Curitiba tecnica SAR -------------------------
    _lin(
        lineage_id="LIN_CUR_0004", entidade="source_footprint", origem_marco="SUSC-18G",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="S18G_CUR_SAR_COMPACT_0001", geometry_type="footprint_tecnico_sar_compacto",
        patch_id="not_available", patch_link_id="not_available", feature_family="sar_sentinel1",
        score_reference="not_available", evidence_tier="C_footprint_tecnico_sar", uncertainty_m="60",
        status="footprint_tecnico_sar_forte_review_only",
        justificativa_tecnica="footprint tecnico SAR compacto para revisao; nao e geometria oficial de ocorrencia e nao e ground truth",
    ),
    _lin(
        lineage_id="LIN_CUR_0005", entidade="derived_patch_link", origem_marco="SUSC-18G",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="S18G_CUR_SAR_COMPACT_0001", geometry_type="footprint_tecnico_sar_compacto",
        patch_id="CUR_01050", patch_link_id="S18G_LINK_CUR_01050", feature_family="sar_sentinel1",
        score_reference="score_v6_candidate", evidence_tier="C_footprint_tecnico_sar", uncertainty_m="60",
        status="overlay_tecnico_sar_review_only",
        justificativa_tecnica="overlay tecnico SAR em CUR_01050 para revisao; nao e geometria oficial e nao vira feature pre-evento",
    ),
    _lin(
        lineage_id="LIN_CUR_0006", entidade="derived_patch_link", origem_marco="SUSC-18G",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="S18G_CUR_SAR_COMPACT_0001", geometry_type="footprint_tecnico_sar_compacto",
        patch_id="CUR_01101", patch_link_id="S18G_LINK_CUR_01101", feature_family="sar_sentinel1",
        score_reference="score_v6_candidate", evidence_tier="C_footprint_tecnico_sar", uncertainty_m="60",
        status="overlay_tecnico_sar_review_only",
        justificativa_tecnica="overlay tecnico SAR em CUR_01101 para revisao; nao e geometria oficial e nao vira feature pre-evento",
    ),
    _lin(
        lineage_id="LIN_CUR_0007", entidade="feature_evidence", origem_marco="SUSC-18F",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="curitiba_patches_43",
        patch_link_id="not_available", feature_family="sar_patch_stats", score_reference="not_available",
        evidence_tier="C_footprint_tecnico_sar", uncertainty_m="60",
        status="patch_stats_sar_real_43_linhas",
        justificativa_tecnica="patch_stats SAR real com 43 linhas; resultado pos-evento para revisao, nao e footprint vetorial e nao vira feature pre-evento",
    ),
    _lin(
        lineage_id="LIN_CUR_0008", entidade="score_evaluation", origem_marco="SUSC-18E",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="curitiba_patches_43",
        patch_link_id="not_available", feature_family="comparacao_sar_score_v6", score_reference="score_v6_candidate",
        evidence_tier="C_footprint_tecnico_sar",
        status="comparacao_diagnostica_com_score_v6",
        justificativa_tecnica="comparacao diagnostica entre footprint SAR e score_v6 apenas para revisao; score_v6 intacto e score_v7 nao criado",
    ),
    _lin(
        lineage_id="LIN_CUR_0009", entidade="gate_status", origem_marco="SUSC-18G",
        region="sul", city="curitiba", candidate_event_id="S17C_REF_0060",
        event_date="2022-01-15", phenomenon_type="inundacao",
        geometry_id="S18G_CUR_SAR_COMPACT_0001", geometry_type="footprint_tecnico_sar_compacto",
        patch_id="not_available", patch_link_id="not_available", feature_family="not_available",
        score_reference="not_available", evidence_tier="C_footprint_tecnico_sar", uncertainty_m="60",
        status="segunda_regiao_tecnica_sem_geometria_oficial",
        justificativa_tecnica="segunda regiao tecnica via SAR para revisao; nao substitui a geometria oficial ainda pendente do 18D",
    ),
    # ------------------------- Petropolis -------------------------
    _lin(
        lineage_id="LIN_PET_0001", entidade="event_record", origem_marco="SUSC-17I",
        region="sudeste", city="petropolis", candidate_event_id="PET_REGISTROS_MISTOS",
        event_date="not_available", phenomenon_type="misto_deslizamento_inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="not_available",
        patch_link_id="not_available", feature_family="not_available", score_reference="not_available",
        evidence_tier="G_rejeitado_ou_insuficiente",
        status="fenomeno_misto_sem_separacao",
        justificativa_tecnica="registros de fenomeno misto (deslizamento e inundacao) sem separacao; nao entra como evidencia de inundacao sem separar o fenomeno",
    ),
    _lin(
        lineage_id="LIN_PET_0002", entidade="event_record", origem_marco="SUSC-18B",
        region="sudeste", city="petropolis", candidate_event_id="PET_2024_CONTEXTUAL",
        event_date="2024", phenomenon_type="inundacao_contextual",
        geometry_id="not_available", geometry_type="none", patch_id="not_available",
        patch_link_id="not_available", feature_family="not_available", score_reference="not_available",
        evidence_tier="E_contexto_rua_bairro",
        status="candidato_contextual_sem_geometria_forte",
        justificativa_tecnica="1 candidato contextual de inundacao de 2024 sem geometria forte; contexto de bairro, nao evidencia forte de ocorrencia",
    ),
    _lin(
        lineage_id="LIN_PET_0003", entidade="gate_status", origem_marco="SUSC-18B",
        region="sudeste", city="petropolis", candidate_event_id="PET_REGISTROS_MISTOS",
        event_date="not_available", phenomenon_type="misto_deslizamento_inundacao",
        geometry_id="not_available", geometry_type="none", patch_id="not_available",
        patch_link_id="not_available", feature_family="not_available", score_reference="not_available",
        evidence_tier="G_rejeitado_ou_insuficiente",
        status="bloqueado_por_fenomeno_e_geometria",
        justificativa_tecnica="bloqueado por fenomeno misto e ausencia de geometria forte; nao promover sem separacao do fenomeno",
    ),
]

# ---------------------------------------------------------------------------
# Matriz de qualidade da evidencia (tiers A..G)
# ---------------------------------------------------------------------------
QUALIDADE_FIELDS = [
    "item_id", "region", "city", "candidate_event_id", "entidade", "evidence_tier",
    "rotulo_tier", "descricao", "review_only", "ground_truth", "eligible_for_training",
    "score_v7_allowed", "status", "justificativa_tecnica",
]

TIER_LABELS = {
    "A_oficial_poligono_observado": "oficial com poligono observado",
    "B_oficial_ponto_observado": "oficial com ponto observado",
    "C_footprint_tecnico_sar": "footprint tecnico SAR somente revisao",
    "D_endereco_oficial_resolvido": "endereco oficial resolvido",
    "E_contexto_rua_bairro": "contexto de rua ou bairro",
    "F_area_risco_ou_suscetibilidade_nao_evento": "area de risco ou suscetibilidade, nao evento",
    "G_rejeitado_ou_insuficiente": "rejeitado ou insuficiente",
}


def _qual(item_id, region, city, event, entidade, tier, descricao, status, justi) -> dict:
    row = dict(HARD_DEFAULTS)
    row.update({
        "item_id": item_id, "region": region, "city": city, "candidate_event_id": event,
        "entidade": entidade, "evidence_tier": tier, "rotulo_tier": TIER_LABELS[tier],
        "descricao": descricao, "status": status, "justificativa_tecnica": justi,
    })
    return row


QUALIDADE_ROWS = [
    _qual("QUAL_REC_01", "nordeste", "recife", "REC_CANARY_SET", "source_footprint",
          "C_footprint_tecnico_sar", "5 canarios fortes com footprint tecnico, features fisicas, espectrais e chuva",
          "referencia_forte_review_only",
          "canario tecnico de Recife entra como evidencia tecnica forte somente revisao; nao e ground truth"),
    _qual("QUAL_REC_02", "nordeste", "recife", "REC_CANARY_SET", "event_record",
          "E_contexto_rua_bairro", "ocorrencias de Recife de 2022 sem coordenada por ocorrencia, usadas como contexto areal",
          "contexto_areal_review_only",
          "ocorrencias nao georreferenciadas por ocorrencia; servem de contexto, nao de evento pontual"),
    _qual("QUAL_CUR_01", "sul", "curitiba", "S17C_REF_0060", "event_record",
          "G_rejeitado_ou_insuficiente", "evento oficial CUR_2022_01_15 reconhecido, porem sem geometria oficial de ocorrencia",
          "bloqueado_sem_geometria_oficial",
          "Curitiba oficial segue bloqueada sem geometria; insuficiente ate resposta do 18D"),
    _qual("QUAL_CUR_02", "sul", "curitiba", "S17C_REF_0060", "source_footprint",
          "C_footprint_tecnico_sar", "footprint tecnico SAR compacto do 18G com 2 overlays em CUR_01050 e CUR_01101",
          "segunda_regiao_tecnica_review_only",
          "Curitiba SAR do 18G entra como footprint tecnico somente revisao; nao e geometria oficial de ocorrencia"),
    _qual("QUAL_CUR_03", "sul", "curitiba", "S17C_REF_0060", "derived_patch_link",
          "F_area_risco_ou_suscetibilidade_nao_evento", "54 prelinks region-only e 43 poligonos de patch preparados",
          "prelink_region_only",
          "prelinks apenas regionais e poligonos de patch nao sao geometria de ocorrencia; nao e evidencia forte"),
    _qual("QUAL_PET_01", "sudeste", "petropolis", "PET_REGISTROS_MISTOS", "event_record",
          "G_rejeitado_ou_insuficiente", "muitos registros de fenomeno misto (deslizamento e inundacao) sem separacao",
          "bloqueado_por_fenomeno_misto",
          "Petropolis misto nao entra como evidencia de inundacao sem separacao do fenomeno"),
    _qual("QUAL_PET_02", "sudeste", "petropolis", "PET_2024_CONTEXTUAL", "event_record",
          "E_contexto_rua_bairro", "1 candidato contextual de inundacao de 2024 sem geometria forte",
          "contextual_sem_geometria",
          "candidato contextual de bairro; nao e evidencia forte de ocorrencia sem geometria"),
]

# ---------------------------------------------------------------------------
# Matriz regional consolidada
# ---------------------------------------------------------------------------
REGIONAL_FIELDS = [
    "regiao", "eventos_fortes", "eventos_tecnicos", "eventos_parciais", "eventos_contextuais",
    "eventos_bloqueados", "patch_links_fortes", "patch_links_tecnicos", "patch_stats_sar",
    "features_fisicas", "features_espectrais", "features_chuva", "status_regional",
    "bloqueio_principal", "proxima_acao",
]

REGIONAL_ROWS = [
    {
        "regiao": "recife", "eventos_fortes": "1", "eventos_tecnicos": "1", "eventos_parciais": "0",
        "eventos_contextuais": "0", "eventos_bloqueados": "0", "patch_links_fortes": "5",
        "patch_links_tecnicos": "5", "patch_stats_sar": "0", "features_fisicas": "5",
        "features_espectrais": "5", "features_chuva": "5",
        "status_regional": "referencia_forte_review_only_uma_regiao_um_evento",
        "bloqueio_principal": "amostra_de_uma_unica_regiao_e_um_unico_evento_forte",
        "proxima_acao": "ampliar_eventos_e_regioes_mantendo_somente_revisao",
    },
    {
        "regiao": "curitiba", "eventos_fortes": "0", "eventos_tecnicos": "1", "eventos_parciais": "1",
        "eventos_contextuais": "0", "eventos_bloqueados": "1", "patch_links_fortes": "0",
        "patch_links_tecnicos": "2", "patch_stats_sar": "43", "features_fisicas": "0",
        "features_espectrais": "0", "features_chuva": "0",
        "status_regional": "segunda_regiao_tecnica_sar_sem_geometria_oficial",
        "bloqueio_principal": "ausencia_de_geometria_oficial_de_ocorrencia_18D_aguardando_resposta",
        "proxima_acao": "ingerir_resposta_oficial_18D_e_consolidar_referencia_tecnica_sar",
    },
    {
        "regiao": "petropolis", "eventos_fortes": "0", "eventos_tecnicos": "0", "eventos_parciais": "0",
        "eventos_contextuais": "1", "eventos_bloqueados": "29", "patch_links_fortes": "0",
        "patch_links_tecnicos": "0", "patch_stats_sar": "0", "features_fisicas": "0",
        "features_espectrais": "0", "features_chuva": "0",
        "status_regional": "bloqueado_por_fenomeno_misto_sem_separacao",
        "bloqueio_principal": "fenomeno_misto_deslizamento_inundacao_sem_separacao_e_sem_geometria_forte",
        "proxima_acao": "separar_fenomeno_e_buscar_evento_2024_com_geometria",
    },
]

# ---------------------------------------------------------------------------
# Diagnostico mestre do 17B (5 cenarios) - nenhum benchmark e criado
# ---------------------------------------------------------------------------
DIAG_FIELDS = [
    "cenario", "descricao", "eventos_distintos", "regioes", "patch_links_fortes",
    "separacao_temporal_possivel", "features_diretas_suficientes", "controles_definidos",
    "ground_truth", "trainable", "score_v7", "score_v6_intacto", "atende_minimos",
    "status_17b", "justificativa_tecnica",
]


def _diag(cenario, desc, eventos, regioes, links, sep, feats, ctrl, atende, status, justi) -> dict:
    return {
        "cenario": cenario, "descricao": desc, "eventos_distintos": str(eventos),
        "regioes": str(regioes), "patch_links_fortes": str(links),
        "separacao_temporal_possivel": sep, "features_diretas_suficientes": feats,
        "controles_definidos": ctrl, "ground_truth": "false", "trainable": "false",
        "score_v7": "false", "score_v6_intacto": "true", "atende_minimos": atende,
        "status_17b": status, "justificativa_tecnica": justi,
    }


DIAG_ROWS = [
    _diag("cenario_1_oficial_estrito",
          "apenas evidencia oficial com geometria observada de ocorrencia",
          1, 1, 0, "false", "false", "false", "false",
          "17B_BLOQUEADO_OFICIALMENTE",
          "so Recife tem referencia utilizavel e sem geometria oficial por ocorrencia; Curitiba oficial bloqueada; abaixo de 3 eventos, 2 regioes e 20 vinculos fortes"),
    _diag("cenario_2_tecnico_review_only",
          "Recife canario tecnico mais Curitiba tecnica SAR, somente revisao",
          2, 2, 7, "true", "true", "parcial", "false",
          "17B_APROXIMACAO_COM_SEGUNDA_REGIAO_TECNICA",
          "duas regioes tecnicas e dois eventos somente revisao; ainda abaixo de 20 vinculos fortes; nao habilita benchmark"),
    _diag("cenario_3_misto_oficial_tecnico",
          "combinacao de evidencia oficial parcial com evidencia tecnica SAR",
          2, 2, 7, "true", "parcial", "parcial", "false",
          "17B_APROXIMACAO_COM_AMOSTRA_PARCIAL",
          "mistura oficial parcial e tecnico reduz bloqueios mas nao atinge os minimos; amostra parcial, sem benchmark"),
    _diag("cenario_4_futuro_resposta_oficial_curitiba",
          "hipotese de chegada da geometria oficial de Curitiba pelo 18D",
          2, 2, 7, "true", "parcial", "parcial", "false",
          "17B_PRONTO_APENAS_PARA_DESENHO_REVIEW_ONLY",
          "resposta oficial de Curitiba habilitaria desenho somente revisao com duas regioes; ainda sem terceiro evento e sem 20 vinculos fortes; sem benchmark"),
    _diag("cenario_5_futuro_petropolis_separado",
          "hipotese de separacao do fenomeno de Petropolis com evento 2024 georreferenciado",
          3, 3, 7, "true", "parcial", "parcial", "false",
          "17B_PRONTO_APENAS_PARA_DESENHO_REVIEW_ONLY",
          "separar Petropolis somaria terceira regiao e terceiro evento; habilitaria desenho somente revisao, ainda nao benchmark, dependente de geometria"),
]

# ---------------------------------------------------------------------------
# Matriz de pendencias priorizadas
# ---------------------------------------------------------------------------
PEND_FIELDS = [
    "prioridade", "frente", "tarefa", "depende_de", "artefato_entrada",
    "artefato_saida_esperado", "comando_ou_acao", "bloqueio", "impacto", "proximo_marco_sugerido",
]

PEND_ROWS = [
    {"prioridade": "P0", "frente": "curitiba_oficial", "tarefa": "aguardar e ingerir a resposta oficial de Curitiba",
     "depende_de": "SUSC-18D", "artefato_entrada": "resposta oficial com geometria de ocorrencia",
     "artefato_saida_esperado": "geometria oficial de ocorrencia ingerida e validada",
     "comando_ou_acao": "acao externa manual: acompanhar resposta do orgao e ingerir quando chegar",
     "bloqueio": "resposta_oficial_ainda_nao_recebida", "impacto": "desbloqueia segunda regiao oficial",
     "proximo_marco_sugerido": "SUSC-18I"},
    {"prioridade": "P0", "frente": "sar_curitiba", "tarefa": "recuperar e validar os exports SAR finais",
     "depende_de": "SUSC-18G", "artefato_entrada": "exports GEE (flood_mask, flood_vector, patch_stats)",
     "artefato_saida_esperado": "vetor SAR final validado sem misturar patch_stats com footprint",
     "comando_ou_acao": "python scripts/suscetibilidade/build_susc_18g_recuperacao_compactacao_vetorial_sar_curitiba.py",
     "bloqueio": "vetor_completo_excedeu_limite_de_feicoes", "impacto": "consolida referencia tecnica SAR",
     "proximo_marco_sugerido": "SUSC-18I"},
    {"prioridade": "P0", "frente": "integridade", "tarefa": "manter patch_stats separado de footprint vetorial",
     "depende_de": "SUSC-18F", "artefato_entrada": "patch_stats SAR real de 43 linhas",
     "artefato_saida_esperado": "matriz que trata patch_stats como estatistica pos-evento, nao como footprint",
     "comando_ou_acao": "revisao das matrizes de linhagem e qualidade",
     "bloqueio": "risco_de_confundir_patch_stats_com_footprint", "impacto": "preserva rigor da evidencia",
     "proximo_marco_sugerido": "SUSC-18I"},
    {"prioridade": "P1", "frente": "sar_curitiba", "tarefa": "consolidar a interpretacao dos 2 overlays SAR",
     "depende_de": "SUSC-18G", "artefato_entrada": "overlays tecnicos CUR_01050 e CUR_01101",
     "artefato_saida_esperado": "leitura tecnica fechada dos overlays somente revisao",
     "comando_ou_acao": "consolidar overlays em relatorio tecnico review-only",
     "bloqueio": "not_available", "impacto": "fecha a evidencia tecnica de Curitiba",
     "proximo_marco_sugerido": "SUSC-18I"},
    {"prioridade": "P1", "frente": "curitiba_oficial", "tarefa": "buscar geometria oficial e resolver patch-links oficiais",
     "depende_de": "SUSC-18C", "artefato_entrada": "43 poligonos de patch CUR e prelinks region-only",
     "artefato_saida_esperado": "patch-links oficiais fortes quando houver geometria",
     "comando_ou_acao": "acao externa manual: buscar geometria oficial de ocorrencia",
     "bloqueio": "ausencia_de_geometria_oficial", "impacto": "eleva Curitiba de tecnico para oficial",
     "proximo_marco_sugerido": "SUSC-19A"},
    {"prioridade": "P2", "frente": "petropolis", "tarefa": "separar o fenomeno de Petropolis (deslizamento e inundacao)",
     "depende_de": "SUSC-18B", "artefato_entrada": "registros de fenomeno misto",
     "artefato_saida_esperado": "subconjunto de inundacao separado do deslizamento",
     "comando_ou_acao": "classificar registros por fenomeno antes de qualquer promocao",
     "bloqueio": "fenomeno_misto_sem_separacao", "impacto": "habilita terceira regiao no futuro",
     "proximo_marco_sugerido": "SUSC-19A"},
    {"prioridade": "P2", "frente": "petropolis", "tarefa": "buscar evento PET 2024 com geometria",
     "depende_de": "SUSC-18B", "artefato_entrada": "1 candidato contextual de 2024",
     "artefato_saida_esperado": "evento 2024 com geometria de ocorrencia",
     "comando_ou_acao": "acao externa manual: buscar geometria oficial do evento 2024",
     "bloqueio": "sem_geometria_forte", "impacto": "converte contexto em evento utilizavel",
     "proximo_marco_sugerido": "SUSC-19A"},
    {"prioridade": "P2", "frente": "recife", "tarefa": "ampliar eventos de Recife alem do canario atual",
     "depende_de": "SUSC-18A", "artefato_entrada": "5 canarios fortes de um unico evento",
     "artefato_saida_esperado": "novos eventos de Recife com evidencia comparavel",
     "comando_ou_acao": "acao externa manual: buscar novos eventos de Recife",
     "bloqueio": "amostra_de_um_unico_evento", "impacto": "reduz dependencia de um evento",
     "proximo_marco_sugerido": "SUSC-19A"},
    {"prioridade": "P3", "frente": "matriz_multimodal", "tarefa": "construir matriz multimodal escalavel por patch",
     "depende_de": "SUSC-18G", "artefato_entrada": "features fisicas, urbanas, espectrais e chuva das regioes",
     "artefato_saida_esperado": "matriz multimodal por patch em todas as regioes",
     "comando_ou_acao": "desenhar matriz por patch com HAND, slope, elevacao, TWI, NDVI, NDWI, MNDWI, NDBI, chuva e score_v6",
     "bloqueio": "cobertura_multimodal_incompleta", "impacto": "vira o eixo estrutural do projeto",
     "proximo_marco_sugerido": "SUSC-19A"},
    {"prioridade": "P3", "frente": "cobertura", "tarefa": "consolidar cobertura multimodal por regiao e feature",
     "depende_de": "SUSC-19A", "artefato_entrada": "matriz multimodal por patch",
     "artefato_saida_esperado": "auditoria de missingness por regiao e feature",
     "comando_ou_acao": "medir completude por regiao e por familia de feature",
     "bloqueio": "not_available", "impacto": "define onde ha lacuna de dado",
     "proximo_marco_sugerido": "SUSC-19B"},
    {"prioridade": "P3", "frente": "estrutura", "tarefa": "preparar embeddings quando houver fluxo consolidado",
     "depende_de": "SUSC-19A", "artefato_entrada": "matriz multimodal por patch",
     "artefato_saida_esperado": "espaco de representacao por patch, se e quando houver fluxo",
     "comando_ou_acao": "avaliar viabilidade de embeddings apos matriz multimodal estavel",
     "bloqueio": "fluxo_de_embeddings_ainda_nao_definido", "impacto": "base estrutural futura",
     "proximo_marco_sugerido": "SUSC-19C"},
]

# ---------------------------------------------------------------------------
# Plano estrategico pos-18G
# ---------------------------------------------------------------------------
PLANO_FIELDS = [
    "ordem", "marco_sugerido", "titulo", "objetivo", "depende_de", "entregaveis",
    "cria_score_v7", "cria_ground_truth", "cria_benchmark_17b", "observacao",
]

PLANO_ROWS = [
    {"ordem": "1", "marco_sugerido": "SUSC-18I", "titulo": "Consolidacao tecnica SAR de Curitiba pos-vetor",
     "objetivo": "fechar a interpretacao dos 2 overlays SAR somente revisao",
     "depende_de": "SUSC-18G", "entregaveis": "leitura tecnica fechada dos overlays; consolidacao da referencia tecnica SAR",
     "cria_score_v7": "false", "cria_ground_truth": "false", "cria_benchmark_17b": "false",
     "observacao": "condicional: executar se for preciso fechar a interpretacao dos overlays"},
    {"ordem": "2", "marco_sugerido": "SUSC-19A", "titulo": "Matriz multimodal escalavel por patch",
     "objetivo": "eixo estrutural principal daqui para frente",
     "depende_de": "SUSC-18I", "entregaveis": "matriz por patch com HAND, slope, elevacao, distancia a agua, TWI, flow_accumulation, urbanizacao, MapBiomas, NDVI, NDWI, MNDWI, NDBI, chuva, score_v6 e documentacao",
     "cria_score_v7": "false", "cria_ground_truth": "false", "cria_benchmark_17b": "false",
     "observacao": "foco principal; nao propor score_v7 ainda"},
    {"ordem": "3", "marco_sugerido": "SUSC-19B", "titulo": "Auditoria de cobertura multimodal",
     "objetivo": "verificar missingness por regiao e por feature",
     "depende_de": "SUSC-19A", "entregaveis": "matriz de completude por regiao e feature",
     "cria_score_v7": "false", "cria_ground_truth": "false", "cria_benchmark_17b": "false",
     "observacao": "mede onde ha lacuna antes de qualquer avaliacao"},
    {"ordem": "4", "marco_sugerido": "SUSC-19C", "titulo": "Avaliacao observacional somente revisao",
     "objetivo": "comparar Recife e Curitiba tecnica com score e features",
     "depende_de": "SUSC-19B", "entregaveis": "comparacao observacional review-only entre regioes",
     "cria_score_v7": "false", "cria_ground_truth": "false", "cria_benchmark_17b": "false",
     "observacao": "review-only; nao e benchmark 17B"},
    {"ordem": "5", "marco_sugerido": "SUSC-19D", "titulo": "Pacote de comunicacao cientifica",
     "objetivo": "figuras, tabelas e narrativa para trabalho de conclusao e apresentacao",
     "depende_de": "SUSC-19C", "entregaveis": "figuras, tabelas e narrativa consolidadas",
     "cria_score_v7": "false", "cria_ground_truth": "false", "cria_benchmark_17b": "false",
     "observacao": "fecha o ciclo de comunicacao sem novos claims"},
]


# ---------------------------------------------------------------------------
# Utilidades git / arquivos
# ---------------------------------------------------------------------------
def _run_git(args: list[str]) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else ""


def _score_v6_changed() -> bool:
    return bool(_run_git(["diff", "--name-only", "--", rel(SCORE_V6)]))


def git_ignored(paths: list[Path]) -> set[str]:
    """Retorna o subconjunto de paths (repo-relativos) ignorados pelo git.

    Usa separacao por NUL (-z) e I/O em bytes para evitar a traducao de fim de
    linha do Windows (\\n -> \\r\\n), que corromperia os paths e quebraria a
    avaliacao das regras de negacao (allowlist) do .gitignore.
    """
    if not paths:
        return set()
    rels = [rel(p) for p in paths]
    r = subprocess.run(
        ["git", "check-ignore", "-z", "--stdin"], cwd=ROOT,
        input=("\0".join(rels) + "\0").encode("utf-8"), capture_output=True, check=False,
    )
    out = r.stdout.decode("utf-8", errors="ignore")
    return {piece.replace("\\", "/") for piece in out.split("\0") if piece.strip()}


def _count_files(base: Path, token: str) -> int:
    if not base.exists():
        return 0
    return sum(1 for p in base.glob("*") if p.is_file() and token in p.name)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _tipo(path: Path) -> str:
    if path.is_dir():
        return "diretorio"
    return path.suffix.lower().lstrip(".") or "arquivo"


# ---------------------------------------------------------------------------
# Construcao das tabelas
# ---------------------------------------------------------------------------
def inventario_rows() -> list[dict]:
    rows = []
    for m in MARCOS:
        pasta = DATA_ROOT / m["pasta"]
        relatorio = REPORTS / m["relatorio"]
        rows.append({
            "marco": m["marco"],
            "titulo_publico": m["titulo_publico"],
            "branch": _marco_branch(pasta),
            "status": m["status"],
            "pasta_outputs": rel(pasta) if pasta.exists() else "lacuna_pasta_ausente",
            "relatorio_publico": rel(relatorio) if relatorio.exists() else "lacuna_relatorio_ausente",
            "scripts": str(_count_files(SCRIPTS_DIR, m["token"])),
            "schemas": str(_count_files(SCHEMAS_DIR, m["token"])),
            "testes": str(_count_files(TESTS_DIR, m["token"])),
            "principais_resultados": m["resultados"],
            "ground_truth_count": "0",
            "trainable_count": "0",
            "score_v7_count": "0",
            "benchmark_17b_criado": "false",
            "score_v6_alterado": "false",
            "observacao": "somente revisao; consolidacao metodologica sem novo dado",
        })
    return rows


def _marco_branch(pasta: Path) -> str:
    for name in ("summary.json",):
        p = pasta / name
        if p.exists():
            try:
                return read_json(p).get("branch", "not_available") or "not_available"
            except Exception:
                pass
    # summaries com nome proprio
    if pasta.exists():
        for p in pasta.glob("*summary.json"):
            try:
                return read_json(p).get("branch", "not_available") or "not_available"
            except Exception:
                continue
    return "not_available"


def manifesto_rows() -> list[dict]:
    entradas: list[tuple[Path, str, str]] = []
    # Relatorio e pasta de cada marco
    for m in MARCOS:
        entradas.append((REPORTS / m["relatorio"], m["marco"], f"relatorio publico do {m['marco']}"))
        entradas.append((DATA_ROOT / m["pasta"], m["marco"], f"pasta de saidas do {m['marco']}"))
    # Saidas do proprio 18H
    descr_18h = {
        INVENTARIO: "inventario mestre dos marcos 17C a 18G",
        LINHAGEM: "linhagem mestre da evidencia observacional",
        QUALIDADE: "matriz de qualidade da evidencia por tier",
        REGIONAL: "matriz regional Recife, Curitiba e Petropolis",
        DIAG_17B: "diagnostico mestre do 17B por cenario",
        PENDENCIAS: "matriz de pendencias priorizadas",
        PLANO_MARCOS: "plano dos proximos marcos pos-18G",
        RESUMO_REGIAO: "resumo por regiao",
        RESUMO_STATUS: "resumo por familia de status",
        SUMMARY: "resumo consolidado da etapa",
        PREFLIGHT: "estado de preflight do repositorio",
        RESUMO_ARTIGO: "sintese tecnica para artigo",
        RESUMO_SLIDES: "sintese visual para apresentacao",
        PLANO_COMMIT: "plano de commit seletivo",
        REPORT: "relatorio publico do 18H",
        SCHEMA: "schema do 18H",
    }
    for path, descr in descr_18h.items():
        entradas.append((path, "SUSC-18H", descr))

    all_paths = [p for p, _, _ in entradas]
    ignored = git_ignored([p for p in all_paths])
    rows = []
    for path, marco, descr in entradas:
        existe = path.exists()
        versionavel = existe and rel(path) not in ignored
        if path.is_dir():
            tamanho = _dir_size(path)
            sha = "not_available"
        elif existe:
            tamanho = path.stat().st_size
            sha = sha256_file(path)
        else:
            tamanho = 0
            sha = "not_available"
        rows.append({
            "path": rel(path),
            "marco": marco,
            "tipo": _tipo(path) if existe else ("diretorio" if path == DATA_ROOT / path.name else "arquivo"),
            "existe": "true" if existe else "false",
            "versionavel_git": "true" if versionavel else "false",
            "tamanho_bytes": str(tamanho),
            "sha256": sha,
            "descricao": descr,
            "incluir_no_commit": "true" if versionavel else "false",
        })
    return rows


def resumo_regiao_rows() -> list[dict]:
    out = []
    for r in REGIONAL_ROWS:
        out.append({
            "regiao": r["regiao"],
            "eventos_fortes": r["eventos_fortes"],
            "eventos_tecnicos": r["eventos_tecnicos"],
            "eventos_parciais": r["eventos_parciais"],
            "eventos_contextuais": r["eventos_contextuais"],
            "eventos_bloqueados": r["eventos_bloqueados"],
            "status_regional": r["status_regional"],
            "proxima_acao": r["proxima_acao"],
        })
    return out


STATUS_FAMILIA = {
    "SUSC-17C": "referencia_forte_review_only",
    "SUSC-17C5": "parcial",
    "SUSC-17D": "parcial",
    "SUSC-17E": "referencia_forte_review_only",
    "SUSC-17F": "referencia_forte_review_only",
    "SUSC-17G": "referencia_forte_review_only",
    "SUSC-17H": "referencia_forte_review_only",
    "SUSC-17I": "parcial",
    "SUSC-18A": "referencia_forte_review_only",
    "SUSC-18B": "parcial",
    "SUSC-18C": "bloqueado",
    "SUSC-18D": "aguardando",
    "SUSC-18E": "preparatorio",
    "SUSC-18E2": "preparatorio",
    "SUSC-18F": "tecnico_review_only",
    "SUSC-18G": "tecnico_review_only",
}


def resumo_status_rows() -> list[dict]:
    counts: dict[str, int] = {}
    for m in MARCOS:
        fam = STATUS_FAMILIA[m["marco"]]
        counts[fam] = counts.get(fam, 0) + 1
    return [{"status_familia": k, "quantidade": str(v)} for k, v in sorted(counts.items())]


# ---------------------------------------------------------------------------
# Documentos markdown
# ---------------------------------------------------------------------------
def resumo_artigo_text(summary: dict) -> str:
    return f"""# Sintese tecnica para artigo - SUSC-18H

## O que o REV-P e agora

O REV-P e um estudo de suscetibilidade multimodal auditavel a inundacao,
organizado por patch e sustentado por evidencia observacional reproduzivel.
A cadeia 17C ate 18G consolidou referencias somente revisao (review-only) em tres
regioes com maturidade diferente.

## Por que nao e previsao operacional

Nao ha previsao operacional. Todas as saidas sao somente revisao, sem alvo de
treino e sem rotulo. O score_v6 permanece intacto e nenhum score_v7 foi criado.

## Por que nao e ground truth

Nenhuma matriz declara ground_truth. As referencias fortes de Recife e a
referencia tecnica SAR de Curitiba sao evidencia para revisao humana, nao verdade
de campo. Curitiba oficial segue bloqueada por ausencia de geometria de ocorrencia.

## Por que e suscetibilidade multimodal auditavel

Cada evidencia tem linhagem separada por entidade: registro de evento, footprint
de fonte, vinculo derivado de patch, evidencia de feature, avaliacao de score e
estado de gate. Tudo e rastreavel a arquivos publicos reproduziveis.

## O que Recife provou

Recife entregou 1 evento com 5 canarios fortes somente revisao, com features
fisicas diretas, espectrais e de chuva, e calibracao forte review-only. A
divergencia fisica (terreno elevado) foi preservada, nao mascarada.

## O que Curitiba tecnica acrescentou

Curitiba adicionou uma segunda regiao tecnica via SAR Sentinel-1: patch_stats
real com 43 linhas e um footprint tecnico compacto com 2 overlays somente
revisao (CUR_01050 e CUR_01101). Nao substitui a geometria oficial, ainda pendente.

## O que Petropolis bloqueia

Petropolis reune muitos registros de fenomeno misto (deslizamento e inundacao)
sem separacao e sem geometria forte. Nao entra como evidencia de inundacao ate a
separacao do fenomeno. Ha 1 candidato contextual de 2024.

## Por que o 17B ainda nao foi criado

O 17B exige no minimo 3 eventos distintos, 2 regioes e 20 vinculos fortes, com
controles definidos. O estado atual e `{summary['status_17b_mestre']}`: duas
regioes tecnicas e dois eventos somente revisao, abaixo dos minimos. Nenhum
benchmark foi criado.

## Qual e a proxima frente cientifica

A proxima frente estrutural e a matriz multimodal escalavel por patch (SUSC-19A):
consolidar features fisicas, urbanas, espectrais e de chuva em todas as regioes,
com auditoria de cobertura, mantendo somente revisao e sem score_v7.
"""


def resumo_slides_text(summary: dict) -> str:
    return f"""# Sintese visual para apresentacao - SUSC-18H

## Slide 1 - O que e o REV-P
- Suscetibilidade multimodal auditavel a inundacao, por patch.
- Somente revisao: sem previsao operacional, sem rotulo, sem treino.

## Slide 2 - Regra de ouro
- score_v6 intacto; score_v7 nao criado; benchmark 17B nao criado.
- ground_truth = falso em toda a cadeia.

## Slide 3 - Recife (referencia forte)
- 1 evento, 5 canarios fortes somente revisao.
- Features fisicas diretas, espectrais e chuva; divergencia fisica preservada.

## Slide 4 - Curitiba (segunda regiao tecnica)
- SAR Sentinel-1: patch_stats real de 43 linhas.
- Footprint tecnico compacto com 2 overlays (CUR_01050, CUR_01101).
- Geometria oficial ainda pendente (18D aguardando resposta).

## Slide 5 - Petropolis (bloqueado)
- Fenomeno misto sem separacao e sem geometria forte.
- 1 candidato contextual de 2024.

## Slide 6 - Estado do 17B
- Status: `{summary['status_17b_mestre']}`.
- Minimos: 3 eventos, 2 regioes, 20 vinculos fortes - ainda nao atingidos.

## Slide 7 - Decisao estrategica
- Deixar de depender do footprint como base central.
- Usar footprint e SAR como canario.
- Avancar para a matriz multimodal escalavel por patch.

## Slide 8 - Proximos marcos
- SUSC-18I: consolidacao tecnica SAR de Curitiba.
- SUSC-19A: matriz multimodal escalavel por patch (eixo principal).
- SUSC-19B, 19C, 19D: cobertura, avaliacao review-only e comunicacao.
"""


def plano_commit_text(summary: dict, manifesto: list[dict]) -> str:
    versionaveis = [r for r in manifesto if r["marco"] == "SUSC-18H" and r["versionavel_git"] == "true"]
    marcos_versionaveis = sorted({r["path"] for r in manifesto if r["versionavel_git"] == "true" and r["marco"] != "SUSC-18H" and r["tipo"] == "diretorio"})
    marcos_ignorados = sorted({r["path"] for r in manifesto if r["versionavel_git"] == "false" and r["marco"] != "SUSC-18H"})
    add_lines = [
        "git add scripts/suscetibilidade/build_susc_18h_consolidacao_mestre_cadeia_observacional.py",
        "git add scripts/suscetibilidade/validate_susc_18h_consolidacao_mestre_cadeia_observacional.py",
        "git add scripts/suscetibilidade/susc_18h_consolidacao_common.py",
        "git add schemas/suscetibilidade/susc_18h_consolidacao_mestre_schema_v1.json",
        "git add tests/suscetibilidade/test_susc_18h_consolidacao_mestre_cadeia_observacional.py",
        "git add outputs_public/data/susc_18h_consolidacao_mestre_cadeia_observacional/",
        "git add outputs_public/reports/SUSC_18H_CONSOLIDACAO_MESTRE_CADEIA_OBSERVACIONAL.md",
        "git add .gitignore",
    ]
    versionaveis_txt = "\n".join(f"- `{r['path']}`" for r in versionaveis) or "- (nenhum)"
    marcos_v_txt = "\n".join(f"- `{p}`" for p in marcos_versionaveis) or "- (nenhum)"
    ignorados_txt = "\n".join(f"- `{p}` (ignorado pelo .gitignore)" for p in marcos_ignorados) or "- (nenhum)"
    add_txt = "\n".join(add_lines)
    return f"""# Plano de commit seletivo pos-18G - SUSC-18H

Este plano descreve o que deve entrar no commit. Nao executa commit.

## 1. Arquivos que devem entrar no commit (codigo e schema)

- `scripts/suscetibilidade/build_susc_18h_consolidacao_mestre_cadeia_observacional.py`
- `scripts/suscetibilidade/validate_susc_18h_consolidacao_mestre_cadeia_observacional.py`
- `scripts/suscetibilidade/susc_18h_consolidacao_common.py`
- `schemas/suscetibilidade/susc_18h_consolidacao_mestre_schema_v1.json`
- `tests/suscetibilidade/test_susc_18h_consolidacao_mestre_cadeia_observacional.py`
- `.gitignore` (nova allowlist do 18H)

## 2. Saidas publicas do 18H que devem entrar

{versionaveis_txt}

## 3. Pastas de marcos anteriores versionaveis (ja no historico)

{marcos_v_txt}

## 4. Saidas que NAO devem entrar (ignoradas pelo .gitignore)

Pastas de marcos fora da allowlist do .gitignore (17E, 17F, 17G, 17H, 17I) e
qualquer conteudo de `local_runs/`, rasters e artefatos pesados:

{ignorados_txt}

- `local_runs/**` nunca entra.
- Rasters `.tif`, `.tiff`, `.vrt` e embeddings `.npy`, `.npz` nunca entram.
- Nenhum path privado ou credencial entra.

## 5. Comando git add sugerido

```
{add_txt}
```

## 6. Ordem de commit sugerida

1. `.gitignore` com a allowlist do 18H.
2. Modulo comum, build e validator do 18H.
3. Schema e testes do 18H.
4. Saidas publicas do 18H em `outputs_public/data/...`.
5. Relatorio publico do 18H em `outputs_public/reports/...`.

## 7. Antes do commit

- `git status --short` para conferir o que esta staged.
- Garantir staging seletivo: nada de `local_runs/`, raster ou artefato pesado.
- Rodar o validator e o pytest do 18H.

Estado consolidado do 17B: `{summary['status_17b_mestre']}` (17B nao criado).
"""


def report_text(summary: dict) -> str:
    linhas_inv = "\n".join(f"| {m['marco']} | {m['titulo_publico']} | `{m['status']}` |" for m in MARCOS)
    return f"""# SUSC-18H - Consolidacao mestre da cadeia observacional

## Resumo executivo

Esta etapa e uma consolidacao metodologica. Nao cria nova geometria, novo score,
novo benchmark nem novo rotulo. Ela fotografa o estado real da cadeia 17C ate 18G,
organiza a evidencia em matrizes auditaveis e define a proxima frente estrutural.

Guardrails permanentes: ground_truth = falso, treino desabilitado, score_v7 nao
permitido, benchmark 17B nao criado e score_v6 intacto.

## Historico 17C ate 18G

| Marco | Titulo | Status |
| --- | --- | --- |
{linhas_inv}

## Estado por regiao

- **Recife**: 1 evento com 5 canarios fortes somente revisao; features fisicas
  diretas, espectrais e de chuva; calibracao forte review-only com divergencia
  fisica preservada. Limite: uma unica regiao e um unico evento.
- **Curitiba**: segunda regiao tecnica via SAR Sentinel-1 (patch_stats real de 43
  linhas e footprint tecnico compacto com 2 overlays em CUR_01050 e CUR_01101).
  A geometria oficial de ocorrencia segue ausente; o pacote formal 18D aguarda
  resposta.
- **Petropolis**: fenomeno misto (deslizamento e inundacao) sem separacao e sem
  geometria forte; 1 candidato contextual de 2024. Nao promovido.

## Estado por evidencia

A qualidade da evidencia foi classificada em tiers de A ate G. O canario tecnico
de Recife e o footprint SAR de Curitiba entram como tier C (footprint tecnico
somente revisao). Curitiba oficial e Petropolis misto permanecem em tier G
(insuficiente) ate haver geometria e separacao de fenomeno.

## Status do 17B

Estado mestre: `{summary['status_17b_mestre']}`. Os cenarios avaliados (oficial
estrito, tecnico review-only, misto, futuro com resposta oficial de Curitiba e
futuro com Petropolis separado) confirmam que os minimos (3 eventos, 2 regioes e
20 vinculos fortes) ainda nao foram atingidos. O 17B nao foi criado.

## Pendencias priorizadas

- **P0**: aguardar e ingerir a resposta oficial de Curitiba (18D); recuperar e
  validar os exports SAR finais; manter patch_stats separado de footprint.
- **P1**: consolidar a interpretacao dos 2 overlays SAR; buscar geometria oficial
  de Curitiba.
- **P2**: separar o fenomeno de Petropolis; buscar evento 2024 com geometria;
  ampliar eventos de Recife.
- **P3**: construir a matriz multimodal escalavel por patch e auditar cobertura.

## Decisao estrategica

1. Parar de depender do footprint como base central.
2. Usar footprint e SAR como canario de evidencia, nao como fundamento.
3. Avancar para a matriz multimodal escalavel por patch como novo eixo estrutural.

## Proximos marcos recomendados

1. **SUSC-18I** - Consolidacao tecnica SAR de Curitiba pos-vetor.
2. **SUSC-19A** - Matriz multimodal escalavel por patch (eixo principal).
3. **SUSC-19B / 19C / 19D** - Cobertura multimodal, avaliacao review-only e
   comunicacao cientifica.

Nenhum desses marcos propoe score_v7.

## Plano de commit seletivo

Entram no commit o codigo, o schema, os testes e as saidas publicas do 18H, alem
da atualizacao de allowlist do `.gitignore`. Nao entram `local_runs/`, rasters,
embeddings nem pastas de marco fora da allowlist. O detalhamento esta em
`plano_commit_seletivo_pos_18g.md`.
"""


# ---------------------------------------------------------------------------
# Preflight / summary / schema
# ---------------------------------------------------------------------------
def preflight_obj() -> dict:
    esperadas = [(m["marco"], DATA_ROOT / m["pasta"]) for m in MARCOS]
    lacunas = [m for m, p in esperadas if not p.exists()]
    # diretorios versionaveis vs ignorados
    dirs = [p for _, p in esperadas]
    ignored = git_ignored(dirs + [OUT])
    versionaveis = [rel(p) for p in dirs if rel(p) not in ignored]
    ignorados = [rel(p) for p in dirs if rel(p) in ignored]
    status_full = _run_git(["status", "--short"]).splitlines()
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "status_short_count": len(status_full),
        "marcos_esperados": [m for m, _ in esperadas],
        "marcos_com_pasta": [m for m, p in esperadas if p.exists()],
        "lacunas_de_pasta": lacunas,
        "diretorios_versionaveis": sorted(versionaveis),
        "diretorios_ignorados_pelo_gitignore": sorted(ignorados),
        "score_v6_path": rel(SCORE_V6),
        "score_v6_changed": _score_v6_changed(),
        "score_v7_path": rel(SCORE_V7),
        "score_v7_created": SCORE_V7.exists(),
    }


def summary_obj() -> dict:
    return {
        "branch": _run_git(["branch", "--show-current"]) or "unknown",
        "head": _run_git(["rev-parse", "--short", "HEAD"]) or "unknown",
        "staged_count": len(_run_git(["diff", "--cached", "--name-only"]).splitlines()),
        "marcos_inventariados": [m["marco"] for m in MARCOS],
        "total_marcos": len(MARCOS),
        "regioes": ["recife", "curitiba", "petropolis"],
        "recife_estado": "referencia_forte_review_only_uma_regiao_um_evento",
        "curitiba_estado": "segunda_regiao_tecnica_sar_sem_geometria_oficial",
        "petropolis_estado": "bloqueado_por_fenomeno_misto_sem_separacao",
        "status_17b_mestre": STATUS_17B_MESTRE,
        "marco_17b_criado": False,
        "benchmark_17b_criado": False,
        "cenarios_17b": [d["cenario"] for d in DIAG_ROWS],
        "proximos_marcos": ["SUSC-18I", "SUSC-19A", "SUSC-19B", "SUSC-19C", "SUSC-19D"],
        "linhagem_rows": len(LINHAGEM_ROWS),
        "qualidade_rows": len(QUALIDADE_ROWS),
        "pendencias_rows": len(PEND_ROWS),
        "ground_truth_true_count": 0,
        "eligible_for_training_true_count": 0,
        "score_v7_allowed_true_count": 0,
        "score_v6_changed": _score_v6_changed(),
        "score_v7_created": SCORE_V7.exists(),
        "review_only": True,
        "ground_truth": False,
        "trainable": False,
        "score_v7_allowed": False,
    }


def schema_obj() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "SUSC-18H consolidacao mestre da cadeia observacional",
        "type": "object",
        "description": "Consolidacao metodologica dos marcos 17C ate 18G, somente revisao.",
        "guardrails": {
            "ground_truth": {"const": "false"},
            "eligible_for_training": {"const": "false"},
            "score_v7_allowed": {"const": "false"},
            "review_only": {"const": "true"},
            "benchmark_17b_criado": {"const": "false"},
            "score_v6_alterado": {"const": "false"},
        },
        "evidence_tiers": sorted(EVIDENCE_TIERS),
        "entidades_linhagem": sorted(ENTIDADES),
        "status_17b_allowed": sorted(STATUS_17B_ALLOWED),
        "status_17b_mestre": STATUS_17B_MESTRE,
        "marcos": [m["marco"] for m in MARCOS],
    }


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_all() -> dict:
    ensure_dir(OUT)
    ensure_dir(REPORTS)
    ensure_dir(SCHEMAS_DIR)

    summary = summary_obj()
    manifesto = manifesto_rows()

    write_csv(INVENTARIO, inventario_rows())
    write_csv(LINHAGEM, LINHAGEM_ROWS, LINHAGEM_FIELDS)
    write_csv(QUALIDADE, QUALIDADE_ROWS, QUALIDADE_FIELDS)
    write_csv(REGIONAL, REGIONAL_ROWS, REGIONAL_FIELDS)
    write_csv(DIAG_17B, DIAG_ROWS, DIAG_FIELDS)
    write_csv(PENDENCIAS, PEND_ROWS, PEND_FIELDS)
    write_csv(PLANO_MARCOS, PLANO_ROWS, PLANO_FIELDS)
    write_csv(MANIFESTO, manifesto, ["path", "marco", "tipo", "existe", "versionavel_git", "tamanho_bytes", "sha256", "descricao", "incluir_no_commit"])
    write_csv(RESUMO_REGIAO, resumo_regiao_rows())
    write_csv(RESUMO_STATUS, resumo_status_rows())

    write_json(SCHEMA, schema_obj())
    write_json(PREFLIGHT, preflight_obj())
    write_json(SUMMARY, summary)

    write_markdown(RESUMO_ARTIGO, resumo_artigo_text(summary))
    write_markdown(RESUMO_SLIDES, resumo_slides_text(summary))
    write_markdown(PLANO_COMMIT, plano_commit_text(summary, manifesto))
    write_markdown(REPORT, report_text(summary))
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


def public_text_violations_text(text: str) -> list[str]:
    return sorted({m.group(0) for m in PUBLIC_FORBIDDEN_RE.finditer(text)})


def validate_public_text() -> list[str]:
    errors = []
    for path in _public_paths():
        hits = public_text_violations_text(path.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            errors.append(f"{rel(path)}:vocabulario_publico_proibido:{','.join(hits)}")
    return errors


def _read_csv_rows(path: Path) -> list[dict]:
    import csv
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def validate_outputs(summary: dict) -> list[str]:
    errors = [f"missing:{rel(p)}" for p in REQUIRED_OUTPUTS if not p.exists()]

    # Guardrails globais
    if summary["score_v6_changed"]:
        errors.append("score_v6_changed_forbidden")
    if summary["score_v7_created"]:
        errors.append("score_v7_created_forbidden")
    if summary["benchmark_17b_criado"]:
        errors.append("benchmark_17b_criado_forbidden")
    if summary["marco_17b_criado"]:
        errors.append("marco_17b_criado_forbidden")
    if summary["status_17b_mestre"] not in STATUS_17B_ALLOWED:
        errors.append(f"status_17b_mestre_fora_enum:{summary['status_17b_mestre']}")

    # Diagnostico 17B: status no enum e flags proibidas
    for row in _read_csv_rows(DIAG_17B):
        if row["status_17b"] not in STATUS_17B_ALLOWED:
            errors.append(f"diagnostico:{row['cenario']}:status_fora_enum:{row['status_17b']}")
        if row["ground_truth"] == "true" or row["trainable"] == "true" or row["score_v7"] == "true":
            errors.append(f"diagnostico:{row['cenario']}:flag_proibida_true")
        if row["score_v6_intacto"] != "true":
            errors.append(f"diagnostico:{row['cenario']}:score_v6_nao_intacto")
        if any("17B_CRIADO" in v or "BENCHMARK_CRIADO" in v for v in row.values()):
            errors.append(f"diagnostico:{row['cenario']}:17b_aparece_criado")

    # Matrizes principais: flags proibidas, justificativa, tier no enum
    for path in (LINHAGEM, QUALIDADE):
        for idx, row in enumerate(_read_csv_rows(path), start=1):
            rid = row.get("lineage_id") or row.get("item_id") or f"linha{idx}"
            for field in ("ground_truth", "eligible_for_training", "score_v7_allowed"):
                if row.get(field) == "true":
                    errors.append(f"{rel(path)}:{rid}:{field}_true")
            if not row.get("justificativa_tecnica", "").strip():
                errors.append(f"{rel(path)}:{rid}:justificativa_vazia")
            tier = row.get("evidence_tier", "")
            if tier and tier not in EVIDENCE_TIERS:
                errors.append(f"{rel(path)}:{rid}:tier_fora_enum:{tier}")
            if row.get("entidade") and row["entidade"] not in ENTIDADES and path == LINHAGEM:
                errors.append(f"{rel(path)}:{rid}:entidade_fora_enum:{row['entidade']}")

    # Curitiba SAR nao pode ser chamada de geometria oficial
    for row in _read_csv_rows(LINHAGEM):
        gid = row.get("geometry_id", "")
        gtype = row.get("geometry_type", "")
        justi = row.get("justificativa_tecnica", "").lower()
        if gid.startswith("S18G_CUR_SAR"):
            if "oficial" in gtype.lower():
                errors.append(f"{row['lineage_id']}:sar_chamado_de_geometria_oficial")
            if "oficial" in justi and "nao" not in justi:
                errors.append(f"{row['lineage_id']}:sar_promovido_a_oficial")
        # patch_stats nao pode ser chamado de footprint vetorial
        if row.get("feature_family") == "sar_patch_stats":
            if "footprint" in justi and "nao e footprint" not in justi:
                errors.append(f"{row['lineage_id']}:patch_stats_chamado_de_footprint")
        # AOI tecnica nao e ocorrencia
        if "aoi" in justi and "ocorrencia" in justi and "nao" not in justi:
            errors.append(f"{row['lineage_id']}:aoi_chamada_de_ocorrencia")
        # Petropolis misto nao vira inundacao sem separacao
        if row.get("city") == "petropolis" and "misto" in row.get("phenomenon_type", ""):
            if row.get("evidence_tier") in {"A_oficial_poligono_observado", "B_oficial_ponto_observado", "C_footprint_tecnico_sar"}:
                errors.append(f"{row['lineage_id']}:petropolis_misto_promovido")
            if "separac" not in justi:
                errors.append(f"{row['lineage_id']}:petropolis_misto_sem_mencao_separacao")

    # Inventario deve conter todos os marcos e nenhum ground_truth/trainable/score_v7
    inv = _read_csv_rows(INVENTARIO)
    marcos_inv = {r["marco"] for r in inv}
    for m in MARCOS:
        if m["marco"] not in marcos_inv:
            errors.append(f"inventario:marco_ausente:{m['marco']}")
    for row in inv:
        if row["ground_truth_count"] != "0" or row["trainable_count"] != "0" or row["score_v7_count"] != "0":
            errors.append(f"inventario:{row['marco']}:contagem_proibida_nao_zero")
        if row["benchmark_17b_criado"] != "false" or row["score_v6_alterado"] != "false":
            errors.append(f"inventario:{row['marco']}:flag_proibida_true")

    # Plano de proximos marcos nao pode criar score_v7/ground_truth/benchmark
    for row in _read_csv_rows(PLANO_MARCOS):
        if row["cria_score_v7"] != "false" or row["cria_ground_truth"] != "false" or row["cria_benchmark_17b"] != "false":
            errors.append(f"plano:{row['marco_sugerido']}:proposta_proibida")

    # Plano de commit deve excluir local_runs explicitamente
    commit_txt = PLANO_COMMIT.read_text(encoding="utf-8", errors="ignore")
    if "local_runs" not in commit_txt or "nunca entra" not in commit_txt:
        errors.append("plano_commit:local_runs_nao_excluido_explicitamente")
    code_blocks = commit_txt.split("```")
    git_add_block = code_blocks[1] if len(code_blocks) > 1 else ""
    if "local_runs" in git_add_block:
        errors.append("plano_commit:git_add_inclui_local_runs")

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
        "18H consolidacao mestre validada: "
        f"marcos={summary['total_marcos']} regioes={len(summary['regioes'])} "
        f"status17B={summary['status_17b_mestre']} 17B_criado={str(summary['marco_17b_criado']).lower()} "
        f"score_v6_changed={str(summary['score_v6_changed']).lower()}"
    )
    return 0


def run_all() -> int:
    summary = build_all()
    print(
        "18H consolidacao mestre gerada: "
        f"marcos={summary['total_marcos']} linhagem={summary['linhagem_rows']} "
        f"pendencias={summary['pendencias_rows']} status17B={summary['status_17b_mestre']}"
    )
    return 0
