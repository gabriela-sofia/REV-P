"""Curadoria e auditoria de evidencias externas (MV1) para o REV-P.

Esta passada e *review-only*. Ela NAO promove nenhuma fonte externa a
ground truth operacional, NAO cria label, NAO cria negativo formal e NAO
libera treino supervisionado. O objetivo e organizar, auditar e documentar
fontes externas (oficiais ou institucionalmente confiaveis) ja presentes no
repositorio em area git-ignored, alem de registrar fontes-candidatas
pesquisadas mas ainda nao baixadas, sempre com proveniencia rastreavel.

Regras de honestidade aplicadas pelo script:
- SHA256 e tamanho sao calculados a partir do arquivo real em disco; nunca
  inventados. Se o arquivo nao existir, o campo fica vazio e o status de
  download e marcado como ausente_local.
- URLs e datas vem de logs de proveniencia locais reais (ex.: v1if) ou de
  busca web registrada (data de acesso 2026-06-18). Nada e fabricado.
- Geometria de suscetibilidade NUNCA e tratada como geometria de evento
  observado. Landslide scar NUNCA e tratado como flood extent. Fonte textual
  NUNCA substitui geometria auditavel.

Saidas (camada publica leve):
- outputs_public/tables/revp_manifesto_evidencias_externas_mv1.csv
- outputs_public/tables/revp_auditoria_fontes_externas_mv1.csv
- outputs_public/tables/revp_indice_eventos_externos_candidatos_mv1.csv
- outputs_public/tables/revp_indice_geometrias_externas_candidatas_mv1.csv
- outputs_public/execution_reports/revp_curadoria_evidencias_externas_mv1.md
- outputs_public/metrics/revp_curadoria_evidencias_externas_mv1.json

Uso:
    python scripts/curadoria_externa/revp_curadoria_evidencias_externas_mv1.py
    python scripts/curadoria_externa/revp_curadoria_evidencias_externas_mv1.py --check
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_ACESSO_WEB = "2026-06-18"
GUARDRAILS = [
    "REV-P permanece review-only; nenhuma fonte externa vira ground truth operacional nesta passada.",
    "Evidencia externa NAO vira label automaticamente.",
    "unknown nunca vira negativo; ausencia de evento nunca vira classe 0.",
    "Curitiba nunca vira negativo formal.",
    "Desastre documentado nao significa label patch-level fechado.",
    "Landslide scar nao prova flood extent; deslizamento e flood nunca sao misturados automaticamente.",
    "Geometria de suscetibilidade nao e geometria de evento observado.",
    "Fonte textual nao substitui geometria auditavel.",
    "Fonte sem licenca clara fica bloqueada para uso publico direto.",
    "Nenhum SHA256, URL ou data foi inventado; valores derivam de arquivo real ou de log/busca registrada.",
]

# ---------------------------------------------------------------------------
# Registro curado de fontes (somente fatos verificados).
# Campos *_local apontam para arquivos reais git-ignored; o hash e calculado
# em tempo de execucao. Fontes web pesquisadas mas nao baixadas tem
# caminho_local vazio e status_download = nao_baixado.
# ---------------------------------------------------------------------------

FONTES: list[dict] = [
    # ---------------------------- RECIFE -----------------------------------
    {
        "id_fonte": "FONTE_REC_001",
        "regiao": "Nordeste", "cidade": "Recife", "estado": "PE", "pais": "Brasil",
        "instituicao": "Defesa Civil do Recife / Prefeitura do Recife",
        "familia_fonte": "base_gis", "tipo_fonte": "mapa_areas_de_risco",
        "titulo": "Areas de risco mapeadas - Recife (pontos)",
        "descricao_curta": "Pontos de areas de risco (niveis) por bairro/regional; suscetibilidade territorial, nao evento observado.",
        "url_origem": "https://www.recife.pe.gov.br",
        "data_acesso": "", "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "suscetibilidade_territorial",
        "formato_original": "GeoJSON",
        "caminho_local": "datasets/external_sources/recife_minimal_tp/raw/recife_defesa_civil_risk_areas_geojson.geojson",
        "licenca_ou_uso": "indefinida_a_confirmar",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_com_ressalvas",
        "observacao": "Geometria de SUSCETIBILIDADE (Area de Risco), nao de evento observado. Proveniencia local: datasets/external_sources/recife_minimal_tp.",
        "tem_geometria": True, "geometria_tipo": "Point", "crs": "EPSG:4326",
        "representa_evento": False, "representa_suscet": True,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_REC_002",
        "regiao": "Nordeste", "cidade": "Recife", "estado": "PE", "pais": "Brasil",
        "instituicao": "International Charter Space and Major Disasters (Ativacao 758)",
        "familia_fonte": "agencia_internacional", "tipo_fonte": "produto_mapa_evento_digitalizado_candidato",
        "titulo": "Charter 758 - poligono candidato digitalizado (produto MEDIA-871-16)",
        "descricao_curta": "MultiPolygon candidato digitalizado a partir de produto publico do Charter; provided_unreviewed, can_be_ground_truth=false.",
        "url_origem": "https://disasterscharter.org/web/guest/activations/-/article/flood-in-brazil-activation-758-",
        "data_acesso": "", "data_publicacao": "2022",
        "periodo_referencia_inicio": "2022-05-24", "periodo_referencia_fim": "2022-05-30",
        "categoria_evento": "desastre_hidrometeorologico_composto",
        "formato_original": "GeoJSON",
        "caminho_local": "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/derived/event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson",
        "licenca_ou_uso": "produto_charter_uso_restrito_a_confirmar",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "requer_revisao_humana",
        "observacao": "Digitalizacao candidata nao revisada (review_status=provided_unreviewed). Produto pode misturar scars de deslizamento com area de inundacao: NAO assumir flood extent.",
        "tem_geometria": True, "geometria_tipo": "MultiPolygon", "crs": "EPSG:4326 (origem EPSG:32725)",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "media", "landslide_flood": "risco_alto",
    },
    {
        "id_fonte": "FONTE_REC_003",
        "regiao": "Nordeste", "cidade": "Recife", "estado": "PE", "pais": "Brasil",
        "instituicao": "SEDEC - Secretaria Executiva de Defesa Civil do Recife",
        "familia_fonte": "defesa_civil", "tipo_fonte": "registro_de_ocorrencia",
        "titulo": "Dados vivos SEDEC Recife - ocorrencias",
        "descricao_curta": "Registros tabulares de ocorrencias da Defesa Civil municipal; evidencia documental/contextual, sem geometria de evento.",
        "url_origem": "http://dados.recife.pe.gov.br",
        "data_acesso": "", "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "alagamento",
        "formato_original": "CSV",
        "caminho_local": "local_runs/protocolo_c/v1np/raw/recife_dados_vivos_sedec_f3b3a0ab-ac8f-4fe8-a1cc-c3afdd20081a_001.csv",
        "licenca_ou_uso": "dados_abertos_municipais_a_confirmar",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Tabular sem geometria; serve como evidencia observacional contextual, nao como geometria de evento.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_REC_004",
        "regiao": "Nordeste", "cidade": "Recife", "estado": "PE", "pais": "Brasil",
        "instituicao": "EMLURB / Servico 156 - Prefeitura do Recife",
        "familia_fonte": "prefeitura", "tipo_fonte": "registro_de_solicitacao_urbana",
        "titulo": "EMLURB 156 Recife - solicitacoes",
        "descricao_curta": "Solicitacoes de servico urbano (156). Inclui muitos itens nao-desastre; uso contextual com forte ressalva.",
        "url_origem": "http://dados.recife.pe.gov.br",
        "data_acesso": "", "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "alagamento",
        "formato_original": "CSV",
        "caminho_local": "local_runs/protocolo_c/v1np/raw/recife_emlurb_156_031c3ad3-265f-4d9c-830a-7d1f85f830fa_007.csv",
        "licenca_ou_uso": "dados_abertos_municipais_a_confirmar",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_com_ressalvas",
        "observacao": "Base de servico urbano generico; nao e registro de desastre por si so. Filtragem tematica obrigatoria antes de qualquer uso contextual.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": True, "representa_contexto": True,
        "circularidade": "media", "landslide_flood": "nao_aplicavel",
    },
    # --------------------------- PETROPOLIS --------------------------------
    {
        "id_fonte": "FONTE_PET_001",
        "regiao": "Sudeste", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "SGB/CPRM (Servico Geologico do Brasil)",
        "familia_fonte": "orgao_federal", "tipo_fonte": "relatorio_tecnico_pos_desastre",
        "titulo": "Avaliacao tecnica pos-desastre: Petropolis, RJ (2022)",
        "descricao_curta": "Relatorio tecnico do SGB/CPRM sobre os deslizamentos de fevereiro/2022. Documento textual; nao e vetor de evento.",
        "url_origem": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "data_acesso": "2026-05-22", "data_publicacao": "2022",
        "periodo_referencia_inicio": "2022-02-15", "periodo_referencia_fim": "2022-03-02",
        "categoria_evento": "deslizamento",
        "formato_original": "PDF",
        "caminho_local": "local_runs/protocolo_c/v1if/raw_official_sources/Relatorio_Tecnico_Petropolis.pdf",
        "licenca_ou_uso": "publicacao_oficial_rigeo_sgb",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Evento de DESLIZAMENTO. Nao confundir com flood extent. Proveniencia: v1if_official_source_download_log.csv (OBS_PET_001).",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "risco_alto",
    },
    {
        "id_fonte": "FONTE_PET_002",
        "regiao": "Sudeste", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "SGB/CPRM (Servico Geologico do Brasil)",
        "familia_fonte": "orgao_federal", "tipo_fonte": "anexos_avaliacao_pos_desastre",
        "titulo": "Anexos da avaliacao pos-desastre Petropolis 2022 (ZIP)",
        "descricao_curta": "ZIP de anexos (~20 MB). Pode conter mapas/croquis georreferenciados nao confirmados como vetor auditavel.",
        "url_origem": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "data_acesso": "2026-05-22", "data_publicacao": "2022",
        "periodo_referencia_inicio": "2022-02-15", "periodo_referencia_fim": "2022-03-02",
        "categoria_evento": "deslizamento",
        "formato_original": "ZIP",
        "caminho_local": "local_runs/protocolo_c/v1if/raw_official_sources/anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip",
        "licenca_ou_uso": "publicacao_oficial_rigeo_sgb",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "requer_revisao_humana",
        "observacao": "Artefato pesado: permanece em local_only. Conteudo vetorial nao confirmado; requer inspecao manual antes de qualquer uso geometrico.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "risco_alto",
    },
    {
        "id_fonte": "FONTE_PET_003",
        "regiao": "Sudeste", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "SGB/CPRM (Servico Geologico do Brasil)",
        "familia_fonte": "orgao_federal", "tipo_fonte": "relatorio_tecnico_bairro",
        "titulo": "ANEXO I - CPRM Petropolis 19-02-22 Bairro Mosella",
        "descricao_curta": "Laudo tecnico de deslizamento por bairro (Mosella). Representativo de 11 anexos bairro a bairro. Textual, sem vetor.",
        "url_origem": "https://rigeo.sgb.gov.br/handle/doc/22668",
        "data_acesso": "2026-05-22", "data_publicacao": "2022",
        "periodo_referencia_inicio": "2022-02-19", "periodo_referencia_fim": "2022-02-19",
        "categoria_evento": "deslizamento",
        "formato_original": "PDF",
        "caminho_local": "local_runs/protocolo_c/v1mc/extracted/PKG_V1MC_0001_24be39ae/ANEXO-I-CPRM_Relatório_Petrópolis_19-02-22_Bairro_Mosella.pdf",
        "licenca_ou_uso": "publicacao_oficial_rigeo_sgb",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Um de 11 anexos por bairro (extraidos do ZIP OBS_PET_002). Evento de deslizamento; nao flood.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "risco_alto",
    },
    {
        "id_fonte": "FONTE_PET_004",
        "regiao": "Sudeste", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "Prefeitura Municipal de Petropolis",
        "familia_fonte": "prefeitura", "tipo_fonte": "diario_oficial_decreto",
        "titulo": "Diario Oficial do Municipio de Petropolis - fevereiro/2022",
        "descricao_curta": "Edicao do Diario Oficial municipal (poder executivo). Suporta contexto de decretos/emergencia; sem geometria.",
        "url_origem": "https://www.petropolis.rj.gov.br",
        "data_acesso": "", "data_publicacao": "2022-02-15",
        "periodo_referencia_inicio": "2022-02-14", "periodo_referencia_fim": "2022-02-15",
        "categoria_evento": "desastre_hidrometeorologico_composto",
        "formato_original": "PDF",
        "caminho_local": "local_runs/protocolo_c/v1na/raw/gazette_issue_2022-02-15_4542_0a45c34e.pdf",
        "licenca_ou_uso": "ato_oficial_publico",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Confirmado como D.O. Municipio de Petropolis (cabecalho da pagina 1). Util para datar decretos de emergencia; nao define geometria de evento.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    # ---------------------------- CURITIBA ---------------------------------
    {
        "id_fonte": "FONTE_CUR_001",
        "regiao": "Sul", "cidade": "Curitiba", "estado": "PR", "pais": "Brasil",
        "instituicao": "Coordenadoria Estadual da Defesa Civil do Parana",
        "familia_fonte": "defesa_civil", "tipo_fonte": "noticia_institucional",
        "titulo": "Chuvas fortes causam transtornos em Curitiba e no Litoral (Defesa Civil PR)",
        "descricao_curta": "Pagina institucional da Defesa Civil estadual sobre chuvas de janeiro/2022. Contexto textual; sem coordenada.",
        "url_origem": "https://www.defesacivil.pr.gov.br",
        "data_acesso": "2026-06-14", "data_publicacao": "2022-01",
        "periodo_referencia_inicio": "2022-01-05", "periodo_referencia_fim": "2022-01-15",
        "categoria_evento": "alagamento",
        "formato_original": "HTML",
        "caminho_local": "local_runs/ground_truth/v2cd/downloaded_sources/SRC_4c0debd1596a.html",
        "licenca_ou_uso": "conteudo_institucional_publico",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Classificada como CONTEXT_ONLY_NO_GEOMETRY na passada v2cd. Nenhuma coordenada extraivel; nao sustenta geometria.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_CUR_002",
        "regiao": "Sul", "cidade": "Curitiba", "estado": "PR", "pais": "Brasil",
        "instituicao": "Fonte oficial (one-hop a partir da Defesa Civil PR)",
        "familia_fonte": "governo_estadual", "tipo_fonte": "documento_pdf_contextual",
        "titulo": "PDF contextual capturado em crawl de 1 salto (v2ce)",
        "descricao_curta": "Recorte PDF (~580 KB) obtido a 1 salto. Conteudo a confirmar; classificado como contexto, sem geometria valida.",
        "url_origem": "https://www.defesacivil.pr.gov.br",
        "data_acesso": "2026-06-14", "data_publicacao": "",
        "periodo_referencia_inicio": "2022-01-05", "periodo_referencia_fim": "2022-01-15",
        "categoria_evento": "alagamento",
        "formato_original": "PDF",
        "caminho_local": "local_runs/ground_truth/v2ce/downloaded_onehop_sources/SRC_9b151d28ab38.pdf",
        "licenca_ou_uso": "indefinida_a_confirmar",
        "status_download": "ja_presente_no_repositorio",
        "status_auditoria": "requer_revisao_humana",
        "observacao": "0 coordenadas validas extraidas (v2ce). Necessita inspecao manual; nao promover.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_CUR_003",
        "regiao": "Sul", "cidade": "Curitiba", "estado": "PR", "pais": "Brasil",
        "instituicao": "IPPUC / GeoCuritiba - Prefeitura de Curitiba",
        "familia_fonte": "base_gis", "tipo_fonte": "mapeamento_alagamento",
        "titulo": "Mapeamento de areas de alagamento Curitiba (IPPUC/GeoCuritiba)",
        "descricao_curta": "Fonte GIS municipal alvo; sem URL de download direto registrada. Requer busca/solicitacao manual.",
        "url_origem": "https://www.ippuc.org.br",
        "data_acesso": "2026-05-22", "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "suscetibilidade_territorial",
        "formato_original": "DESCONHECIDO",
        "caminho_local": "",
        "licenca_ou_uso": "indefinida_a_confirmar",
        "status_download": "nao_baixado",
        "status_auditoria": "bloqueada_sem_origem_rastreavel",
        "observacao": "Sem URL direta no log v1if (OBS_CUR_001). Curitiba nunca vira negativo formal por ausencia de evidencia.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": True,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    # ------------------- FONTES NACIONAIS (pesquisadas) --------------------
    {
        "id_fonte": "FONTE_NAC_001",
        "regiao": "Nacional", "cidade": "", "estado": "", "pais": "Brasil",
        "instituicao": "CEMADEN/MCTI - Centro Nacional de Monitoramento e Alertas de Desastres Naturais",
        "familia_fonte": "instituto_pesquisa", "tipo_fonte": "mapa_interativo_alertas_suscetibilidade",
        "titulo": "CEMADEN - Mapa Interativo / GeoRisk",
        "descricao_curta": "Portal de alertas e areas suscetiveis (deslizamento/inundacao). Pesquisado; nao baixado nesta passada.",
        "url_origem": "https://mapainterativo.cemaden.gov.br/",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "suscetibilidade_territorial",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "uso_publico_termos_a_confirmar",
        "status_download": "nao_baixado",
        "status_auditoria": "requer_revisao_humana",
        "observacao": "Tambem: GeoRisk (georisk.cemaden.gov.br) e gov.br/cemaden. Suscetibilidade nao e geometria de evento observado.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": True,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_NAC_002",
        "regiao": "Nacional", "cidade": "", "estado": "", "pais": "Brasil",
        "instituicao": "ANA - Agencia Nacional de Aguas e Saneamento Basico (SNIRH/HidroWeb)",
        "familia_fonte": "orgao_federal", "tipo_fonte": "serie_hidrologica",
        "titulo": "ANA HidroWeb / SNIRH - series hidrometeorologicas",
        "descricao_curta": "Series de nivel/vazao/pluviometria (ex.: bacia do Capibaribe). Pesquisado; nao baixado nesta passada.",
        "url_origem": "https://www.snirh.gov.br/hidroweb/",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "chuva_extrema",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "dados_abertos_ana",
        "status_download": "nao_baixado",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Tambem dadosabertos.ana.gov.br. Serie pontual/estacao; nao e geometria de extensao de evento.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_NAC_003",
        "regiao": "Nacional", "cidade": "", "estado": "", "pais": "Brasil",
        "instituicao": "MapBiomas",
        "familia_fonte": "instituto_pesquisa", "tipo_fonte": "uso_cobertura_solo",
        "titulo": "MapBiomas - Colecao 10.1 (uso e cobertura do solo)",
        "descricao_curta": "Mapas anuais de uso/cobertura do solo (Brasil). Contexto territorial; pesquisado, nao baixado.",
        "url_origem": "https://brasil.mapbiomas.org/",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "2026",
        "periodo_referencia_inicio": "1985", "periodo_referencia_fim": "2024",
        "categoria_evento": "suscetibilidade_territorial",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "CC_BY_SA_mapbiomas_a_confirmar_versao",
        "status_download": "nao_baixado",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Contexto de cobertura, nao geometria de evento. Verificar versao/licenca antes de uso publico.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": True,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_NAC_004",
        "regiao": "Nacional", "cidade": "", "estado": "", "pais": "Brasil",
        "instituicao": "IBGE - Instituto Brasileiro de Geografia e Estatistica",
        "familia_fonte": "orgao_federal", "tipo_fonte": "malha_setor_censitario",
        "titulo": "IBGE - Malha de Setores Censitarios / Downloads Geociencias",
        "descricao_curta": "Malha de setores censitarios e bases territoriais. Contexto socioterritorial; pesquisado, nao baixado.",
        "url_origem": "https://www.ibge.gov.br/geociencias/downloads-geociencias.html",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "vulnerabilidade_socioambiental",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "dados_publicos_ibge",
        "status_download": "nao_baixado",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Geometria administrativa/censitaria (contexto), nao geometria de evento observado.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": True, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_NAC_005",
        "regiao": "Nordeste", "cidade": "Recife", "estado": "PE", "pais": "Brasil",
        "instituicao": "APAC - Agencia Pernambucana de Aguas e Clima",
        "familia_fonte": "governo_estadual", "tipo_fonte": "boletim_pluviometrico_hidrologico",
        "titulo": "APAC - boletins de chuva/rios/reservatorios (PE)",
        "descricao_curta": "Boletins e monitoramento pluvio-hidrologico de Pernambuco. Pesquisado; nao baixado nesta passada.",
        "url_origem": "https://www.apac.pe.gov.br",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "chuva_extrema",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "uso_publico_termos_a_confirmar",
        "status_download": "nao_baixado",
        "status_auditoria": "requer_revisao_humana",
        "observacao": "Boletim/serie regional; suporte contextual de chuva, nao geometria de evento.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": False, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "nao_aplicavel",
    },
    {
        "id_fonte": "FONTE_NAC_006",
        "regiao": "Sudeste", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "DRM-RJ / NADE - Servico Geologico do Estado do Rio de Janeiro",
        "familia_fonte": "governo_estadual", "tipo_fonte": "relatorio_pos_desastre",
        "titulo": "DRM-RJ/NADE - relatorio pos-desastre Petropolis 2022",
        "descricao_curta": "Relatorio estadual pos-desastre; sem URL direta. Requer solicitacao formal.",
        "url_origem": "https://www.drm.rj.gov.br",
        "data_acesso": "2026-05-22", "data_publicacao": "",
        "periodo_referencia_inicio": "2022-02-15", "periodo_referencia_fim": "",
        "categoria_evento": "deslizamento",
        "formato_original": "DESCONHECIDO",
        "caminho_local": "",
        "licenca_ou_uso": "indefinida_a_confirmar",
        "status_download": "nao_baixado",
        "status_auditoria": "bloqueada_sem_origem_rastreavel",
        "observacao": "Log v1if (OBS_PET_003): NO_URL, requer solicitacao formal DRM-RJ/NADE.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "risco_alto",
    },
    # ------------------ FONTES INTERNACIONAIS (pesquisadas) ----------------
    {
        "id_fonte": "FONTE_INT_001",
        "regiao": "Internacional", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "Copernicus Emergency Management Service (EMS) Rapid Mapping",
        "familia_fonte": "agencia_internacional", "tipo_fonte": "mapa_resposta_emergencial",
        "titulo": "Copernicus EMS Rapid Mapping - ativacao Petropolis 2022 (a confirmar)",
        "descricao_curta": "Servico europeu de mapeamento rapido de emergencia. Numero de ativacao para Petropolis nao confirmado.",
        "url_origem": "https://rapidmapping.emergency.copernicus.eu/",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "2022",
        "periodo_referencia_inicio": "2022-02-15", "periodo_referencia_fim": "",
        "categoria_evento": "deslizamento",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "copernicus_ems_full_open_a_confirmar",
        "status_download": "nao_baixado",
        "status_auditoria": "requer_revisao_humana",
        "observacao": "ID exato da ativacao (EMSR) para Petropolis nao foi confirmado na busca; nao inventar. Produto pode delimitar deslizamento, nao flood.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "risco_alto",
    },
    {
        "id_fonte": "FONTE_INT_002",
        "regiao": "Internacional", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
        "instituicao": "NHESS (Copernicus Publications) - artigo academico",
        "familia_fonte": "artigo_academico", "tipo_fonte": "artigo_revisado_por_pares",
        "titulo": "Deadly disasters... flash floods and landslides of February 2022 in Petropolis (NHESS, 2023)",
        "descricao_curta": "Artigo academico sobre o evento de Petropolis. Fonte documental secundaria, nao geometria operacional.",
        "url_origem": "https://nhess.copernicus.org/articles/23/1157/2023/",
        "data_acesso": DATA_ACESSO_WEB, "data_publicacao": "2023",
        "periodo_referencia_inicio": "2022-02-15", "periodo_referencia_fim": "2022-02-15",
        "categoria_evento": "desastre_hidrometeorologico_composto",
        "formato_original": "PORTAL_WEB",
        "caminho_local": "",
        "licenca_ou_uso": "CC_BY_4.0",
        "status_download": "nao_baixado",
        "status_auditoria": "aprovada_com_ressalvas",
        "observacao": "NHESS vol. 23, p. 1157, 2023. Usar como referencia documental; nao tratar figuras do artigo como geometria auditavel.",
        "tem_geometria": False, "geometria_tipo": "", "crs": "",
        "representa_evento": True, "representa_suscet": False,
        "representa_vulner": False, "representa_contexto": True,
        "circularidade": "baixa", "landslide_flood": "risco_alto",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_of(rel_path: str) -> tuple[str, int]:
    """Retorna (sha256, tamanho_bytes) do arquivo real ou ("", -1) se ausente."""
    if not rel_path:
        return "", -1
    p = ROOT / rel_path
    if not p.is_file():
        return "", -1
    h = hashlib.sha256()
    size = 0
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


MANIFEST_COLS = [
    "id_fonte", "regiao", "cidade", "estado", "pais", "instituicao",
    "familia_fonte", "tipo_fonte", "titulo", "descricao_curta", "url_origem",
    "data_acesso", "data_publicacao", "periodo_referencia_inicio",
    "periodo_referencia_fim", "categoria_evento", "formato_original",
    "caminho_local_quarentena", "sha256_arquivo", "licenca_ou_uso",
    "status_download", "status_auditoria", "observacao",
]

AUDIT_COLS = ["id_fonte", "criterio", "status", "evidencia", "risco", "severidade", "acao_recomendada"]

EVENT_COLS = [
    "id_evento_externo", "id_fonte", "cidade", "regiao", "categoria_evento",
    "data_evento_inicio", "data_evento_fim", "janela_temporal_status",
    "descricao_evento", "fonte_observacional_independente", "geometria_disponivel",
    "geometria_tipo", "crs", "escala_espacial", "pode_sustentar_label_patch_level",
    "motivo_nao_label", "status_evento", "observacao",
]

GEOM_COLS = [
    "id_geometria", "id_fonte", "cidade", "regiao", "tipo_geometria", "formato",
    "crs", "escala", "caminho_local_quarentena", "sha256_arquivo",
    "representa_evento_observado", "representa_suscetibilidade",
    "representa_vulnerabilidade", "representa_contexto",
    "adequada_para_overlay_patch", "motivo_bloqueio_overlay", "status_geometria",
    "observacao",
]


def build_rows():
    manifest, audit, events, geoms = [], [], [], []

    for f in FONTES:
        sha, size = sha256_of(f["caminho_local"])
        status_download = f["status_download"]
        if f["caminho_local"] and size < 0:
            status_download = "ausente_local"

        manifest.append({
            "id_fonte": f["id_fonte"], "regiao": f["regiao"], "cidade": f["cidade"],
            "estado": f["estado"], "pais": f["pais"], "instituicao": f["instituicao"],
            "familia_fonte": f["familia_fonte"], "tipo_fonte": f["tipo_fonte"],
            "titulo": f["titulo"], "descricao_curta": f["descricao_curta"],
            "url_origem": f["url_origem"], "data_acesso": f["data_acesso"],
            "data_publicacao": f["data_publicacao"],
            "periodo_referencia_inicio": f["periodo_referencia_inicio"],
            "periodo_referencia_fim": f["periodo_referencia_fim"],
            "categoria_evento": f["categoria_evento"],
            "formato_original": f["formato_original"],
            "caminho_local_quarentena": f["caminho_local"],
            "sha256_arquivo": sha, "licenca_ou_uso": f["licenca_ou_uso"],
            "status_download": status_download,
            "status_auditoria": f["status_auditoria"],
            "observacao": f["observacao"],
        })

        audit.extend(build_audit_rows(f))

        # Evento externo candidato (somente fontes que representam um evento observado).
        if f["representa_evento"]:
            janela = "fechada" if f["periodo_referencia_inicio"] and f["periodo_referencia_fim"] else "aberta_ou_indefinida"
            pode = decide_pode_label(f, janela)
            events.append({
                "id_evento_externo": "EV_" + f["id_fonte"],
                "id_fonte": f["id_fonte"], "cidade": f["cidade"], "regiao": f["regiao"],
                "categoria_evento": f["categoria_evento"],
                "data_evento_inicio": f["periodo_referencia_inicio"],
                "data_evento_fim": f["periodo_referencia_fim"],
                "janela_temporal_status": janela,
                "descricao_evento": f["titulo"],
                "fonte_observacional_independente": "sim" if f["familia_fonte"] in {"agencia_internacional", "orgao_federal", "instituto_pesquisa"} else "parcial",
                "geometria_disponivel": "sim" if f["tem_geometria"] else "nao",
                "geometria_tipo": f["geometria_tipo"], "crs": f["crs"],
                "escala_espacial": "municipal_ou_bairro",
                "pode_sustentar_label_patch_level": "true" if pode else "false",
                "motivo_nao_label": motivo_nao_label(f, janela, pode),
                "status_evento": "candidato_para_revisao_futura",
                "observacao": f["observacao"],
            })

        # Geometria externa candidata (somente fontes com geometria real).
        if f["tem_geometria"]:
            _, motivo = avalia_overlay(f)
            geoms.append({
                "id_geometria": "GEO_" + f["id_fonte"],
                "id_fonte": f["id_fonte"], "cidade": f["cidade"], "regiao": f["regiao"],
                "tipo_geometria": f["geometria_tipo"], "formato": f["formato_original"],
                "crs": f["crs"], "escala": "local",
                "caminho_local_quarentena": f["caminho_local"], "sha256_arquivo": sha,
                "representa_evento_observado": "true" if f["representa_evento"] else "false",
                "representa_suscetibilidade": "true" if f["representa_suscet"] else "false",
                "representa_vulnerabilidade": "true" if f["representa_vulner"] else "false",
                "representa_contexto": "true" if f["representa_contexto"] else "false",
                "adequada_para_overlay_patch": "false",  # nesta tarefa nenhum overlay e feito
                "motivo_bloqueio_overlay": motivo,
                "status_geometria": "geometria_candidata_nao_promovida",
                "observacao": f["observacao"],
            })

    return manifest, audit, events, geoms


def decide_pode_label(f: dict, janela: str) -> bool:
    """pode_sustentar_label_patch_level: false por padrao; true exige tudo verde."""
    return (
        f["representa_evento"]
        and f["tem_geometria"]
        and janela == "fechada"
        and bool(f["crs"])
        and f["circularidade"] == "baixa"
        and f["landslide_flood"] != "risco_alto"
        and f["status_auditoria"] not in {"bloqueada_sem_origem_rastreavel", "bloqueada_sem_metadados"}
    )


def motivo_nao_label(f: dict, janela: str, pode: bool) -> str:
    if pode:
        return "candidato_apenas_para_revisao_futura_nao_libera_treino"
    motivos = []
    if not f["tem_geometria"]:
        motivos.append("sem_geometria_auditavel")
    if janela != "fechada":
        motivos.append("janela_temporal_nao_fechada")
    if not f["crs"]:
        motivos.append("sem_crs_declarado")
    if f["landslide_flood"] == "risco_alto":
        motivos.append("risco_landslide_vs_flood")
    if f["status_auditoria"].startswith("bloqueada"):
        motivos.append("fonte_bloqueada_na_auditoria")
    return ";".join(motivos) or "evidencia_apenas_contextual"


def avalia_overlay(f: dict) -> tuple[bool, str]:
    if f["representa_suscet"]:
        return False, "geometria_de_suscetibilidade_nao_e_evento_observado"
    if f["landslide_flood"] == "risco_alto":
        return False, "produto_pode_misturar_deslizamento_e_inundacao_revisao_obrigatoria"
    if f["status_auditoria"] == "requer_revisao_humana":
        return False, "geometria_nao_revisada_provided_unreviewed"
    return False, "overlay_nao_executado_nesta_tarefa_review_only"


def build_audit_rows(f: dict) -> list[dict]:
    rows = []

    def add(criterio, status, evidencia, risco, severidade, acao):
        rows.append({"id_fonte": f["id_fonte"], "criterio": criterio, "status": status,
                     "evidencia": evidencia, "risco": risco, "severidade": severidade,
                     "acao_recomendada": acao})

    # origem institucional rastreavel
    rastreavel = bool(f["url_origem"]) and f["instituicao"]
    add("origem_institucional_rastreavel",
        "ok" if rastreavel else "falha",
        f"instituicao={f['instituicao']}; url={f['url_origem'] or 'nenhuma'}",
        "fonte_nao_atribuivel" if not rastreavel else "nenhum",
        "alta" if not rastreavel else "baixa",
        "registrar_origem" if not rastreavel else "manter")

    # data de acesso registrada
    add("data_de_acesso_registrada",
        "ok" if f["data_acesso"] else "parcial",
        f"data_acesso={f['data_acesso'] or 'nao_registrada_em_log'}",
        "proveniencia_temporal_incompleta" if not f["data_acesso"] else "nenhum",
        "media" if not f["data_acesso"] else "baixa",
        "recuperar_data_de_acesso_da_proveniencia_local" if not f["data_acesso"] else "manter")

    # periodo de referencia claro
    periodo_ok = bool(f["periodo_referencia_inicio"])
    add("periodo_de_referencia_claro",
        "ok" if periodo_ok else "parcial",
        f"inicio={f['periodo_referencia_inicio'] or '-'}; fim={f['periodo_referencia_fim'] or '-'}",
        "janela_temporal_ambigua" if not periodo_ok else "nenhum",
        "media" if not periodo_ok else "baixa",
        "definir_janela_temporal" if not periodo_ok else "manter")

    # categoria de evento clara
    add("categoria_de_evento_clara", "ok",
        f"categoria_evento={f['categoria_evento']}", "nenhum", "baixa", "manter")

    # geometria disponivel ou ausente
    add("geometria_disponivel_ou_ausente",
        "ok" if f["tem_geometria"] else "ausente",
        f"tem_geometria={str(f['tem_geometria']).lower()}; tipo={f['geometria_tipo'] or '-'}",
        "fonte_documental_sem_geometria" if not f["tem_geometria"] else "nenhum",
        "baixa", "classificar_como_documental_contextual" if not f["tem_geometria"] else "manter_como_geometria_candidata")

    # CRS declarado se houver geometria
    if f["tem_geometria"]:
        add("crs_declarado_quando_ha_geometria",
            "ok" if f["crs"] else "falha",
            f"crs={f['crs'] or 'nao_declarado'}",
            "georreferenciamento_nao_auditavel" if not f["crs"] else "nenhum",
            "alta" if not f["crs"] else "baixa",
            "declarar_crs" if not f["crs"] else "manter")

    # licenca/uso claro
    licenca_clara = "a_confirmar" not in f["licenca_ou_uso"] and "indefinida" not in f["licenca_ou_uso"]
    add("licenca_ou_uso_claro",
        "ok" if licenca_clara else "parcial",
        f"licenca={f['licenca_ou_uso']}",
        "uso_publico_direto_bloqueado" if not licenca_clara else "nenhum",
        "media" if not licenca_clara else "baixa",
        "bloquear_para_uso_publico_ate_confirmar_licenca" if not licenca_clara else "manter")

    # risco de circularidade
    add("risco_de_circularidade",
        "ok" if f["circularidade"] == "baixa" else "atencao",
        f"nivel={f['circularidade']}",
        "evidencia_pode_derivar_do_proprio_pipeline" if f["circularidade"] != "baixa" else "nenhum",
        "alta" if f["circularidade"] == "alta" else ("media" if f["circularidade"] == "media" else "baixa"),
        "isolar_fonte_independente" if f["circularidade"] != "baixa" else "manter")

    # risco de confundir landslide com flood
    add("risco_landslide_vs_flood",
        "atencao" if f["landslide_flood"] == "risco_alto" else "ok",
        f"avaliacao={f['landslide_flood']}",
        "scar_de_deslizamento_tratado_como_flood_extent" if f["landslide_flood"] == "risco_alto" else "nenhum",
        "alta" if f["landslide_flood"] == "risco_alto" else "baixa",
        "nunca_misturar_deslizamento_com_inundacao" if f["landslide_flood"] == "risco_alto" else "manter")

    # risco de usar evidencia contextual como label
    add("risco_evidencia_contextual_como_label",
        "atencao" if f["representa_contexto"] else "ok",
        "fonte marcada como contexto/suscetibilidade",
        "promover_contexto_a_label_patch_level",
        "alta",
        "manter_como_contexto_bloqueado_para_label")

    # compatibilidade com Protocolo C
    add("compatibilidade_com_protocolo_c", "ok",
        "review-only; sem label/negativo/treino nesta passada",
        "nenhum", "baixa", "manter")

    # compatibilidade com ground truth futuro
    gt_futuro = f["tem_geometria"] and f["representa_evento"] and f["circularidade"] == "baixa"
    add("compatibilidade_com_ground_truth_futuro",
        "potencial" if gt_futuro else "limitada",
        "tem geometria de evento + baixa circularidade" if gt_futuro else "sem geometria de evento auditavel",
        "nenhum", "baixa",
        "encaminhar_para_revisao_humana_futura" if gt_futuro else "tratar_como_suporte_documental")

    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, cols: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def build_metrics(manifest, events, geoms) -> dict:
    baixadas = [m for m in manifest if m["status_download"] == "ja_presente_no_repositorio"]
    aprovadas = [m for m in manifest if m["status_auditoria"] in {"aprovada_para_contexto", "aprovada_com_ressalvas"}]
    bloqueadas = [m for m in manifest if m["status_auditoria"].startswith("bloqueada")]
    com_geom = [m for m in manifest if any(g["id_fonte"] == m["id_fonte"] for g in geoms)]
    com_crs = [g for g in geoms if g["crs"]]
    eventos_revisaveis = [e for e in events if e["pode_sustentar_label_patch_level"] == "true"]
    return {
        "total_fontes_pesquisadas": len(manifest),
        "total_fontes_baixadas": len(baixadas),
        "total_fontes_aprovadas_para_contexto": len(aprovadas),
        "total_fontes_bloqueadas": len(bloqueadas),
        "total_eventos_candidatos": len(events),
        "total_geometrias_candidatas": len(geoms),
        "contagem_por_cidade": dict(Counter(m["cidade"] or "nacional_ou_internacional" for m in manifest)),
        "contagem_por_familia_fonte": dict(Counter(m["familia_fonte"] for m in manifest)),
        "contagem_por_categoria_evento": dict(Counter(m["categoria_evento"] for m in manifest)),
        "fontes_com_geometria": len(com_geom),
        "fontes_sem_geometria": len(manifest) - len(com_geom),
        "fontes_com_crs_declarado": len(com_crs),
        "fontes_sem_crs_declarado": len(geoms) - len(com_crs),
        "eventos_potencialmente_revisaveis": len(eventos_revisaveis),
        "eventos_que_nao_podem_virar_label": len(events) - len(eventos_revisaveis),
        "guardrails_preservados": GUARDRAILS,
        "proximo_passo_recomendado": (
            "Submeter geometrias candidatas (Charter 758 / areas de risco Recife) a revisao humana "
            "e solicitar formalmente fontes bloqueadas (DRM-RJ, IPPUC, Defesa Civil PE), sem promover "
            "nada a ground truth nesta passada."
        ),
    }


def write_report(path: Path, manifest, events, geoms, metrics) -> None:
    aprovadas = [m for m in manifest if m["status_auditoria"].startswith("aprovada")]
    bloqueadas = [m for m in manifest if m["status_auditoria"].startswith("bloqueada")]
    revisao = [m for m in manifest if m["status_auditoria"] == "requer_revisao_humana"]
    baixadas = [m for m in manifest if m["status_download"] == "ja_presente_no_repositorio"]
    nao_baixadas = [m for m in manifest if m["status_download"] == "nao_baixado"]

    def linha(m):
        return f"- `{m['id_fonte']}` — {m['instituicao']} — {m['titulo']} ({m['status_auditoria']})"

    L = []
    L.append("# Curadoria de evidencias externas (MV1)\n")
    L.append("> Passada review-only. Nenhuma fonte externa foi promovida a ground truth operacional, "
             "nenhum label foi criado, nenhum negativo formal foi criado e nenhum treino foi liberado.\n")

    L.append("## 1. Escopo da curadoria externa\n")
    L.append("Baixar (quando ja disponivel localmente), auditar, organizar e documentar fontes externas "
             "oficiais ou institucionalmente confiaveis para Recife, Petropolis e Curitiba, alem de fontes "
             "nacionais/internacionais de contexto. Toda evidencia serve como suporte observacional/contextual "
             "e NAO vira label automaticamente.\n")

    L.append("## 2. Relacao com o marco label-free MV1\n")
    L.append("Esta curadoria e paralela e nao interfere no fechamento do marco "
             "`marco/validacao-label-free-evidencia-estrutural-mv1`. Ela apenas prepara um pacote auditavel de "
             "evidencias externas para uso futuro, preservando todos os guardrails do REV-P (Sentinel-first, "
             "DINOv2 congelado, sem ground truth operacional patch-level, sem classe binaria, sem negativo formal).\n")

    L.append("## 3. Fontes pesquisadas\n")
    L.append(f"Total de {metrics['total_fontes_pesquisadas']} fontes inventariadas. "
             "Fontes locais ja presentes no repositorio foram auditadas a partir do arquivo real (SHA256 calculado). "
             "Fontes nacionais/internacionais foram pesquisadas via busca web (data de acesso "
             f"{DATA_ACESSO_WEB}) e registradas como candidatas nao baixadas.\n")
    L.extend(linha(m) for m in manifest)
    L.append("")

    L.append("## 4. Fontes baixadas\n")
    L.append(f"{len(baixadas)} fontes ja presentes no repositorio (area git-ignored), com SHA256 calculado:\n")
    L.extend(f"- `{m['id_fonte']}` — `{m['caminho_local_quarentena']}` — SHA256 `{m['sha256_arquivo'][:16]}...`"
             for m in baixadas)
    L.append("")

    L.append("## 5. Fontes nao baixadas e motivo\n")
    L.extend(f"- `{m['id_fonte']}` — {m['instituicao']} — {m['observacao']}" for m in nao_baixadas)
    L.append("")

    L.append("## 6. Organizacao dos arquivos em quarentena\n")
    L.append("A estrutura `local_only/evidencias_externas_quarentena/{recife,petropolis,curitiba,"
             "fontes_nacionais,fontes_internacionais}/` foi criada para ingestao de novos downloads. "
             "Os arquivos brutos ja existentes permanecem em sua localizacao git-ignored original "
             "(`local_runs/...`, `datasets/external_sources/...`) e sao referenciados por caminho relativo "
             "no manifesto, para evitar duplicacao de artefatos pesados. Apenas manifestos, auditorias, "
             "indices e checksums foram publicados em `outputs_public/`.\n")

    L.append("## 7. Fontes aprovadas para contexto\n")
    L.extend(linha(m) for m in aprovadas)
    L.append("")

    L.append("## 8. Fontes bloqueadas\n")
    L.extend(linha(m) for m in bloqueadas)
    if not bloqueadas:
        L.append("- (nenhuma)")
    L.append("\n### Fontes que requerem revisao humana\n")
    L.extend(linha(m) for m in revisao)
    L.append("")

    L.append("## 9. Eventos externos candidatos\n")
    L.append(f"{metrics['total_eventos_candidatos']} eventos candidatos; "
             f"{metrics['eventos_potencialmente_revisaveis']} potencialmente revisaveis (apenas para revisao "
             "futura, sem liberar treino) e "
             f"{metrics['eventos_que_nao_podem_virar_label']} que nao podem virar label.\n")
    L.extend(f"- `{e['id_evento_externo']}` ({e['cidade']}, {e['categoria_evento']}) — "
             f"pode_label={e['pode_sustentar_label_patch_level']} — motivo: {e['motivo_nao_label']}"
             for e in events)
    L.append("")

    L.append("## 10. Geometrias externas candidatas\n")
    L.extend(f"- `{g['id_geometria']}` ({g['cidade']}, {g['tipo_geometria']}, {g['crs']}) — "
             f"overlay={g['adequada_para_overlay_patch']} — bloqueio: {g['motivo_bloqueio_overlay']}"
             for g in geoms)
    if not geoms:
        L.append("- (nenhuma geometria candidata)")
    L.append("")

    L.append("## 11. Riscos metodologicos\n")
    L.append("- Evidencia administrativa/textual nao fecha geometria patch-level.\n"
             "- Suscetibilidade (areas de risco) nao e o mesmo que extensao de evento observado.\n"
             "- Produtos de midia/charter podem ser parcialmente digitalizados e nao revisados.\n")

    L.append("## 12. Risco de circularidade\n")
    L.append("Fontes derivadas do proprio pipeline ou que reaproveitam saidas internas podem inflar concordancia. "
             "Nenhuma fonte de alta circularidade foi aprovada para uso direto; bases de servico urbano (EMLURB 156) "
             "exigem filtragem tematica.\n")

    L.append("## 13. Risco landslide vs flood\n")
    L.append("As fontes de Petropolis e a ativacao Charter 758 referem-se predominantemente a DESLIZAMENTO. "
             "Scar de deslizamento NAO prova flood extent. Deslizamento e inundacao nunca sao combinados "
             "automaticamente.\n")

    L.append("## 14. Risco de evidencia textual sem geometria\n")
    L.append("A maioria das fontes (relatorios CPRM, diarios oficiais, boletins, noticias) e textual e foi "
             "classificada como documental/contextual. Texto nao substitui geometria auditavel.\n")

    L.append("## 15. Risco de licenca/uso\n")
    L.append("Varias fontes tem licenca a confirmar (Charter, portais municipais, CEMADEN, APAC). "
             "Toda fonte sem licenca clara permanece bloqueada para uso publico direto.\n")

    L.append("## 16. Como essas fontes podem ajudar o ground truth futuro\n")
    L.append("- Geometrias candidatas (Charter 758, areas de risco Recife) podem, apos revisao humana e "
             "confirmacao de CRS/janela temporal, sustentar candidatos para revisao futura.\n"
             "- Series ANA/APAC e relatorios CPRM ancoram a janela temporal e a natureza fisica do evento.\n"
             "- Bases IBGE/MapBiomas fornecem contexto territorial para estratificacao futura.\n")

    L.append("## 17. O que ainda falta\n")
    L.append("- Geometria de EVENTO observado, georreferenciada, revisada e com CRS auditavel para as tres cidades.\n"
             "- Confirmacao de licenca das fontes marcadas como a_confirmar.\n"
             "- Obtencao formal das fontes bloqueadas (DRM-RJ, IPPUC/GeoCuritiba, Defesa Civil PE).\n"
             "- Confirmacao do ID de ativacao Copernicus EMS para Petropolis.\n")

    L.append("## 18. Guardrails preservados\n")
    L.extend(f"- {g}" for g in GUARDRAILS)
    L.append("")

    L.append("## 19. Proximos passos recomendados\n")
    L.append(f"{metrics['proximo_passo_recomendado']}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# Orquestracao
# ---------------------------------------------------------------------------

OUT_TABLES = ROOT / "outputs_public" / "tables"
OUT_REPORTS = ROOT / "outputs_public" / "execution_reports"
OUT_METRICS = ROOT / "outputs_public" / "metrics"

P_MANIFEST = OUT_TABLES / "revp_manifesto_evidencias_externas_mv1.csv"
P_AUDIT = OUT_TABLES / "revp_auditoria_fontes_externas_mv1.csv"
P_EVENTS = OUT_TABLES / "revp_indice_eventos_externos_candidatos_mv1.csv"
P_GEOMS = OUT_TABLES / "revp_indice_geometrias_externas_candidatas_mv1.csv"
P_REPORT = OUT_REPORTS / "revp_curadoria_evidencias_externas_mv1.md"
P_METRICS = OUT_METRICS / "revp_curadoria_evidencias_externas_mv1.json"


def run() -> dict:
    manifest, audit, events, geoms = build_rows()
    metrics = build_metrics(manifest, events, geoms)
    write_csv(P_MANIFEST, MANIFEST_COLS, manifest)
    write_csv(P_AUDIT, AUDIT_COLS, audit)
    write_csv(P_EVENTS, EVENT_COLS, events)
    write_csv(P_GEOMS, GEOM_COLS, geoms)
    P_METRICS.parent.mkdir(parents=True, exist_ok=True)
    P_METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(P_REPORT, manifest, events, geoms, metrics)
    return metrics


def check() -> int:
    """Validacao simples: arquivos existem, colunas obrigatorias, JSON valido, guardrails no relatorio."""
    problems = []
    for path, cols in [(P_MANIFEST, MANIFEST_COLS), (P_AUDIT, AUDIT_COLS),
                        (P_EVENTS, EVENT_COLS), (P_GEOMS, GEOM_COLS)]:
        if not path.is_file():
            problems.append(f"CSV ausente: {path}")
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh))
        faltando = [c for c in cols if c not in header]
        if faltando:
            problems.append(f"{path.name}: colunas faltando {faltando}")

    if not P_METRICS.is_file():
        problems.append(f"JSON ausente: {P_METRICS}")
    else:
        try:
            data = json.loads(P_METRICS.read_text(encoding="utf-8"))
            obrig = ["total_fontes_pesquisadas", "guardrails_preservados", "proximo_passo_recomendado"]
            for k in obrig:
                if k not in data:
                    problems.append(f"JSON sem chave obrigatoria: {k}")
        except json.JSONDecodeError as e:
            problems.append(f"JSON invalido: {e}")

    if not P_REPORT.is_file():
        problems.append(f"Relatorio ausente: {P_REPORT}")
    else:
        txt = P_REPORT.read_text(encoding="utf-8")
        if "Guardrails preservados" not in txt:
            problems.append("Relatorio sem secao de guardrails")
        if "label-free" not in txt and "review-only" not in txt:
            problems.append("Relatorio sem mencao a review-only/label-free")

    if problems:
        print("CHECK: FALHOU")
        for p in problems:
            print("  -", p)
        return 1
    print("CHECK: OK (CSVs, JSON, relatorio e guardrails validados)")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    m = run()
    print("Curadoria concluida.")
    print(json.dumps({k: m[k] for k in [
        "total_fontes_pesquisadas", "total_fontes_baixadas",
        "total_fontes_aprovadas_para_contexto", "total_fontes_bloqueadas",
        "total_eventos_candidatos", "total_geometrias_candidatas",
        "eventos_potencialmente_revisaveis", "eventos_que_nao_podem_virar_label",
    ]}, ensure_ascii=False, indent=2))
