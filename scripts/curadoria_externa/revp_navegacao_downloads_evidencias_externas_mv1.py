"""Navegacao real nos portais oficiais e download efetivo de evidencias externas (MV1).

Esta passada e *review-only*. Ela NAO promove nenhuma fonte externa a
ground truth operacional, NAO cria label, NAO cria negativo formal e NAO
libera treino supervisionado.

A rodada anterior havia deixado 7 fontes apenas como `requer_download_manual`.
Aqui cada fonte pendente passou por navegacao real (URL do manifesto,
navegacao no portal, busca interna no HTML, busca web, inspecao de endpoints
ArcGIS REST/CKAN/FTP/API, fonte oficial alternativa) e terminou com uma
classificacao detalhada e auditavel. Arquivos reais foram baixados sempre que
tecnicamente possivel, salvos apenas na quarentena local (git-ignored), com
SHA256 calculado do arquivo em disco (nunca inventado).

Reutiliza o modulo de curadoria original para auditoria, eventos e geometrias.

Uso:
    python scripts/curadoria_externa/revp_navegacao_downloads_evidencias_externas_mv1.py
    python scripts/curadoria_externa/revp_navegacao_downloads_evidencias_externas_mv1.py --check
"""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())
ORIG_SCRIPT = ROOT / "scripts" / "curadoria_externa" / "revp_curadoria_evidencias_externas_mv1.py"
QUARENTENA = ROOT / "local_only" / "evidencias_externas_quarentena"
DATA_ACESSO = "2026-06-18"

_spec = importlib.util.spec_from_file_location("revp_curadoria_externa_mv1", ORIG_SCRIPT)
assert _spec and _spec.loader
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)


GUARDRAILS = [
    "Sem treino supervisionado; sem label binario; sem positivo formal; sem negativo formal.",
    "Sem ground truth operacional patch-level nesta passada.",
    "unknown nao vira negativo; ausencia de evidencia nao vira classe 0.",
    "Curitiba nao vira negativo formal.",
    "Evidencia externa nao vira label automaticamente.",
    "DINOv2 nao prova inundacao.",
    "Fonte textual sem geometria nao fecha patch-level.",
    "Landslide scar nao prova flood extent.",
    "Geometria de suscetibilidade nao e geometria de evento observado.",
    "Download de fonte nao significa validacao operacional.",
]

PROXIMO_PASSO = (
    "Solicitar formalmente os dados sob formulario/e-mail (CEMADEN) e os portais sem arquivo estatico "
    "(APAC, Copernicus EMS rapid mapping); complementar a serie hidrologica da ANA via HidroWeb por estacao "
    "do Capibaribe; e submeter as geometrias candidatas (bacias GeoCuritiba, malhas IBGE, carta de "
    "suscetibilidade SGB) a revisao humana. Nenhuma fonte e promovida a label nesta passada."
)

# ---------------------------------------------------------------------------
# Arquivos efetivamente baixados nesta rodada (quarentena local, git-ignored).
# SHA256 e tamanho sao recalculados do arquivo real em tempo de execucao.
# ---------------------------------------------------------------------------

ARQUIVOS_BAIXADOS: list[dict] = [
    {
        "id_arquivo": "ARQ_IBGE_PET", "id_fonte": "FONTE_NAC_004", "cidade": "Petropolis", "regiao": "Sudeste",
        "instituicao": "IBGE", "titulo_arquivo": "Malha municipal de Petropolis (3303906)",
        "tipo_arquivo": "limite_municipal", "formato": "GeoJSON",
        "url_pagina_origem": "https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/26565-malhas-de-setores-censitarios-divisoes-intramunicipais.html",
        "url_final_download": "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/3303906?formato=application/vnd.geo+json&qualidade=intermediaria",
        "caminho_local": "local_only/evidencias_externas_quarentena/petropolis/ibge_malha_municipal_petropolis_3303906.geojson",
        "content_type": "application/geo+json", "licenca_ou_uso": "dados_publicos_ibge",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "vulnerabilidade_socioambiental", "tem_geometria": True, "crs": "EPSG:4326",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Limite municipal oficial via API de malhas do IBGE; contexto territorial, nao evento.",
    },
    {
        "id_arquivo": "ARQ_IBGE_REC", "id_fonte": "FONTE_NAC_004", "cidade": "Recife", "regiao": "Nordeste",
        "instituicao": "IBGE", "titulo_arquivo": "Malha municipal de Recife (2611606)",
        "tipo_arquivo": "limite_municipal", "formato": "GeoJSON",
        "url_pagina_origem": "https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/26565-malhas-de-setores-censitarios-divisoes-intramunicipais.html",
        "url_final_download": "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/2611606?formato=application/vnd.geo+json&qualidade=intermediaria",
        "caminho_local": "local_only/evidencias_externas_quarentena/recife/ibge_malha_municipal_recife_2611606.geojson",
        "content_type": "application/geo+json", "licenca_ou_uso": "dados_publicos_ibge",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "vulnerabilidade_socioambiental", "tem_geometria": True, "crs": "EPSG:4326",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Limite municipal oficial via API de malhas do IBGE; contexto territorial, nao evento.",
    },
    {
        "id_arquivo": "ARQ_IBGE_CUR", "id_fonte": "FONTE_NAC_004", "cidade": "Curitiba", "regiao": "Sul",
        "instituicao": "IBGE", "titulo_arquivo": "Malha municipal de Curitiba (4106902)",
        "tipo_arquivo": "limite_municipal", "formato": "GeoJSON",
        "url_pagina_origem": "https://www.ibge.gov.br/geociencias/organizacao-do-territorio/estrutura-territorial/26565-malhas-de-setores-censitarios-divisoes-intramunicipais.html",
        "url_final_download": "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/4106902?formato=application/vnd.geo+json&qualidade=intermediaria",
        "caminho_local": "local_only/evidencias_externas_quarentena/curitiba/ibge_malha_municipal_curitiba_4106902.geojson",
        "content_type": "application/geo+json", "licenca_ou_uso": "dados_publicos_ibge",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "vulnerabilidade_socioambiental", "tem_geometria": True, "crs": "EPSG:4326",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Limite municipal oficial via API de malhas do IBGE; contexto territorial, nao evento.",
    },
    {
        "id_arquivo": "ARQ_GEOCWB_BACIA", "id_fonte": "FONTE_CUR_003", "cidade": "Curitiba", "regiao": "Sul",
        "instituicao": "IPPUC / GeoCuritiba", "titulo_arquivo": "Bacias Hidrograficas de Curitiba (camada 54)",
        "tipo_arquivo": "bacia_hidrografica", "formato": "GeoJSON",
        "url_pagina_origem": "https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/54",
        "url_final_download": "https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/54/query?where=1%3D1&outFields=*&outSR=4326&f=geojson",
        "caminho_local": "local_only/evidencias_externas_quarentena/curitiba/geocuritiba_bacia_hidrografica_l54.geojson",
        "content_type": "application/geo+json", "licenca_ou_uso": "dados_publicos_geocuritiba_a_confirmar",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "suscetibilidade_territorial", "tem_geometria": True, "crs": "EPSG:4326",
        "status_auditoria": "aprovada_com_ressalvas",
        "observacao": "6 poligonos de bacia hidrografica via ArcGIS REST; contexto de drenagem, nao evento observado.",
    },
    {
        "id_arquivo": "ARQ_ANA_INVENTARIO", "id_fonte": "FONTE_NAC_002", "cidade": "", "regiao": "Nacional",
        "instituicao": "ANA - Agencia Nacional de Aguas e Saneamento Basico", "titulo_arquivo": "Inventario das Estacoes Pluviometricas",
        "tipo_arquivo": "inventario_estacoes", "formato": "PDF",
        "url_pagina_origem": "https://www.snirh.gov.br/hidroweb/",
        "url_final_download": "https://arquivos.ana.gov.br/infohidrologicas/InventariodasEstacoesPluviometricas.pdf",
        "caminho_local": "local_only/evidencias_externas_quarentena/fontes_nacionais/ana_inventario_estacoes_pluviometricas.pdf",
        "content_type": "application/pdf", "licenca_ou_uso": "dados_abertos_ana",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "chuva_extrema", "tem_geometria": False, "crs": "",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Inventario nacional de estacoes pluviometricas; serie especifica do Capibaribe ainda pendente via HidroWeb.",
    },
    {
        "id_arquivo": "ARQ_MAPBIOMAS_DHN250", "id_fonte": "FONTE_NAC_003", "cidade": "", "regiao": "Nacional",
        "instituicao": "MapBiomas", "titulo_arquivo": "Estatisticas de cobertura COL.10.1 por divisao hidrografica (DHN250)",
        "tipo_arquivo": "estatistica_cobertura_solo", "formato": "XLSX",
        "url_pagina_origem": "https://brasil.mapbiomas.org/estatisticas/",
        "url_final_download": "https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2026/02/MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.10.1-DHN250_STATE_BIOME.xlsx",
        "caminho_local": "local_only/evidencias_externas_quarentena/fontes_nacionais/mapbiomas_col10_1_cobertura_dhn250_estado_bioma.xlsx",
        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "licenca_ou_uso": "CC_BY_SA_mapbiomas", "periodo_referencia_inicio": "1985", "periodo_referencia_fim": "2024",
        "categoria_evento": "suscetibilidade_territorial", "tem_geometria": False, "crs": "",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Estatisticas de cobertura por divisao hidrografica nacional; contexto territorial, nao geometria de evento.",
    },
    {
        "id_arquivo": "ARQ_SGB_CARTA_PET", "id_fonte": "FONTE_PET_005", "cidade": "Petropolis", "regiao": "Sudeste",
        "instituicao": "SGB/CPRM (Servico Geologico do Brasil)", "titulo_arquivo": "Carta de Suscetibilidade a Movimentos Gravitacionais de Massa e Inundacao - Petropolis/RJ",
        "tipo_arquivo": "carta_suscetibilidade", "formato": "PDF",
        "url_pagina_origem": "https://rigeo.sgb.gov.br/handle/doc/15692",
        "url_final_download": "https://rigeo.sgb.gov.br/bitstreams/a7728930-5553-40d7-8021-52b6f85c3412/download",
        "caminho_local": "local_only/evidencias_externas_quarentena/petropolis/sgb_cprm_carta_suscetibilidade_petropolis_cs.pdf",
        "content_type": "application/pdf", "licenca_ou_uso": "publicacao_oficial_rigeo_sgb",
        "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
        "categoria_evento": "suscetibilidade_territorial", "tem_geometria": False, "crs": "EPSG:4326 (mapa; SIG vetorial 1.8GB nao baixado)",
        "status_auditoria": "aprovada_para_contexto",
        "observacao": "Carta de suscetibilidade (PDF do mapa). Suscetibilidade != evento observado; deslizamento != flood. SIG vetorial completo (1.8 GB) nao baixado por tamanho.",
    },
    {
        "id_arquivo": "ARQ_NHESS", "id_fonte": "FONTE_INT_002", "cidade": "Petropolis", "regiao": "Internacional",
        "instituicao": "NHESS (Copernicus Publications)", "titulo_arquivo": "Artigo NHESS 23/1157/2023 - Petropolis fev/2022",
        "tipo_arquivo": "artigo_academico", "formato": "PDF",
        "url_pagina_origem": "https://nhess.copernicus.org/articles/23/1157/2023/",
        "url_final_download": "https://nhess.copernicus.org/articles/23/1157/2023/nhess-23-1157-2023.pdf",
        "caminho_local": "local_only/evidencias_externas_quarentena/fontes_internacionais/nhess_23_1157_2023_petropolis_fev2022.pdf",
        "content_type": "application/pdf", "licenca_ou_uso": "CC_BY_4.0",
        "periodo_referencia_inicio": "2022-02-15", "periodo_referencia_fim": "2022-02-15",
        "categoria_evento": "desastre_hidrometeorologico_composto", "tem_geometria": False, "crs": "",
        "status_auditoria": "aprovada_com_ressalvas",
        "observacao": "Fonte documental secundaria (baixada em rodada anterior); figuras do artigo nao sao geometria auditavel.",
    },
]

# Nova fonte derivada da navegacao (carta de suscetibilidade SGB para Petropolis),
# alternativa oficial para o papel de suscetibilidade do CEMADEN/DRM-RJ.
FONTE_NOVA_SGB = {
    "id_fonte": "FONTE_PET_005",
    "regiao": "Sudeste", "cidade": "Petropolis", "estado": "RJ", "pais": "Brasil",
    "instituicao": "SGB/CPRM (Servico Geologico do Brasil)",
    "familia_fonte": "orgao_federal", "tipo_fonte": "carta_suscetibilidade",
    "titulo": "Carta de Suscetibilidade a Movimentos de Massa e Inundacao - Petropolis/RJ",
    "descricao_curta": "Carta oficial de suscetibilidade (PDF) baixada do RIGEO/SGB; alternativa oficial para o papel de suscetibilidade.",
    "url_origem": "https://rigeo.sgb.gov.br/handle/doc/15692",
    "data_acesso": DATA_ACESSO, "data_publicacao": "",
    "periodo_referencia_inicio": "", "periodo_referencia_fim": "",
    "categoria_evento": "suscetibilidade_territorial", "formato_original": "PDF",
    "caminho_local": "local_only/evidencias_externas_quarentena/petropolis/sgb_cprm_carta_suscetibilidade_petropolis_cs.pdf",
    "licenca_ou_uso": "publicacao_oficial_rigeo_sgb",
    "status_download": "baixado", "status_auditoria": "aprovada_para_contexto",
    "observacao": "Suscetibilidade != evento observado; deslizamento != flood extent. SIG vetorial (1.8 GB) nao baixado por tamanho.",
    "tem_geometria": False, "geometria_tipo": "", "crs": "",
    "representa_evento": False, "representa_suscet": True, "representa_vulner": False, "representa_contexto": True,
    "circularidade": "baixa", "landslide_flood": "risco_alto",
}

# ---------------------------------------------------------------------------
# Resultado de navegacao por fonte pendente: status final + tentativas (log).
# Cada fonte tem >= 1 tentativa registrada.
# ---------------------------------------------------------------------------

NAV: dict[str, dict] = {
    "FONTE_NAC_004": {  # IBGE
        "status": "baixado",
        "tentativas": [
            ("url_manifesto", "https://www.ibge.gov.br/geociencias/downloads-geociencias.html", "", "pagina HTML de indice (portal)"),
            ("endpoint_api", "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/3303906?formato=application/vnd.geo+json", "Petropolis 3303906", "GeoJSON oficial baixado"),
            ("endpoint_api", "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/2611606?formato=application/vnd.geo+json", "Recife 2611606", "GeoJSON oficial baixado"),
            ("endpoint_api", "https://servicodados.ibge.gov.br/api/v3/malhas/municipios/4106902?formato=application/vnd.geo+json", "Curitiba 4106902", "GeoJSON oficial baixado"),
        ],
        "motivo": "",
    },
    "FONTE_CUR_003": {  # IPPUC/GeoCuritiba
        "status": "baixado",
        "tentativas": [
            ("url_manifesto", "https://www.ippuc.org.br", "", "homepage HTML (portal)"),
            ("arcgis_rest", "https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/layers?f=json", "risco/hidro/drenag/bacia", "ArcGIS REST com camadas Hidrografia/Drenagem/Bacia"),
            ("arcgis_rest", "https://geocuritiba.ippuc.org.br/server/rest/services/GeoCuritiba/Publico_Interno_GeoCuritiba_BaseCartografica_para_MC/MapServer/54/query?where=1%3D1&outFields=*&outSR=4326&f=geojson", "Bacia Hidrografica", "GeoJSON com 6 bacias baixado"),
        ],
        "motivo": "",
    },
    "FONTE_NAC_002": {  # ANA
        "status": "baixado_parcialmente",
        "tentativas": [
            ("url_manifesto", "https://www.snirh.gov.br/hidroweb/", "", "portal HidroWeb (busca por estacao)"),
            ("busca_web", "https://arquivos.ana.gov.br/infohidrologicas/InventariodasEstacoesPluviometricas.pdf", "inventario estacoes", "Inventario pluviometrico PDF baixado"),
            ("dados_abertos", "https://dadosabertos.ana.gov.br/datasets/", "Capibaribe Recife serie", "serie por estacao exige consulta HidroWeb (pendente)"),
        ],
        "motivo": "Inventario oficial baixado; serie historica especifica do Capibaribe ainda exige consulta por estacao no HidroWeb.",
    },
    "FONTE_NAC_003": {  # MapBiomas
        "status": "baixado",
        "tentativas": [
            ("url_manifesto", "https://brasil.mapbiomas.org/", "", "portal HTML"),
            ("inspecao_html", "https://brasil.mapbiomas.org/estatisticas/", "xlsx cobertura", "links XLSX diretos localizados no HTML"),
            ("download_direto", "https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2026/02/MAPBIOMAS_BRAZIL-COVERAGE_STATISTICS-COL.10.1-DHN250_STATE_BIOME.xlsx", "DHN250 cobertura", "XLSX de cobertura por divisao hidrografica baixado"),
        ],
        "motivo": "",
    },
    "FONTE_NAC_001": {  # CEMADEN
        "status": "requer_solicitacao_formal",
        "tentativas": [
            ("url_manifesto", "https://mapainterativo.cemaden.gov.br/", "", "mapa interativo (aplicacao JavaScript)"),
            ("navegacao_portal", "http://www2.cemaden.gov.br/mapainterativo/download/downpluv.php", "download pluviometrico", "endpoint retorna formulario que exige e-mail para enviar link"),
            ("fonte_alternativa_oficial", "https://rigeo.sgb.gov.br/handle/doc/15692", "suscetibilidade", "papel de suscetibilidade coberto pela carta SGB (FONTE_PET_005)"),
        ],
        "motivo": "Download do CEMADEN depende de formulario com e-mail (link enviado por e-mail); requer solicitacao. Suscetibilidade coberta por fonte alternativa oficial (SGB).",
    },
    "FONTE_NAC_005": {  # APAC
        "status": "bloqueado_apos_navegacao_real",
        "tentativas": [
            ("url_manifesto", "https://www.apac.pe.gov.br", "", "homepage HTML (Joomla/JS)"),
            ("navegacao_portal", "https://www.apac.pe.gov.br/boletins", "boletim", "retorna HTTP 404"),
            ("navegacao_portal", "http://old.apac.pe.gov.br/meteorologia/chuvas-rmr.php", "chuva RMR", "tabela de PCD em HTML, sem PDF/CSV estatico"),
            ("inspecao_html", "https://www.apac.pe.gov.br/", "pdf/csv/xls", "nenhum link de arquivo estatico nos menus"),
        ],
        "motivo": "Portal Joomla/JS sem arquivo estatico de boletim localizavel apos navegacao; dados via PCD em tempo real. Angulo de chuva parcialmente coberto pelo inventario ANA.",
    },
    "FONTE_NAC_006": {  # DRM-RJ
        "status": "fonte_alternativa_oficial_baixada",
        "tentativas": [
            ("url_manifesto", "https://www.drm.rj.gov.br", "", "retorna HTTP 503 (Service Unavailable)"),
            ("fonte_alternativa_oficial", "https://rigeo.sgb.gov.br/handle/doc/15692", "suscetibilidade Petropolis", "Carta de Suscetibilidade SGB/CPRM de Petropolis baixada (FONTE_PET_005)"),
        ],
        "motivo": "Portal DRM-RJ indisponivel (503); equivalente oficial federal (SGB/CPRM) para o mesmo municipio/tema baixado.",
    },
    "FONTE_INT_001": {  # Copernicus EMS
        "status": "bloqueado_apos_navegacao_real",
        "tentativas": [
            ("url_manifesto", "https://rapidmapping.emergency.copernicus.eu/", "", "redireciona para lista de ativacoes (HTML/JS)"),
            ("busca_web", "EMSR Petropolis 2022", "ativacao Petropolis", "nenhum codigo EMSR de rapid mapping para Petropolis confirmado"),
            ("navegacao_portal", "https://global-flood.emergency.copernicus.eu/.../99-floods-and-landslides-in-rio-de-janeiro-state-brazil-february-to-march-2022/", "Petropolis 2022", "pagina React (CEMS-Floods), sem produto vetorial/raster baixavel"),
        ],
        "motivo": "Nenhum produto rapid mapping vetorial publico para Petropolis localizado apos navegacao; GloFAS/GFM e noticia/monitoramento sem download de delimitacao.",
    },
}

# Mapeamento de cidade -> regiao para a fonte nova / contexto.


def _sha(rel: str):
    return M.sha256_of(rel)


# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------

LOG_COLS = [
    "id_fonte", "nome_fonte", "url_inicial", "url_visitada", "tipo_tentativa",
    "termo_busca_ou_filtro", "resultado", "url_final_arquivo", "status_download",
    "codigo_http", "content_type", "tamanho_bytes", "caminho_local_quarentena",
    "sha256_arquivo", "motivo_falha_ou_bloqueio", "data_acesso", "observacao",
]

ARQ_COLS = [
    "id_arquivo", "id_fonte", "cidade", "regiao", "instituicao", "titulo_arquivo",
    "tipo_arquivo", "formato", "url_pagina_origem", "url_final_download",
    "caminho_local_quarentena", "sha256_arquivo", "tamanho_bytes", "content_type",
    "data_acesso", "licenca_ou_uso", "periodo_referencia_inicio",
    "periodo_referencia_fim", "categoria_evento", "tem_geometria", "crs",
    "status_auditoria", "observacao",
]

INTEGRACAO_COLS = [
    "item", "tipo", "origem", "artefato_relacionado", "status",
    "pode_apoiar_ground_truth_futuro", "pode_virar_label_agora", "bloqueador",
    "acao_recomendada", "observacao_metodologica",
]

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _arquivos_por_fonte() -> dict[str, list[dict]]:
    d: dict[str, list[dict]] = {}
    for a in ARQUIVOS_BAIXADOS:
        d.setdefault(a["id_fonte"], []).append(a)
    return d


def construir():
    fontes = copy.deepcopy(M.FONTES) + [copy.deepcopy(FONTE_NOVA_SGB)]
    arq_por_fonte = _arquivos_por_fonte()

    # Atualiza status_download e caminho das fontes navegadas.
    for f in fontes:
        fid = f["id_fonte"]
        if f["status_download"] == "ja_presente_no_repositorio":
            f["status_download"] = "ja_existia_localmente"
        if fid in NAV:
            f["status_download"] = NAV[fid]["status"]
            arqs = arq_por_fonte.get(fid, [])
            if arqs:
                f["caminho_local"] = arqs[0]["caminho_local"]
                if len(arqs) > 1:
                    f["observacao"] = f["observacao"] + f" [{len(arqs)} arquivos baixados]"
                if any(a["tem_geometria"] for a in arqs):
                    f["tem_geometria"] = True
                    f["geometria_tipo"] = "Polygon"
                    f["crs"] = f["crs"] or "EPSG:4326"
                    f["representa_contexto"] = True

    # Manifesto / auditoria / eventos / geometrias.
    manifest, audit, events, geoms = [], [], [], []
    for f in fontes:
        sha, _ = _sha(f["caminho_local"])
        manifest.append({
            "id_fonte": f["id_fonte"], "regiao": f["regiao"], "cidade": f["cidade"],
            "estado": f["estado"], "pais": f["pais"], "instituicao": f["instituicao"],
            "familia_fonte": f["familia_fonte"], "tipo_fonte": f["tipo_fonte"],
            "titulo": f["titulo"], "descricao_curta": f["descricao_curta"],
            "url_origem": f["url_origem"], "data_acesso": f["data_acesso"],
            "data_publicacao": f["data_publicacao"],
            "periodo_referencia_inicio": f["periodo_referencia_inicio"],
            "periodo_referencia_fim": f["periodo_referencia_fim"],
            "categoria_evento": f["categoria_evento"], "formato_original": f["formato_original"],
            "caminho_local_quarentena": f["caminho_local"], "sha256_arquivo": sha,
            "licenca_ou_uso": f["licenca_ou_uso"], "status_download": f["status_download"],
            "status_auditoria": f["status_auditoria"], "observacao": f["observacao"],
        })
        audit.extend(M.build_audit_rows(f))
        if f["representa_evento"]:
            janela = "fechada" if f["periodo_referencia_inicio"] and f["periodo_referencia_fim"] else "aberta_ou_indefinida"
            pode = M.decide_pode_label(f, janela)
            events.append({
                "id_evento_externo": "EV_" + f["id_fonte"], "id_fonte": f["id_fonte"], "cidade": f["cidade"],
                "regiao": f["regiao"], "categoria_evento": f["categoria_evento"],
                "data_evento_inicio": f["periodo_referencia_inicio"], "data_evento_fim": f["periodo_referencia_fim"],
                "janela_temporal_status": janela, "descricao_evento": f["titulo"],
                "fonte_observacional_independente": "sim" if f["familia_fonte"] in {"agencia_internacional", "orgao_federal", "instituto_pesquisa"} else "parcial",
                "geometria_disponivel": "sim" if f["tem_geometria"] else "nao",
                "geometria_tipo": f["geometria_tipo"], "crs": f["crs"], "escala_espacial": "municipal_ou_bairro",
                "pode_sustentar_label_patch_level": "true" if pode else "false",
                "motivo_nao_label": M.motivo_nao_label(f, janela, pode),
                "status_evento": "candidato_para_revisao_futura", "observacao": f["observacao"],
            })

    # Geometrias: por arquivo baixado com geometria + candidatas originais (Charter, risco Recife).
    for a in ARQUIVOS_BAIXADOS:
        if a["tem_geometria"]:
            sha, _ = _sha(a["caminho_local"])
            geoms.append({
                "id_geometria": "GEO_" + a["id_arquivo"], "id_fonte": a["id_fonte"], "cidade": a["cidade"],
                "regiao": a["regiao"], "tipo_geometria": "Polygon", "formato": a["formato"], "crs": a["crs"],
                "escala": "municipal_ou_bacia",
                "caminho_local_quarentena": a["caminho_local"], "sha256_arquivo": sha,
                "representa_evento_observado": "false", "representa_suscetibilidade": "true" if a["categoria_evento"] == "suscetibilidade_territorial" else "false",
                "representa_vulnerabilidade": "true" if a["categoria_evento"] == "vulnerabilidade_socioambiental" else "false",
                "representa_contexto": "true",
                "adequada_para_overlay_patch": "false",
                "motivo_bloqueio_overlay": "geometria_de_contexto_territorial_nao_e_evento_observado",
                "status_geometria": "geometria_candidata_nao_promovida", "observacao": a["observacao"],
            })
    # Candidatas originais (Charter 758 e areas de risco Recife) preservadas.
    for f in M.FONTES:
        if f["tem_geometria"]:
            sha, _ = _sha(f["caminho_local"])
            _, motivo = M.avalia_overlay(f)
            geoms.append({
                "id_geometria": "GEO_" + f["id_fonte"], "id_fonte": f["id_fonte"], "cidade": f["cidade"],
                "regiao": f["regiao"], "tipo_geometria": f["geometria_tipo"], "formato": f["formato_original"],
                "crs": f["crs"], "escala": "local",
                "caminho_local_quarentena": f["caminho_local"], "sha256_arquivo": sha,
                "representa_evento_observado": "true" if f["representa_evento"] else "false",
                "representa_suscetibilidade": "true" if f["representa_suscet"] else "false",
                "representa_vulnerabilidade": "true" if f["representa_vulner"] else "false",
                "representa_contexto": "true" if f["representa_contexto"] else "false",
                "adequada_para_overlay_patch": "false", "motivo_bloqueio_overlay": motivo,
                "status_geometria": "geometria_candidata_nao_promovida", "observacao": f["observacao"],
            })

    return fontes, manifest, audit, events, geoms


def construir_log(fontes):
    nome_por_id = {f["id_fonte"]: f["instituicao"] for f in fontes}
    arq_por_fonte = _arquivos_por_fonte()
    rows = []
    for fid, nav in NAV.items():
        nome = nome_por_id.get(fid, fid)
        url_inicial = nav["tentativas"][0][1] if nav["tentativas"] else ""
        arqs = {a["url_final_download"]: a for a in arq_por_fonte.get(fid, [])}
        for (tipo, url, termo, resultado) in nav["tentativas"]:
            arq = arqs.get(url)
            if arq:
                sha, size = _sha(arq["caminho_local"])
                rows.append({
                    "id_fonte": fid, "nome_fonte": nome, "url_inicial": url_inicial, "url_visitada": url,
                    "tipo_tentativa": tipo, "termo_busca_ou_filtro": termo, "resultado": resultado,
                    "url_final_arquivo": url, "status_download": "baixado", "codigo_http": "200",
                    "content_type": arq["content_type"], "tamanho_bytes": size if size >= 0 else "",
                    "caminho_local_quarentena": arq["caminho_local"], "sha256_arquivo": sha,
                    "motivo_falha_ou_bloqueio": "", "data_acesso": DATA_ACESSO,
                    "observacao": arq["observacao"],
                })
            else:
                rows.append({
                    "id_fonte": fid, "nome_fonte": nome, "url_inicial": url_inicial, "url_visitada": url,
                    "tipo_tentativa": tipo, "termo_busca_ou_filtro": termo, "resultado": resultado,
                    "url_final_arquivo": "", "status_download": nav["status"], "codigo_http": "",
                    "content_type": "", "tamanho_bytes": "", "caminho_local_quarentena": "",
                    "sha256_arquivo": "", "motivo_falha_ou_bloqueio": nav["motivo"], "data_acesso": DATA_ACESSO,
                    "observacao": resultado,
                })
    return rows


def construir_arquivos_index():
    rows = []
    for a in ARQUIVOS_BAIXADOS:
        sha, size = _sha(a["caminho_local"])
        rows.append({
            "id_arquivo": a["id_arquivo"], "id_fonte": a["id_fonte"], "cidade": a["cidade"], "regiao": a["regiao"],
            "instituicao": a["instituicao"], "titulo_arquivo": a["titulo_arquivo"], "tipo_arquivo": a["tipo_arquivo"],
            "formato": a["formato"], "url_pagina_origem": a["url_pagina_origem"], "url_final_download": a["url_final_download"],
            "caminho_local_quarentena": a["caminho_local"], "sha256_arquivo": sha,
            "tamanho_bytes": size if size >= 0 else "", "content_type": a["content_type"], "data_acesso": DATA_ACESSO,
            "licenca_ou_uso": a["licenca_ou_uso"], "periodo_referencia_inicio": a["periodo_referencia_inicio"],
            "periodo_referencia_fim": a["periodo_referencia_fim"], "categoria_evento": a["categoria_evento"],
            "tem_geometria": "true" if a["tem_geometria"] else "false", "crs": a["crs"],
            "status_auditoria": a["status_auditoria"], "observacao": a["observacao"],
        })
    return rows


# ---------------------------------------------------------------------------
# Metricas / integracao
# ---------------------------------------------------------------------------

def metricas(manifest, events, geoms, arq_rows) -> dict:
    status_count = {}
    for f in manifest:
        status_count[f["status_download"]] = status_count.get(f["status_download"], 0) + 1
    pendentes = list(NAV.keys())
    resolvidas = [fid for fid in pendentes if NAV[fid]["status"] == "baixado"]
    parciais = [fid for fid in pendentes if NAV[fid]["status"] == "baixado_parcialmente"]
    alternativas = [fid for fid in pendentes if NAV[fid]["status"] == "fonte_alternativa_oficial_baixada"]
    bloqueadas = [fid for fid in pendentes if NAV[fid]["status"] == "bloqueado_apos_navegacao_real"]
    formais = [fid for fid in pendentes if NAV[fid]["status"] == "requer_solicitacao_formal"]
    return {
        "total_fontes_pendentes_iniciais": len(pendentes),
        "total_fontes_resolvidas_com_download": len(resolvidas),
        "total_fontes_baixadas_parcialmente": len(parciais),
        "total_fontes_alternativas_oficiais_baixadas": len(alternativas),
        "total_fontes_bloqueadas_apos_navegacao_real": len(bloqueadas),
        "total_fontes_requerem_solicitacao_formal": len(formais),
        "total_arquivos_baixados": len(arq_rows),
        "total_arquivos_com_sha256": sum(1 for a in arq_rows if a["sha256_arquivo"]),
        "total_geometrias_candidatas": len(geoms),
        "total_eventos_candidatos": len(events),
        "contagem_por_instituicao": _count(arq_rows, "instituicao"),
        "contagem_por_cidade": _count(arq_rows, lambda r: r["cidade"] or "nacional_ou_internacional"),
        "contagem_por_formato": _count(arq_rows, "formato"),
        "distribuicao_status_manifesto": status_count,
        "guardrails_preservados": GUARDRAILS,
        "proximo_passo_recomendado": PROXIMO_PASSO,
    }


def _count(rows, key):
    out = {}
    for r in rows:
        k = key(r) if callable(key) else r[key]
        out[k] = out.get(k, 0) + 1
    return out


MARCO_ARTEFATOS = [
    ROOT / "outputs_public" / "execution_reports" / "revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md",
    ROOT / "outputs_public" / "tables" / "revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv",
    ROOT / "outputs_public" / "metrics" / "revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json",
    ROOT / "outputs_public" / "tables" / "revp_proximos_passos_pos_marco_label_free_mv1.csv",
]


def construir_integracao(manifest, geoms, arq_rows):
    rows = []
    for p in MARCO_ARTEFATOS:
        rows.append({
            "item": p.name, "tipo": "artefato_marco_label_free", "origem": "marco_label_free",
            "artefato_relacionado": p.relative_to(ROOT).as_posix(),
            "status": "encontrado" if p.is_file() else "ausente",
            "pode_apoiar_ground_truth_futuro": "sim", "pode_virar_label_agora": "false",
            "bloqueador": "marco_metodologico_interno_review_only",
            "acao_recomendada": "manter_como_referencia_metodologica",
            "observacao_metodologica": "Fechamento interno label-free; nao define geometria de evento.",
        })
    for a in arq_rows:
        rows.append({
            "item": a["id_arquivo"], "tipo": "arquivo_externo_baixado", "origem": "navegacao_downloads",
            "artefato_relacionado": a["caminho_local_quarentena"], "status": "baixado",
            "pode_apoiar_ground_truth_futuro": "sim" if a["tem_geometria"] == "true" else "potencial",
            "pode_virar_label_agora": "false",
            "bloqueador": "geometria_de_contexto_requer_revisao_humana" if a["tem_geometria"] == "true" else "evidencia_contextual_sem_geometria_de_evento",
            "acao_recomendada": "encaminhar_para_revisao_humana" if a["tem_geometria"] == "true" else "tratar_como_contexto_documental",
            "observacao_metodologica": a["observacao"][:240],
        })
    for f in manifest:
        if f["id_fonte"] in NAV and NAV[f["id_fonte"]]["status"] in {"bloqueado_apos_navegacao_real", "requer_solicitacao_formal"}:
            rows.append({
                "item": f["id_fonte"], "tipo": "fonte_pendente_pos_navegacao", "origem": "navegacao_downloads",
                "artefato_relacionado": f["url_origem"], "status": f["status_download"],
                "pode_apoiar_ground_truth_futuro": "potencial", "pode_virar_label_agora": "false",
                "bloqueador": f["status_download"],
                "acao_recomendada": "solicitacao_formal_ou_acesso_interativo",
                "observacao_metodologica": NAV[f["id_fonte"]]["motivo"][:240],
            })
    return rows


def metricas_integracao(events, geoms, met) -> dict:
    return {
        "status_integracao": "integrado_review_only",
        "artefatos_marco_encontrados": [p.name for p in MARCO_ARTEFATOS if p.is_file()],
        "total_fontes_pendentes_iniciais": met["total_fontes_pendentes_iniciais"],
        "total_fontes_resolvidas_com_download": met["total_fontes_resolvidas_com_download"],
        "total_fontes_baixadas_parcialmente": met["total_fontes_baixadas_parcialmente"],
        "total_fontes_alternativas_oficiais_baixadas": met["total_fontes_alternativas_oficiais_baixadas"],
        "total_fontes_bloqueadas_apos_navegacao_real": met["total_fontes_bloqueadas_apos_navegacao_real"],
        "total_fontes_requerem_solicitacao_formal": met["total_fontes_requerem_solicitacao_formal"],
        "total_arquivos_baixados": met["total_arquivos_baixados"],
        "total_geometrias_candidatas": len(geoms),
        "total_eventos_candidatos": len(events),
        "itens_potenciais_para_revisao_humana": len(geoms),
        "pode_virar_label_agora": "false",
        "guardrails_preservados": GUARDRAILS,
        "proximo_passo_recomendado": PROXIMO_PASSO,
    }


# ---------------------------------------------------------------------------
# Escrita
# ---------------------------------------------------------------------------

OUT_T = ROOT / "outputs_public" / "tables"
OUT_R = ROOT / "outputs_public" / "execution_reports"
OUT_M = ROOT / "outputs_public" / "metrics"

P_MANIFEST = OUT_T / "revp_manifesto_evidencias_externas_navegacao_mv1.csv"
P_AUDIT = OUT_T / "revp_auditoria_fontes_externas_navegacao_mv1.csv"
P_LOG = OUT_T / "revp_log_navegacao_downloads_evidencias_externas_mv1.csv"
P_ARQ = OUT_T / "revp_indice_arquivos_baixados_evidencias_externas_mv1.csv"
P_EVENTS = OUT_T / "revp_indice_eventos_externos_candidatos_navegacao_mv1.csv"
P_GEOMS = OUT_T / "revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv"
P_REPORT = OUT_R / "revp_navegacao_downloads_evidencias_externas_mv1.md"
P_METRICS = OUT_M / "revp_navegacao_downloads_evidencias_externas_mv1.json"

P_INT_TABLE = OUT_T / "revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.csv"
P_INT_METRICS = OUT_M / "revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.json"
P_INT_REPORT = OUT_R / "revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.md"


def _wcsv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _report(manifest, arq_rows, geoms, events, met):
    L = ["# Navegacao real e download de evidencias externas (MV1)\n"]
    L.append("> Passada review-only. Nenhuma fonte foi promovida a label, negativo formal ou ground truth. "
             "Arquivos salvos apenas em quarentena local (git-ignored); SHA256 calculado do arquivo real.\n")
    L.append("## 1. Escopo\n")
    L.append("Resolver as 7 fontes pendentes da rodada anterior por navegacao real nos portais oficiais, "
             "baixando arquivos quando tecnicamente possivel.\n")
    L.append("## 2. Por que a rodada anterior era insuficiente\n")
    L.append("A rodada anterior classificou CEMADEN, ANA, MapBiomas, IBGE, APAC, Copernicus e IPPUC apenas como "
             "`requer_download_manual`, sem navegar dentro dos portais nem testar endpoints (API, ArcGIS REST, "
             "dados abertos, FTP).\n")
    L.append("## 3. Metodo de navegacao real\n")
    L.append("Para cada fonte: URL do manifesto -> navegacao no portal -> inspecao do HTML -> busca web -> "
             "teste de endpoints (API IBGE, ArcGIS REST GeoCuritiba, arquivos diretos ANA/MapBiomas, RIGEO/SGB) "
             "-> download direto ou fonte oficial alternativa.\n")
    L.append("## 4. Fontes resolvidas com download\n")
    for fid, nav in NAV.items():
        if nav["status"] == "baixado":
            L.append(f"- `{fid}` ({_nome(manifest, fid)})")
    L.append("## 5. Fontes parcialmente resolvidas\n")
    for fid, nav in NAV.items():
        if nav["status"] == "baixado_parcialmente":
            L.append(f"- `{fid}` ({_nome(manifest, fid)}) — {nav['motivo']}")
    L.append("## 6. Fontes alternativas oficiais encontradas\n")
    for fid, nav in NAV.items():
        if nav["status"] == "fonte_alternativa_oficial_baixada":
            L.append(f"- `{fid}` ({_nome(manifest, fid)}) — {nav['motivo']}")
    L.append("## 7. Fontes bloqueadas apos navegacao real\n")
    for fid, nav in NAV.items():
        if nav["status"] == "bloqueado_apos_navegacao_real":
            L.append(f"- `{fid}` ({_nome(manifest, fid)}) — {nav['motivo']}")
    L.append("## 8. Fontes que requerem solicitacao formal\n")
    for fid, nav in NAV.items():
        if nav["status"] == "requer_solicitacao_formal":
            L.append(f"- `{fid}` ({_nome(manifest, fid)}) — {nav['motivo']}")
    L.append("## 9. Arquivos baixados\n")
    for a in arq_rows:
        L.append(f"- `{a['id_arquivo']}` {a['titulo_arquivo']} ({a['formato']}, {a['tamanho_bytes']} bytes) "
                 f"-> {a['caminho_local_quarentena']}")
    L.append("## 10. Hashes calculados\n")
    for a in arq_rows:
        L.append(f"- `{a['id_arquivo']}` SHA256 `{a['sha256_arquivo']}`")
    L.append("## 11. Geometrias candidatas\n")
    for g in geoms:
        L.append(f"- `{g['id_geometria']}` ({g['cidade']}, {g['tipo_geometria']}, {g['crs']}) — "
                 f"overlay bloqueado: {g['motivo_bloqueio_overlay']}")
    L.append("## 12. Eventos candidatos\n")
    L.append(f"- {len(events)} eventos candidatos; nenhum pode sustentar label patch-level nesta passada.")
    L.append("## 13. Limitacoes\n")
    L.append("- Series hidrologicas por estacao (ANA/HidroWeb) e boletins APAC dependem de consulta interativa.\n"
             "- Copernicus EMS nao tem produto rapid mapping vetorial publico para Petropolis.\n"
             "- O SIG vetorial completo da carta SGB (1.8 GB) nao foi baixado por tamanho; usado o PDF do mapa.\n"
             "- Geometrias baixadas sao de contexto territorial/drenagem, nao de evento observado.\n")
    L.append("## 14. Proximas acoes manuais inevitaveis\n")
    L.append(PROXIMO_PASSO + "\n")
    L.append("## 15. Guardrails preservados\n")
    L.extend(f"- {g}" for g in GUARDRAILS)
    L.append("\n## 16. Integracao com o marco label-free\n")
    L.append("Ver `revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.md`. "
             "Todos os itens integrados tem `pode_virar_label_agora=false`.\n")
    P_REPORT.parent.mkdir(parents=True, exist_ok=True)
    P_REPORT.write_text("\n".join(L), encoding="utf-8")


def _nome(manifest, fid):
    for f in manifest:
        if f["id_fonte"] == fid:
            return f["instituicao"]
    return fid


def _report_integracao(met_int):
    L = ["# Integracao: marco label-free MV1 + evidencias externas (pos-navegacao)\n"]
    L.append("> Passada review-only. `pode_virar_label_agora` e sempre `false`. Nenhum evento candidato e "
             "promovido automaticamente a label.\n")
    L.append("## Artefatos do marco label-free encontrados\n")
    L.extend(f"- `{n}`" for n in met_int["artefatos_marco_encontrados"])
    L.append("\n## Resultado da navegacao\n")
    L.append(f"- Pendentes iniciais: {met_int['total_fontes_pendentes_iniciais']}")
    L.append(f"- Resolvidas com download: {met_int['total_fontes_resolvidas_com_download']}")
    L.append(f"- Parcialmente resolvidas: {met_int['total_fontes_baixadas_parcialmente']}")
    L.append(f"- Alternativas oficiais baixadas: {met_int['total_fontes_alternativas_oficiais_baixadas']}")
    L.append(f"- Bloqueadas apos navegacao real: {met_int['total_fontes_bloqueadas_apos_navegacao_real']}")
    L.append(f"- Requerem solicitacao formal: {met_int['total_fontes_requerem_solicitacao_formal']}")
    L.append(f"- Arquivos baixados: {met_int['total_arquivos_baixados']}")
    L.append(f"- Geometrias candidatas: {met_int['total_geometrias_candidatas']}")
    L.append("\n## Guardrails preservados\n")
    L.extend(f"- {g}" for g in GUARDRAILS)
    L.append(f"\n## Proximo passo recomendado\n\n{PROXIMO_PASSO}\n")
    P_INT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    P_INT_REPORT.write_text("\n".join(L), encoding="utf-8")


def run() -> dict:
    fontes, manifest, audit, events, geoms = construir()
    log_rows = construir_log(fontes)
    arq_rows = construir_arquivos_index()
    met = metricas(manifest, events, geoms, arq_rows)

    _wcsv(P_MANIFEST, M.MANIFEST_COLS, manifest)
    _wcsv(P_AUDIT, M.AUDIT_COLS, audit)
    _wcsv(P_LOG, LOG_COLS, log_rows)
    _wcsv(P_ARQ, ARQ_COLS, arq_rows)
    _wcsv(P_EVENTS, M.EVENT_COLS, events)
    _wcsv(P_GEOMS, M.GEOM_COLS, geoms)
    P_METRICS.parent.mkdir(parents=True, exist_ok=True)
    P_METRICS.write_text(json.dumps(met, ensure_ascii=False, indent=2), encoding="utf-8")
    _report(manifest, arq_rows, geoms, events, met)

    int_rows = construir_integracao(manifest, geoms, arq_rows)
    met_int = metricas_integracao(events, geoms, met)
    _wcsv(P_INT_TABLE, INTEGRACAO_COLS, int_rows)
    P_INT_METRICS.write_text(json.dumps(met_int, ensure_ascii=False, indent=2), encoding="utf-8")
    _report_integracao(met_int)
    return met


def check() -> int:
    problems = []
    for path, cols in [(P_MANIFEST, M.MANIFEST_COLS), (P_AUDIT, M.AUDIT_COLS), (P_LOG, LOG_COLS),
                       (P_ARQ, ARQ_COLS), (P_EVENTS, M.EVENT_COLS), (P_GEOMS, M.GEOM_COLS),
                       (P_INT_TABLE, INTEGRACAO_COLS)]:
        if not path.is_file():
            problems.append(f"CSV ausente: {path}")
            continue
        with path.open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh))
        faltando = [c for c in cols if c not in header]
        if faltando:
            problems.append(f"{path.name}: colunas faltando {faltando}")
    for path in (P_METRICS, P_INT_METRICS):
        if not path.is_file():
            problems.append(f"JSON ausente: {path}")
        else:
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                problems.append(f"JSON invalido {path.name}: {e}")
    for path in (P_REPORT, P_INT_REPORT):
        if not path.is_file() or "Guardrails preservados" not in path.read_text(encoding="utf-8"):
            problems.append(f"Relatorio sem guardrails: {path.name}")
    if problems:
        print("CHECK: FALHOU")
        for p in problems:
            print("  -", p)
        return 1
    print("CHECK: OK")
    return 0


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    m = run()
    print("Navegacao concluida.")
    print(json.dumps({k: m[k] for k in [
        "total_fontes_pendentes_iniciais", "total_fontes_resolvidas_com_download",
        "total_fontes_baixadas_parcialmente", "total_fontes_alternativas_oficiais_baixadas",
        "total_fontes_bloqueadas_apos_navegacao_real", "total_fontes_requerem_solicitacao_formal",
        "total_arquivos_baixados", "total_arquivos_com_sha256", "total_geometrias_candidatas",
    ]}, ensure_ascii=False, indent=2))
