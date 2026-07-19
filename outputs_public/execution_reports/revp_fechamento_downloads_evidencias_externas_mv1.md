# Fechamento dos downloads de evidencias externas (MV1)

> Passada review-only. Nenhuma fonte foi promovida a label, negativo formal ou ground truth. Downloads salvos apenas em quarentena local (git-ignored); SHA256 calculado do arquivo real.

## Resumo por status de download

- `ja_existia_localmente`: 10
- `requer_download_manual`: 7
- `falha_download`: 1
- `baixado`: 1

## Fontes baixadas nesta rodada

- `FONTE_INT_002` — local_only/evidencias_externas_quarentena/fontes_internacionais/nhess_23_1157_2023_petropolis_fev2022.pdf — SHA256 `a8f1468feb77c412...` (7765136 bytes)

## Fontes que ja existiam localmente

- `FONTE_REC_001` — datasets/external_sources/recife_minimal_tp/raw/recife_defesa_civil_risk_areas_geojson.geojson — SHA256 `0f8036b19ec60481...`
- `FONTE_REC_002` — datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758/derived/event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson — SHA256 `8e833d449effb837...`
- `FONTE_REC_003` — local_runs/protocolo_c/v1np/raw/recife_dados_vivos_sedec_f3b3a0ab-ac8f-4fe8-a1cc-c3afdd20081a_001.csv — SHA256 `e26f80ddada0a6c7...`
- `FONTE_REC_004` — local_runs/protocolo_c/v1np/raw/recife_emlurb_156_031c3ad3-265f-4d9c-830a-7d1f85f830fa_007.csv — SHA256 `dd2f31e3f53c3aa9...`
- `FONTE_PET_001` — local_runs/protocolo_c/v1if/raw_official_sources/Relatorio_Tecnico_Petropolis.pdf — SHA256 `45d1a9802c648570...`
- `FONTE_PET_002` — local_runs/protocolo_c/v1if/raw_official_sources/anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip — SHA256 `78a185ba06b6f5d4...`
- `FONTE_PET_003` — local_runs/protocolo_c/v1mc/extracted/PKG_V1MC_0001_24be39ae/ANEXO-I-CPRM_Relatório_Petrópolis_19-02-22_Bairro_Mosella.pdf — SHA256 `053fa4cbccac09a6...`
- `FONTE_PET_004` — local_runs/protocolo_c/v1na/raw/gazette_issue_2022-02-15_4542_0a45c34e.pdf — SHA256 `3399af178d756a19...`
- `FONTE_CUR_001` — local_runs/ground_truth/v2cd/downloaded_sources/SRC_4c0debd1596a.html — SHA256 `8997b1edd6025e46...`
- `FONTE_CUR_002` — local_runs/ground_truth/v2ce/downloaded_onehop_sources/SRC_9b151d28ab38.pdf — SHA256 `4db40582b10238be...`

## Fontes que exigem download manual

- `FONTE_CUR_003` — https://www.ippuc.org.br — Portal IPPUC/GeoCuritiba retorna HTML; o dataset exige navegacao/selecao humana, sem URL de arquivo direto.
- `FONTE_NAC_001` — https://mapainterativo.cemaden.gov.br/ — CEMADEN Mapa Interativo e aplicacao JavaScript; o download exige interacao humana, nao ha URL de arquivo direto.
- `FONTE_NAC_002` — https://www.snirh.gov.br/hidroweb/ — ANA HidroWeb/SNIRH exige busca por estacao e formulario; a URL retorna HTML de portal, nao o dataset.
- `FONTE_NAC_003` — https://brasil.mapbiomas.org/ — MapBiomas exige selecao de colecao/recorte (Toolkit/GEE ou formulario); a URL retorna HTML, nao o arquivo de cobertura.
- `FONTE_NAC_004` — https://www.ibge.gov.br/geociencias/downloads-geociencias.html — IBGE Downloads Geociencias e pagina-indice/FTP; exige navegacao ate o arquivo de malha de setores censitarios.
- `FONTE_NAC_005` — https://www.apac.pe.gov.br — APAC retorna portal HTML; os boletins pluviometricos exigem navegacao/selecao de data.
- `FONTE_INT_001` — https://rapidmapping.emergency.copernicus.eu/ — Copernicus EMS Rapid Mapping exige selecao da ativacao (EMSR) e download por produto; o ID exato para Petropolis nao foi confirmado.

## Falhas de download

- `FONTE_NAC_006` — HTTP 503 — DRM-RJ retornou HTTP 503 (Service Unavailable); alem disso nao ha URL de arquivo direto (requer solicitacao formal).

## Guardrails preservados

- Sem treino supervisionado; sem label binario; sem positivo formal; sem negativo formal.
- Sem ground truth operacional patch-level nesta passada.
- unknown nao vira negativo; ausencia de evidencia nao vira classe 0.
- Curitiba nao vira negativo formal.
- Evidencia externa nao vira label automaticamente.
- DINOv2 nao prova inundacao.
- Fonte textual sem geometria nao fecha patch-level.
- Landslide scar nao prova flood extent.
- Geometria de suscetibilidade nao e geometria de evento observado.
- Download de fonte nao significa validacao operacional.

## Proximo passo recomendado

Executar manualmente os downloads marcados como requer_download_manual (CEMADEN, ANA/HidroWeb, MapBiomas, IBGE, APAC, Copernicus EMS, IPPUC), salvar o bruto na quarentena e re-rodar este script; e solicitar formalmente as fontes em falha/bloqueio (DRM-RJ). Nenhuma fonte e promovida a label nesta passada.
