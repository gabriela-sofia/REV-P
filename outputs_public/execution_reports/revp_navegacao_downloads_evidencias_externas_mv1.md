# Navegacao real e download de evidencias externas (MV1)

> Passada review-only. Nenhuma fonte foi promovida a label, negativo formal ou ground truth. Arquivos salvos apenas em quarentena local (git-ignored); SHA256 calculado do arquivo real.

## 1. Escopo

Resolver as 7 fontes pendentes da rodada anterior por navegacao real nos portais oficiais, baixando arquivos quando tecnicamente possivel.

## 2. Por que a rodada anterior era insuficiente

A rodada anterior classificou CEMADEN, ANA, MapBiomas, IBGE, APAC, Copernicus e IPPUC apenas como `requer_download_manual`, sem navegar dentro dos portais nem testar endpoints (API, ArcGIS REST, dados abertos, FTP).

## 3. Metodo de navegacao real

Para cada fonte: URL do manifesto -> navegacao no portal -> inspecao do HTML -> busca web -> teste de endpoints (API IBGE, ArcGIS REST GeoCuritiba, arquivos diretos ANA/MapBiomas, RIGEO/SGB) -> download direto ou fonte oficial alternativa.

## 4. Fontes resolvidas com download

- `FONTE_NAC_004` (IBGE - Instituto Brasileiro de Geografia e Estatistica)
- `FONTE_CUR_003` (IPPUC / GeoCuritiba - Prefeitura de Curitiba)
- `FONTE_NAC_003` (MapBiomas)
## 5. Fontes parcialmente resolvidas

- `FONTE_NAC_002` (ANA - Agencia Nacional de Aguas e Saneamento Basico (SNIRH/HidroWeb)) — Inventario oficial baixado; serie historica especifica do Capibaribe ainda exige consulta por estacao no HidroWeb.
## 6. Fontes alternativas oficiais encontradas

- `FONTE_NAC_006` (DRM-RJ / NADE - Servico Geologico do Estado do Rio de Janeiro) — Portal DRM-RJ indisponivel (503); equivalente oficial federal (SGB/CPRM) para o mesmo municipio/tema baixado.
## 7. Fontes bloqueadas apos navegacao real

- `FONTE_NAC_005` (APAC - Agencia Pernambucana de Aguas e Clima) — Portal Joomla/JS sem arquivo estatico de boletim localizavel apos navegacao; dados via PCD em tempo real. Angulo de chuva parcialmente coberto pelo inventario ANA.
- `FONTE_INT_001` (Copernicus Emergency Management Service (EMS) Rapid Mapping) — Nenhum produto rapid mapping vetorial publico para Petropolis localizado apos navegacao; GloFAS/GFM e noticia/monitoramento sem download de delimitacao.
## 8. Fontes que requerem solicitacao formal

- `FONTE_NAC_001` (CEMADEN/MCTI - Centro Nacional de Monitoramento e Alertas de Desastres Naturais) — Download do CEMADEN depende de formulario com e-mail (link enviado por e-mail); requer solicitacao. Suscetibilidade coberta por fonte alternativa oficial (SGB).
## 9. Arquivos baixados

- `ARQ_IBGE_PET` Malha municipal de Petropolis (3303906) (GeoJSON, 2371 bytes) -> local_only/evidencias_externas_quarentena/petropolis/ibge_malha_municipal_petropolis_3303906.geojson
- `ARQ_IBGE_REC` Malha municipal de Recife (2611606) (GeoJSON, 1110 bytes) -> local_only/evidencias_externas_quarentena/recife/ibge_malha_municipal_recife_2611606.geojson
- `ARQ_IBGE_CUR` Malha municipal de Curitiba (4106902) (GeoJSON, 1773 bytes) -> local_only/evidencias_externas_quarentena/curitiba/ibge_malha_municipal_curitiba_4106902.geojson
- `ARQ_GEOCWB_BACIA` Bacias Hidrograficas de Curitiba (camada 54) (GeoJSON, 804561 bytes) -> local_only/evidencias_externas_quarentena/curitiba/geocuritiba_bacia_hidrografica_l54.geojson
- `ARQ_ANA_INVENTARIO` Inventario das Estacoes Pluviometricas (PDF, 4350035 bytes) -> local_only/evidencias_externas_quarentena/fontes_nacionais/ana_inventario_estacoes_pluviometricas.pdf
- `ARQ_MAPBIOMAS_DHN250` Estatisticas de cobertura COL.10.1 por divisao hidrografica (DHN250) (XLSX, 9177772 bytes) -> local_only/evidencias_externas_quarentena/fontes_nacionais/mapbiomas_col10_1_cobertura_dhn250_estado_bioma.xlsx
- `ARQ_SGB_CARTA_PET` Carta de Suscetibilidade a Movimentos Gravitacionais de Massa e Inundacao - Petropolis/RJ (PDF, 13811060 bytes) -> local_only/evidencias_externas_quarentena/petropolis/sgb_cprm_carta_suscetibilidade_petropolis_cs.pdf
- `ARQ_NHESS` Artigo NHESS 23/1157/2023 - Petropolis fev/2022 (PDF, 7765136 bytes) -> local_only/evidencias_externas_quarentena/fontes_internacionais/nhess_23_1157_2023_petropolis_fev2022.pdf
## 10. Hashes calculados

- `ARQ_IBGE_PET` SHA256 `11b41120f8c51c5fede86e1702deee9577e808a433bd3fca3908c9abb4c3c4a9`
- `ARQ_IBGE_REC` SHA256 `df703841e180eb5afff728077f9ba4a2c8e2fd08a77d3c85f439082dbd31c0e4`
- `ARQ_IBGE_CUR` SHA256 `ecf25a7c5bc2134e0326c2e6fbe4781f83e631c107affc5f5cf3d00f6df52beb`
- `ARQ_GEOCWB_BACIA` SHA256 `653b9e15abe261690d5d060f427ec136611d3f9abef2ed824451ca5e4bb8d4b6`
- `ARQ_ANA_INVENTARIO` SHA256 `b54f5c3e4ada405ffd4b92af41e47bb85703061b2e184c6200fc95291f0ffe6c`
- `ARQ_MAPBIOMAS_DHN250` SHA256 `b93e268b1cff7ce5da5d57339d77a0e4098918dff6b86122ab5ca04811b053bb`
- `ARQ_SGB_CARTA_PET` SHA256 `97880242981d26caa7487dcf1e1c7c2f0857f991a5c662c2379b9671dcd6d665`
- `ARQ_NHESS` SHA256 `a8f1468feb77c412acc8d0e508dc78deaf91fc6f2b53e219496516b902dcfc56`
## 11. Geometrias candidatas

- `GEO_ARQ_IBGE_PET` (Petropolis, Polygon, EPSG:4326) — overlay bloqueado: geometria_de_contexto_territorial_nao_e_evento_observado
- `GEO_ARQ_IBGE_REC` (Recife, Polygon, EPSG:4326) — overlay bloqueado: geometria_de_contexto_territorial_nao_e_evento_observado
- `GEO_ARQ_IBGE_CUR` (Curitiba, Polygon, EPSG:4326) — overlay bloqueado: geometria_de_contexto_territorial_nao_e_evento_observado
- `GEO_ARQ_GEOCWB_BACIA` (Curitiba, Polygon, EPSG:4326) — overlay bloqueado: geometria_de_contexto_territorial_nao_e_evento_observado
- `GEO_FONTE_REC_001` (Recife, Point, EPSG:4326) — overlay bloqueado: geometria_de_suscetibilidade_nao_e_evento_observado
- `GEO_FONTE_REC_002` (Recife, MultiPolygon, EPSG:4326 (origem EPSG:32725)) — overlay bloqueado: produto_pode_misturar_deslizamento_e_inundacao_revisao_obrigatoria
## 12. Eventos candidatos

- 8 eventos candidatos; nenhum pode sustentar label patch-level nesta passada.
## 13. Limitacoes

- Series hidrologicas por estacao (ANA/HidroWeb) e boletins APAC dependem de consulta interativa.
- Copernicus EMS nao tem produto rapid mapping vetorial publico para Petropolis.
- O SIG vetorial completo da carta SGB (1.8 GB) nao foi baixado por tamanho; usado o PDF do mapa.
- Geometrias baixadas sao de contexto territorial/drenagem, nao de evento observado.

## 14. Proximas acoes manuais inevitaveis

Solicitar formalmente os dados sob formulario/e-mail (CEMADEN) e os portais sem arquivo estatico (APAC, Copernicus EMS rapid mapping); complementar a serie hidrologica da ANA via HidroWeb por estacao do Capibaribe; e submeter as geometrias candidatas (bacias GeoCuritiba, malhas IBGE, carta de suscetibilidade SGB) a revisao humana. Nenhuma fonte e promovida a label nesta passada.

## 15. Guardrails preservados

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

## 16. Integracao com o marco label-free

Ver `revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.md`. Todos os itens integrados tem `pode_virar_label_agora=false`.
