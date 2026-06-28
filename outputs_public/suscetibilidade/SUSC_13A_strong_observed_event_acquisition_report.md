# SUSC-13A - Aquisicao forte de eventos observados reais

Status: **review-only** | `can_be_ground_truth=false` | `allowed_for_training=false`

O SUSC-13A busca evidências observacionais mais fortes de alagamento/inundação para reduzir a fragilidade detectada em SUSC-11/12. Mesmo eventos fortes permanecem review-only nesta etapa e não criam ground truth, treino supervisionado ou confirmação operacional automática por patch.

## 1. Objetivo
Fortalecer a camada observacional usada em SUSC-11/12, separando eventos fortes,
moderados, fracos e documentais sem promover nenhum item a rotulo operacional.

## 2. Fontes registradas
Total de fontes-alvo: **15**.

| regiao | fontes |
|---|---|
| curitiba | 5 |
| petropolis | 5 |
| recife | 5 |

## 3. Politica de aquisicao controlada
Downloads automaticos so ocorrem com URL direta leve, download permitido e
extensao permitida. Raster, Sentinel bruto, API com chave, scraping agressivo e
arquivos acima de 100MB permanecem bloqueados.

## 4. Resultado de downloads
Tentativas registradas: **15**.

| status | n |
|---|---|
| not_attempted_no_direct_url_or_no_network | 15 |

## 5. Fontes manuais
O diretorio `datasets/suscetibilidade/observed_event_sources_susc13a/` aceita CSV,
GeoJSON, KML/KMZ, SHP ZIP, GPKG, WKT, PDF pequeno, XLSX e TXT com data, geometria,
tipo de evento, fonte e instituicao. Fontes incompletas exigem revisao manual.

## 6. Eventos parseados
Total de registros parseados/reclassificados: **13**.

| nivel | n |
|---|---|
| moderate_official_occurrence_point | 4 |
| weak_administrative_context | 3 |
| weak_risk_area_context | 3 |
| documentary_only | 2 |
| moderate_official_flood_bbox | 1 |

## 7. Eventos por regiao
| regiao | n |
|---|---|
| recife | 8 |
| petropolis | 5 |

## 8. Eventos fortes e moderados
Eventos fortes encontrados: **0**.
Eventos moderados encontrados: **5**.
Eventos fortes exigem data/periodo e geometria explicita de alagamento/inundacao.
Registros moderados continuam uteis para avaliacao observacional, mas nao sao GT.

## 9. Linkage patch-evento
Linhas de linkage: **39**; links fortes/moderados: **1**;
linhas permitidas para avaliacao observacional review-only: **30**.

| relacao espacial | n |
|---|---|
| near_patch_buffer_candidate | 29 |
| insufficient_for_patch_link | 7 |
| same_region_period_context | 2 |
| moderate_point_inside_patch | 1 |

## 10. Regioes com melhora observacional
Regioes com algum vinculo forte/moderado ou avaliavel: **petropolis**.
Melhora aqui significa apenas mais aderencia observacional review-only, nao
confirmacao operacional por patch.

## 11. Lacunas restantes
Recife: buscar pontos de ocorrencia/alagamento com data e lat/lon ou shapefile/CSV oficial.
Petropolis: buscar footprint/ocorrencia 2022 com coordenada/poligono.
Curitiba: buscar ocorrencias oficiais GeoCuritiba/IPPUC/Defesa Civil com geometria.

## 12. Governanca e limites
- `can_be_ground_truth=false` em todos os artefatos.
- `allowed_for_training=false` em todos os artefatos.
- `review_only=true` em todos os artefatos.
- Risco, alerta e contexto administrativo nao sao evento observado.
- Nenhum score v7, modelo, treino supervisionado ou confirmacao automatica foi criado.
