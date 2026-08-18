# SUSC-16A - descoberta de footprints externos

Status: review-only.

O SUSC-16A substitui a tentativa de geocodificacao textual por uma estrategia de footprints observacionais, combinando geometrias locais, fontes oficiais/tecnicas e planejamento Sentinel/SAR. A etapa mantem todos os vinculos review-only, nao cria ground truth, nao libera treino supervisionado e nao cria score v7 automatico.

## Fontes registradas
- `S16AEXT_001` Copernicus EMS On Demand Mapping: https://mapping.emergency.copernicus.eu/ (technical_flood_footprint; portal_search_required).
- `S16AEXT_002` Dartmouth Flood Observatory flood records: https://floodobservatory.colorado.edu/wiki/FloodRecords (technical_flood_footprint; global_records_filter_required).
- `S16AEXT_003` Global Flood Database MODIS events: https://developers.google.com/earth-engine/datasets/catalog/GLOBAL_FLOOD_DB_MODIS_EVENTS_V1 (remote_sensing_candidate_footprint; gee_execution_required).
- `S16AEXT_004` Recife Defesa Civil dados abertos: https://dados.recife.pe.gov.br/dataset/?organization=secretaria-executiva-de-defesa-civil (official_observed_event_point; ckan_resource_filter_required).
- `S16AEXT_005` Copernicus GloFAS Global Flood Monitoring: https://global-flood.emergency.copernicus.eu/react/technical-information/glofas-gfm/ (remote_sensing_candidate_footprint; service_query_required).
- `S16AEXT_006` CEMADEN mapa interativo: https://mapainterativo.cemaden.gov.br/ (official_rain_hydro_context; manual_form_required).
- `S16AEXT_007` SGB/CPRM geoportal: https://geoportal.sgb.gov.br/ (official_risk_context; manual_layer_discovery_required).
- `S16AEXT_008` INEA dados geoespaciais: https://www.inea.rj.gov.br/ (official_risk_context; manual_layer_discovery_required).

## Decisao
As fontes foram registradas como candidatas externas. Nenhum raster pesado, imagem bruta ou HTML sem vetor foi promovido. Aquisicao vetorial direta fica bloqueada ate haver URL pequena e auditavel.
