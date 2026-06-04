# Protocolo C v1uj — Relatorio Focused Public Source Deepening
Gerado: 2026-06-03 21:08:06
Protocolo: v1uj

## Guardrails [ENFORCED]
  ground_truth_operational = False
  can_create_ground_reference = False
  can_create_training_label = False
  can_reopen_protocol_b = False
  dino_usage = SUPPORT_ONLY
  no_overlay_executed = True
  no_coordinates_invented = True
  supervisor_review_completed = False
  route = PUBLIC_OFFICIAL_DISCOVERY
  formal_request_path = LEGACY_SECONDARY_ONLY
  operational_product_max_status = OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW

## O que a v1ui-live encontrou (baseline)
  - 10 fontes publicas acessadas; 21/23 URLs HTTP 200; 497 links; 16 PDFs (33.2MB); 76 assets DOCUMENT_ONLY; 0 candidato geometrico.

## O que a v1uj aprofundou
  - Resolvers focados source-specific: Copernicus EMS, GeoSGB ArcGIS REST, CKAN/Dados Recife, S2iD/dados.gov.br, RIGeo bitstreams e deep links de PDF.
  - Correcao metodologica Copernicus: EMSR564 e EMSR602 foram auditados como off-target (Madagascar/Spain) e nao sao tratados como event-specific para PET/REC.
  - Produtos explicitamente nao-event-specific ficam bloqueados para download.

## Achado principal: CKAN Recife
  - Recursos CKAN totais: 361
  - Candidatos geoespaciais nao contextuais: 147
  - CSV no registry CKAN: 194
  - GeoJSON no registry CKAN: 12
  - URLs unicas tratadas no downloader: 64
  - DOWNLOAD_OK: 49
  - ALREADY_EXISTS_SAME_URL_SAME_HASH: 0
  - DUPLICATE_CONTENT_DIFFERENT_URL: 15
  - Bloqueios/falhas explicitos: 0
  - Grupos de colisao corrigidos por filename seguro: 9
  - Linhas com colisao no audit: 24
  - CSV inventariados: 59
  - GeoJSON inventariados: 5
  - TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW: 2
  - DOCUMENTED_OCCURRENCE_TABLE_NO_GEOMETRY: 28
  - CONTEXTUAL_OFFICIAL_LAYER: 33
  - CONTEXT_ONLY: 1
  - Promotion audit CANDIDATE_WITH_BLOCKERS: 24
  - Promotion audit NOT_A_GEOMETRY_CANDIDATE: 40

## Fontes especificas e respostas
  [PET_2022_02_15] copernicus_prod=0 (vec=0) | geosgb_layers=0 (obs=0) | ckan=0 (geo=0) | s2id=0 | rigeo=0 (geo=0) | pdf_links=17 | downloaded=0 | obs_for_review=0
  [PET_2024_03_21_28] copernicus_prod=0 (vec=0) | geosgb_layers=0 (obs=0) | ckan=0 (geo=0) | s2id=0 | rigeo=0 (geo=0) | pdf_links=0 | downloaded=0 | obs_for_review=0
  [REC_2022_05_24_30] copernicus_prod=0 (vec=0) | geosgb_layers=0 (obs=0) | ckan=343 (geo=147) | s2id=0 | rigeo=0 (geo=0) | pdf_links=6 | downloaded=64 | obs_for_review=0

## Apareceu artefato vetorial?
  - Eventos com artefato vetorial detectado: 1/3

## Candidato pronto para revisao?
  - OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW (total): 0
  - REC_2022_05_24_30 path_to_supervisor_review: NOT_YET
  - REC_2022_05_24_30 status: Sem candidato observado; aprofundar fonte focada ou rodada regional

## Evento mais promissor
  - REC_2022_05_24_30 (fonte: ckan)

## O que falta para ground reference
  - G11 supervisor_review_required: pendente (sempre FAIL nesta etapa)
  - G12 overlay_not_executed: nenhum overlay executado (bloqueia ground reference)
  - G13 label_forbidden: rotulo proibido
  - Revisao supervisora humana ainda nao realizada.

## Recomendacao para v1uk
  - v1uk - Recife CKAN Schema Deep Audit

## Invariantes
  - Nenhum ground reference criado
  - Nenhum label de treinamento criado
  - Nenhum overlay executado
  - Nenhuma coordenada inventada
  - Nenhum dado bruto versionado
  - quickview/suscetibilidade NAO viram ocorrencia observada
  - produto operacional publico no maximo OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW