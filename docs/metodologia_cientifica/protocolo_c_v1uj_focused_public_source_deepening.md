# Protocolo C — v1uj: Focused Public Official Source Deepening

## Contexto e motivação

A v1ui-live executou descoberta pública genérica: 10 fontes oficiais acessadas,
21/23 URLs com HTTP 200, 497 links extraídos, 16 PDFs baixados (33.2MB) e 76
assets inventariados — todos classificados como `DOCUMENT_ONLY`, com **0
candidatos a geometria observada** e 0 prontos para revisão supervisora.

Diagnóstico: a busca genérica funcionou mecanicamente, mas só retornou PDFs
genéricos. A v1uj **não** retorna ao pedido formal e **não** é um overlay. A
v1uj aprofunda, de forma source-specific, as fontes públicas com maior
probabilidade de conter geometria observada ou produtos operacionais
baixáveis (ZIP/GeoJSON/SHP/GPKG/KML/KMZ/CSV/XLSX georreferenciado).

Rota principal: `PUBLIC_OFFICIAL_DISCOVERY`.
`formal_request_path = LEGACY_SECONDARY_ONLY`.

## Guardrails permanentes

- `ground_truth_operational = false`
- `can_create_ground_reference = false`
- `can_create_training_label = false`
- `can_reopen_protocol_b = false`
- `dino_usage = SUPPORT_ONLY`
- `no_overlay_executed = true`
- `no_coordinates_invented = true`
- `supervisor_review_completed = false`
- Sem login/autenticação, sem bypass de bloqueio, sem scraping agressivo
- Nenhum dado bruto versionado
- quickview **não** vira ground truth
- suscetibilidade/modelagem **não** vira ocorrência observada
- produto operacional público pode virar no máximo `OBSERVED_GEOMETRY_CANDIDATE`,
  nunca ground reference nesta etapa

## Eventos mínimos

- `PET_2022_02_15`
- `PET_2024_03_21_28`
- `REC_2022_05_24_30`

## Resolvers focados

| Resolver | Alvo | Saída |
|---|---|---|
| Copernicus EMS | Ativações Rapid Mapping (EMSR564, EMSR602, termos de busca) | Produtos classificados (delineation/grading/reference/map_pdf/vector_package/raster_package/quicklook/metadata_only) |
| GeoSGB ArcGIS REST | Paths REST reais/prováveis (`/arcgis/rest/services`, `MapServer`, `FeatureServer`) | Metadata de layers (geometry_type, fields, extent, spatialReference) sem baixar features |
| CKAN / Dados Recife | `package_search` por evento/termos | Resources classificados por formato (geo/tabular/contexto) |
| S2iD / dados.gov.br | Desastres reconhecidos, COBRADE, decretos | Registros documentais/temporais; município ≠ geometria |
| RIGeo / SGB | Item pages e bitstreams (DSpace) | Bitstreams classificados (relatório PDF vs pacote de anexos vs geodados) |
| PDF deep link | PDFs da v1ui-live | Links internos e menções a anexos/shapefile/geodados |

### A. Copernicus EMS
Descobre páginas de ativação, produtos e downloads. Diferencia map PDF,
product package, vector package, raster package, delineation/grading e
quicklook. Baixa apenas artefatos públicos permitidos dentro do limite.
Quicklook nunca vira ground truth.

### B. GeoSGB
A v1ui falhou em path genérico. A v1uj testa bases ArcGIS REST reais/prováveis,
consulta `?f=pjson`, lista services/layers e registra metadata. Modelagem/
suscetibilidade é classificada como contexto, nunca ocorrência.

### C. CKAN / Dados Recife
Resolve `dados.recife.pe.gov.br` (e equivalentes) via `package_search`.
Drenagem/infraestrutura genérica não é tratada como ocorrência observada.

### D. S2iD / dados.gov.br
Datasets públicos de desastres reconhecidos. CSV/XLSX com geocódigo municipal/
data são evidência documental/temporal, não geometria de ocorrência.

### E. RIGeo / SGB
Aprofunda busca por avaliação pós-desastre, anexos, shapefile, geodados.
ZIP só com PDF é `DOCUMENT_ONLY`; ZIP com vetor é auditado.

### F. PDF deep link extraction
Extrai texto (pypdf) dos PDFs já baixados, procura URLs internas e termos de
geodados. Sem OCR massivo.

## Gates de promoção (audit)

`revp_v1uj_observed_candidate_promotion_audit.py` aplica 13 gates:

- G1 public_official_traceable
- G2 event_specific_or_regionally_relevant
- G3 artifact_downloaded_or_service_metadata_available
- G4 geometry_or_coordinate_table_available
- G5 crs_or_coordinate_reference_available
- G6 event_date_available
- G7 event_date_compatible
- G8 hazard_or_phenomenon_available
- G9 phenomenon_not_only_susceptibility
- G10 not_static_map_only
- G11 supervisor_review_required (sempre FAIL / pendente)
- G12 overlay_not_executed (sempre FAIL para ground reference)
- G13 label_forbidden (sempre FAIL)

Status máximo: `OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW`.
Nunca `GROUND_REFERENCE`, `GROUND_TRUTH` ou `LABEL`.

## Pipeline e reprodução

Ver `protocolo_c_relatorio_v1uj_focused_public_source_deepening.md` (relatório
gerado) e `protocolo_c_status_atual_v1uj.md` (status corrente). Os comandos
PowerShell de execução live estão registrados no pacote da etapa.

## Armazenamento

Downloads brutos vão para
`local_only/protocolo_c/focused_public_artifacts/raw/v1uj/<source>/<event>/`,
nunca versionados. SHA256 sempre calculado. Apenas registries CSV, configs YAML
e docs são versionáveis.
