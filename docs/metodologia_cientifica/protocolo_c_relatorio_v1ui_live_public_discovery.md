# Relatorio v1ui-live — Public Official Web Discovery (Execucao Real)

Execucao: 2026-06-03
Modo: --allow-web --download (live, nao DRY_RUN)

## Guardrails

- ground_truth_operational=false [ENFORCED]
- can_create_ground_reference=false [ENFORCED]
- can_create_training_label=false [ENFORCED]
- no_overlay_executed=true [ENFORCED]
- no_coordinates_invented=true [ENFORCED]
- formal_request_path=LEGACY_SECONDARY_ONLY [ENFORCED]

## 1. Fontes Publicas Acessadas

23 discoveries registradas para 10 fontes e 3 eventos.

### Fontes que responderam (HTTP 200):
- SGB_CPRM_RIGEO (rigeo.sgb.gov.br) — 200
- SGB_CPRM_GEOSGB (geosgb.sgb.gov.br) — 200
- PREFEITURA_PETROPOLIS (petropolis.rj.gov.br) — 200
- DADOS_ABERTOS_RECIFE (dados.recife.pe.gov.br) — 200
- DADOS_ABERTOS_RECIFE (www.recife.pe.gov.br) — 200
- CEMADEN_PORTAL (www2.cemaden.gov.br, www.cemaden.gov.br) — 200
- DADOS_GOV_BR (dados.gov.br) — 200
- COPERNICUS_EMS_PUBLIC (emergency.copernicus.eu) — 200
- DEFESA_CIVIL_RJ_ESTADO (www.defesacivil.rj.gov.br) — 200

### Fontes que falharam:
- DRM_RJ_PORTAL (www.drm.rj.gov.br) — HTTP 0 (timeout/SSL/connection refused)
- EMLURB_RECIFE (www.emlurb.recife.pe.gov.br) — HTTP 0 (timeout/SSL/connection refused)

## 2. Crawler — Links Extraidos

497 links extraidos de portais publicos (depth=1, max-depth=2).
16 links classificados como CANDIDATE (todos PDFs por extensao .pdf).

### Distribuicao por fonte:
- PREFEITURA_PETROPOLIS: 7 PDFs
- DADOS_ABERTOS_RECIFE: 4 PDFs
- DEFESA_CIVIL_RJ_ESTADO: 3 PDFs
- CEMADEN_PORTAL: 1 PDF
- COPERNICUS_EMS_PUBLIC: 1 PDF

### Conteudo dos PDFs encontrados:
Todos os 16 candidatos sao documentos genericos (certificados, cartilhas de vacinacao,
calendarios de eventos turisticos, regulamentos, planos de integridade, notas tecnicas
de outras bacias). Nenhum e especifico de evento de inundacao/deslizamento dos 3 eventos-alvo.

## 3. ArcGIS/GeoServer

3 tentativas de resolver servicos ArcGIS REST no GeoSGB.
Todas FETCH_FAILED — o endpoint raiz (geosgb.sgb.gov.br) responde HTML mas nao retorna
JSON de metadata no path generico. Necessario: descobrir paths ArcGIS REST especificos
(ex: /geosgb/rest/services/...).

## 4. Downloads

16 artefatos baixados, 33.2MB total.
Todos PDFs. Nenhum ZIP, SHP, GPKG, GeoJSON, KML, CSV ou XLSX.

## 5. Inventario

76 assets inventariados (16 PDFs descomprimidos em subcomponentes pelo inventory).
Todas as 76 entradas classificadas como DOCUMENT_ONLY.
Nenhuma geometria vetorial, nenhuma tabela com coordenadas.

## 6. Candidatos a Geometria Observada

0 (zero) candidatos a geometria observada.
76 assets avaliados, todos DOCUMENT_ONLY.
Nenhum asset continha geometria, coordenadas, ou campos de data/fenomeno.

## 7. Gate Delta v1uh -> v1ui

4 GAINS:
- PET_2022_02_15: public_artifact_found (NO -> YES), event_specificity_improved (NO -> YES)
- REC_2022_05_24_30: public_artifact_found (NO -> YES), event_specificity_improved (NO -> YES)

26 NO_CHANGE — geometry, CRS, date, hazard, locality, supervisor_review, ground_reference, label
permaneceram todos em NO.

PET_2024_03_21_28: nenhum ganho (nenhum artefato encontrado especifico para este evento).

## 8. Status por Evento

| Evento | Artefatos | Geometria | Gate Delta | Pronto p/ Review |
|--------|-----------|-----------|------------|-------------------|
| PET_2022_02_15 | 12 PDFs | 0 | 2 GAINS | NAO |
| PET_2024_03_21_28 | 0 | 0 | 0 GAINS | NAO |
| REC_2022_05_24_30 | 4 PDFs | 0 | 2 GAINS | NAO |

## 9. Fila Supervisora

76 entradas, todas NOT_REVIEWABLE (DOCUMENT_ONLY).
0 prontos para revisao.

## 10. Conclusao

A varredura v1ui-live confirmou que os portais publicos genéricos das fontes configuradas
NAO disponibilizam diretamente artefatos com geometria de evento. Os PDFs encontrados
sao documentos institucionais genericos, nao relatorios tecnicos de desastre com anexos vetoriais.

### Candidato suficiente para v1uj?
**NAO.** Nenhum candidato a geometria observada foi encontrado.

### Proximo passo recomendado:
Aprofundar busca publica regional focada:
1. Navegar manualmente no GeoSGB para encontrar paths ArcGIS REST especificos de Petropolis
2. Buscar no Copernicus EMS activations especificas: EMSR564 (Petropolis 2022), EMSR602 (Recife 2022)
3. Buscar no portal dados.recife.pe.gov.br por datasets especificos de alagamento/SEDEC
4. Buscar relatorios SGB/CPRM de levantamento pos-desastre Petropolis (RIGeo por titulo)
5. Buscar dados S2iD (Sistema Integrado de Informacoes sobre Desastres) via dados.gov.br
6. Considerar resolucao de paths CKAN especificos em dados.recife.pe.gov.br/dataset/

## Invariantes Confirmados

- Nenhum ground truth criado
- Nenhum ground reference criado
- Nenhum label de treinamento criado
- Nenhum overlay executado
- Nenhuma coordenada inventada
- Nenhum dado bruto versionado (PDFs em local_only/)
- formal_request_path=LEGACY_SECONDARY_ONLY
