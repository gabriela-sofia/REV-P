# SUSC-17C3 - Official Source Acquisition & Patch Coverage Audit (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `eligible_for_17b_now=false`.

O SUSC-17C3 audita aquisicao de fonte oficial e cobertura de patches review-only. Mapa de risco/suscetibilidade nunca e evento observado; PE3D/MDE e camada fisica contextual, nao evento; bairro/centroide nunca e geometria forte; PDF sem vetor/coordenada fica bloqueado. O 17C3 nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo, label ou ground truth, nao inventa pacote, coordenada, poligono, data ou fonte, e nao executa o benchmark 17B.

## 1. O que o NotebookLM encontrou

Fontes oficiais reais ja referenciadas no Protocolo C: relatorios e anexos SGB/CPRM (Petropolis 2022), pacote formal DRM-RJ `PKG_FR_PET_001`, pacote formal COMPDEC/Defesa Civil PE `PKG_FR_REC_002`, evento Petropolis 2024 (Valparaiso-Floresta), o mapa de extensao de inundacao do International Charter (Recife 2022-05) e a camada topografica PE3D/MDE.

## 2. O que e realmente utilizavel

Apenas o footprint do International Charter (Recife 2022-05) traz geometria de evento utilizavel hoje - e ainda assim coarse, sem QA e fora da grade de patches. Os demais sao PDFs sem vetor ou pacotes nao baixados.

## 3. O que e so contexto

PE3D/MDE e camada fisica contextual (terreno), nunca evidencia de evento observado. Entra como `contextual_physical_layer`, fora do caminho de evidencia forte.

## 4. Por que `ground reference` foi rebaixado para `reference_evidence`

O Protocolo C usava o termo `ground_reference`, mas no SUSC-17A o protocolo foi formalizado como `reference_evidence` review-only: footprints e geometrias oficiais sao evidencia candidata, nunca ground truth nem label. O 17C3 mantem essa disciplina (`ground_truth=false` em todas as linhas).

## 5. Por que DRM-RJ / PKG_FR_PET_001 e P0

E o pacote formal com maior chance de trazer setores/poligonos oficiais do evento Petropolis 2022-02-15 dentro da malha urbana (potencial intersecao com a grade de patches), hoje `PENDING_FORMAL_REQUEST`.

## 6. Por que COMPDEC Recife e P0

O `PKG_FR_REC_002` (COMPDEC/Defesa Civil PE) pode trazer pontos/enderecos oficiais do evento Recife 2022-05 dentro da grade, complementando o footprint coarse do International Charter; hoje `PENDING_FORMAL_REQUEST`.

## 7. Por que PE3D/MDE e contextual, nao evento

PE3D/MDE e modelo de terreno (MDT/MDS): descreve a fisiografia, nao registra onde/quando houve inundacao. Usar como evento confundiria suscetibilidade fisica com ocorrencia.

## 8. Por que SGB/CPRM PDF esta bloqueado ate vetor/coordenada

Os artefatos SGB/CPRM existem localmente (`DOWNLOAD_OK`, `INGESTED`) mas sao PDF/ZIP sem vetor. Sem extracao de coordenada/vetor auditavel ficam `pdf_only` e nunca viram avaliacao forte.

## 9. Qual candidato intersecta ou nao a grade de patches

- `CUR_HISTORICAL` (Curitiba): no_geometry_to_test (intersecoes=0, nearest=not_available)
- `PET_2022_02_15` (Petropolis): no_geometry_to_test (intersecoes=0, nearest=not_available)
- `PET_2024_03_21_28` (Petropolis): no_geometry_to_test (intersecoes=0, nearest=not_available)
- `REC_2022_05_24_30` (Recife): outside_patch_grid (intersecoes=0, nearest=1398.4)

A unica geometria (International Charter) cai FORA da grade Recife (adjacente a borda NE). Os demais nao tem geometria para testar.

## 10. Proximo passo mais racional

P0 - ingestao/solicitacao dos pacotes oficiais:
- `P0_DRM_RJ_PKG_FR_PET_001` -> PKG_FR_PET_001 (DRM-RJ/NADE)
- `P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES` -> PKG_FR_REC_002 (COMPDEC/Defesa Civil PE)

SUSC-17C4 Official Artifact Ingestion (ingerir/solicitar PKG_FR_PET_001 DRM-RJ e PKG_FR_REC_002 COMPDEC, P0); em paralelo SUSC-17C4 Patch Grid Expansion Review para a geometria do International Charter que cai fora da grade; SAR runtime e QA dependem dessas etapas

`blocks_17b`: ["no_qa_accepted_yet", "p0_official_packages_missing_drm_rj_and_compdec", "only_event_geometry_charter_falls_outside_patch_grid", "sar_runtime_unavailable"].
