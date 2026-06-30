# SUSC-17C4 - Official Artifact Ingestion (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `eligible_for_17b_now=false`.

O SUSC-17C4 ingere e classifica artefatos oficiais review-only sem inventar geometria, data, coordenada ou footprint. PDF/ZIP sem vetor/coordenada explicita fica bloqueado (sem OCR pesado, sem raster); PE3D/MDE e camada contextual, nao evento; centroide de bairro nunca vira geometria forte; geometria fora da grade nao cria patch-link. Nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo, label ou ground truth, e nao executa o benchmark 17B nem o SAR runtime.

## 1. Artefatos oficiais encontrados localmente

1 artefato local utilizavel: o GeoJSON digitalizado do Charter 758 (REC_2022_05_24_30), CRS EPSG:4326, tracked no repo. Os PDFs/ZIP SGB/CPRM existem apenas em `local_runs/` (privado, gitignored), como ponteiros.

## 2. Quais estao ausentes

5 artefatos ausentes (`missing_source_artifact`): pacotes formais nao publicos e eventos sem geometria baixada (DRM-RJ `PKG_FR_PET_001`, COMPDEC `PKG_FR_REC_002`, Defesa Civil Petropolis, Curitiba/IPPUC, PET_2024 Valparaiso).

## 3. Quais dependem de solicitacao formal

7 solicitacoes formais na fila (2 P0):
- `P0_DRM_RJ_PKG_FR_PET_001` -> DRM-RJ/NADE / PKG_FR_PET_001 (queued_needs_human_send)
- `P0_COMPDEC_RECIFE_2022_POINTS_OR_ADDRESSES` -> COMPDEC/Defesa Civil PE / PKG_FR_REC_002 (queued_needs_human_send)

## 4. Status do PKG_FR_PET_001

DRM-RJ/NADE `PKG_FR_PET_001`: `missing_source_artifact`, `PENDING_FORMAL_REQUEST`. Nao existe localmente; bloqueia evidencia forte para PET_2022_02_15. Solicitacao P0 gerada.

## 5. Status do PKG_FR_REC_002

COMPDEC/Defesa Civil PE `PKG_FR_REC_002`: `missing_source_artifact`, `PENDING_FORMAL_REQUEST`. Nao existe localmente; complementaria o footprint coarse do Charter para REC_2022_05_24_30. Solicitacao P0 gerada.

## 6. Status dos PDFs/ZIP SGB/CPRM

`Relatorio_Tecnico_Petropolis.pdf` = `pdf_only`: tentativa `pdf_text_scan` -> `blocked_pdf_only_no_vector` (sem coordenada/vetor explicito; sem OCR pesado). `anexos_avaliacao_pos_desastre_petropolis_rj_2022.zip` = `zip_only`: `zip_inventory` -> `blocked_unknown_format` (arquivo abre como BadZipFile / sem vetor extraivel). Ambos viram solicitacao P1 de vetor/coordenadas.

## 7. Por que PE3D/MDE e contextual, nao evento

PE3D/MDE e modelo de terreno (MDT/MDS): descreve a fisiografia, nao registra onde/quando houve inundacao. Tentativa `not_attempted_context_only` -> `blocked_context_layer_not_event`.

## 8. Alguma geometria candidata foi extraida

Sim: 1 candidato de referencia (`official_observed_event_polygon`) a partir do MultiPolygon real do Charter 758, `qa_status=needs_review`, sem invencao. 1 geometria no GeoJSON candidato.

## 9. Alguma geometria intersecta patches

Nao: 0 patch-links. O MultiPolygon do Charter 758 cai FORA da grade SUSC `recife_*` (mesma area NE da borda). O `patch_id=REC_00019` da property pertence ao esquema do Protocolo C, nao a grade de suscetibilidade.

## 10. Por que 17B continua bloqueado

["no_qa_accepted_yet", "p0_official_packages_missing_drm_rj_and_compdec", "candidate_geometry_falls_outside_susc_patch_grid", "sar_runtime_unavailable"]. Ha geometria candidata, mas sem QA aceito, sem patch-link na grade SUSC e com os pacotes P0 ainda ausentes.

## 11. Proximo marco recomendado

SUSC-17C5 Patch Grid Expansion Review (geometria oficial extraida do Charter 758 cai fora da grade SUSC); em paralelo SUSC-17C5 Formal Request Package para os pacotes P0 DRM-RJ PKG_FR_PET_001 e COMPDEC PKG_FR_REC_002 ainda ausentes.
