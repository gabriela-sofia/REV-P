# SUSC-17C2 - Sentinel-1/SAR Footprint Execution (review-only)

Status: review-only. `trainable=false`; `ground_truth=false`; `score_v6_changed=false`; `score_v7_created=false`; `technical_footprint_created_count=0`; `qa_accepted_count=0`; `readiness_status=FOOTPRINT_CANDIDATES_CREATED_NEEDS_QA`.

O SUSC-17C2 executa de forma controlada e review-only os canaries SAR do SUSC-17C. Footprint SAR candidato e evidencia observacional candidata, nunca ground truth, nunca label, nunca score; so vira forte com QA humano. O 17C2 nao altera o score v6 oficial, nao cria score v7, nao cria treino, modelo ou ground truth, nao baixa raster bruto, nao inventa geometria e nao executa o benchmark 17B.

## 1. O que o 17C2 tentou executar

Processar os 5 canaries priorizados pelo 17C: 1 via geometria oficial (International Charter) e 4 via Sentinel-1/SAR (janelas Recife 2014).

## 2. Canaries processados ou bloqueados

- `S17C_E_SUSC13A_00001` (Recife, 2022-05-24): `local_existing_artifact` -> `executed_candidate_created`
- `S17C_W_S16AWIN_00003` (recife, 2014-01-15): `gee_task_spec` -> `blocked_no_runtime_access` (no_gee_or_stac_credential_and_no_opt_in)
- `S17C_W_S16AWIN_00004` (recife, 2014-01-16): `gee_task_spec` -> `blocked_no_runtime_access` (no_gee_or_stac_credential_and_no_opt_in)
- `S17C_W_S16AWIN_00005` (recife, 2014-01-21): `gee_task_spec` -> `blocked_no_runtime_access` (no_gee_or_stac_credential_and_no_opt_in)
- `S17C_W_S16AWIN_00006` (recife, 2014-01-23): `gee_task_spec` -> `blocked_no_runtime_access` (no_gee_or_stac_credential_and_no_opt_in)

## 3. SAR real foi executado ou so planejado

SAR real NAO foi executado. As bibliotecas GEE/STAC existem no ambiente, mas sem credencial e sem opt-in `SUSC_17C2_SAR_RUNTIME=1`. Os 4 canaries SAR ficam `blocked_no_runtime_access` com `query_plan`/`processing_plan` prontos e compativeis com os stubs `scripts/suscetibilidade/susc_16a_gee_sentinel1_flood_mapping_stub.js` e `scripts/suscetibilidade/susc_16a_stac_sentinel1_query_stub.py`.

## 4. Algum footprint candidato foi criado

1 footprint candidato: `S17C2_FP_S17C_E_SUSC13A_00001` (International Charter, caminho de geometria oficial). Materializado a partir do bbox oficial real ja commitado no SUSC-13A, como vetor leve, sem raster e sem invencao. `technical_footprint_created_count=0` porque nenhuma deteccao de mudanca SAR foi de fato executada.

## 5. Por que footprints candidatos nao sao ground truth

Sao geometrias candidatas (oficiais coarse ou, no futuro, derivadas de SAR), sem confirmacao QA, sem validacao patch-a-patch independente, com incerteza nao quantificada. `evidence_class=technical_remote_sensing_flood_footprint_candidate`, `qa_status=needs_review`, `ground_truth=false`, `not_ground_truth_reason` preenchido.

## 6. Por que o QA humano ainda bloqueia o 17B

Nenhum footprint foi aceito (`qa_accepted_count=0`). `eligible_for_strong_evaluation`, `eligible_for_calibration` e `eligible_for_17b` permanecem `false` ate QA `accepted`. Alem disso o footprint do International Charter cai fora da grade de patches atual (0 patch-links), entao nao alimenta o 17B mesmo apos QA sem ajuste de cobertura.

## 7. Como usar cada output no proximo marco

- `canary_execution_registry`: rastreia o estado de cada canary.
- `sentinel1_query_plan` + `sar_processing_plan`: specs prontas para o `SUSC-17C3 SAR Runtime Integration`.
- `candidate_footprint_registry` + `candidate_footprints.geojson`: entram no `SUSC-17D Human QA Protocol`.
- `candidate_patch_links`: vazio aqui (cobertura), mas e o gancho para o 17B quando houver footprint dentro da grade.
- `human_qa_update`: fila de decisoes humanas.

## 8. Readiness

- QA humano: PRONTO para revisar 1 footprint candidato e 4 bloqueios SAR.
- 17B benchmark: BLOQUEADO (["no_qa_accepted_yet", "no_candidate_patch_link_charter_footprint_outside_patch_grid", "no_sar_footprint_executed_no_runtime", "strong_references_still_concentrated_in_recife"]).
- Expansao regional: ainda concentrado em Recife; depende de novos canaries datados em Curitiba/Petropolis.

## 9. Proximo marco recomendado

SUSC-17D Human QA Protocol para revisar os footprints/geometria oficial candidatos; em paralelo SUSC-17C3 SAR Runtime Integration para destravar os canaries SAR bloqueados por falta de runtime.
