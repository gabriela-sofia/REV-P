# REV-P - retomada apos perda de contexto

Status: review-only. Sem push. Sem ground truth. Sem treino supervisionado.
Sem score v7 automatico.

## 1. Branch

Branch atual: `marco/pre-unificacao-gates-mv1`.

## 2. Ultimos commits

- `084a122 feat: resgata coordenadas precisas de eventos SUSC-15A`
- `650dd93 feat: expande referencias oficiais SUSC-14C`
- `32dc08c feat: associa ocorrencias a referencias oficiais SUSC-14B`
- `783e234 feat: resgata evidencias observacionais SUSC-14A`

## 3. SUSC-15A-PRECISION existe

SUSC-15A-PRECISION existe no repositorio, com scripts, datasets, outputs,
validador e teste focado.

## 4. SUSC-15A-PRECISION foi commitado

Sim. Commit detectado em `HEAD`: `084a122 feat: resgata coordenadas precisas de
eventos SUSC-15A`.

Observacao de estado local: ha uma modificacao nao stageada em
`datasets/suscetibilidade/susc_15a_precision_geocoded_occurrences_v1.csv`,
restrita a duas alteracoes de `candidate_count` em linhas T5. Essa alteracao nao
foi promovida, stageada ou commitada neste marco.

## 5. Metricas principais do SUSC-15A

- `precision_events_total`: 4412
- `calibration_eligible_events`: 0
- `T0_official_event_polygon`: 0
- `T1_official_event_point`: 0
- `T2_official_address_point_or_parcel`: 0
- `T3_official_intersection_point`: 0
- `T4_official_house_number_linear_reference`: 0
- `T5_official_street_segment_candidate`: 31, apenas como candidato fraco
- `T6_neighborhood_context_only`: 1598
- `precision_patch_links`: 0
- `unique_patches_observational`: 0
- `score_v6_mean_event_links`: null
- `hit_rate_top_10`: 0.0
- `hit_rate_top_20`: 0.0
- `hit_rate_top_30`: 0.0

## 6. Validacoes executadas

- `python scripts/suscetibilidade/validate_susc_features_by_patch_v1.py`: passou.
- `python scripts/suscetibilidade/validate_susc_10a_score_v6_candidate.py`: passou.
- `python scripts/suscetibilidade/validate_susc_14c_official_reference_expansion.py`: passou.
- `python scripts/suscetibilidade/validate_susc_15a_precision_coordinate_rescue.py`: passou.
- `python -m pytest tests/suscetibilidade/test_susc_15a_precision_coordinate_rescue.py -q`: nao concluiu em 120s nem em 300s; o primeiro teste reexecuta a pipeline 15A completa por subprocesso.
- `python -m pytest tests/suscetibilidade/test_susc_15a_precision_coordinate_rescue.py -q -k "not pipeline_and_validator_pass"`: 4 passed, 1 deselected.
- `python -m pytest tests/suscetibilidade/test_susc_15b_forensic_precision_rescue.py -q`: 4 passed.
- `python scripts/suscetibilidade/validate_susc_15b_forensic_precision_rescue.py`: passou.

## 7. Estado de readiness

- `ready_for_12a`: false
- `ready_for_12b`: false
- `ready_for_12c`: false
- `ready_for_score_v7`: false
- `score_v7_created`: false

## 8. Decisao automatica tomada

Caso 2 da arvore de decisao: `calibration_eligible_events < 10` e
`precision_patch_links < 10`.

## 9. Proximo marco executado

SUSC-15B - Forensic Precision Rescue.

Artefatos criados:

- `scripts/suscetibilidade/audit_susc_15b_hidden_coordinate_and_id_fields.py`
- `scripts/suscetibilidade/discover_susc_15b_nonobvious_event_layers.py`
- `scripts/suscetibilidade/acquire_susc_15b_precision_sources_deep.py`
- `scripts/suscetibilidade/parse_susc_15b_precision_sources_deep.py`
- `scripts/suscetibilidade/join_susc_15b_occurrence_ids_to_official_tables.py`
- `scripts/suscetibilidade/build_susc_15b_precision_event_catalog.py`
- `scripts/suscetibilidade/build_susc_15b_precision_event_patch_linkage.py`
- `scripts/suscetibilidade/run_susc_15b_precision_readiness.py`
- `scripts/suscetibilidade/validate_susc_15b_forensic_precision_rescue.py`
- `tests/suscetibilidade/test_susc_15b_forensic_precision_rescue.py`
- `outputs_public/suscetibilidade/SUSC_15B_hidden_coordinate_and_id_audit.csv`
- `outputs_public/suscetibilidade/SUSC_15B_nonobvious_layer_discovery_report.md`
- `outputs_public/suscetibilidade/SUSC_15B_precision_source_download_manifest.csv`
- `datasets/suscetibilidade/susc_15b_precision_event_catalog_v1.csv`
- `datasets/suscetibilidade/susc_15b_precision_event_patch_linkage_v1.csv`
- `outputs_public/suscetibilidade/SUSC_15B_precision_readiness.md`
- `outputs_public/suscetibilidade/SUSC_15B_forensic_precision_rescue_report.md`

## 10. Limitacoes

SUSC-15B nao materializou nova fonte oficial precisa. IDs internos foram
retidos para revisao manual, mas nao houve tabela oficial auxiliar local que
desbloqueasse T0/T1/T2/T3/T4. PDFs/anexos, ArcGIS/WFS/FeatureServer, pontos
criticos, numeracao predial e footprints externos foram registrados como
bloqueados ate aquisicao oficial controlada.

## 11. Confirmacao sem push

Nenhum push foi executado.

## 12. Confirmacao sem ground truth

Nenhum ground truth foi criado. Todos os artefatos permanecem
`can_be_ground_truth=false` e `can_be_used_as_ground_truth=false`.

## 13. Confirmacao sem treino supervisionado

Nenhum treino supervisionado foi criado ou liberado. Todos os artefatos
permanecem `allowed_for_training=false`.

## 14. Confirmacao sem score v7 automatico

Score v7 nao foi criado.
