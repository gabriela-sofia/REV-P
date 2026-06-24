# Auditoria inicial pre-unificacao REV-P

## Estado Git atual
- branch atual: analysis/temporal-asset-readiness-mv1
- top commit: 67d8cfd docs: consolida curadoria publica e linha metodologica PT-BR
- staged files: 0
- entradas untracked/modificadas relevantes: 327

## Worktrees detectados
```text
C:/Users/gabriela/Documents/REV-P                                       67d8cfd [analysis/temporal-asset-readiness-mv1]
C:/Users/gabriela/.codex/worktrees/0475/REV-P                           4c7bad4 [public-docs-consolidation]
C:/Users/gabriela/Documents/REV-P/.claude/worktrees/angry-curran-de22fa 869bde6 [marco/mv2-observational-closure-contract]
C:/Users/gabriela/Documents/REV-P-mv2-01-reconciliado                   817ed17 [marco/mv2-12-reconstrucao-espectral-sentinel-baseline]
C:/Users/gabriela/Documents/REV-P-v2cx-v2dd-clean-20260617_154005       8b8c278 [feat/v2de-v2dk-trainability-mv1]
```

## MV2 worktree esperado
- path: C:\Users\gabriela\Documents\REV-P-mv2-01-reconciliado
- existe: True
- branch: marco/mv2-12-reconstrucao-espectral-sentinel-baseline
- top commit: 817ed17 MV2-11: rebalanceamento representacional regional label-free
- staged files: 0

## Presenca de blocos
- DATA-05: PRESENTE (outputs_public/mv2_data_temporal_window_intake)
- MV2-12 Data Readiness: PRESENTE (outputs_public/mv2_data_readiness)
- MV2-12 Spectral Reconstruction no MV2 worktree: PRESENTE (C:\Users\gabriela\Documents\REV-P-mv2-01-reconciliado\outputs_public\mv2_spectral_reconstruction)

## Riscos detectados
- arquivos MV2-13/MV2-14/MV2-15 misturados: 55
- risco local_only/rasters/crops/pesados em status: 2
- acao segura: excluir MV2-13+ da consolidacao e manter rasters/crops fora de outputs_public.

## Status curto capturado
```text
M .gitignore
?? datasets/schemas/schema_mv2_13_binding_risk_matrix.json
?? datasets/schemas/schema_mv2_13_day10_spectral_gate.json
?? datasets/schemas/schema_mv2_13_manual_recovery_queue.json
?? datasets/schemas/schema_mv2_13_sentinel_binding.json
?? datasets/schemas/schema_mv2_13_stac_gate.json
?? datasets/schemas/schema_mv2_14_day10_post_lineage_gate.json
?? datasets/schemas/schema_mv2_14_lineage_binding_matrix.json
?? datasets/schemas/schema_mv2_14_lineage_candidates.json
?? datasets/schemas/schema_mv2_14_manual_lineage_template.json
?? datasets/schemas/schema_mv2_14_post_lineage_stac_gate.json
?? datasets/schemas/schema_mv2_15_schedule_state.json
?? datasets/schemas/schema_mv2_data_01_lineage_targets.json
?? datasets/schemas/schema_mv2_data_01_manual_gee_template.json
?? datasets/schemas/schema_mv2_data_01_metadata_query_plan.json
?? datasets/schemas/schema_mv2_data_02_api_lineage_resolution.json
?? datasets/schemas/schema_mv2_data_02_api_provider_registry.json
?? datasets/schemas/schema_mv2_data_02_private_raster_validation.json
?? datasets/schemas/schema_mv2_data_02_raster_canary_execution_manifest.json
?? datasets/schemas/schema_mv2_data_02_raster_canary_plan.json
?? datasets/schemas/schema_mv2_data_03_canary_metadata_consensus.json
?? datasets/schemas/schema_mv2_data_03_corpus_impact_matrix.json
?? datasets/schemas/schema_mv2_data_03_official_canary_candidates.json
?? datasets/schemas/schema_mv2_data_03_private_raster_canary_execution_manifest.json
?? datasets/schemas/schema_mv2_data_03_private_raster_canary_validation.json
?? datasets/schemas/schema_mv2_data_04_corpus_api_lineage_resolution.json
?? datasets/schemas/schema_mv2_data_04_corpus_metadata_batch.json
?? datasets/schemas/schema_mv2_data_04_temporal_window_gap_matrix.json
?? datasets/schemas/schema_mv2_data_04_temporal_window_template.json
?? datasets/schemas/schema_mv2_data_05_normalized_temporal_windows.json
?? datasets/schemas/schema_mv2_data_05_probe_ready_batch.json
?? datasets/schemas/schema_mv2_data_05_temporal_evidence_validation.json
?? datasets/schemas/schema_mv2_data_05_temporal_promotion_gate.json
?? docs/metodologia_cientifica/revp_ontologia_labels_ground_truth_mv1.md
?? docs/metodologia_cientifica/revp_politica_anti_leakage_mv1.md
?? docs/metodologia_cientifica/revp_politica_evidencia_negativa_mv1.md
?? outputs_public/audits/
?? outputs_public/execution_reports/revp_auditoria_critica_banca_marco_mv1.md
?? outputs_public/execution_reports/revp_auditoria_integral_estado_atual.md
?? outputs_public/execution_reports/revp_auditoria_prontidao_temporal_assets_mv1.md
?? outputs_public/execution_reports/revp_auditoria_reprodutibilidade_externa_marco_mv1.md
?? outputs_public/execution_reports/revp_curadoria_evidencias_externas_mv1.md
?? outputs_public/execution_reports/revp_current_project_state_after_ptbr_curation.md
?? outputs_public/execution_reports/revp_diagnostico_linguagem_publica_ptbr.md
?? outputs_public/execution_reports/revp_fechamento_downloads_evidencias_externas_mv1.md
?? outputs_public/execution_reports/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md
?? outputs_public/execution_reports/revp_hardening_reprodutibilidade_publica_marco_mv1.md
?? outputs_public/execution_reports/revp_integracao_final_marco_mv1_maturidade_revisao_humana.md
?? outputs_public/execution_reports/revp_integracao_marco_label_free_evidencias_externas_mv1.md
?? outputs_public/execution_reports/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.md
?? outputs_public/execution_reports/revp_navegacao_downloads_evidencias_externas_mv1.md
?? outputs_public/execution_reports/revp_normalizacao_evidencias_externas_publicas_mv1.md
?? outputs_public/execution_reports/revp_normalized_temporal_asset_manifest_mv1.md
?? outputs_public/execution_reports/revp_plano_stage_seletivo_marco_mv1.md
?? outputs_public/execution_reports/revp_protocolo_ground_truth_fail_closed_mv1.md
?? outputs_public/execution_reports/revp_restauracao_manual_v2dz_v2ef.md
?? outputs_public/execution_reports/revp_temporal_acquisition_gap_plan_mv1.md
?? outputs_public/execution_reports/revp_temporal_asset_backfill_queue_mv1.md
?? outputs_public/execution_reports/revp_temporal_asset_readiness_mv1.md
?? outputs_public/execution_reports/revp_temporal_backfill_request_manifest_mv1.md
?? outputs_public/execution_reports/revp_temporal_metadata_repair_candidates_mv1.md
?? outputs_public/execution_reports/revp_temporal_sensor_family_resolution_mv1.md
?? outputs_public/execution_reports/revp_temporal_source_sensor_provenance_mv1.md
?? outputs_public/execution_reports/revp_validacao_label_free_evidencia_estrutural_mv1.md
?? outputs_public/metrics/revp_auditoria_critica_banca_marco_mv1.json
?? outputs_public/metrics/revp_auditoria_prontidao_temporal_assets_mv1.json
?? outputs_public/metrics/revp_auditoria_reprodutibilidade_externa_marco_mv1.json
?? outputs_public/metrics/revp_curadoria_evidencias_externas_mv1.json
?? outputs_public/metrics/revp_fechamento_downloads_evidencias_externas_mv1.json
?? outputs_public/metrics/revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json
?? outputs_public/metrics/revp_hardening_reprodutibilidade_publica_marco_mv1.json
?? outputs_public/metrics/revp_integracao_final_marco_mv1_maturidade_revisao_humana.json
?? outputs_public/metrics/revp_integracao_marco_label_free_evidencias_externas_mv1.json
?? outputs_public/metrics/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.json
?? outputs_public/metrics/revp_navegacao_downloads_evidencias_externas_mv1.json
?? outputs_public/metrics/revp_normalizacao_evidencias_externas_publicas_mv1.json
?? outputs_public/metrics/revp_normalized_temporal_asset_manifest_mv1.json
?? outputs_public/metrics/revp_plano_stage_seletivo_marco_mv1.json
?? outputs_public/metrics/revp_protocolo_ground_truth_fail_closed_mv1.json
?? outputs_public/metrics/revp_temporal_acquisition_gap_plan_mv1.json
?? outputs_public/metrics/revp_temporal_asset_backfill_queue_mv1.json
?? outputs_public/metrics/revp_temporal_asset_readiness_mv1.json
?? outputs_public/metrics/revp_temporal_backfill_request_manifest_mv1.json
?? outputs_public/metrics/revp_temporal_metadata_repair_candidates_mv1.json
?? outputs_public/metrics/revp_temporal_sensor_family_resolution_mv1.json
?? outputs_public/metrics/revp_temporal_source_sensor_provenance_mv1.json
?? outputs_public/metrics/revp_validacao_label_free_evidencia_estrutural_mv1.json
?? outputs_public/mv2_cronograma_engine/
?? outputs_public/mv2_data_api_raster_harness/
?? outputs_public/mv2_data_corpus_metadata_probe/
?? outputs_public/mv2_data_gee_lineage_targets/
?? outputs_public/mv2_data_live_api_canary/
?? outputs_public/mv2_data_readiness/
?? outputs_public/mv2_data_temporal_window_intake/
?? outputs_public/mv2_gee_lineage_recovery/
?? outputs_public/mv2_sentinel_binding/
?? outputs_public/tables/revp_arquivos_cruciais_para_defesa.csv
?? outputs_public/tables/revp_arquivos_historicos_ou_auxiliares.csv
?? outputs_public/tables/revp_auditoria_critica_banca_marco_mv1.csv
?? outputs_public/tables/revp_auditoria_fontes_externas_downloads_mv1.csv
?? outputs_public/tables/revp_auditoria_fontes_externas_mv1.csv
?? outputs_public/tables/revp_auditoria_fontes_externas_navegacao_mv1.csv
?? outputs_public/tables/revp_auditoria_fontes_externas_normalizada_mv1.csv
?? outputs_public/tables/revp_auditoria_prontidao_temporal_assets_mv1.csv
?? outputs_public/tables/revp_bloqueadores_finais_ground_truth_treino_mv1.csv
?? outputs_public/tables/revp_checklist_pre_stage_marco_mv1.csv
?? outputs_public/tables/revp_checklist_reproducao_publica_marco_mv1.csv
?? outputs_public/tables/revp_dashboard_bloqueio_treino_ground_truth_mv1.csv
?? outputs_public/tables/revp_dependencias_local_only_marco_mv1.csv
?? outputs_public/tables/revp_diagnostico_linguagem_publica_ptbr.csv
?? outputs_public/tables/revp_estado_git_branches_curadoria_ptbr.csv
?? outputs_public/tables/revp_evidence_packet_registry_v2ea.csv
?? outputs_public/tables/revp_exclusoes_stage_marco_mv1.csv
?? outputs_public/tables/revp_fila_revisao_humana_candidatos_mv1.csv
?? outputs_public/tables/revp_formal_label_gate_evaluator_v2ee.csv
?? outputs_public/tables/revp_gates_readiness_treino_mv1.csv
?? outputs_public/tables/revp_ground_truth_closure_dashboard_v2ef.csv
?? outputs_public/tables/revp_guardrails_validacao_label_free_mv1.csv
?? outputs_public/tables/revp_human_review_queue_v2ed.csv
?? outputs_public/tables/revp_indice_arquivos_baixados_evidencias_externas_mv1.csv
?? outputs_public/tables/revp_indice_eventos_externos_candidatos_downloads_mv1.csv
?? outputs_public/tables/revp_indice_eventos_externos_candidatos_mv1.csv
?? outputs_public/tables/revp_indice_eventos_externos_candidatos_navegacao_mv1.csv
?? outputs_public/tables/revp_indice_geometrias_externas_candidatas_downloads_mv1.csv
?? outputs_public/tables/revp_indice_geometrias_externas_candidatas_mv1.csv
?? outputs_public/tables/revp_indice_geometrias_externas_candidatas_navegacao_mv1.csv
?? outputs_public/tables/revp_indice_reprodutibilidade_marco_mv1.csv
?? outputs_public/tables/revp_integracao_marco_label_free_evidencias_externas_mv1.csv
?? outputs_public/tables/revp_integracao_marco_label_free_evidencias_externas_navegacao_mv1.csv
?? outputs_public/tables/revp_log_downloads_evidencias_externas_mv1.csv
?? outputs_public/tables/revp_log_navegacao_downloads_evidencias_externas_mv1.csv
?? outputs_public/tables/revp_manifesto_evidencias_externas_downloads_mv1.csv
?? outputs_public/tables/revp_manifesto_evidencias_externas_mv1.csv
?? outputs_public/tables/revp_manifesto_evidencias_externas_navegacao_mv1.csv
?? outputs_public/tables/revp_manifesto_evidencias_externas_normalizado_mv1.csv
?? outputs_public/tables/revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv
?? outputs_public/tables/revp_manifesto_mestre_marco_mv1.csv
?? outputs_public/tables/revp_manifesto_publico_arquivos_externos_url_hash_mv1.csv
?? outputs_public/tables/revp_mapa_pipeline_real_atual.csv
?? outputs_public/tables/revp_matriz_evidencias_externas_gates_mv1.csv
?? outputs_public/tables/revp_matriz_maturidade_metodologica_mv1.csv
?? outputs_public/tables/revp_matriz_topologica_cidades_mv1.csv
?? outputs_public/tables/revp_normalizacao_bloqueadores_evidencias_externas_mv1.csv
?? outputs_public/tables/revp_normalized_temporal_asset_manifest_mv1.csv
?? outputs_public/tables/revp_normalized_temporal_patch_coverage_mv1.csv
?? outputs_public/tables/revp_observed_event_registry_v2dz.csv
?? outputs_public/tables/revp_ontologia_estados_label_mv1.csv
?? outputs_public/tables/revp_patch_event_spatial_binding_v2ec.csv
?? outputs_public/tables/revp_patch_event_temporal_alignment_v2eb.csv
?? outputs_public/tables/revp_plano_pacote_externo_reprodutibilidade_mv1.csv
?? outputs_public/tables/revp_plano_stage_seletivo_marco_mv1.csv
?? outputs_public/tables/revp_proximos_passos_pos_marco_label_free_mv1.csv
?? outputs_public/tables/revp_recomendacoes_pacote_externo_marco_mv1.csv
?? outputs_public/tables/revp_recomendacoes_pacote_externo_normalizada_mv1.csv
?? outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_candidatos.csv
?? outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_manifesto.csv
?? outputs_public/tables/revp_restauracao_manual_v2dz_v2ef_validacao.csv
?? outputs_public/tables/revp_temporal_acquisition_gap_plan_mv1.csv
?? outputs_public/tables/revp_temporal_acquisition_requirements_mv1.csv
?? outputs_public/tables/revp_temporal_asset_backfill_queue_mv1.csv
?? outputs_public/tables/revp_temporal_asset_readiness_mv1.csv
?? outputs_public/tables/revp_temporal_backfill_batch_targets_mv1.csv
?? outputs_public/tables/revp_temporal_backfill_request_manifest_mv1.csv
?? outputs_public/tables/revp_temporal_metadata_repair_candidates_mv1.csv
?? outputs_public/tables/revp_temporal_resolved_acquisition_requirements_mv1.csv
?? outputs_public/tables/revp_temporal_sensor_family_resolution_mv1.csv
?? outputs_public/tables/revp_temporal_source_sensor_blockers_mv1.csv
?? outputs_public/tables/revp_temporal_source_sensor_provenance_mv1.csv
?? outputs_public/tables/revp_validacao_label_free_evidencia_estrutural_mv1.csv
?? outputs_public/tables/revp_verificacao_arquivos_local_only_marco_mv1.csv
?? outputs_public/tables/revp_verificacao_fontes_externas_marco_mv1.csv
?? outputs_public/tables/revp_vizinhos_embeddings_label_free_mv1.csv
?? scripts/curadoria_externa/
?? scripts/ground_truth/revp_auditoria_prontidao_temporal_assets_mv1.py
?? scripts/ground_truth/revp_hardening_reprodutibilidade_publica_marco_mv1.py
?? scripts/ground_truth/revp_integracao_final_marco_mv1_maturidade_revisao_humana.py
?? scripts/ground_truth/revp_normalized_temporal_asset_manifest_mv1.py
?? scripts/ground_truth/revp_protocolo_ground_truth_fail_closed_mv1.py
?? scripts/ground_truth/revp_temporal_acquisition_gap_plan_mv1.py
?? scripts/ground_truth/revp_temporal_asset_backfill_queue_mv1.py
?? scripts/ground_truth/revp_temporal_asset_readiness_audit_mv1.py
?? scripts/ground_truth/revp_temporal_backfill_request_manifest_mv1.py
?? scripts/ground_truth/revp_temporal_metadata_repair_candidates_mv1.py
?? scripts/ground_truth/revp_temporal_sensor_family_resolution_mv1.py
?? scripts/ground_truth/revp_temporal_source_sensor_provenance_audit_mv1.py
?? scripts/ground_truth/revp_v2dz_to_v2ef_common.py
?? scripts/ground_truth/revp_v2dz_to_v2ef_orchestrator.py
?? scripts/ground_truth/revp_validacao_label_free_evidencia_estrutural_mv1.py
?? scripts/mv2_12_build_download_readiness.py
?? scripts/mv2_12_build_missing_data_matrix.py
?? scripts/mv2_12_data_readiness_common.py
?? scripts/mv2_12_event_geometry_backlog.py
?? scripts/mv2_12_scan_local_data_candidates.py
?? scripts/mv2_12_sentinel_native_raster_backlog.py
?? scripts/mv2_12_validate_no_heavy_public_outputs.py
?? scripts/mv2_13_build_binding_risk_matrix.py
?? scripts/mv2_13_build_day10_spectral_gate.py
?? scripts/mv2_13_build_manual_recovery_queue.py
?? scripts/mv2_13_build_stac_gate.py
?? scripts/mv2_13_build_summaries.py
?? scripts/mv2_13_discover_binding_inputs.py
?? scripts/mv2_13_normalize_asset_inventory.py
?? scripts/mv2_13_normalize_patch_inventory.py
?? scripts/mv2_13_normalize_sentinel_anchors.py
?? scripts/mv2_13_reconcile_anchor_asset_patch.py
?? scripts/mv2_13_resolve_geometry_aoi_crs.py
?? scripts/mv2_13_resolve_temporal_scene_lineage.py
?? scripts/mv2_13_run_sentinel_binding.py
?? scripts/mv2_13_sentinel_binding_common.py
?? scripts/mv2_13_validate_sentinel_binding_guardrails.py
?? scripts/mv2_14_build_day10_post_lineage_gate.py
?? scripts/mv2_14_build_gee_manual_recovery_queue.py
?? scripts/mv2_14_build_lineage_risk_matrix.py
?? scripts/mv2_14_build_manual_lineage_template.py
?? scripts/mv2_14_build_medium_binding_index.py
?? scripts/mv2_14_build_post_lineage_stac_gate.py
?? scripts/mv2_14_build_summaries.py
?? scripts/mv2_14_discover_gee_lineage_sources.py
?? scripts/mv2_14_extract_lineage_candidates.py
?? scripts/mv2_14_gee_lineage_common.py
?? scripts/mv2_14_reconcile_lineage_to_bindings.py
?? scripts/mv2_14_review_strong_scene_anchors.py
?? scripts/mv2_14_run_gee_lineage_recovery.py
?? scripts/mv2_14_validate_gee_lineage_guardrails.py
?? scripts/mv2_15_build_allowed_programming_actions.py
?? scripts/mv2_15_build_artifact_registry.py
?? scripts/mv2_15_build_engineering_backlog.py
?? scripts/mv2_15_build_gate_engine.py
?? scripts/mv2_15_build_schedule_state.py
?? scripts/mv2_15_build_scientific_risk_summary.py
?? scripts/mv2_15_build_summaries.py
?? scripts/mv2_15_build_worktree_integration_matrix.py
?? scripts/mv2_15_cronograma_common.py
?? scripts/mv2_15_discover_mv2_outputs.py
?? scripts/mv2_15_run_cronograma_engine.py
?? scripts/mv2_15_validate_cronograma_engine.py
?? scripts/mv2_data_01_build_gee_metadata_query_plan.py
?? scripts/mv2_data_01_build_lineage_targets.py
?? scripts/mv2_data_01_build_manual_recovery_checklist.py
?? scripts/mv2_data_01_build_risk_matrix.py
?? scripts/mv2_data_01_build_stac_odata_metadata_plan.py
?? scripts/mv2_data_01_build_summary.py
?? scripts/mv2_data_01_common.py
?? scripts/mv2_data_01_create_manual_gee_template.py
?? scripts/mv2_data_01_run_gee_lineage_targets.py
?? scripts/mv2_data_01_validate_lineage_targets.py
?? scripts/mv2_data_02_api_config.py
?? scripts/mv2_data_02_build_api_lineage_resolution.py
?? scripts/mv2_data_02_build_api_provider_registry.py
?? scripts/mv2_data_02_build_api_raster_risk_matrix.py
?? scripts/mv2_data_02_build_raster_canary_plan.py
?? scripts/mv2_data_02_build_summary.py
?? scripts/mv2_data_02_cdse_odata_client.py
?? scripts/mv2_data_02_cdse_stac_client.py
?? scripts/mv2_data_02_common.py
?? scripts/mv2_data_02_execute_raster_canary.py
?? scripts/mv2_data_02_gee_metadata_client.py
?? scripts/mv2_data_02_run_api_raster_harness.py
?? scripts/mv2_data_02_validate_api_raster_harness.py
?? scripts/mv2_data_02_validate_private_raster_intake.py
?? scripts/mv2_data_03_build_corpus_impact_matrix.py
?? scripts/mv2_data_03_build_live_api_canary_risk_matrix.py
?? scripts/mv2_data_03_build_summary.py
?? scripts/mv2_data_03_common.py
?? scripts/mv2_data_03_config_env_preflight.py
?? scripts/mv2_data_03_execute_private_raster_canary.py
?? scripts/mv2_data_03_live_cdse_odata_metadata_probe.py
?? scripts/mv2_data_03_live_cdse_stac_metadata_probe.py
?? scripts/mv2_data_03_live_gee_metadata_probe.py
?? scripts/mv2_data_03_resolve_canary_metadata_consensus.py
?? scripts/mv2_data_03_run_live_api_canary.py
?? scripts/mv2_data_03_select_official_canary_candidates.py
?? scripts/mv2_data_03_validate_live_api_canary.py
?? scripts/mv2_data_03_validate_private_raster_canary.py
?? scripts/mv2_data_04_build_risk_matrix.py
?? scripts/mv2_data_04_build_summary.py
?? scripts/mv2_data_04_build_temporal_window_gap_matrix.py
?? scripts/mv2_data_04_cdse_odata_corpus_metadata_probe.py
?? scripts/mv2_data_04_cdse_stac_corpus_metadata_probe.py
?? scripts/mv2_data_04_common.py
?? scripts/mv2_data_04_config_env_preflight.py
?? scripts/mv2_data_04_create_temporal_window_template.py
?? scripts/mv2_data_04_gee_corpus_metadata_probe.py
?? scripts/mv2_data_04_resolve_corpus_api_lineage.py
?? scripts/mv2_data_04_run_corpus_metadata_probe.py
?? scripts/mv2_data_04_select_corpus_metadata_batch.py
?? scripts/mv2_data_04_validate_corpus_metadata_probe.py
?? scripts/mv2_data_05_build_probe_ready_batch.py
?? scripts/mv2_data_05_build_risk_matrix.py
?? scripts/mv2_data_05_build_summary.py
?? scripts/mv2_data_05_build_temporal_promotion_gate.py
?? scripts/mv2_data_05_common.py
?? scripts/mv2_data_05_create_temporal_window_correction_template.py
?? scripts/mv2_data_05_discover_temporal_inputs.py
?? scripts/mv2_data_05_normalize_temporal_windows.py
?? scripts/mv2_data_05_run_temporal_window_intake.py
?? scripts/mv2_data_05_validate_temporal_evidence.py
?? scripts/mv2_data_05_validate_temporal_window_intake.py
?? scripts/mv2_pre_unification_run.py
?? tests/test_mv2_12_data_readiness.py
?? tests/test_mv2_13_sentinel_binding.py
?? tests/test_mv2_14_gee_lineage_recovery.py
?? tests/test_mv2_15_cronograma_engine.py
?? tests/test_mv2_data_01_gee_lineage_targets.py
?? tests/test_mv2_data_02_api_raster_harness.py
?? tests/test_mv2_data_03_live_api_canary.py
?? tests/test_mv2_data_04_corpus_metadata_probe.py
?? tests/test_mv2_data_05_temporal_window_intake.py
?? tests/test_mv2_pre_unification_contracts.py
?? tests/test_revp_auditoria_prontidao_temporal_assets_mv1.py
?? tests/test_revp_curadoria_evidencias_externas_mv1.py
?? tests/test_revp_fechamento_downloads_e_integracao_evidencias_externas_mv1.py
?? tests/test_revp_hardening_reprodutibilidade_publica_marco_mv1.py
?? tests/test_revp_integracao_final_marco_mv1_maturidade_revisao_humana.py
?? tests/test_revp_navegacao_downloads_evidencias_externas_mv1.py
?? tests/test_revp_normalizacao_evidencias_externas_publicas_mv1.py
?? tests/test_revp_normalized_temporal_asset_manifest_mv1.py
?? tests/test_revp_protocolo_ground_truth_fail_closed_mv1.py
?? tests/test_revp_temporal_acquisition_gap_plan_mv1.py
?? tests/test_revp_temporal_asset_backfill_queue_mv1.py
?? tests/test_revp_temporal_asset_readiness_audit_mv1.py
?? tests/test_revp_temporal_backfill_request_manifest_mv1.py
?? tests/test_revp_temporal_metadata_repair_candidates_mv1.py
?? tests/test_revp_temporal_sensor_family_resolution_mv1.py
?? tests/test_revp_temporal_source_sensor_provenance_audit_mv1.py
?? tests/test_revp_v2dz_to_v2ef_orchestrator.py
?? tests/test_revp_validacao_label_free_evidencia_estrutural_mv1.py
```

## Ultimos commits
```text
67d8cfd docs: consolida curadoria publica e linha metodologica PT-BR
afff542 analise: audita recuperabilidade da base original v2dz-v2ef
58ed64a curadoria: organiza repositorio publico em portugues
af729bf curadoria: organiza repositorio publico em portugues
faa9156 chore: curadoria da camada publica do repositorio
```
