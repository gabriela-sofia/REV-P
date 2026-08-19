# Embeddings DINOv2 — análise estrutural auxiliar (datasets/)
DINOv2 é um encoder visual pré-treinado e congelado, usado só para análise estrutural exploratória (similaridade, k-NN, PCA, clusters, outliers) — nunca como variável causal do modelo de suscetibilidade. Esta pasta reúne as saídas de ~40 micro-etapas de execução desse pipeline auxiliar (extração de embedding, checks de ambiente/execução, e as análises em si).

## O resultado que importa

O teste que decide se DINOv2 entra no modelo causal está em [`dino_v12_ab_comparison_summary_v1r5.csv`](dino_v12_ab_comparison_summary_v1r5.csv): comparando o modelo físico (`rain_decay_index_api_chirps, twi_dinf, slope_deg`) com e sem duas componentes PCA do DINOv2, o teste de razão de verossimilhança (LRT) deu significativo (p=0,0048) — **mas** 109 pontos mapeiam para só 23 vetores DINO únicos (até 10 pontos compartilhando o mesmo vetor), então essa significância é efeito de pseudorreplicação em nível de patch, não sinal real. Decisão registrada: `DINO_LRT_SIGNIFICANT_BUT_CONFOUNDED_BY_PATCH_LEVEL_PSEUDOREPLICATION` — DINO não virou feature de treino (`used_as_training_feature=false`, `labels_created=0`). É esse resultado que justifica DINO ficar de fora do modelo causal, com o motivo estatístico exato, não só "não ajudou".

## Limpeza aplicada (auditoria de 2026-08-19)

21 arquivos de rodadas de teste ("smoke test") que processaram 0 itens (`_results`, `_failures` e `_summary` todos vazios ou "0 items processed") foram removidos — nenhuma informação real existia neles. Restam 115 arquivos, ~2 MB no total (o volume aqui nunca foi um problema de espaço, é puramente de navegação).

## Índice por etapa

| Etapa | Arquivos |
|---|---|
| `(sem código)` | dino_c3_anchor_review_triage_registry.csv, dino_control_candidate_review_queue.csv, dino_embedding_training_boundary_matrix.csv |
| `v1pg` | dino_artifact_discovery_summary_v1pg.csv, dino_artifact_discovery_v1pg.csv |
| `v1ph` | dino_embedding_feature_store_registry_v1ph.csv, dino_embedding_feature_store_summary_v1ph.csv |
| `v1pi` | dino_embedding_quality_summary_v1pi.csv |
| `v1pj` | dino_similarity_matrix_long_v1pj.csv, dino_similarity_neighbors_v1pj.csv, dino_similarity_summary_v1pj.csv |
| `v1pk` | dino_cluster_exploratory_v1pk.csv, dino_pca_cluster_summary_v1pk.csv, dino_pca_projection_v1pk.csv |
| `v1pl` | dino_protocol_c_crosswalk_summary_v1pl.csv |
| `v1pm` | dino_tcc_results_manifest_v1pm.csv, dino_tcc_results_scientific_summary_v1pm.csv |
| `v1pn` | dino_patch_visual_asset_inventory_summary_v1pn.csv, dino_patch_visual_asset_inventory_v1pn.csv |
| `v1po` | dino_embedding_execution_queue_summary_v1po.csv |
| `v1pp` | dino_backend_model_probe_summary_v1pp.csv, dino_backend_model_probe_v1pp.csv |
| `v1pq` | dino_controlled_smoke_embedding_summary_v1pq.csv |
| `v1pr` | dino_smoke_embedding_feature_store_summary_v1pr.csv |
| `v1ps` | dino_smoke_review_products_summary_v1ps.csv |
| `v1pt` | dino_execution_manifest_v1pt.csv, dino_execution_quality_checks_v1pt.csv, dino_execution_scientific_summary_v1pt.csv |
| `v1pu` | dino_visual_asset_eligibility_audit_v1pu.csv, dino_visual_asset_eligibility_summary_v1pu.csv |
| `v1pv` | dino_patch_visual_linkage_registry_v1pv.csv, dino_patch_visual_linkage_summary_v1pv.csv |
| `v1pw` | dino_review_only_execution_queue_expanded_summary_v1pw.csv, dino_review_only_execution_queue_expanded_v1pw.csv |
| `v1px` | dino_queue_leakage_audit_v1px.csv, dino_queue_leakage_summary_v1px.csv |
| `v1py` | dino_tcc_table_review_queue_v1py.csv, dino_tcc_table_visual_asset_eligibility_v1py.csv |
| `v1pz` | dino_visual_eligibility_bundle_manifest_v1pz.csv, dino_visual_eligibility_scientific_summary_v1pz.csv |
| `v1qa` | dino_execution_queue_from_visual_expansion_summary_v1qa.csv, dino_execution_queue_from_visual_expansion_v1qa.csv |
| `v1qb` | dino_execution_readiness_audit_v1qb.csv, dino_execution_readiness_summary_v1qb.csv |
| `v1qc` | dino_dry_run_execution_commands_v1qc.csv, dino_dry_run_execution_plan_v1qc.csv |
| `v1qd` | dino_executor_compatibility_report_v1qd.csv, dino_executor_compatibility_summary_v1qd.csv |
| `v1qe` | dino_tcc_table_execution_readiness_v1qe.csv, dino_tcc_table_execution_safety_v1qe.csv |
| `v1qf` | dino_execution_bridge_manifest_v1qf.csv, dino_execution_bridge_scientific_summary_v1qf.csv |
| `v1qg` | dino_local_model_offline_audit_v1qg.csv, dino_local_model_offline_summary_v1qg.csv |
| `v1qh` | dino_smoke_sample_selection_v1qh.csv, dino_smoke_sample_summary_v1qh.csv |
| `v1qi` | dino_local_asset_preprocessing_audit_v1qi.csv, dino_local_asset_preprocessing_summary_v1qi.csv |
| `v1qj` | dino_smoke_embedding_execution_manifest_v1qj.csv, dino_smoke_embedding_summary_v1qj.csv, dino_smoke_embeddings_feature_store_v1qj.csv |
| `v1qk` | dino_representation_feature_store_with_smoke_summary_v1qk.csv |
| `v1ql` | dino_smoke_review_products_summary_v1ql.csv |
| `v1qm` | dino_smoke_embedding_bundle_manifest_v1qm.csv, dino_smoke_embedding_quality_checks_v1qm.csv, dino_smoke_embedding_scientific_summary_v1qm.csv, dino_tcc_table_smoke_embedding_results_v1qm.csv |
| `v1qn` | dino_local_root_environment_audit_v1qn.csv, dino_local_root_environment_summary_v1qn.csv |
| `v1qo` | dino_smoke_asset_local_reconciliation_summary_v1qo.csv, dino_smoke_asset_local_reconciliation_v1qo.csv |
| `v1qp` | dino_manifest_crosswalk_repair_suggestions_v1qp.csv, dino_manifest_crosswalk_repair_summary_v1qp.csv |
| `v1qr` | dino_local_smoke_run_readiness_gate_v1qr.csv, dino_local_smoke_run_readiness_summary_v1qr.csv |
| `v1qs` | dino_tcc_table_local_blockers_v1qs.csv, dino_tcc_table_local_readiness_v1qs.csv |
| `v1qt` | dino_local_readiness_manifest_v1qt.csv, dino_local_readiness_quality_checks_v1qt.csv, dino_local_readiness_scientific_summary_v1qt.csv |
| `v1qu` | dino_smoke_relative_path_linker_summary_v1qu.csv, dino_smoke_sample_selection_linked_v1qu.csv |
| `v1qv` | dino_attention_rollout_manifest_v1qv.csv, dino_attention_rollout_summary_v1qv.csv |
| `v1qw` | dino_petropolis_terrain_overlap_embeddings_v1qw.csv, dino_petropolis_terrain_overlap_linked_v1qw.csv |
| `v1qx` | dino_physical_correlation_pairwise_v1qx.csv, dino_physical_correlation_pca_v1qx.csv, dino_physical_correlation_summary_v1qx.csv |
| `v1qy` | dino_recife_sedec_matched_embeddings_v1qy.csv, dino_recife_sedec_matched_linked_v1qy.csv |
| `v1qz` | dino_sedec_recife_join_summary_v1qz.csv, dino_sedec_recife_join_v1qz.csv |
| `v1r0` | dino_recife_new_official_patches_embeddings_v1r0.csv, dino_recife_new_official_patches_linked_v1r0.csv |
| `v1r1` | dino_recife_sedec_full_embeddings_v1r1.csv |
| `v1r2` | dino_recife_neg_evidence_embeddings_v1r2.csv, dino_recife_neg_evidence_patches_linked_v1r2.csv |
| `v1r3` | dino_recife_sedec_all_embeddings_v1r3.csv |
| `v1r4` | dino_sedec_extended_firth_dino_only_v1r4.csv, dino_sedec_extended_summary_v1r4.csv, dino_sedec_extended_univariate_v1r4.csv |
| `v1r5` | dino_v12_ab_comparison_summary_v1r5.csv, dino_v12_ab_firth_model_coefs_v1r5.csv |
| `v1r6` | dino_v12_cluster_robust_sensitivity_v1r6.csv |
| `v1r7` | dino_evidence_refinement_boundary_matrix_v1r7.csv, dino_recife_evidence_refinement_qa_v1r7.csv, dino_recife_evidence_refinement_review_queue_v1r7.csv, dino_recife_evidence_refinement_scores_v1r7.csv |
| `v1r8` | dino_evidence_refinement_boundary_matrix_v1r8.csv, dino_recife_coverage_expansion_audit_v1r8.csv, dino_recife_coverage_expansion_embeddings_v1r8.csv, dino_recife_expanded_refinement_qa_v1r8.csv, dino_recife_expanded_refinement_review_queue_v1r8.csv, dino_recife_expanded_refinement_scores_v1r8.csv |

## Como navegar

O prefixo `v1p*`/`v1q*`/`v1r*` indica a etapa que gerou o arquivo, em ordem cronológica aproximada (`v1pg` mais antigo → `v1r8` mais recente). Pares `*_summary` costumam ser o resumo agregado do arquivo detalhado de mesmo prefixo. Os arquivos `dino_similarity_*`, `dino_pca_*`, `dino_cluster_*` e `dino_v12_ab_*` são os de conteúdo analítico direto; o restante é auditoria/preparação de execução do pipeline de embeddings.
