# v2an - orchestrator run manifest

Etapas executadas: 15. Nenhuma operacao git foi executada.

| ordem | etapa | status | outputs |
| --- | --- | --- | --- |
| 1 | candidate_inventory_normalizer | OK | datasets/protocolo_c/v2an_observed_candidate_inventory_normalized.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_observed_candidate_inventory_normalized.md |
| 2 | source_access_probe | OK | datasets/protocolo_c/v2an_source_access_probe.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_source_access_probe.md |
| 3 | document_metadata_extractor | OK | datasets/protocolo_c/v2an_document_metadata_registry.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_document_metadata_registry.md |
| 4 | spatial_anchor_extractor | OK | datasets/protocolo_c/v2an_spatial_anchor_registry.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_spatial_anchor_registry.md |
| 5 | temporal_sentinel_crosswalk_audit | OK | datasets/protocolo_c/v2an_temporal_sentinel_crosswalk_audit.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_temporal_sentinel_crosswalk_audit.md |
| 6 | patch_link_readiness_audit | OK | datasets/protocolo_c/v2an_patch_link_readiness_audit.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_patch_link_readiness_audit.md |
| 7 | gate_closure_matrix | OK | datasets/protocolo_c/v2an_gate_closure_matrix.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_gate_closure_matrix.md |
| 8 | readiness_scorer | OK | datasets/protocolo_c/v2an_ground_reference_readiness_scores.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_ground_reference_readiness_scores.md |
| 9 | candidate_dossier_builder | OK | datasets/protocolo_c/v2an_candidate_dossier_index.csv |
| 10 | human_review_package | OK | datasets/protocolo_c/v2an_human_ground_reference_review_package.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_human_ground_reference_review_package.md |
| 11 | validation_decision_registry | OK | datasets/protocolo_c/v2an_validation_decision_registry.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_validation_decision_registry.md |
| 12 | ground_truth_blocker_audit | OK | datasets/protocolo_c/v2an_ground_truth_blocker_audit.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_ground_truth_blocker_audit.md |
| 13 | guardrail_regression | OK | datasets/protocolo_c/v2an_guardrail_regression.csv |
| 14 | next_action_ranker | OK | datasets/protocolo_c/v2an_next_actions_registry.csv |
| 15 | completion_report | OK | datasets/protocolo_c/v2an_completion_report.csv\|docs/protocolo_c/v2an_ground_reference_validation_sprint/v2an_completion_report.md |
