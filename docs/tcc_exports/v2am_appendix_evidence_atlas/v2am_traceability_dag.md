# Protocolo C v2am - DAG de rastreabilidade

Grafo de rastreabilidade entre etapas e artefatos. Cada aresta preserva
guardrails e nao cria promocao operacional.

## Nodes
| node_id | stage | label | artifact_path | claim_safety_status |
| --- | --- | --- | --- | --- |
| N_v2ah_stop_gate | v2ah | v2ah stop gate | datasets/protocolo_c/v2ah_ground_truth_search_stop_gate.csv | review_only_no_promotion |
| N_v2ah_review_queue | v2ah | v2ah review queue | datasets/protocolo_c/v2ah_candidate_reference_review_queue.csv | review_only_no_promotion |
| N_v2ai_assignments | v2ai | v2ai assignments | datasets/protocolo_c/v2ai_review_assignment_registry.csv | review_only_no_promotion |
| N_v2ai_adjudication | v2ai | v2ai adjudication queue | datasets/protocolo_c/v2ai_adjudication_queue.csv | review_only_no_promotion |
| N_v2aj_claims | v2aj | v2aj claims matrix | datasets/protocolo_c/v2aj_tcc_protocol_c_claims_matrix.csv | review_only_no_promotion |
| N_v2aj_summary | v2aj | v2aj evidence summary | datasets/protocolo_c/v2aj_tcc_evidence_summary_table.csv | review_only_no_promotion |
| N_v2ak_drafts | v2ak | v2ak drafts | docs/tcc_exports/protocolo_c_v2ak_metodologia_draft.md | review_only_no_promotion |
| N_v2al_bundles | v2al | v2al bundles | docs/tcc_exports/v2al_manuscript_integration/v2al_metodologia_section_candidate.md | review_only_no_promotion |
| N_v2am_atlas | v2am | v2am atlas | docs/tcc_exports/v2am_appendix_evidence_atlas/v2am_protocol_c_evidence_atlas.md | review_only_no_promotion |

## Edges
| edge_id | source | target | relationship | guardrail_preserved | promotion_created |
| --- | --- | --- | --- | --- | --- |
| EDG_v2am_000 | N_v2ah_stop_gate | N_v2ah_review_queue | stop_gate_bounds_queue | true | false |
| EDG_v2am_001 | N_v2ah_review_queue | N_v2ai_assignments | v2ah_to_v2ai | true | false |
| EDG_v2am_002 | N_v2ai_assignments | N_v2ai_adjudication | assignments_feed_adjudication | true | false |
| EDG_v2am_003 | N_v2ai_adjudication | N_v2aj_claims | v2ai_to_v2aj | true | false |
| EDG_v2am_004 | N_v2aj_claims | N_v2aj_summary | claims_inform_summary | true | false |
| EDG_v2am_005 | N_v2aj_summary | N_v2ak_drafts | v2aj_to_v2ak | true | false |
| EDG_v2am_006 | N_v2ak_drafts | N_v2al_bundles | v2ak_to_v2al | true | false |
| EDG_v2am_007 | N_v2al_bundles | N_v2am_atlas | v2al_to_v2am | true | false |
