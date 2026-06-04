# Protocolo C v1uk Acceptance Audit

- v1uk_exists: true
- expected_outputs_present: 10/10
- tables_audited: 59
- total_rows: 2744469
- rows_in_REC_2022_05_24_30_window: 11163
- rows_with_hazard_terms: 45145
- rows_with_bairro_locality_address: 4202901
- rows_with_coordinates: 436824
- candidates_for_review: 6672
- absolute_path_public_csv: false
- sensitive_literal_public_csv: false
- guardrail_violation: false
- acceptance_status: V1UK_COMPLETE

Missing outputs:
- none

Guardrails:
- ground_truth_operational=false
- can_create_ground_reference=false
- can_create_training_label=false
- can_reopen_protocol_b=false
- dino_usage=SUPPORT_ONLY
- no_overlay_executed=true
- no_coordinates_invented=true
- supervisor_review_completed=false
