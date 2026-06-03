# v1ug Completion Report
Generated: 2026-06-03 17:30:15
Protocol version: v1ug

## Guardrails
  ground_truth_operational = False [PASS]
  can_create_ground_reference = False [PASS]
  can_create_training_label = False [PASS]
  can_reopen_protocol_b = False [PASS]
  no_overlay_executed = True [PASS]
  no_coordinates_invented = True [PASS]
  review_package_only = True [PASS]
  formal_request_only = True [PASS]

## Event Summary
  PET_2022_02_15:
    review_package_status: BLOCKED_PHENOMENON_SEPARATION_REQUIRED
    overall_readiness: WAITING_PHENOMENON_SEPARATION
    blocking_dimensions: 4
    priority_rank: #2
    can_create_ground_reference: false
  PET_2024_03_21_28:
    review_package_status: BLOCKED_PHENOMENON_SEPARATION_REQUIRED
    overall_readiness: WAITING_PHENOMENON_SEPARATION
    blocking_dimensions: 4
    priority_rank: #3
    can_create_ground_reference: false
  REC_2022_05_24_30:
    review_package_status: BLOCKED_INSUFFICIENT_COVERAGE
    overall_readiness: WAITING_OBSERVED_GEOMETRY
    blocking_dimensions: 3
    priority_rank: #1
    can_create_ground_reference: false

## Gap Matrix Summary
  Total gaps: 36
  FAIL: 16 | PASS: 16 | REVIEW/NA: 4
  training_label_allowed: FAIL (all events)
  observed_geometry_available: FAIL (all events)

## Formal Request Queue
  Total requests: 11
  REQ_v1ug_0000: PET_2022_02_15 -> SGB_CPRM [PENDING_HUMAN_ACTION]
  REQ_v1ug_0001: PET_2022_02_15 -> DRM_RJ_NADE [PENDING_HUMAN_ACTION]
  REQ_v1ug_0002: PET_2022_02_15 -> DEFESA_CIVIL_PETROPOLIS [PENDING_HUMAN_ACTION]
  REQ_v1ug_0003: PET_2024_03_21_28 -> DEFESA_CIVIL_PETROPOLIS [PENDING_HUMAN_ACTION]
  REQ_v1ug_0004: REC_2022_05_24_30 -> COMPDEC_RECIFE_PE [PENDING_HUMAN_ACTION]
  REQ_v1ug_0005: PET_2022_02_15 -> CEMADEN [PENDING_HUMAN_ACTION]
  REQ_v1ug_0006: PET_2024_03_21_28 -> CEMADEN [PENDING_HUMAN_ACTION]
  REQ_v1ug_0007: REC_2022_05_24_30 -> CEMADEN [PENDING_HUMAN_ACTION]
  REQ_v1ug_0008: PET_2022_02_15 -> ANA_HIDROWEB [PENDING_HUMAN_ACTION]
  REQ_v1ug_0009: PET_2024_03_21_28 -> ANA_HIDROWEB [PENDING_HUMAN_ACTION]
  REQ_v1ug_0010: REC_2022_05_24_30 -> ANA_HIDROWEB [PENDING_HUMAN_ACTION]

## Supervisor Review Checklist
  Total entries: 45
  NOT_EVALUATED: 42
  supervisor_review_completed: false (all)

## Artifact Manifest
  [EXISTS] datasets/protocolo_c/v1ug_event_gap_matrix.csv (sha256: bfd079da9805146b)
  [EXISTS] datasets/protocolo_c/v1ug_event_review_package_registry.csv (sha256: 0698113cbbbffbbc)
  [EXISTS] datasets/protocolo_c/v1ug_formal_request_queue.csv (sha256: f03dde2746bd8454)
  [EXISTS] datasets/protocolo_c/v1ug_supervisor_review_checklist.csv (sha256: 32f59243c6ecff06)
  [EXISTS] datasets/protocolo_c/v1ug_ground_reference_readiness_matrix.csv (sha256: a1d60f58cc4b90bd)
  [EXISTS] datasets/protocolo_c/v1ug_event_priority_queue.csv (sha256: eb939e3f400c2d83)

## Scripts
  [EXISTS] scripts/protocolo_c/revp_v1ug_event_gap_matrix_builder.py
  [EXISTS] scripts/protocolo_c/revp_v1ug_human_review_package_builder.py
  [EXISTS] scripts/protocolo_c/revp_v1ug_formal_request_finalizer.py
  [EXISTS] scripts/protocolo_c/revp_v1ug_supervisor_review_checklist.py
  [EXISTS] scripts/protocolo_c/revp_v1ug_ground_reference_readiness_matrix.py
  [EXISTS] scripts/protocolo_c/revp_v1ug_event_priority_queue.py
  [EXISTS] scripts/protocolo_c/revp_v1ug_completion_report.py

## Configs
  [EXISTS] configs/protocolo_c/v1ug_formal_request_targets.yaml
  [EXISTS] configs/protocolo_c/v1ug_ground_reference_readiness_policy.yaml
  [EXISTS] configs/protocolo_c/v1ug_review_package_policy.yaml
  [EXISTS] configs/protocolo_c/v1ug_supervisor_review_policy.yaml

## Next Steps (Human Action Required)
  1. Enviar pedidos formais às instituições listadas na fila de requisições
  2. Registrar respostas recebidas em datasets/protocolo_c/
  3. Completar checklist de revisão supervisora quando evidência suficiente
  4. NÃO criar ground reference, labels ou overlays nesta etapa

## Invariants
  - Nenhum evento atingiu READY_FOR_GROUND_REFERENCE
  - Nenhuma geometria observada adquirida
  - Nenhum overlay executado
  - Nenhuma coordenada inventada
  - Nenhum label de treinamento criado