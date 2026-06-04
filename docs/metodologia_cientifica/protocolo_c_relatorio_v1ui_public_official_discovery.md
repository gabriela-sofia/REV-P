# v1ui Completion Report — Public Official Discovery
Generated: 2026-06-03 19:42:21
Protocol: v1ui

## Guardrails
  ground_truth_operational = False [ENFORCED]
  can_create_ground_reference = False [ENFORCED]
  can_create_training_label = False [ENFORCED]
  can_reopen_protocol_b = False [ENFORCED]
  no_overlay_executed = True [ENFORCED]
  no_coordinates_invented = True [ENFORCED]
  public_artifact_discovery = True [ENFORCED]
  formal_request_path = LEGACY_SECONDARY_ONLY [ENFORCED]

## Discovery Summary
  Public sources registered: 23
  Artifacts inventoried: 76
  ArcGIS/GeoServer layers: 3
  Geometry candidates: 0
  Gate deltas (gains): 4
  Ready for supervisor review: 0

## Why No Ground Truth Yet
  - G12 supervisor_review_pending: always FAIL
  - G13 patch_overlay_not_executed: always FAIL
  - G14 label_forbidden: always FAIL
  - can_create_ground_reference=false at all stages

## Invariants
  - Nenhum ground reference criado
  - Nenhum label de treinamento criado
  - Nenhum overlay executado
  - Nenhuma coordenada inventada
  - Nenhum dado bruto versionado
  - formal_request_path=LEGACY_SECONDARY_ONLY