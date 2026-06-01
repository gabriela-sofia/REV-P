# v1qy — Ground Reference Adjudication Decision Registry

## Objetivo

Consolidar scores observacionais em decisoes adjudicaveis sem criar label. Todo candidato C3 exige supervisor_review_required=true. C4 nunca e aberto sem fonte formal negativa explicita (esperado: formal_negative=false).

## Decisoes

KEEP_C1_CONTEXTUAL, KEEP_C2_REVIEW_ONLY, PROMOTE_TO_C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR, BLOCK_C3_INSUFFICIENT_TEMPORAL_PRECISION, BLOCK_C3_INSUFFICIENT_SPATIAL_PRECISION, BLOCK_C3_SOURCE_WEAK, BLOCK_C4_NO_FORMAL_NEGATIVE_SOURCE.

## Resultado

Adjudicados: 0. C3 needing supervisor: 0. Blocked C3: 0. C4 formal negatives: 0.

## Guardrails

can_create_operational_label=false em todas as linhas. C4 fechado sem fonte formal. Nenhum target, ground truth ou label criado.
