# v1rj — Supervisor Review Packet Generator

## Objetivo

Gerar pacotes para o supervisor quando uma revisao completa alcanca estado C3-candidate ou desacordo e ainda precisa decisao humana final. Sem revisoes completas, SUPERVISOR_PACKETS_WAITING_COMPLETED_REVIEWS.

## Acoes do supervisor

APPROVE_C3_CANDIDATE_REVIEW_ONLY, KEEP_C2_REVIEW_ONLY, BLOCK_C3_NEEDS_MORE_SOURCE, BLOCK_C3_NEEDS_BETTER_TEMPORAL_PRECISION, BLOCK_C3_NEEDS_BETTER_SPATIAL_PRECISION, REQUEST_ADDITIONAL_REVIEW.

## Resultado

Status: SUPERVISOR_PACKETS_WAITING_COMPLETED_REVIEWS. Pacotes: 0. Promoviveis a C3 candidate: 0.

## Guardrails

can_promote_to_c3_candidate=true significa apenas elegibilidade review-only. can_create_operational_label=false sempre. Desacordo exige revisao adicional.
