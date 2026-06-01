# v1rn — Protocol C State Machine Registry

## Objetivo

Registrar de forma auditavel os estados do Protocolo C (C1, C2, C3-candidate, C4-blocked, blocked), transicoes permitidas/proibidas, gates necessarios e os outputs que comprovam cada gate.

## Invariantes

Nenhum estado e label operacional. C3-candidate exige supervisor e permanece review-only. C4 nunca abre automaticamente nem por ausencia de evidencia. Promocao automatica para label e proibida.

## Resultado

Estados: 5. Estados abertos (operacionais): 0. Estados de label operacional: 0.

## Guardrails

review_only=true. can_create_operational_label=false. ground_truth_operational=false.
