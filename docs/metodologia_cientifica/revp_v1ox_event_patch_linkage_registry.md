# v1ox — Event-Patch Linkage Registry

## Objetivo

Vincular eventos observados (v1ov) a patches por região, alias e localização aproximada. Usa fail-closed: sem scene_date Sentinel confirmada (v1og-v1ot), temporal linkage fica BLOCKED_SENTINEL_SCENE_DATE_MISSING.

## Resultado

Status de recuperação temporal (v1ot): TEMPORAL_RECOVERY_FAIL_CLOSED. Linkages totais: 12. Temporal bloqueado: 12. Temporal confirmado: 0.

## Por que temporal permanece bloqueado

v1og-v1ot confirmou TEMPORAL_RECOVERY_FAIL_CLOSED: 0 product_dates confirmadas em 2.654 patches avaliados. Sem cadeia patch→asset→produto Sentinel confirmada, qualquer temporal linkage seria baseado em data inferida — proibido neste protocolo.

## Guardrails

can_create_label=false, can_train_model=false. allowed_use máximo: REVIEW_ONLY ou CONTEXTUAL_ONLY.
