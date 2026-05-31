# v1oy — Ground Truth Candidate Decision Audit

## Objetivo

Consolidar decisões C1/C2/C3/C4 com base em v1ov-v1ox e v1og-v1ot. Gerar decisão auditável para cada evento/patch/linkage.

## Resultado

C1 (contextual): 2. C2 (review-only candidate): 2. C3+ não alcançado. C4 fechado.

## Por que C3+ não é alcançado

C3+ requer scene_date Sentinel confirmada. v1og-v1ot: product_dates_confirmed_real=0 em 2.654 patches. Sem cadeia temporal confirmada, C3+ é metodologicamente impossível.

## Por que C4 permanece fechado

C4 requer negativo formal explícito (formal_negative_count=0). Nenhuma declaração oficial de não-ocorrência confirmada para Recife.

## Guardrails

can_be_used_for_training=false, can_create_operational_label=false em todos os registros.
