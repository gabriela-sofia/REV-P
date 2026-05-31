# v1ow — Evidence Strength and Precision Scoring

## Objetivo

Pontuar cada evento/evidência do v1ov com critérios auditáveis de precisão temporal, precisão espacial, confiabilidade da fonte, especificidade do evento e independência. Não transforma score em label.

## Critérios de pontuação

- Temporal: HIGH=3, MODERATE=2, LOW=1, NONE=0
- Espacial: POINT_EXPLICIT=3, ADDRESS=2, ADMINISTRATIVE=1, NONE=0
- Confiabilidade: OFFICIAL_HIGH=3, NEWS/CONTEXTUAL=1, UNKNOWN=0
- Especificidade: CONFIRMED=3, PROBABLE=2, CONTEXTUAL=1, BLOCKED=0
- Independência: 0 (fonte única disponível)

## Tiers

STRONG_REVIEW_ONLY (≥10), MODERATE_REVIEW_ONLY (≥6), LIMITED_CONTEXTUAL (≥3), CONTEXTUAL_GAP (≥1), BLOCKED (0).

## Guardrails

can_promote_to_label=false e can_train_model=false em todos os registros. Score alto não autoriza label ou treino.
