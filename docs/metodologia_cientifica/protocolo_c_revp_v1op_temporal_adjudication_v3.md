# Protocolo C v1op — Temporal Adjudication v3

## Objetivo

v1op recalcula adjudicação temporal usando os dados do resolver v3 (v1oo). Fail-closed.

## Regras

- can_support_c3_plus=true SOMENTE se scene_date_status==PRODUCT_DATE_CONFIRMED e categoria temporal for strong/moderate/contextual e houver formal negative.
- C4 fechado se formal_negative_count==0.
- can_create_operational_label e can_train_model sempre false.
- Probabilistic (PROBABLE) downgradeado para review_only_probable.

## Categorias temporais

- strong
- moderate
- contextual
- review_only_probable
- weak
- unknown_blocked

## Resultado

Total: 0. can_support_c3_plus=true: 0. unknown_blocked: 0.
