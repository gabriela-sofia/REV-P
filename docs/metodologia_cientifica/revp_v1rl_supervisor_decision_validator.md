# v1rl — Supervisor Decision Validator

## Objetivo

Validar decisoes do supervisor preenchidas manualmente (REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH). Sem o arquivo, SUPERVISOR_DECISIONS_WAITING_MANUAL_INPUT.

## Garantias

Aprovar C3 candidate permanece review-only (can_create_operational_label=false, ground_truth_operational=false). C4 nunca aberto sem fonte formal negativa.

## Resultado

Status: SUPERVISOR_DECISIONS_WAITING_MANUAL_INPUT. Decisoes: 0. Checagens: 0 (passou 0, falhou 0).

## Guardrails

review_only=true. formal_negative=false esperado. Nenhum label/target/ground truth.
