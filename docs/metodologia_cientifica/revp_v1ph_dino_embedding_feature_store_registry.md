# v1ph — DINO Embedding Feature Store Registry

## Objetivo

Registrar vetores de embedding REAIS descobertos em v1pg. Nunca inventa vetor: se nenhum embedding real for parseado, o registry é gravado vazio com header (fail-closed).

## Regras de validação

Dimensão 768 → `VALID_REVIEW_ONLY`. Dimensão diferente → `BLOCKED_INVALID_DIMENSION`. NaN/inf/zero → bloqueado. Duplicata → flag de revisão, nunca label.

## Guardrails DINO

`dino_can_create_label`, `dino_can_train_model` e `dino_target_field_created` são sempre false. Embeddings são representação visual auto-supervisionada, não rótulo.

## Resultado

Embeddings parseados: 0. Válidos 768D: 0.
