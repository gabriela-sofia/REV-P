# v1qa — Expanded Queue Import Bridge

## Objetivo

Converter fila visual expandida v1pw para formato compatível com executor v1pq. Preserva priority/reason/linkage_confidence. Tudo review-only.

## Guardrails

can_create_label, can_train_model e target_created sempre false. ready_for_real_execution=false enquanto modelo indisponível.

## Resultado

Itens importados: 100. Prontos dry-run: 100. Prontos execução real: 0.
