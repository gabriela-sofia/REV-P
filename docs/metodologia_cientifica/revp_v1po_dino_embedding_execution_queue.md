# v1po — DINO Embedding Execution Queue

## Objetivo

Construir fila priorizada de assets visuais para geração de embedding. Não executa embedding. Tudo review-only.

## Prioridade

1 = patch em fila DINO review v1oz. 2 = asset elegível restante.

## Guardrails

`can_create_label`, `can_train_model` e `target_created` sempre false.

## Resultado

Itens na fila: 0. Prioridade 1 (Protocol C): 0.
