# v1qd — Executor Compatibility Patch Report

## Objetivo

Verificar que v1pq aceita fila v1qa via REVP_V1PQ_QUEUE_PATH sem quebrar comportamento original (v1po). Mudanças mínimas e auditáveis.

## Mudanças em v1pq

1. Adicionado suporte a REVP_V1PQ_QUEUE_PATH para override de IN_QUEUE. 2. qid lê execution_queue_id e source_queue_id como fallbacks. 3. Comportamento antigo (queue_id, v1po path) preservado.

## Resultado

Checks: 6. PASS: 6. FAIL: 0.
